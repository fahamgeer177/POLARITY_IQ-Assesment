from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from polarity_iq.query_logic import compact_results, filter_retrieved, is_found_in_dataset
from polarity_iq.rag_engine import RagEngine, load_dataset, load_store, save_store


def cmd_build(args: argparse.Namespace) -> int:
    engine = RagEngine(
        embedding_backend=args.embedding_backend,
        embedding_model=args.embedding_model,
        chat_model=args.chat_model,
    )
    df = load_dataset(args.data_path, sheet_name=args.sheet_name, header_row=args.header_row)
    store = engine.build_store(df)
    save_store(store, args.artifacts_dir)
    print(
        f"Built index with {len(store.documents)} records ({store.config.get('embedding_backend')}) -> {args.artifacts_dir}"
    )
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    store = load_store(args.artifacts_dir)
    if args.embedding_backend and args.embedding_backend != store.config.get("embedding_backend"):
        raise SystemExit(
            f"Embedding backend mismatch: artifacts use '{store.config.get('embedding_backend')}', but you requested '{args.embedding_backend}'."
        )

    engine = RagEngine(
        embedding_backend=str(store.config.get("embedding_backend", "openai")),
        embedding_model=args.embedding_model,
        chat_model=args.chat_model,
    )
    retrieved = engine.query(store, args.query, k=args.k)
    retrieved = filter_retrieved(args.query, store, retrieved, strict_mode=not args.loose)

    if args.raw:
        print(json.dumps(retrieved, ensure_ascii=False, indent=2))
        return 0

    if not is_found_in_dataset(args.query, retrieved):
        print("Not found in dataset")
        return 0

    if args.no_llm:
        print(json.dumps(compact_results(retrieved), ensure_ascii=False, indent=2))
        return 0

    answer = engine.generate_answer(args.query, retrieved)
    print(answer)
    return 0


def cmd_test(args: argparse.Namespace) -> int:
    queries = [
        "Family offices investing in AI in the United States",
        "Which family offices focus on healthcare investments?",
        "Family offices with high confidence data",
    ]

    store = load_store(args.artifacts_dir)
    engine = RagEngine(
        embedding_backend=str(store.config.get("embedding_backend", "openai")),
        embedding_model=args.embedding_model,
        chat_model=args.chat_model,
    )

    for q in queries:
        print("\n=== QUERY ===")
        print(q)
        retrieved = engine.query(store, q, k=args.k)
        retrieved = filter_retrieved(q, store, retrieved, strict_mode=not args.loose)
        print("\n--- TOP MATCH METAS ---")
        for r in retrieved:
            meta = r["meta"]
            print(
                f"#{r['rank']} score={r['score']:.4f} :: {meta.get('family_office_name')} :: {meta.get('investing_sectors')} :: {meta.get('confidence_level')}"
            )
        if args.no_llm:
            continue

        print("\n--- LLM ANSWER ---")
        print(engine.generate_answer(q, retrieved))

    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    engine = RagEngine(
        embedding_backend="openai",
        embedding_model=args.embedding_model,
        chat_model=args.chat_model,
    )
    report = engine.openai_diagnostics(
        probe_embeddings=bool(args.probe_embeddings),
        probe_chat=bool(args.probe_chat),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))

    combined_msgs = "\n".join(
        [
            str(report.get("openai_error_message") or ""),
            str(report.get("probe_embeddings_error_message") or ""),
            str(report.get("probe_chat_error_message") or ""),
        ]
    )
    if "insufficient_quota" in combined_msgs or "exceeded your current quota" in combined_msgs:
        print(
            "\nNext steps for 'insufficient_quota':\n"
            "- Confirm you're using the OpenAI API (platform.openai.com) billing, not ChatGPT credits.\n"
            "- Ensure the API key belongs to the org/project that has billing enabled and a non-zero budget/spend limit.\n"
            "- If models list works but probes fail, the key is valid but the project likely has no budget/billing enabled.\n"
            "- Create a NEW API key inside the billed project (recommended).\n"
            "- If you must target a specific org/project explicitly, set OPENAI_PROJECT_ID / OPENAI_ORG_ID in .env.\n"
            "- As a workaround, use: python app.py build --embedding-backend tfidf (offline)"
        )

    if "invalid_request_error" in combined_msgs or "Invalid" in combined_msgs:
        print(
            "\nIf you see invalid_request_error, that's usually request format/model name. "
            "For quota errors (429 insufficient_quota), message format is almost never the cause."
        )

    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Minimal RAG pipeline (Excel -> embeddings -> FAISS -> grounded answers)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument(
            "--embedding-backend",
            default=None,
            choices=["openai", "tfidf"],
            help="Embedding backend. For ask/test it must match built artifacts. Use tfidf if OpenAI quota is unavailable.",
        )
        sp.add_argument(
            "--embedding-model",
            default="text-embedding-3-small",
            help="OpenAI embedding model",
        )
        sp.add_argument(
            "--chat-model",
            default="gpt-4o-mini",
            help="OpenAI chat model for final answer",
        )
        sp.add_argument(
            "--artifacts-dir",
            default=str(Path("artifacts")),
            help="Directory to store/load FAISS index + docs",
        )
        sp.add_argument(
            "--loose",
            action="store_true",
            help="Disable strict mode filtering (more semantic, more false positives)",
        )

    b = sub.add_parser("build", help="Build FAISS index from dataset")
    add_common(b)
    b.set_defaults(embedding_backend="openai")
    b.add_argument(
        "--data-path",
        default="FO-Decision-Grade-Dataset-Fahamgeer-v3-FINAL.xlsx",
        help="Path to dataset (.xlsx or .csv)",
    )
    b.add_argument(
        "--sheet-name",
        default="Family Office Intelligence",
        help="Excel sheet name",
    )
    b.add_argument(
        "--header-row",
        type=int,
        default=2,
        help="0-based header row index for Excel",
    )
    b.set_defaults(func=cmd_build)

    a = sub.add_parser("ask", help="Ask a question using an existing index")
    add_common(a)
    a.add_argument("query", help="Natural-language question")
    a.add_argument("-k", type=int, default=5, help="Number of records to retrieve")
    a.add_argument(
        "--raw",
        action="store_true",
        help="Print raw retrieved docs/metas instead of LLM answer",
    )
    a.add_argument(
        "--no-llm",
        action="store_true",
        help="Do not call chat model; only print retrieved results",
    )
    a.set_defaults(func=cmd_ask)

    t = sub.add_parser("test", help="Run the 3 suggested test queries")
    add_common(t)
    t.add_argument("-k", type=int, default=5, help="Number of records to retrieve")
    t.add_argument(
        "--no-llm",
        action="store_true",
        help="Do not call chat model; only print retrieval diagnostics",
    )
    t.set_defaults(func=cmd_test)

    d = sub.add_parser("doctor", help="Diagnose OpenAI configuration/quota issues")
    add_common(d)
    d.add_argument(
        "--probe-embeddings",
        action="store_true",
        help="Also make a tiny embeddings request (consumes a small amount of quota)",
    )
    d.add_argument(
        "--probe-chat",
        action="store_true",
        help="Also make a tiny chat request (consumes a small amount of quota)",
    )
    d.set_defaults(func=cmd_doctor)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return int(args.func(args))
