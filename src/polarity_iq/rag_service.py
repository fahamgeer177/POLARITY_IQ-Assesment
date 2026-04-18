from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from polarity_iq.query_logic import compact_results, filter_retrieved, is_found_in_dataset
from polarity_iq.rag_engine import RagEngine, load_store


@dataclass(frozen=True)
class AskResponse:
    results: list[dict[str, Any]]
    not_found: bool
    elapsed_ms: int
    retrieved_before_filter: int
    strict_mode: bool


def ask(
    query: str,
    *,
    artifacts_dir: str = "artifacts",
    k: int = 5,
    strict_mode: bool = True,
    embedding_model: str = "text-embedding-3-small",
    chat_model: str = "gpt-4o-mini",
    no_llm: bool = True,
) -> AskResponse:
    """Deterministic retrieval + deterministic filtering.

    - Uses the stored artifacts backend (OpenAI or TF-IDF) for retrieval.
    - Applies strict deterministic filtering to reduce false positives.
    - Returns structured JSON-ready results (with required fields + evidence).

    no_llm is kept for forward compatibility; UI uses retrieval-only by default.
    """

    start = time.perf_counter()
    store = load_store(artifacts_dir)

    engine = RagEngine(
        embedding_backend=str(store.config.get("embedding_backend", "openai")),
        embedding_model=embedding_model,
        chat_model=chat_model,
    )

    retrieved = engine.query(store, query, k=k)
    retrieved_before_filter = len(retrieved)
    retrieved = filter_retrieved(query, store, retrieved, strict_mode=strict_mode)

    not_found = not is_found_in_dataset(query, retrieved)
    results = [] if not_found else compact_results(retrieved)

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    return AskResponse(
        results=results,
        not_found=not_found,
        elapsed_ms=elapsed_ms,
        retrieved_before_filter=retrieved_before_filter,
        strict_mode=strict_mode,
    )
