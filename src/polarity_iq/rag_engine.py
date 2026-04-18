from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None  # type: ignore

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


DEFAULT_SHEET_NAME = "Family Office Intelligence"
DEFAULT_HEADER_ROW = 2
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_EMBEDDING_BACKEND: Literal["openai", "tfidf"] = "openai"


def _require_faiss() -> Any:
    if faiss is None:
        raise RuntimeError("FAISS import failed. Ensure 'faiss-cpu' is installed in your environment.")
    return faiss


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def row_to_text(row: pd.Series) -> str:
    parts = [
        f"Record: {_safe_str(row.get('#'))}",
        f"Name: {_safe_str(row.get('Family Office Name'))}",
        f"Type: {_safe_str(row.get('Family Office Type'))}",
        f"Location: {_safe_str(row.get('Family Office City'))}, {_safe_str(row.get('Family Office State / Region'))}, {_safe_str(row.get('Family Office Country'))}",
        f"Description: {_safe_str(row.get('Family Office Description'))}",
        f"Investment Sectors: {_safe_str(row.get('Investing Sectors'))}",
        f"Investment Thesis: {_safe_str(row.get('Investment Thesis'))}",
        f"AUM: {_safe_str(row.get('Estimated AUM Range'))}",
        f"Domain: {_safe_str(row.get('Family Office Domain'))}",
        f"Website: {_safe_str(row.get('Family Office Website URL'))}",
        f"LinkedIn: {_safe_str(row.get('Corporate LinkedIn'))}",
        f"Signals: {_safe_str(row.get('Methodology Notes'))}",
        f"Confidence: {_safe_str(row.get('Confidence Level'))}",
    ]
    parts = [p for p in parts if not p.endswith(": ") and not p.endswith(":") and p.split(":", 1)[-1].strip()]
    return "\n".join(parts) + "\n"


def load_dataset(
    data_path: str | Path,
    *,
    sheet_name: str = DEFAULT_SHEET_NAME,
    header_row: int = DEFAULT_HEADER_ROW,
) -> pd.DataFrame:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path, sheet_name=sheet_name, header=header_row)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported dataset type: {path.suffix}")

    if "Family Office Name" not in df.columns:
        raise ValueError(
            "Could not find expected column 'Family Office Name'. "
            "If you are loading the provided Excel, ensure header_row=2 and sheet_name='Family Office Intelligence'."
        )

    df = df.dropna(subset=["Family Office Name"]).copy()
    df.reset_index(drop=True, inplace=True)
    return df


@dataclass
class RagStore:
    index: Any
    documents: list[str]
    metas: list[dict[str, Any]]
    config: dict[str, Any]
    vectorizer: TfidfVectorizer | None = None


class RagEngine:
    def __init__(
        self,
        *,
        embedding_backend: Literal["openai", "tfidf"] = DEFAULT_EMBEDDING_BACKEND,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        chat_model: str = DEFAULT_CHAT_MODEL,
        openai_client: Any | None = None,
    ) -> None:
        load_dotenv()

        self.embedding_backend = embedding_backend
        self._api_key = os.getenv("OPENAI_API_KEY")
        self._org_id = os.getenv("OPENAI_ORG_ID") or os.getenv("OPENAI_ORGANIZATION")
        self._project_id = os.getenv("OPENAI_PROJECT_ID") or os.getenv("OPENAI_PROJECT")

        self.embedding_model = embedding_model
        self.chat_model = chat_model

        self.client: Any | None = openai_client
        needs_openai = self.embedding_backend == "openai" or bool(self.chat_model)
        if needs_openai and self.client is None:
            if not self._api_key:
                self.client = None
            else:
                if OpenAI is None:
                    raise RuntimeError("OpenAI SDK is not installed. Install 'openai' or use embedding_backend='tfidf' and --no-llm.")

                client_kwargs: dict[str, Any] = {"api_key": self._api_key}
                if self._org_id:
                    client_kwargs["organization"] = self._org_id
                if self._project_id:
                    client_kwargs["project"] = self._project_id
                self.client = OpenAI(**client_kwargs)  # type: ignore[misc]

    def _require_openai(self) -> Any:
        if self.client is None:
            raise RuntimeError(
                "OpenAI client is not configured. Set OPENAI_API_KEY or use embedding_backend='tfidf' and/or --no-llm."
            )
        return self.client

    def openai_diagnostics(
        self,
        *,
        probe_embeddings: bool = False,
        probe_chat: bool = False,
    ) -> dict[str, Any]:
        has_key = bool(self._api_key)
        key_prefix = (self._api_key or "")[:7] if has_key else ""
        key_suffix = (self._api_key or "")[-4:] if has_key else ""
        info: dict[str, Any] = {
            "has_openai_api_key": has_key,
            "openai_api_key_fingerprint": f"{key_prefix}…{key_suffix}" if has_key else None,
            "openai_org_id_set": bool(self._org_id),
            "openai_project_id_set": bool(self._project_id),
            "embedding_backend": self.embedding_backend,
            "embedding_model": self.embedding_model,
            "chat_model": self.chat_model,
        }

        if not has_key:
            info["openai_check"] = "skipped (no key)"
            return info

        try:
            client = self._require_openai()
            models = client.models.list()
            info["openai_check"] = "ok"
            info["model_count"] = len(getattr(models, "data", []) or [])
        except Exception as exc:
            info["openai_check"] = "error"
            info["openai_error_type"] = type(exc).__name__
            info["openai_error_message"] = str(exc)

        if probe_embeddings:
            try:
                client = self._require_openai()
                _ = client.embeddings.create(model=self.embedding_model, input=["ping"])
                info["probe_embeddings"] = "ok"
            except Exception as exc:
                info["probe_embeddings"] = "error"
                info["probe_embeddings_error_type"] = type(exc).__name__
                info["probe_embeddings_error_message"] = str(exc)

        if probe_chat:
            try:
                client = self._require_openai()
                _ = client.chat.completions.create(
                    model=self.chat_model,
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=1,
                )
                info["probe_chat"] = "ok"
            except Exception as exc:
                info["probe_chat"] = "error"
                info["probe_chat_error_type"] = type(exc).__name__
                info["probe_chat_error_message"] = str(exc)

        return info

    def embed_texts_openai(self, texts: list[str], *, batch_size: int = 64) -> np.ndarray:
        client = self._require_openai()
        vectors: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = client.embeddings.create(model=self.embedding_model, input=batch)
            vectors.extend([item.embedding for item in resp.data])
        arr = np.array(vectors, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] != len(texts):
            raise RuntimeError("Unexpected embeddings shape returned by API")
        return arr

    def embed_texts_tfidf(self, texts: list[str]) -> tuple[np.ndarray, TfidfVectorizer]:
        vectorizer = TfidfVectorizer(stop_words="english")
        mat = vectorizer.fit_transform(texts)
        arr = mat.toarray().astype(np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        arr = arr / norms
        return arr, vectorizer

    def build_faiss_index(self, embeddings: np.ndarray, *, metric: Literal["l2", "ip"] = "l2") -> Any:
        f = _require_faiss()
        if embeddings.size == 0:
            raise ValueError("No embeddings provided")
        dim = int(embeddings.shape[1])
        if metric == "l2":
            index = f.IndexFlatL2(dim)
        elif metric == "ip":
            index = f.IndexFlatIP(dim)
        else:
            raise ValueError(f"Unsupported FAISS metric: {metric}")
        index.add(embeddings)
        return index

    def build_store(
        self,
        df: pd.DataFrame,
        *,
        text_mode: Literal["row_to_text"] = "row_to_text",
    ) -> RagStore:
        if text_mode != "row_to_text":
            raise ValueError("Only text_mode='row_to_text' is supported")

        documents = df.apply(row_to_text, axis=1).tolist()
        metas: list[dict[str, Any]] = []
        cols = list(df.columns)
        for _, row in df.iterrows():
            row_dict = {col: _safe_str(row.get(col)) for col in cols}
            metas.append(
                {
                    "record": _safe_str(row.get("#")),
                    "family_office_name": _safe_str(row.get("Family Office Name")),
                    "family_office_type": _safe_str(row.get("Family Office Type")),
                    "city": _safe_str(row.get("Family Office City")),
                    "state_region": _safe_str(row.get("Family Office State / Region")),
                    "country": _safe_str(row.get("Family Office Country")),
                    "investing_sectors": _safe_str(row.get("Investing Sectors")),
                    "investment_thesis": _safe_str(row.get("Investment Thesis")),
                    "estimated_aum_range": _safe_str(row.get("Estimated AUM Range")),
                    "confidence_level": _safe_str(row.get("Confidence Level")),
                    "row": row_dict,
                }
            )

        vectorizer: TfidfVectorizer | None = None
        if self.embedding_backend == "openai":
            embeddings = self.embed_texts_openai(documents)
            index = self.build_faiss_index(embeddings, metric="l2")
            config = {
                "embedding_backend": "openai",
                "embedding_model": self.embedding_model,
                "faiss_metric": "l2",
            }
        elif self.embedding_backend == "tfidf":
            embeddings, vectorizer = self.embed_texts_tfidf(documents)
            index = self.build_faiss_index(embeddings, metric="ip")
            config = {
                "embedding_backend": "tfidf",
                "embedding_model": "tfidf",
                "faiss_metric": "ip",
            }
        else:
            raise ValueError(f"Unsupported embedding backend: {self.embedding_backend}")

        return RagStore(index=index, documents=documents, metas=metas, config=config, vectorizer=vectorizer)

    def query(self, store: RagStore, query: str, *, k: int = 5) -> list[dict[str, Any]]:
        backend = str(store.config.get("embedding_backend", "openai"))
        if backend == "openai":
            query_vec = self.embed_texts_openai([query])[0]
        elif backend == "tfidf":
            if store.vectorizer is None:
                raise RuntimeError("TF-IDF vectorizer missing from store")
            mat = store.vectorizer.transform([query]).toarray().astype(np.float32)
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            mat = mat / norms
            query_vec = mat[0]
        else:
            raise ValueError(f"Unknown embedding backend in artifacts: {backend}")

        scores, indices = store.index.search(np.array([query_vec], dtype=np.float32), k)

        results: list[dict[str, Any]] = []
        for rank, idx in enumerate(indices[0].tolist()):
            if idx < 0 or idx >= len(store.documents):
                continue
            results.append(
                {
                    "rank": rank + 1,
                    "score": float(scores[0][rank]),
                    "meta": store.metas[idx],
                    "text": store.documents[idx],
                }
            )
        return results

    def generate_answer(self, query: str, retrieved: list[dict[str, Any]]) -> str:
        client = self._require_openai()
        context = [
            {
                "rank": r["rank"],
                "score": r["score"],
                "meta": r["meta"],
                "text": r["text"],
            }
            for r in retrieved
        ]

        prompt = (
            "Use ONLY the provided dataset records (Data).\n"
            "If the information is not found in the dataset, output EXACTLY this string and nothing else:\n"
            "Not found in dataset\n\n"
            f"Query: {query}\n\n"
            "Data (top matches):\n"
            f"{json.dumps(context, ensure_ascii=False)}\n\n"
            "Return a JSON array of matching family offices. Each item must include: "
            "family_office_name, family_office_type, location, investing_sectors, confidence_level, evidence (a short quote from Data.text)."
        )

        resp = client.chat.completions.create(
            model=self.chat_model,
            messages=[{"role": "user", "content": prompt}],
        )
        return cast(str, resp.choices[0].message.content or "")


def save_store(store: RagStore, artifacts_dir: str | Path) -> None:
    f = _require_faiss()
    out_dir = Path(artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    f.write_index(store.index, str(out_dir / "index.faiss"))

    with (out_dir / "config.json").open("w", encoding="utf-8") as fp:
        json.dump(store.config, fp, ensure_ascii=False, indent=2)

    with (out_dir / "docs.jsonl").open("w", encoding="utf-8") as fp:
        for doc, meta in zip(store.documents, store.metas, strict=True):
            fp.write(json.dumps({"text": doc, "meta": meta}, ensure_ascii=False) + "\n")

    if str(store.config.get("embedding_backend")) == "tfidf":
        if store.vectorizer is None:
            raise RuntimeError("TF-IDF vectorizer missing; cannot save")
        joblib.dump(store.vectorizer, out_dir / "vectorizer.joblib")


def load_store(artifacts_dir: str | Path) -> RagStore:
    f = _require_faiss()
    in_dir = Path(artifacts_dir)
    index_path = in_dir / "index.faiss"
    docs_path = in_dir / "docs.jsonl"
    config_path = in_dir / "config.json"

    if not index_path.exists() or not docs_path.exists() or not config_path.exists():
        raise FileNotFoundError(
            f"Artifacts not found in {in_dir}. Expected {index_path.name}, {docs_path.name}, and {config_path.name}."
        )

    index = f.read_index(str(index_path))

    config = json.loads(config_path.read_text(encoding="utf-8"))

    documents: list[str] = []
    metas: list[dict[str, Any]] = []
    with docs_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            obj = json.loads(line)
            documents.append(obj["text"])
            metas.append(obj["meta"])

    vectorizer: TfidfVectorizer | None = None
    if str(config.get("embedding_backend")) == "tfidf":
        vec_path = in_dir / "vectorizer.joblib"
        if not vec_path.exists():
            raise FileNotFoundError(f"Missing TF-IDF vectorizer: {vec_path}")
        vectorizer = joblib.load(vec_path)

    return RagStore(index=index, documents=documents, metas=metas, config=config, vectorizer=vectorizer)
