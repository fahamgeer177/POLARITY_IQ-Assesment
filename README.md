# Polarity IQ Assessment — Minimal RAG (FAISS + deterministic filtering)

A minimal Retrieval-Augmented Generation (RAG) pipeline built for **trustworthy, reproducible retrieval**.

**What it does**

- Loads an Excel dataset and converts each row into a rich text “document”
- Builds embeddings (OpenAI embeddings *or* offline TF‑IDF)
- Stores vectors in FAISS
- Retrieves top‑k matches deterministically
- Applies deterministic post-filtering (no LLM reasoning)
- Optionally produces a grounded, structured JSON answer using only retrieved rows

---

## Quickstart

### 1) Install

```bash
pip install -r requirements.txt
```

### 2) Configure (optional for offline mode)

Copy `.env.example` → `.env` and set:

```bash
OPENAI_API_KEY=...
```

If you use multiple OpenAI orgs/projects (project-scoped keys), you may also set:

- `OPENAI_ORG_ID`
- `OPENAI_PROJECT_ID`

### 3) Build the index

This repo expects your dataset locally (the Excel file is intentionally not tracked in git).

OpenAI embeddings:

```bash
python app.py build --data-path "path/to/your-dataset.xlsx"
```

Offline TF‑IDF embeddings (no API quota required):

```bash
python app.py build --embedding-backend tfidf --data-path "path/to/your-dataset.xlsx"
```

Artifacts are written to `./artifacts/` (and are gitignored by default).

---

## Ask a question

Default mode (may use chat model for a structured answer):

```bash
python app.py ask "Family offices investing in AI in the United States"
```

Retrieval-only (no chat model call):

```bash
python app.py ask "Family offices investing in AI in the United States" --no-llm
```

`--no-llm` prints compact JSON: key fields + short evidence + the full dataset `row` payload.

Inspect raw retrieved text blocks (debugging):

```bash
python app.py ask "Family offices investing in AI in the United States" --raw
```

---

## Suggested tests

```bash
python app.py test
```

Retrieval-only mode:

```bash
python app.py test --no-llm
```

---

## Demo UI (Streamlit)

The Streamlit UI calls a local Python wrapper (`ask(query)`) that runs deterministic retrieval + filtering.

1) Build artifacts (offline TF‑IDF recommended if OpenAI quota is unavailable):

```bash
python app.py build --embedding-backend tfidf --data-path "path/to/your-dataset.xlsx"
```

2) Start the UI:

```bash
streamlit run streamlit_app.py
```

Response JSON format: see [sample_response.json](sample_response.json).

---

## Repo layout

- `app.py` — CLI entrypoints (`build`, `ask`, `test`, `doctor`)
- `rag_engine.py` — embeddings + indexing + retrieval
- `rag_service.py` — public service wrapper used by UI / potential API
- `streamlit_app.py` — demo UI
- `artifacts/` — generated FAISS index + docs (gitignored)

---

## Notes

- The Excel’s real header row is `header_row=2` (0-based) on the `Family Office Intelligence` sheet.
- Defaults: `text-embedding-3-small` (embeddings) and `gpt-4o-mini` (chat). Override via `--embedding-model` / `--chat-model`.

---

## Troubleshooting OpenAI quota

If you see `429 insufficient_quota`, it means **API billing/quota** (platform.openai.com) is not available for the org/project tied to your API key (separate from ChatGPT credits).

Run:

```bash
python app.py doctor
```

Definitive probe of quota-spending endpoints:

```bash
python app.py doctor --probe-embeddings --probe-chat
```

If probes fail with `insufficient_quota`, check:

- You created the key in the correct **Project** with billing enabled
- Project **budget/spend limits** are not $0 / not exhausted
- The organization has an active billing method for the API
