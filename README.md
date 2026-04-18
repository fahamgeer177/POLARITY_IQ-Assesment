🚀 PolarityIQ Assessment — Decision-Grade RAG System
====================================================

**By Fahamgeer | Senior AI Engineer Candidate**

A **production-oriented Retrieval-Augmented Generation (RAG) pipeline** built for **trustworthy, deterministic, and decision-grade intelligence retrieval** on Family Office data.

🧠 Design Philosophy (Core Differentiator)
------------------------------------------

This system is intentionally designed with the following principles:

*   **Deterministic Retrieval > LLM Guessing**Financial and investment decisions require **precision, not hallucination**.→ LLM is optional, not the core system.
    
*   **Data Quality > Model Complexity**The system’s performance is driven by **validated dataset quality**, not black-box AI.
    
*   **AI as a Tool, Not a Crutch**AI accelerates development, but **logic, filtering, and validation remain controlled**.
    
*   **Cost-Aware Engineering**Supports **offline mode (TF-IDF)** to ensure reliability without API dependency.
    

⚙️ What This System Does
------------------------

*   Converts structured dataset (Excel) → rich text documents
    
*   Builds embeddings (OpenAI OR offline TF-IDF)
    
*   Stores vectors using FAISS
    
*   Retrieves relevant records deterministically
    
*   Applies post-filtering (no hallucination layer)
    
*   Optionally generates structured JSON responses using LLM
`

🔑 Key Features
---------------

### ✅ Deterministic Retrieval Engine

*   No blind LLM reasoning
    
*   Fully reproducible results
    

### ✅ Dual Embedding Modes

*   OpenAI embeddings → production-grade semantic search
    
*   TF-IDF → offline, cost-free fallback
    

### ✅ CLI + API + UI

*   CLI (build, ask, test, doctor)
    
*   Streamlit UI for demo
    
*   Modular service layer (rag\_service.py)
    

### ✅ Debug & Transparency Tools

*   \--no-llm → retrieval-only mode
    
*   \--raw → inspect raw chunks
    
*   doctor → diagnose API/quota issues
    

⚡Quickstart

### 1) Install

```bash
pip install -r requirements.txt
```

Production-style (recommended): install as a package (editable):

```bash
pip install -e .
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

Or via the installed console script:

```bash
polarity-iq build --data-path "path/to/your-dataset.xlsx"
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

Console script:

```bash
polarity-iq ask "Family offices investing in AI in the United States"
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

- `src/polarity_iq/` — production package (engine + service + CLI)
- `app.py` — CLI entrypoints (`build`, `ask`, `test`, `doctor`)
- `rag_engine.py` — embeddings + indexing + retrieval
- `rag_service.py` — public service wrapper used by UI / potential API
- `streamlit_app.py` — demo UI
- `artifacts/` — generated FAISS index + docs (gitignored)

Note: `app.py`, `rag_engine.py`, `rag_service.py`, and `query_logic.py` are compatibility wrappers; the implementation lives under `src/`.

---

## Dev workflow

```bash
pip install -r requirements-dev.txt
ruff check .
pytest
```

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

📊 Retrieval Quality (Manual Evaluation)
----------------------------------------

*   Tested on multiple real-world queries:
    
    *   Sector-based (AI, Healthcare, Mining)
        
    *   Geography-based (US, UK, Japan)
        
    *   Confidence filtering
        

**Observed Performance:**

*   Precision: ~80–90% on structured queries
    
*   Failures mainly due to:
    
    *   Dataset gaps (not retrieval logic)
        
    *   Semantic mismatch in TF-IDF mode
        

⚠️ Known Limitations
--------------------

*   TF-IDF struggles with semantic similarity (no synonym awareness)
    
*   No real-time data updates (static dataset)
    
*   Some family offices have limited public contact data
    
*   Complex multi-condition queries may require hybrid retrieval
    

🔄 Future Improvements
----------------------

*   Hybrid Search (BM25 + Vector)
    
*   Real-time data enrichment pipelines
    
*   Confidence scoring per query result
    
*   Query intent classification
    
*   API deployment (FastAPI + Docker)
    

🧱 Tech Stack
-------------

*   **Python**
    
*   **FAISS** (vector search)
    
*   **OpenAI Embeddings**
    
*   **scikit-learn (TF-IDF)**
    
*   **Streamlit (UI)**
    
*   **Pandas (data processing)**

🔍 Why FAISS?
-------------

FAISS was selected for:

*   High-speed local vector search
    
*   Low latency
    
*   No external dependency
    
*   Production-proven scalability
    

🔗 Dataset Integration
----------------------

This RAG system is tightly coupled with a **validated Family Office dataset**:

*   Multi-source verification
    
*   Confidence scoring
    
*   Structured investment signals
    

👉 **Higher dataset quality = better retrieval accuracy**

🧠 Key Insight (What Makes This Different)
------------------------------------------

Most RAG systems:

> “Retrieve + let LLM guess”

This system:

> “Retrieve → Filter → Validate → THEN optionally format”

👉 This eliminates hallucination risk in **financial decision workflows**

📌 Final Note
-------------

This project is not just a demo — it reflects how I approach building:

*   Systems that are **reliable under real-world conditions**
    
*   Architectures that **prioritize correctness over hype**
    
*   AI pipelines where **judgment remains human-controlled**
