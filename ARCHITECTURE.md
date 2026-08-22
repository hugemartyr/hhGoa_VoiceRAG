# Repository Architecture Overview 📂

> **Purpose**: This document provides a high‑level, yet detailed, walkthrough of the **hhGoa_VoiceRAG** codebase. It is intended for new developers joining the project to quickly understand where key pieces live, how they interact, and where to extend or debug functionality.

---

## Table of Contents
1. [Project Layout](#project-layout)
2. [Core Application Flow](#core-application-flow)
3. [Configuration & Settings](#configuration--settings)
4. [Pipeline Components](#pipeline-components)
   - [Input Guard](#input-guard)
   - [Speech‑to‑Text (STT)](#speech-to-text-stt)
   - [Embedder (Dense + Sparse)](#embedder-dense--sparse)
   - [Retriever](#retriever)
   - [Reranker (Cross‑Encoder)](#reranker-cross-encoder)
   - [Output Guard (Confidence & Grounding)](#output-guard-confidence--grounding)
   - [Generator (LLM)](#generator-llm)
5. [Ingestion Sub‑system](#ingestion-subsystem)
6. [Utility Scripts & Tests](#utility-scripts--tests)
7. [Docker & Deployment](#docker--deployment)
8. [Static Assets & UI](#static-assets--ui)
9. [Extending the System](#extending-the-system)
10. [Glossary](#glossary)

---

## Project Layout
```
hhGoa_VoiceRAG/
├── .env.example            # Example env file with keys placeholders
├── Dockerfile              # Container image definition
├── docker-compose.yml      # Spins up Qdrant & optional services
├── README.md               # User‑facing README (this repo)
├── requirements.txt        # Python dependencies
├── scripts/                # Helper scripts (ingest, test, benchmark…)
│   ├── ingest.py           # Data ingestion pipeline
│   ├── test_*.py           # Unit & integration tests
│   └── *.py                # Misc utilities (calibrate_confidence, etc.)
├── data/                   # Persistent data (vocab, IDF, checkpoints)
├── app/                    # **FastAPI** application source
│   ├── __init__.py
│   ├── config.py           # Pydantic Settings (environment variables)
│   ├── main.py             # FastAPI entry‑point, routes, startup hooks
│   ├── models.py           # Pydantic request/response models
│   ├── guardrails/         # Safety & policy checks
│   │   ├── input_guard.py
│   │   └── output_guard.py
│   ├── ingestion/          # Ingestion helpers used by scripts/ingest.py
│   │   └── indexer.py
│   ├── pipeline/           # Core RAG pipeline modules
│   │   ├── stt.py           # Sarvam STT + translation wrapper
│   │   ├── embedder.py      # Dense embeddings + BM25 sparse encoder
│   │   ├── retriever.py    # Qdrant retrieval + parent‑expansion
│   │   ├── reranker.py     # Cross‑encoder scoring & RRF fusion
│   │   ├── generator.py    # Groq LLM call (structured JSON mode)
│   │   └── orchestrator.py # Orchestration of the end‑to‑end flow
│   └── static/             # Front‑end assets served by FastAPI
│       └── index.html
└── results/                # Output artifacts from benchmarks/tests
```

---

## Core Application Flow
1. **User submits an audio query** (via UI – `static/index.html`).
2. **Input Guard** (`app/guardrails/input_guard.py`) validates request size, language, and basic profanity‑checks.
3. **STT + Translation** (`app/pipeline/stt.py`) calls Sarvam AI to transcribe and translate the audio into English text.
4. **Embedding** (`app/pipeline/embedder.py`) creates:
   - Dense vector using `sentence‑transformers/all‑MiniLM‑L6‑v2`.
   - Sparse BM25 representation (offline vocabulary built during ingestion).
5. **Retrieval** (`app/pipeline/retriever.py`) queries Qdrant, applies Reciprocal Rank Fusion (RRF) with the sparse results, and expands parent passages.
6. **Reranker** (`app/pipeline/reranker.py`) (optional, flag‑controlled) re‑scores top passages with `cross‑encoder/ms‑marco‑MiniLM‑L‑6‑v2`.
7. **Output Guard** (`app/guardrails/output_guard.py`) executes two safety checks:
   - **Confidence Gate** – early rejection if the top score is below a calibrated threshold.
   - **Grounding Validation** – forces LLM to emit a JSON payload of citation IDs and a `grounded` flag; non‑grounded answers are discarded.
8. **Generator** (`app/pipeline/generator.py`) sends the curated context to Groq (`openai/gpt‑oss‑20b` by default) and receives a structured answer.
9. **FastAPI** (`app/main.py`) returns the final JSON to the front‑end, which renders the answer and citations.

---

## Configuration & Settings
All runtime configuration lives in **`app/config.py`**, a thin Pydantic `BaseSettings` wrapper. Key groups:
| Category | Important Settings |
|---|---|
| **API Keys** | `sarvam_api_key`, `groq_api_key` |
| **Qdrant** | `qdrant_url`, `qdrant_collection` |
| **Model Names** | `embedding_model`, `reranker_model`, `llm_model` |
| **Retrieval Parameters** | `dense_top_k`, `sparse_top_k`, `rrf_k`, `final_top_k` |
| **Feature Flags** | `enable_reranker`, `enable_sparse_retrieval` |
| **Chunking** | `max_passage_tokens`, `sentence_overlap` |
| **Confidence Gate** | `confidence_threshold`, `confidence_threshold_no_reranker` |
| **Ingestion** | `batch_size`, `max_rows`, `checkpoint_interval` |
| **BM25** | `bm25_k1`, `bm25_b` |
| **Timeouts & Retries** | `max_retries`, `stt_timeout_s`, `llm_timeout_s`, `retrieval_timeout_s`, `reranker_timeout_s` |

Environment variables are loaded from `.env` (or `.env.example` for templates). The singleton `settings = Settings()` is imported throughout the codebase.

---

## Pipeline Components
### Input Guard
- Located at `app/guardrails/input_guard.py`.
- Performs lightweight checks (payload size, UTF‑8 validity, profanity regex) before any heavy processing.
- Returns a 4xx error early, saving compute.

### Speech‑to‑Text (STT)
- `app/pipeline/stt.py` wraps the **Sarvam Saaras v3** API.
- Handles transcription, language detection, and translation to English in a single request.
- Retries with exponential back‑off according to `settings.max_retries`.

### Embedder (Dense + Sparse)
- **Dense**: Uses `sentence‑transformers` to produce 384‑dim vectors.
- **Sparse**: Pre‑computed BM25 vocab & IDF stored under `data/` (generated by `scripts/test_sparse.py`).
- Returns a tuple `(dense_vec, sparse_vector)` for downstream fusion.

### Retriever
- `app/pipeline/retriever.py` queries Qdrant (`settings.qdrant_url`).
- Retrieves `dense_top_k` dense hits and `sparse_top_k` BM25 hits.
- Applies **Reciprocal Rank Fusion (RRF)** with `settings.rrf_k`.
- Performs **parent expansion**: if a child sentence is selected, its original passage is also added to preserve context.

### Reranker (Cross‑Encoder)
- Optional step (`settings.enable_reranker`).
- Scores the top `rrf_k` passages using `cross‑encoder/ms‑marco‑MiniLM‑L‑6‑v2`.
- Selects the best `final_top_k` passages for the LLM.

### Output Guard (Confidence & Grounding)
- **Confidence Gate**: compares the top score (reranker or RRF) against `confidence_threshold` (or `_no_reranker`). Low‑confidence queries are rejected early, returning a friendly *“I’m not sure how to answer that.”* response.
- **Grounding Validation**: the LLM is forced to emit a JSON object containing:
  ```json
  { "citations": ["doc_id1", "doc_id2"], "grounded": true }
  ```
  If `grounded` is false or citation IDs don’t match retrieved passages, the answer is discarded and the request is marked as ungrounded.

### Generator (LLM)
- `app/pipeline/generator.py` calls **Groq** with the selected passages as system prompt and the user query as user prompt.
- Uses **structured‑JSON mode** (requires the model to output a JSON schema) to simplify grounding checks.
- Timeout governed by `settings.llm_timeout_s`.

---

## Ingestion Sub‑system
- **Entry point**: `scripts/ingest.py` (CLI tool).
- Reads the **MSMARCO‑XI** dataset (or a mock subset with `--mock`).
- For each document:
  1. Chunk via `app/pipeline/orchestrator.py` → `embedder` → `retriever` (store in Qdrant).
  2. Build BM25 vocabulary & IDF (`scripts/test_sparse.py`).
  3. Persist checkpoints every `settings.checkpoint_interval` batches.
- Configuration (batch size, max rows, checkpoint interval) lives in `Settings`.

---

## Utility Scripts & Tests
| Script | Description |
|---|---|
| `scripts/ingest.py` | Populate Qdrant with documents. |
| `scripts/test_chunker.py` | Unit tests for passage‑splitting logic. |
| `scripts/test_sparse.py` | Verify BM25 vocab/IDF generation. |
| `scripts/calibrate_confidence.py` | Produce calibrated confidence thresholds used by the Output Guard. |
| `scripts/benchmark.py` | End‑to‑end latency benchmark (produces the table shown in README). |
| `scripts/test_orchestrator.py` | Integration test of the full pipeline orchestration. |

Tests are runnable via `python -m pytest` (pytest listed in `requirements.txt`).

---

## Docker & Deployment
- **Dockerfile** builds a minimal Python image with all dependencies.
- **docker-compose.yml** spins up:
  - `qdrant` container (vector DB).
  - Optional `redis` or other services can be added.
- To run locally:
  ```bash
  docker-compose up -d           # start Qdrant
  python -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  uvicorn app.main:app --host 0.0.0.0 --port 8000
  ```
- Production deployments typically replace the Groq model with a higher‑tier Llama‑3.1 variant and use a managed Qdrant service.

---

## Static Assets & UI
- The minimal front‑end lives under `app/static/`.
- Served by FastAPI's `StaticFiles` mount in `app/main.py`.
- Contains a basic HTML page with a microphone button, a text area for the transcript, and a results pane that renders citations.
- UI is intentionally lightweight – the heavy lifting occurs server‑side.

---

## Extending the System
1. **Add a new model** – Update `config.py` (`embedding_model`, `reranker_model`, or `llm_model`) and ensure the model is compatible with the existing inference wrappers.
2. **Swap the STT provider** – Replace the implementation in `pipeline/stt.py` and adjust the expected JSON schema.
3. **Introduce a new guardrail** – Create a module under `guardrails/` and plug it into `app/main.py` request middleware.
4. **Custom UI** – Extend `app/static/` and update the FastAPI route to serve additional static files or a React/Vite bundle.
5. **Scale out** – Deploy Qdrant as a managed cluster, move the FastAPI container to Kubernetes, and configure environment variables through a secret manager.

---

## Glossary
- **RAG** – Retrieval‑Augmented Generation; combines external knowledge retrieval with LLM generation.
- **BM25** – Classic probabilistic sparse retrieval algorithm.
- **RRF** – Reciprocal Rank Fusion, a simple method to combine rankings from multiple retrieval systems.
- **Groq** – Hosted inference service offering low‑latency LLM endpoints.
- **Sarvam Saaras v3** – Multilingual speech‑to‑text and translation API used for the voice front‑end.
- **Qdrant** – Vector‑search database that stores dense embeddings.
- **FastAPI** – Modern, async Python web framework powering the HTTP API.

---

*Happy hacking! 🎉*
