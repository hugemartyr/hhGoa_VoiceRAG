"""Centralized configuration using Pydantic BaseSettings.

All settings are loaded from environment variables (via .env file).
Defaults are provided for non-secret values.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # --- API Keys ---
    sarvam_api_key: str = Field(default="", description="Sarvam AI API key for Saaras v3 STT")
    groq_api_key: str = Field(default="", description="Groq API key for LLM inference")

    # --- Qdrant ---
    qdrant_url: str = Field(default="http://localhost:6333", description="Qdrant server URL")
    qdrant_collection: str = Field(default="msmarco_rag", description="Qdrant collection for passage chunks")
    qdrant_query_collection: str = Field(
        default="msmarco_queries",
        description="Qdrant collection for query-level retrieval (Eng_Query → Eng_Answer)",
    )
    qdrant_api_key: str = Field(default="", description="Qdrant Cloud API key (if using Qdrant Cloud)")
    retrieval_mode: str = Field(
        default="query",
        description="Retrieval strategy: 'query' (fast path, no LLM) or 'passage' (classic RAG with LLM)",
    )

    # --- Model Names ---
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Sentence-transformers model for dense embeddings (384-dim)",
    )
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model for reranking",
    )
    llm_model: str = Field(
        default="openai/gpt-oss-20b",
        description="LLM model identifier in Groq.",
    )

    # --- Retrieval Parameters ---
    dense_top_k: int = Field(default=20, description="Number of dense retrieval results")
    sparse_top_k: int = Field(default=20, description="Number of sparse BM25 retrieval results")
    rrf_k: int = Field(default=60, description="RRF constant k for score fusion")
    final_top_k: int = Field(default=5, description="Number of context passages sent to LLM")

    # --- Feature Flags ---
    enable_reranker: bool = Field(default=True, description="Enable cross-encoder reranking")
    enable_sparse_retrieval: bool = Field(default=True, description="Enable BM25 sparse retrieval")

    # --- Chunking Parameters ---
    max_passage_tokens: int = Field(
        default=256,
        description="Passages exceeding this token count are split into sentence children",
    )
    sentence_overlap: int = Field(
        default=1,
        description="Number of overlapping sentences between consecutive sentence chunks",
    )

    # --- Confidence Gate ---
    confidence_threshold: float = Field(
        default=0.0,
        description=(
            "Calibrated threshold for answerability gate. "
            "Applied to top reranker score (or top RRF score if reranker disabled). "
            "Default is a placeholder — must be calibrated via scripts/calibrate_confidence.py"
        ),
    )
    confidence_threshold_no_reranker: float = Field(
        default=0.3,
        description="Fallback confidence threshold when reranker is disabled (calibrated separately)",
    )

    # --- Ingestion ---
    batch_size: int = Field(default=1000, description="Batch size for Qdrant upserts")
    max_rows: Optional[int] = Field(
        default=None,
        description="Max dataset rows to process (None = full dataset)",
    )
    checkpoint_interval: int = Field(
        default=100,
        description="Save checkpoint every N batches during ingestion",
    )

    # --- BM25 Parameters ---
    bm25_k1: float = Field(default=1.2, description="BM25 k1 parameter")
    bm25_b: float = Field(default=0.75, description="BM25 b parameter")

    # --- Retry / Timeout ---
    max_retries: int = Field(default=2, description="Max retry attempts for external API calls")
    stt_timeout_s: float = Field(default=5.0, description="Sarvam STT API timeout (seconds)")
    llm_timeout_s: float = Field(default=3.0, description="Groq LLM API timeout (seconds)")
    retrieval_timeout_s: float = Field(default=0.5, description="Qdrant retrieval timeout (seconds)")
    reranker_timeout_s: float = Field(default=0.2, description="Reranker inference timeout (seconds)")

    # --- Data Paths ---
    data_dir: str = Field(default="data", description="Directory for persisted artifacts (vocab, IDF, checkpoints)")

    # --- Sarvam STT ---
    sarvam_api_url: str = Field(
        default="https://api.sarvam.ai/speech-to-text-translate",
        description="Sarvam Saaras v3 API endpoint",
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }


# Singleton instance — import this throughout the app
settings = Settings()
