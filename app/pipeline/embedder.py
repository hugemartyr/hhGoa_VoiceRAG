"""Query embedding logic for the retrieval pipeline.

Generates dense embeddings and optional sparse embeddings. The Vercel runtime
uses FastEmbed to avoid shipping the PyTorch / Transformers stack.
"""

import os
import time
from typing import Tuple

from app.config import settings
from app.ingestion.sparse_encoder import BM25SparseEncoder
from app.models import SparseVectorData


class _FastEmbedModel:
    def __init__(self):
        try:
            from fastembed import TextEmbedding
        except ImportError as e:
            raise ImportError(
                "The 'fastembed' package is required for the default embedding backend. "
                "Install it with: pip install fastembed"
            ) from e

        cache_dir = os.environ.get("FASTEMBED_CACHE_PATH") or os.environ.get("HF_HOME")
        self.model = TextEmbedding(
            model_name=settings.embedding_model,
            cache_dir=cache_dir,
            lazy_load=True,
        )

    def encode(self, text: str) -> list[float]:
        return list(next(self.model.query_embed([text])))


class _SentenceTransformersModel:
    def __init__(self):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "The 'sentence-transformers' package is required when "
                "EMBEDDING_BACKEND=sentence-transformers."
            ) from e

        self.model = SentenceTransformer(settings.embedding_model)

    def encode(self, text: str) -> list[float]:
        return self.model.encode(text, convert_to_numpy=True).tolist()


class QueryEmbedder:
    """Singleton-like class for generating query embeddings."""

    def __init__(self):
        # Respect HF_HOME if set (used to point to a bundled model cache on Vercel)
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            os.environ.setdefault("TRANSFORMERS_CACHE", hf_home)
            os.environ.setdefault("HF_HOME", hf_home)

        backend = settings.embedding_backend.lower().replace("_", "-")
        if backend == "fastembed":
            self.dense_model = _FastEmbedModel()
        elif backend in {"sentence-transformers", "sentence_transformers"}:
            self.dense_model = _SentenceTransformersModel()
        else:
            raise ValueError(
                "Unsupported EMBEDDING_BACKEND. Use 'fastembed' or 'sentence-transformers'."
            )

        self.sparse_encoder = None
        if settings.retrieval_mode == "passage" and settings.enable_sparse_retrieval:
            self.sparse_encoder = BM25SparseEncoder.load(settings.data_dir)

    def embed_query(self, query: str) -> Tuple[list[float], SparseVectorData, float]:
        """Encode a query into dense and sparse representations.
        
        Returns:
            (dense_vector, sparse_vector, latency_ms)
        """
        start_time = time.perf_counter()
        
        dense_vector = self.dense_model.encode(query)
        
        if self.sparse_encoder is not None:
            sparse_vector = self.sparse_encoder.encode_query(query)
        else:
            sparse_vector = SparseVectorData(indices=[], values=[])
        
        latency = (time.perf_counter() - start_time) * 1000
        return dense_vector, sparse_vector, latency
