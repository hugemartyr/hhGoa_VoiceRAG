"""Query embedding logic for the retrieval pipeline.

Generates dense embeddings (SentenceTransformers) and sparse embeddings (BM25).
Loads models once into memory.
"""

import time
from typing import Tuple

from sentence_transformers import SentenceTransformer

from app.config import settings
from app.ingestion.sparse_encoder import BM25SparseEncoder
from app.models import SparseVectorData


class QueryEmbedder:
    """Singleton-like class for generating query embeddings."""

    def __init__(self):
        self.dense_model = SentenceTransformer(settings.embedding_model)
        self.sparse_encoder = BM25SparseEncoder.load(settings.data_dir)

    def embed_query(self, query: str) -> Tuple[list[float], SparseVectorData, float]:
        """Encode a query into dense and sparse representations.
        
        Returns:
            (dense_vector, sparse_vector, latency_ms)
        """
        start_time = time.perf_counter()
        
        # Dense encoding
        dense_vector = self.dense_model.encode(query, convert_to_numpy=True).tolist()
        
        # Sparse encoding
        sparse_vector = self.sparse_encoder.encode_query(query)
        
        latency = (time.perf_counter() - start_time) * 1000
        return dense_vector, sparse_vector, latency
