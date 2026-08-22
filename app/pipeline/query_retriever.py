"""Query-level dense retrieval against the msmarco_queries collection.

Matches the user's English question to indexed Eng_Query embeddings
and returns pre-written Eng_Answer candidates.
"""

import time
from typing import List

from qdrant_client import QdrantClient

from app.config import settings
from app.models import QueryRetrievalCandidate, QueryRetrievalResult


class QueryRetriever:
    """Dense retrieval over indexed MSMARCO-XI queries."""

    def __init__(self):
        self.client = QdrantClient(url=settings.qdrant_url, timeout=settings.retrieval_timeout_s)
        self.collection_name = settings.qdrant_query_collection

    def search(self, dense_vector: List[float], limit: int = 20) -> QueryRetrievalResult:
        """Retrieve the nearest Eng_Query matches for the user query embedding."""
        start_time = time.perf_counter()

        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=dense_vector,
            limit=limit,
            with_payload=True,
        ).points

        candidates = []
        for hit in hits:
            payload = hit.payload or {}
            candidates.append(
                QueryRetrievalCandidate(
                    query_id=int(payload.get("query_id", 0)),
                    eng_query=str(payload.get("eng_query", "")),
                    eng_answer=str(payload.get("eng_answer", "")),
                    answer=str(payload.get("answer", "")),
                    dense_score=float(hit.score or 0.0),
                    query_type=str(payload.get("query_type", "")),
                )
            )

        latency = (time.perf_counter() - start_time) * 1000
        return QueryRetrievalResult(candidates=candidates, latency_ms=latency)
