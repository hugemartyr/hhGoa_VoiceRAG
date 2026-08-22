"""Query-level dense retrieval against the msmarco_queries collection.

Matches the user's English question to indexed Eng_Query embeddings
and returns pre-written Eng_Answer candidates.
"""

import time
from typing import List

import httpx

from app.config import settings
from app.models import QueryRetrievalCandidate, QueryRetrievalResult


class QueryRetriever:
    """Dense retrieval over indexed MSMARCO-XI queries."""

    def __init__(self):
        self.base_url = settings.qdrant_url.rstrip("/")
        self.collection_name = settings.qdrant_query_collection
        self.headers = {}
        if settings.qdrant_api_key:
            self.headers["api-key"] = settings.qdrant_api_key

    def search(self, dense_vector: List[float], limit: int = 20) -> QueryRetrievalResult:
        """Retrieve the nearest Eng_Query matches for the user query embedding."""
        start_time = time.perf_counter()

        url = f"{self.base_url}/collections/{self.collection_name}/points/query"
        payload = {
            "query": dense_vector,
            "limit": limit,
            "with_payload": True,
        }
        response = httpx.post(
            url,
            headers=self.headers,
            json=payload,
            timeout=settings.retrieval_timeout_s,
        )
        response.raise_for_status()

        result = response.json().get("result", {})
        hits = result.get("points", result if isinstance(result, list) else [])

        candidates = []
        for hit in hits:
            payload = hit.get("payload") or {}
            candidates.append(
                QueryRetrievalCandidate(
                    query_id=int(payload.get("query_id", 0)),
                    eng_query=str(payload.get("eng_query", "")),
                    eng_answer=str(payload.get("eng_answer", "")),
                    answer=str(payload.get("answer", "")),
                    dense_score=float(hit.get("score") or 0.0),
                    query_type=str(payload.get("query_type", "")),
                )
            )

        latency = (time.perf_counter() - start_time) * 1000
        return QueryRetrievalResult(candidates=candidates, latency_ms=latency)
