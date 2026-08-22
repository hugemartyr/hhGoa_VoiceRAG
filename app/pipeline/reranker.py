"""Cross-encoder reranking for retrieval candidates.

Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` to score (query, context) pairs.
"""

import time
from typing import List

from sentence_transformers import CrossEncoder

from app.config import settings
from app.models import RerankResult, RetrievalCandidate, QueryRetrievalCandidate


class QueryReranker:
    """Singleton-like class for loading and running the cross-encoder."""
    
    def __init__(self):
        self.model = None
        
    def _lazy_load(self):
        if self.model is None and settings.enable_reranker:
            self.model = CrossEncoder(settings.reranker_model, max_length=512)

    def rerank(self, query: str, candidates: List[RetrievalCandidate]) -> RerankResult:
        """Rerank retrieval candidates using a cross-encoder model."""
        start_time = time.perf_counter()
        
        if not settings.enable_reranker or not candidates:
            # If disabled or no candidates, just return them in their current (RRF) order
            return RerankResult(
                candidates=candidates,
                latency_ms=(time.perf_counter() - start_time) * 1000,
                confidence_score=candidates[0].rrf_score if candidates else 0.0
            )

        self._lazy_load()
        
        # Prepare pairs of (Query, Context)
        # We use `context_text` since that's what the LLM will see
        pairs = [[query, c.context_text] for c in candidates]
        
        # Predict logits
        scores = self.model.predict(pairs)
        
        # Update scores and sort
        for candidate, score in zip(candidates, scores):
            candidate.rerank_score = float(score)
            
        candidates.sort(key=lambda x: x.rerank_score, reverse=True)
        
        # Keep only the top-k required for generation
        final_candidates = candidates[:settings.final_top_k]
        
        # The confidence score for the entire retrieval step is the score of the best match
        best_score = final_candidates[0].rerank_score if final_candidates else 0.0

        latency = (time.perf_counter() - start_time) * 1000
        
        return RerankResult(
            candidates=final_candidates,
            latency_ms=latency,
            confidence_score=best_score
        )

    def rerank_queries(
        self, query: str, candidates: List[QueryRetrievalCandidate]
    ) -> tuple[List[QueryRetrievalCandidate], float, float]:
        """Rerank query-level candidates by scoring (user_query, eng_query) pairs.

        Returns:
            (sorted_candidates, latency_ms, confidence_score)
        """
        start_time = time.perf_counter()

        if not candidates:
            return [], (time.perf_counter() - start_time) * 1000, 0.0

        if not settings.enable_reranker:
            best = candidates[0].dense_score
            return candidates[: settings.final_top_k], (time.perf_counter() - start_time) * 1000, best

        self._lazy_load()
        pairs = [[query, c.eng_query] for c in candidates]
        scores = self.model.predict(pairs)

        for candidate, score in zip(candidates, scores):
            candidate.rerank_score = float(score)

        candidates.sort(key=lambda x: x.rerank_score or 0.0, reverse=True)
        final = candidates[: settings.final_top_k]
        best_score = final[0].rerank_score if final else 0.0
        latency = (time.perf_counter() - start_time) * 1000
        return final, latency, best_score
