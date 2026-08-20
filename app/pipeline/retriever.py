"""Hybrid retrieval and context expansion logic.

Performs:
  1. Dense + Sparse concurrent querying in Qdrant.
  2. Reciprocal Rank Fusion (RRF).
  3. Parent Context Expansion (fetches parent passages for matching sentences).
  4. Deduplication of identical parents.
"""

import time
from typing import Dict, List, Set

from qdrant_client import QdrantClient
from qdrant_client import models as qmodels

from app.config import settings
from app.models import RetrievalCandidate, RetrievalResult, SparseVectorData


class HybridRetriever:
    def __init__(self):
        self.client = QdrantClient(url=settings.qdrant_url, timeout=settings.retrieval_timeout_s)
        self.collection_name = settings.qdrant_collection

    def _rrf_fusion(self, dense_results, sparse_results, k: int = 60) -> List[Dict]:
        """Reciprocal Rank Fusion.
        
        RRF_score = sum(1 / (k + rank)) for each ranking list the item appears in.
        """
        scores: Dict[str, float] = {}
        payloads: Dict[str, dict] = {}
        
        for rank, hit in enumerate(dense_results, start=1):
            chunk_id = str(hit.id)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + (1.0 / (k + rank))
            if chunk_id not in payloads:
                payloads[chunk_id] = hit.payload
                
        for rank, hit in enumerate(sparse_results, start=1):
            chunk_id = str(hit.id)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + (1.0 / (k + rank))
            if chunk_id not in payloads:
                payloads[chunk_id] = hit.payload
                
        # Sort by RRF score descending
        sorted_hits = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        
        fused = []
        for chunk_id, rrf_score in sorted_hits:
            fused.append({
                "chunk_id": chunk_id,
                "score": rrf_score,
                "payload": payloads[chunk_id]
            })
            
        return fused

    def search(self, dense_vector: List[float], sparse_vector: SparseVectorData, limit: int = 20) -> RetrievalResult:
        """Execute hybrid search with parent expansion and deduplication."""
        start_time = time.perf_counter()

        # Batch request for both dense and sparse to minimize network roundtrips
        # We only want to retrieve `is_retrieval_unit = True` chunks
        filter_cond = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="is_retrieval_unit",
                    match=qmodels.MatchValue(value=True)
                )
            ]
        )

        requests = []
        
        # Dense Query
        requests.append(qmodels.QueryRequest(
            using="dense",
            query=dense_vector,
            filter=filter_cond,
            limit=limit,
            with_payload=True
        ))
        
        # Sparse Query
        # If the query had no recognized BM25 terms, sparse vector will be empty.
        if sparse_vector.indices and settings.enable_sparse_retrieval:
            requests.append(qmodels.QueryRequest(
                using="sparse",
                query=qmodels.SparseVector(
                    indices=sparse_vector.indices,
                    values=sparse_vector.values
                ),
                filter=filter_cond,
                limit=limit,
                with_payload=True
            ))

        # Perform searches
        batch_results = self.client.query_batch_points(
            collection_name=self.collection_name,
            requests=requests
        )

        dense_hits = batch_results[0].points
        sparse_hits = batch_results[1].points if len(batch_results) > 1 else []

        # Fuse rankings via RRF
        fused_results = self._rrf_fusion(dense_hits, sparse_hits, k=settings.rrf_k)
        
        # Process parent expansion and deduplication
        # We need to fetch missing parents by ID from Qdrant
        parent_ids_to_fetch = set()
        for hit in fused_results:
            payload = hit["payload"]
            if payload.get("chunk_type") == "sentence" and payload.get("parent_id"):
                parent_ids_to_fetch.add(payload["parent_id"])
                
        fetched_parents = {}
        if parent_ids_to_fetch:
            parent_points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=list(parent_ids_to_fetch),
                with_payload=True,
                with_vectors=False
            )
            for p in parent_points:
                fetched_parents[str(p.id)] = p.payload

        final_candidates = []
        seen_contexts: Set[str] = set()

        for hit in fused_results:
            payload = hit["payload"]
            chunk_type = payload.get("chunk_type", "passage")
            
            context_text = payload.get("text", "")
            parent_id = payload.get("parent_id")
            
            if chunk_type == "sentence" and parent_id:
                if parent_id in fetched_parents:
                    context_text = fetched_parents[parent_id].get("text", "")
            
            # Deduplicate - if multiple sentences from the same parent match,
            # we only yield the parent context once (with the highest RRF score).
            context_hash = hash(context_text)
            if context_hash in seen_contexts:
                continue
                
            seen_contexts.add(context_hash)
            
            final_candidates.append(
                RetrievalCandidate(
                    chunk_id=hit["chunk_id"],
                    chunk_text=payload.get("text", ""),
                    context_text=context_text,
                    rrf_score=hit["score"],
                    chunk_type=chunk_type,
                    parent_id=parent_id,
                    metadata=payload
                )
            )
            
            if len(final_candidates) >= settings.final_top_k * 4:
                # Keep plenty of candidates for the reranker, but not too many
                break

        latency = (time.perf_counter() - start_time) * 1000
        
        return RetrievalResult(
            candidates=final_candidates,
            latency_ms=latency
        )
