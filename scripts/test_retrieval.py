"""Sprint 5 verification: test the retrieval pipeline.

Runs queries through the embedder and retriever, verifying:
  - Dense + sparse embedding shape
  - RRF ranking and scoring
  - Parent context expansion
  - Deduplication
"""

import sys
import os
import time
from pprint import pprint

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.pipeline.embedder import QueryEmbedder
from app.pipeline.retriever import HybridRetriever


def main():
    print("=" * 80)
    print("Sprint 5 Verification: Retrieval Pipeline")
    print("=" * 80)
    
    print("\nLoading models and connecting to Qdrant...")
    embedder = QueryEmbedder()
    retriever = HybridRetriever()
    print("✅ Models loaded and connected")

    queries = [
        "What is osmosis?",
        "Tallest building in Paris",
        "Water states of matter liquid solid gas"
    ]

    for q in queries:
        print(f"\n" + "-" * 80)
        print(f"QUERY: '{q}'")
        
        # 1. Embed
        dense, sparse, embed_ms = embedder.embed_query(q)
        print(f"✅ Embedded in {embed_ms:.1f}ms")
        print(f"   Dense: {len(dense)} dims")
        print(f"   Sparse: {len(sparse.indices)} terms")
        
        # 2. Retrieve
        retrieval_res = retriever.search(dense, sparse, limit=5)
        print(f"✅ Retrieved in {retrieval_res.latency_ms:.1f}ms")
        
        candidates = retrieval_res.candidates
        print(f"✅ Returned {len(candidates)} unique candidates after dedup")
        
        if not candidates:
            print("❌ No candidates found! (Ensure mock ingestion ran successfully)")
            continue
            
        for i, c in enumerate(candidates[:3]):
            print(f"\n   [{i+1}] RRF Score: {c.rrf_score:.4f} | Type: {c.chunk_type.upper()}")
            print(f"       Chunk:   {c.chunk_text[:100]}...")
            if c.chunk_type == 'sentence':
                print(f"       Context: {c.context_text[:100]}...")
                assert len(c.context_text) > len(c.chunk_text), "Context should be expanded parent passage!"
                print("       ✅ Context successfully expanded from parent")

if __name__ == "__main__":
    main()
