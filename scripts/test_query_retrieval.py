"""Sprint verification: test query-level retrieval (Option B fast path)."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.pipeline.embedder import QueryEmbedder
from app.pipeline.query_retriever import QueryRetriever


def main():
    print("Query-Level Retrieval Verification")
    print("=" * 50)

    embedder = QueryEmbedder()
    retriever = QueryRetriever()

    queries = [
        "What is the chemical formula for water?",
        "Where is the Eiffel Tower located?",
        "Explain photosynthesis",
    ]

    for q in queries:
        print(f"\nQuery: {q}")
        dense, _, embed_ms = embedder.embed_query(q)
        result = retriever.search(dense, limit=3)
        print(f"  Embed: {embed_ms:.1f}ms | Retrieve: {result.latency_ms:.1f}ms")

        if not result.candidates:
            print("  ❌ No candidates returned")
            continue

        best = result.candidates[0]
        print(f"  ✅ Top match (query_id={best.query_id}, score={best.dense_score:.4f})")
        print(f"     Eng_Query:  {best.eng_query[:80]}...")
        print(f"     Eng_Answer: {best.eng_answer[:120]}...")


if __name__ == "__main__":
    main()
