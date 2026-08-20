"""Calibrate the confidence threshold for the cross-encoder.

This script runs known "in-domain" answerable queries and known
"out-of-domain" unanswerable queries through the retrieval+reranking pipeline.

By comparing the logit scores from the cross-encoder, we can determine
a calibrated `confidence_threshold` to place in our configuration to reject
unanswerable queries before calling the LLM.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.pipeline.embedder import QueryEmbedder
from app.pipeline.retriever import HybridRetriever
from app.pipeline.reranker import QueryReranker

IN_DOMAIN_QUERIES = [
    "What is osmosis?",
    "Where is the Eiffel Tower located?",
    "What is the chemical formula for water?",
    "Tallest building in Paris"
]

OUT_OF_DOMAIN_QUERIES = [
    "What is the capital of Mars?",
    "How do I bake a chocolate cake?",
    "Who won the Superbowl in 2024?",
    "Give me Python code for a binary search tree",
    "fjkdlsafjkdslajfklsdajlk" # gibberish
]


def evaluate_queries(queries: list[str], label: str, embedder, retriever, reranker):
    print(f"\n--- Evaluating {label} Queries ---")
    scores = []
    
    for q in queries:
        # Pipeline execution
        dense, sparse, _ = embedder.embed_query(q)
        retrieval_res = retriever.search(dense, sparse, limit=5)
        
        # We need to make sure we don't crash if nothing is retrieved 
        # (though out of domain usually retrieves *something*, just irrelevant)
        if not retrieval_res.candidates:
            print(f"[{q[:30]}...] -> NO CANDIDATES RETRIEVED (Score: -99.0)")
            scores.append(-99.0)
            continue
            
        rerank_res = reranker.rerank(q, retrieval_res.candidates)
        score = rerank_res.confidence_score
        scores.append(score)
        
        top_context = rerank_res.candidates[0].context_text[:60].replace('\n', ' ')
        print(f"[{q[:30]:<30}] -> Score: {score:>6.2f} | Context: {top_context}...")
        
    avg_score = sum(scores) / len(scores) if scores else -99.0
    print(f"Average {label} Score: {avg_score:.2f}")
    return scores


def main():
    print("Initializing models...")
    embedder = QueryEmbedder()
    retriever = HybridRetriever()
    reranker = QueryReranker()
    print("Models initialized.\n")

    in_domain_scores = evaluate_queries(IN_DOMAIN_QUERIES, "IN-DOMAIN", embedder, retriever, reranker)
    ood_scores = evaluate_queries(OUT_OF_DOMAIN_QUERIES, "OUT-OF-DOMAIN", embedder, retriever, reranker)
    
    min_in_domain = min(in_domain_scores) if in_domain_scores else 0.0
    max_ood = max(ood_scores) if ood_scores else 0.0
    
    print("\n" + "="*50)
    print("CALIBRATION RESULTS")
    print("="*50)
    print(f"Lowest In-Domain Score:      {min_in_domain:.2f}")
    print(f"Highest Out-of-Domain Score: {max_ood:.2f}")
    
    if min_in_domain > max_ood:
        suggested = max_ood + ((min_in_domain - max_ood) / 2)
        print(f"\n✅ Perfect separation! Suggested threshold: {suggested:.2f}")
    else:
        print("\n⚠️ Overlap detected. You may need to tune the threshold carefully.")
        print(f"Suggested safe threshold (favoring recall): {min_in_domain:.2f}")
        print(f"Suggested strict threshold (favoring precision): {max_ood + 0.5:.2f}")


if __name__ == "__main__":
    main()
