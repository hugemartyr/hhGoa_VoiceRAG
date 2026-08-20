"""Benchmark script to measure and profile the end-to-end RAG latency.

Fires a series of valid, unanswerable, and edge-case queries through the PipelineOrchestrator
and computes statistical latency percentiles (P50, P90, P99) for each stage to prove
sub-2-second target compliance.
"""

import sys
import os
import time
import statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.pipeline.orchestrator import PipelineOrchestrator

QUERIES = [
    "What is the chemical formula for water?",
    "Tallest building in Paris",
    "Where is the Eiffel Tower located?",
    "Explain the process of photosynthesis.",
    "Who is the current president of the United States?",
    "Tell me about the chemical formula for photosynthesis on Mars",
    "How do I bake a chocolate cake?",
    "fjkdlsafjkdslajfklsdajlk",
    "hi",
    "Water cycle",
]

def format_percentiles(latencies):
    if not latencies:
        return "N/A"
    latencies.sort()
    p50 = statistics.median(latencies)
    
    # Calculate P90 manually or use a standard formula
    p90_idx = int(len(latencies) * 0.90)
    p90_idx = min(p90_idx, len(latencies) - 1)
    p90 = latencies[p90_idx]
    
    p99_idx = int(len(latencies) * 0.99)
    p99_idx = min(p99_idx, len(latencies) - 1)
    p99 = latencies[p99_idx]
    
    return f"P50: {p50:>5.1f}ms | P90: {p90:>5.1f}ms | P99: {p99:>5.1f}ms"

def main():
    print("Initializing Pipeline for Benchmarking...")
    orchestrator = PipelineOrchestrator()
    print("Warming up models...")
    # Warmup query
    orchestrator.process_query("What is photosynthesis?")
    
    print("\nStarting Benchmark Run (10 queries x 3 iterations)...")
    
    # Run multiple iterations to gather more data points
    all_queries = QUERIES * 3
    
    metrics = {
        "input_guard_ms": [],
        "embedding_ms": [],
        "retrieval_ms": [],
        "reranking_ms": [],
        "confidence_gate_ms": [],
        "generation_ms": [],
        "grounding_ms": [],
        "rag_total_ms": [],
    }
    
    success_count = 0
    reject_count = 0
    
    start_time = time.perf_counter()
    
    for i, q in enumerate(all_queries):
        res = orchestrator.process_query(q)
        if res.rejected:
            reject_count += 1
        else:
            success_count += 1
            
        l = res.latency
        metrics["input_guard_ms"].append(l.input_guard_ms)
        metrics["rag_total_ms"].append(l.rag_total_ms)
        
        # Only log stages that actually ran (e.g. fast-rejections skip generation)
        if l.embedding_ms > 0: metrics["embedding_ms"].append(l.embedding_ms)
        if l.retrieval_ms > 0: metrics["retrieval_ms"].append(l.retrieval_ms)
        if l.reranking_ms is not None and l.reranking_ms > 0: metrics["reranking_ms"].append(l.reranking_ms)
        if l.confidence_gate_ms > 0: metrics["confidence_gate_ms"].append(l.confidence_gate_ms)
        if l.generation_ms > 0: metrics["generation_ms"].append(l.generation_ms)
        if l.grounding_ms > 0: metrics["grounding_ms"].append(l.grounding_ms)
        
        # Simple progress bar
        sys.stdout.write(f"\rProcessed {i+1}/{len(all_queries)}")
        sys.stdout.flush()
        
    total_time = time.perf_counter() - start_time
    
    print("\n\n" + "="*60)
    print("⏱️  LATENCY BENCHMARK RESULTS")
    print("="*60)
    print(f"Total Queries Processed: {len(all_queries)}")
    print(f"Time Elapsed:            {total_time:.2f}s")
    print(f"Queries / Second:        {len(all_queries)/total_time:.2f} qps")
    print(f"Successful Generations:  {success_count}")
    print(f"Early Rejections (Fast): {reject_count}")
    
    print("\n--- Latency Breakdown (Percentiles) ---")
    print(f"{'Input Guard:':<18} {format_percentiles(metrics['input_guard_ms'])}")
    print(f"{'Embedding:':<18} {format_percentiles(metrics['embedding_ms'])}")
    print(f"{'Retrieval:':<18} {format_percentiles(metrics['retrieval_ms'])}")
    print(f"{'Reranking:':<18} {format_percentiles(metrics['reranking_ms'])}")
    print(f"{'Confidence Gate:':<18} {format_percentiles(metrics['confidence_gate_ms'])}")
    print(f"{'LLM Generation:':<18} {format_percentiles(metrics['generation_ms'])}")
    print(f"{'Grounding Check:':<18} {format_percentiles(metrics['grounding_ms'])}")
    print("-" * 60)
    print(f"{'TOTAL RAG FLOW:':<18} {format_percentiles(metrics['rag_total_ms'])}")
    print("="*60)

if __name__ == "__main__":
    main()
