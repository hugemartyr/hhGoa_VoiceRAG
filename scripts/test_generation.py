"""Sprint 7 verification: test the LLM Generation and Grounding pipeline.

Queries Groq to generate a JSON response with citations, and then
passes it through the programmatic grounding checker.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models import RetrievalCandidate
from app.pipeline.generator import GroqGenerator
from app.guardrails.output_guard import OutputGuard


def test_generation():
    print("=" * 80)
    print("Sprint 7 Verification: LLM Generation & Grounding")
    print("=" * 80)

    generator = GroqGenerator()

    # Mock candidates
    candidates = [
        RetrievalCandidate(
            chunk_id="c1",
            chunk_text="The capital of France is Paris.",
            context_text="Paris is the capital and most populous city of France.",
            rrf_score=0.1,
            rerank_score=2.5,
            chunk_type="passage"
        ),
        RetrievalCandidate(
            chunk_id="c2",
            chunk_text="It is located on the Seine River.",
            context_text="The city is located on the Seine River, in the north of the country.",
            rrf_score=0.08,
            rerank_score=1.5,
            chunk_type="passage"
        )
    ]

    # Test 1: Answerable Query
    query1 = "What is the capital of France and what river is it on?"
    print(f"\nQUERY 1: '{query1}'")
    ans1 = generator.generate(query1, candidates)
    
    print(f"Generated Answer: {ans1.answer}")
    print(f"Grounded Flag: {ans1.grounded}")
    print(f"Confidence: {ans1.confidence}")
    print(f"Sources Cited: {ans1.sources}")
    print(f"Generation Latency: {ans1.latency_ms:.1f}ms")

    guard1 = OutputGuard.check_grounding(ans1, len(candidates))
    print(f"Grounding Passed: {guard1.passed}")
    if not guard1.passed:
        print(f"Rejection Reason: {guard1.rejection_reason}")

    # Test 2: Unanswerable Query (LLM should realize it can't answer from context)
    query2 = "What is the capital of Germany?"
    print(f"\nQUERY 2: '{query2}'")
    ans2 = generator.generate(query2, candidates)
    
    print(f"Generated Answer: {ans2.answer}")
    print(f"Grounded Flag: {ans2.grounded}")
    print(f"Confidence: {ans2.confidence}")
    print(f"Sources Cited: {ans2.sources}")
    print(f"Generation Latency: {ans2.latency_ms:.1f}ms")

    guard2 = OutputGuard.check_grounding(ans2, len(candidates))
    print(f"Grounding Passed: {guard2.passed}")
    if not guard2.passed:
        print(f"Rejection Reason: {guard2.rejection_reason}")


if __name__ == "__main__":
    test_generation()
