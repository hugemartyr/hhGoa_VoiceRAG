"""Guardrails for outputs (retrieval confidence and final grounding).

Confidence Gate: Rejects answers if the top retrieved document score falls
                 below a calibrated threshold (avoids out-of-domain hallucinations).

Grounding Gate: Validates if the final LLM response is grounded in the retrieved text.
"""

from typing import Tuple

from app.config import settings

class OutputGuard:
    @staticmethod
    def check_retrieval_confidence(best_score: float) -> Tuple[bool, str]:
        """Check if the best retrieval score passes the confidence threshold.
        
        Returns:
            (is_confident, rejection_reason)
        """
        if not settings.enable_reranker:
            threshold = settings.confidence_threshold_no_reranker
            if best_score < threshold:
                return False, (
                    f"Retrieval confidence ({best_score:.2f}) below threshold "
                    f"({threshold:.2f}) with reranker disabled."
                )
            return True, ""
            
        if best_score < settings.confidence_threshold:
            return False, f"Retrieval confidence ({best_score:.2f}) below threshold ({settings.confidence_threshold:.2f})."
            
        return True, ""

    @staticmethod
    def check_grounding(gen_answer: 'GeneratedAnswer', num_candidates: int) -> 'GroundingResult':
        """Verify the generated answer is grounded using simple programmatic checks.
        
        Per user constraints: "Do NOT use token overlap percentage. Use citation existence 
        check + evidence presence check. No second LLM call."
        """
        import time
        from app.models import GroundingResult
        start_time = time.perf_counter()
        
        # 1. Did the LLM self-report that it could not answer the question?
        if not gen_answer.grounded:
            return GroundingResult(
                passed=False,
                grounded=False,
                rejection_reason="LLM self-reported that context was insufficient.",
                latency_ms=(time.perf_counter() - start_time) * 1000
            )
            
        # 2. Did the LLM cite any sources?
        if not gen_answer.sources:
            return GroundingResult(
                passed=False,
                grounded=False,
                rejection_reason="LLM generated an answer but failed to cite any supporting sources.",
                latency_ms=(time.perf_counter() - start_time) * 1000
            )
            
        # 3. Are the cited sources valid (within the range of provided candidates)?
        for source_idx in gen_answer.sources:
            if not isinstance(source_idx, int) or source_idx < 1 or source_idx > num_candidates:
                return GroundingResult(
                    passed=False,
                    grounded=False,
                    rejection_reason=f"LLM cited invalid source index: {source_idx}",
                    latency_ms=(time.perf_counter() - start_time) * 1000
                )
                
        # Passed all checks
        return GroundingResult(
            passed=True,
            grounded=True,
            rejection_reason=None,
            latency_ms=(time.perf_counter() - start_time) * 1000
        )
