"""Full RAG Pipeline Orchestrator.

Manages the data flow from text query to final answer, wiring together
the input guard, embedding, retrieval, reranking, and generation stages.
It carefully tracks latency at every stage to meet the sub-2 second target.
"""

import time
import logging

from app.models import PipelineResult, LatencyBreakdown
from app.guardrails.input_guard import InputGuard
from app.guardrails.output_guard import OutputGuard
from app.pipeline.embedder import QueryEmbedder
from app.pipeline.retriever import HybridRetriever
from app.pipeline.reranker import QueryReranker
from app.pipeline.generator import GroqGenerator
from app.config import settings

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Singleton-like harness for the end-to-end text-to-answer pipeline."""

    def __init__(self):
        # We initialize all models/clients to keep them warm
        self.embedder = QueryEmbedder()
        self.retriever = HybridRetriever()
        self.reranker = QueryReranker()
        self.generator = GroqGenerator()

    def process_query(self, query: str, stt_latency: float = 0.0) -> PipelineResult:
        """Run the full text-to-answer RAG pipeline.
        
        Args:
            query: The transcribed (and potentially translated) English text.
            stt_latency: Time taken by the STT step (passed in for metrics).
        """
        rag_start_time = time.perf_counter()
        latencies = LatencyBreakdown(stt_ms=stt_latency)
        stages = []
        
        # We initialize the result with rejection defaults
        result = PipelineResult(
            transcribed_text=query,
            rejected=False,
            latency=latencies
        )

        try:
            # ---------------------------------------------------------
            # 1. Input Guard
            # ---------------------------------------------------------
            stages.append("Input Guard")
            t0 = time.perf_counter()
            ig_res = InputGuard.validate(query)
            latencies.input_guard_ms = (time.perf_counter() - t0) * 1000
            
            if not ig_res.passed:
                result.rejected = True
                result.rejection_reason = ig_res.rejection_reason
                result.answer = "I'm sorry, I cannot process this query."
                self._finalize_latency(rag_start_time, latencies)
                result.stages_trace = stages
                return result

            # ---------------------------------------------------------
            # 2. Embedding (Dense + Sparse)
            # ---------------------------------------------------------
            stages.append("Embedding")
            dense_vec, sparse_vec, embed_ms = self.embedder.embed_query(query)
            latencies.embedding_ms = embed_ms

            # ---------------------------------------------------------
            # 3. Retrieval
            # ---------------------------------------------------------
            stages.append("Retrieval")
            ret_res = self.retriever.search(
                dense_vector=dense_vec,
                sparse_vector=sparse_vec,
                limit=20
            )
            latencies.retrieval_ms = ret_res.latency_ms
            
            # ---------------------------------------------------------
            # 4. Reranking
            # ---------------------------------------------------------
            stages.append("Reranking")
            rerank_res = self.reranker.rerank(query, ret_res.candidates)
            latencies.reranking_ms = rerank_res.latency_ms
            
            # ---------------------------------------------------------
            # 5. Confidence Gate (Output Guard)
            # ---------------------------------------------------------
            stages.append("Confidence Gate")
            t0 = time.perf_counter()
            is_confident, conf_reason = OutputGuard.check_retrieval_confidence(rerank_res.confidence_score)
            latencies.confidence_gate_ms = (time.perf_counter() - t0) * 1000
            
            result.confidence = rerank_res.confidence_score
            result.retrieved_passages = [c.context_text for c in rerank_res.candidates]
            
            if not is_confident:
                result.rejected = True
                result.rejection_reason = conf_reason
                result.answer = "I'm sorry, I don't have enough context to answer that accurately."
                self._finalize_latency(rag_start_time, latencies)
                result.stages_trace = stages
                return result

            # ---------------------------------------------------------
            # 6. LLM Generation
            # ---------------------------------------------------------
            stages.append("Generation")
            gen_res = self.generator.generate(query, rerank_res.candidates)
            latencies.generation_ms = gen_res.latency_ms
            result.answer = gen_res.answer
            
            # ---------------------------------------------------------
            # 7. Grounding Gate
            # ---------------------------------------------------------
            stages.append("Grounding Gate")
            t0 = time.perf_counter()
            grounding_res = OutputGuard.check_grounding(gen_res, len(rerank_res.candidates))
            latencies.grounding_ms = (time.perf_counter() - t0) * 1000
            
            result.grounded = grounding_res.grounded
            result.sources = gen_res.sources
            
            if not grounding_res.passed:
                result.rejected = True
                result.rejection_reason = grounding_res.rejection_reason
                result.answer = "I apologize, but I could not find a reliable answer in my knowledge base."

            # Success
            self._finalize_latency(rag_start_time, latencies)
            result.stages_trace = stages
            return result

        except Exception as e:
            logger.exception("Pipeline failed")
            result.errors.append(str(e))
            result.rejected = True
            result.rejection_reason = "Internal pipeline error"
            result.answer = "I encountered an internal error while processing."
            self._finalize_latency(rag_start_time, latencies)
            result.stages_trace = stages
            return result

    def _finalize_latency(self, rag_start_time: float, latencies: LatencyBreakdown):
        """Calculate final totals."""
        rag_total = (time.perf_counter() - rag_start_time) * 1000
        latencies.rag_total_ms = rag_total
        latencies.e2e_total_ms = latencies.stt_ms + rag_total
