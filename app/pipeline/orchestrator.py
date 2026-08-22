"""Full RAG Pipeline Orchestrator.

Manages the data flow from text query to final answer, wiring together
the input guard, embedding, retrieval, reranking, and generation stages.
It carefully tracks latency at every stage to meet the sub-2 second target.

When ``retrieval_mode=query`` (default), uses query-level retrieval and
returns pre-written Eng_Answer values without an LLM call.
"""

import time
import logging

from app.models import PipelineResult, LatencyBreakdown
from app.guardrails.input_guard import InputGuard
from app.guardrails.output_guard import OutputGuard
from app.pipeline.embedder import QueryEmbedder
from app.pipeline.query_retriever import QueryRetriever
from app.pipeline.reranker import QueryReranker
from app.config import settings

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Singleton-like harness for the end-to-end text-to-answer pipeline."""

    def __init__(self):
        self.embedder = QueryEmbedder()
        self.query_retriever = QueryRetriever()
        self.reranker = QueryReranker()
        self._retriever = None
        self._generator = None

    def _get_retriever(self):
        if self._retriever is None:
            from app.pipeline.retriever import HybridRetriever
            self._retriever = HybridRetriever()
        return self._retriever

    def _get_generator(self):
        if self._generator is None:
            from app.pipeline.generator import GroqGenerator
            self._generator = GroqGenerator()
        return self._generator

    def process_query(self, query: str, stt_latency: float = 0.0) -> PipelineResult:
        """Run the full text-to-answer pipeline."""
        if settings.retrieval_mode == "query":
            return self._process_query_level(query, stt_latency)
        return self._process_passage_level(query, stt_latency)

    def _process_query_level(self, query: str, stt_latency: float) -> PipelineResult:
        """Fast path: match Eng_Query and return stored Eng_Answer (no LLM)."""
        rag_start_time = time.perf_counter()
        latencies = LatencyBreakdown(stt_ms=stt_latency)
        stages = []

        result = PipelineResult(
            transcribed_text=query,
            rejected=False,
            latency=latencies,
        )

        try:
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

            stages.append("Embedding")
            dense_vec, _, embed_ms = self.embedder.embed_query(query)
            latencies.embedding_ms = embed_ms

            stages.append("Query Retrieval")
            ret_res = self.query_retriever.search(dense_vector=dense_vec, limit=settings.dense_top_k)
            latencies.retrieval_ms = ret_res.latency_ms

            if not ret_res.candidates:
                result.rejected = True
                result.rejection_reason = "No matching queries found in the knowledge base."
                result.answer = "I'm sorry, I don't have enough context to answer that accurately."
                self._finalize_latency(rag_start_time, latencies)
                result.stages_trace = stages
                return result

            stages.append("Reranking")
            reranked, rerank_ms, confidence = self.reranker.rerank_queries(query, ret_res.candidates)
            latencies.reranking_ms = rerank_ms
            result.confidence = confidence
            result.retrieved_passages = [c.eng_query for c in reranked]

            stages.append("Confidence Gate")
            t0 = time.perf_counter()
            is_confident, conf_reason = OutputGuard.check_retrieval_confidence(confidence)
            latencies.confidence_gate_ms = (time.perf_counter() - t0) * 1000

            if not is_confident or not reranked:
                result.rejected = True
                result.rejection_reason = conf_reason or "No confident query match."
                result.answer = "I'm sorry, I don't have enough context to answer that accurately."
                self._finalize_latency(rag_start_time, latencies)
                result.stages_trace = stages
                return result

            best = reranked[0]
            stages.append("Answer Lookup")
            latencies.generation_ms = 0.0
            latencies.grounding_ms = 0.0

            result.answer = best.eng_answer
            result.grounded = True
            result.sources = [best.query_id]
            result.rejected = False

            self._finalize_latency(rag_start_time, latencies)
            result.stages_trace = stages
            return result

        except Exception as e:
            logger.exception("Query-level pipeline failed")
            result.errors.append(str(e))
            result.rejected = True
            result.rejection_reason = "Internal pipeline error"
            result.answer = "I encountered an internal error while processing."
            self._finalize_latency(rag_start_time, latencies)
            result.stages_trace = stages
            return result

    def _process_passage_level(self, query: str, stt_latency: float) -> PipelineResult:
        """Classic RAG path: passage retrieval + LLM generation."""
        rag_start_time = time.perf_counter()
        latencies = LatencyBreakdown(stt_ms=stt_latency)
        stages = []

        result = PipelineResult(
            transcribed_text=query,
            rejected=False,
            latency=latencies,
        )

        try:
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

            stages.append("Embedding")
            dense_vec, sparse_vec, embed_ms = self.embedder.embed_query(query)
            latencies.embedding_ms = embed_ms

            stages.append("Retrieval")
            ret_res = self._get_retriever().search(
                dense_vector=dense_vec,
                sparse_vector=sparse_vec,
                limit=20,
            )
            latencies.retrieval_ms = ret_res.latency_ms

            stages.append("Reranking")
            rerank_res = self.reranker.rerank(query, ret_res.candidates)
            latencies.reranking_ms = rerank_res.latency_ms

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

            stages.append("Generation")
            gen_res = self._get_generator().generate(query, rerank_res.candidates)
            latencies.generation_ms = gen_res.latency_ms
            result.answer = gen_res.answer

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
