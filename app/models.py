"""Pydantic models shared across all pipeline stages.

Every stage uses typed input/output models. This file is the single source of truth
for all data structures flowing through the pipeline.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, List


# =============================================================================
# Chunking / Ingestion Models
# =============================================================================


class ChunkMetadata(BaseModel):
    """Metadata attached to every chunk during ingestion."""

    chunk_id: str = Field(description="Deterministic UUID: hash(query_id, passage_index, chunk_index)")
    chunk_type: str = Field(description="'passage' | 'sentence' | 'parent'")
    is_retrieval_unit: bool = Field(description="True for passage/sentence chunks, False for parent-only objects")
    parent_id: Optional[str] = Field(default=None, description="chunk_id of parent passage (for sentence chunks)")
    passage_index: int = Field(description="Index in the original English_passages array")
    is_selected: int = Field(description="MSMARCO ground truth: was this passage selected as relevant")
    query_type: str = Field(default="", description="Query type from the source MSMARCO row")
    query_id: int = Field(description="Original MSMARCO query ID")
    token_count: int = Field(description="Pre-computed token count of the chunk text")


class Chunk(BaseModel):
    """A single chunk produced by the chunking engine."""

    text: str = Field(description="Chunk text content")
    metadata: ChunkMetadata


# =============================================================================
# Sparse Vector Model
# =============================================================================


class SparseVectorData(BaseModel):
    """BM25-weighted sparse vector for Qdrant sparse vector indexing."""

    indices: List[int] = Field(description="Term IDs from the vocabulary")
    values: List[float] = Field(description="BM25 weights for each term")


# =============================================================================
# STT Models
# =============================================================================


class STTResponse(BaseModel):
    """Response from Sarvam Saaras v3 speech-to-text."""

    text: str = Field(description="Transcribed/translated English text")
    source_language: str = Field(default="en", description="Detected source language code")
    was_translated: bool = Field(default=False, description="True if non-English input was translated")
    confidence: float = Field(default=1.0, description="Transcription confidence score")
    latency_ms: float = Field(default=0.0, description="STT API call latency in milliseconds")


# =============================================================================
# Input Guardrail Models
# =============================================================================


class InputGuardResult(BaseModel):
    """Result from cheap pre-retrieval input validation."""

    passed: bool = Field(description="True if the input passed all validation checks")
    rejection_reason: Optional[str] = Field(default=None, description="Why the input was rejected")


# =============================================================================
# Retrieval Models
# =============================================================================


class RetrievalCandidate(BaseModel):
    """A single candidate from hybrid retrieval, after RRF fusion and parent expansion."""

    chunk_id: str = Field(description="Unique chunk ID")
    chunk_text: str = Field(description="Original chunk text (the retrieval unit)")
    context_text: str = Field(description="Expanded context: parent passage for sentence chunks, same as chunk_text for passage chunks")
    rrf_score: float = Field(default=0.0, description="Reciprocal Rank Fusion score")
    rerank_score: Optional[float] = Field(default=None, description="Cross-encoder reranker score (if reranking was performed)")
    chunk_type: str = Field(description="'passage' or 'sentence'")
    parent_id: Optional[str] = Field(default=None, description="Parent chunk ID (for sentence chunks)")
    metadata: Dict = Field(default_factory=dict, description="Additional metadata from Qdrant payload")


class RetrievalResult(BaseModel):
    """Result from the hybrid retrieval stage."""

    candidates: List[RetrievalCandidate] = Field(default_factory=list)
    latency_ms: float = Field(default=0.0, description="Total retrieval latency in milliseconds")


# =============================================================================
# Reranker Models
# =============================================================================


class RerankResult(BaseModel):
    """Result from the cross-encoder reranking stage."""

    candidates: List[RetrievalCandidate] = Field(default_factory=list, description="Reranked candidates with rerank_score populated")
    latency_ms: float = Field(default=0.0, description="Reranker inference latency in milliseconds")
    confidence_score: float = Field(default=0.0, description="Highest reranker score across candidates")


# =============================================================================
# Generation Models
# =============================================================================


class GeneratedAnswer(BaseModel):
    """Structured output from the Groq LLM."""

    answer: str = Field(description="The generated answer text")
    grounded: bool = Field(description="Whether the model believes the answer is grounded in context")
    confidence: float = Field(default=0.0, description="Model's self-assessed confidence (0.0–1.0)")
    sources: List[int] = Field(default_factory=list, description="1-indexed passage numbers cited by the model")
    latency_ms: float = Field(default=0.0, description="LLM API call latency in milliseconds")


# =============================================================================
# Grounding Validation Models
# =============================================================================


class GroundingResult(BaseModel):
    """Result from the lightweight grounding validation check."""

    passed: bool = Field(description="True if grounding validation passed")
    grounded: bool = Field(description="Whether the answer is considered grounded")
    rejection_reason: Optional[str] = Field(default=None, description="Why grounding failed")
    latency_ms: float = Field(default=0.0, description="Grounding check latency in milliseconds")


# =============================================================================
# Latency Breakdown
# =============================================================================


class LatencyBreakdown(BaseModel):
    """Per-stage latency tracking for the full pipeline."""

    stt_ms: float = Field(default=0.0, description="Speech-to-text latency")
    input_guard_ms: float = Field(default=0.0, description="Input validation latency")
    embedding_ms: float = Field(default=0.0, description="Query encoding (dense + sparse) latency")
    retrieval_ms: float = Field(default=0.0, description="Qdrant hybrid retrieval + RRF + dedup latency")
    reranking_ms: Optional[float] = Field(default=None, description="Cross-encoder reranking latency (None if skipped)")
    confidence_gate_ms: float = Field(default=0.0, description="Confidence gate evaluation latency")
    generation_ms: float = Field(default=0.0, description="Groq LLM generation latency")
    grounding_ms: float = Field(default=0.0, description="Grounding validation latency")
    rag_total_ms: float = Field(default=0.0, description="Total RAG pipeline latency (excludes STT)")
    e2e_total_ms: float = Field(default=0.0, description="Full end-to-end latency (includes STT)")


# =============================================================================
# Pipeline Result (Top-Level Response)
# =============================================================================


class PipelineResult(BaseModel):
    """Top-level response from the VoiceRAG orchestrator.

    This is what the API returns to the client.
    """

    # Answer
    answer: str = Field(default="", description="Generated answer or rejection message")
    grounded: bool = Field(default=False, description="Whether the answer is grounded in retrieved context")
    confidence: float = Field(default=0.0, description="Confidence score (from reranker or RRF)")
    sources: List[int] = Field(default_factory=list, description="1-indexed passage numbers supporting the answer")

    # Rejection
    rejected: bool = Field(default=False, description="True if query was rejected by input guard or confidence gate")
    rejection_reason: Optional[str] = Field(default=None, description="Reason for rejection")

    # Language
    source_language: str = Field(default="en", description="Detected language of voice input")
    was_translated: bool = Field(default=False, description="Whether voice input was translated to English")
    transcribed_text: str = Field(default="", description="The English query text (after transcription/translation)")

    # Retrieved context (for transparency / debugging)
    retrieved_passages: List[str] = Field(default_factory=list, description="Context passages sent to the LLM")

    # Latency
    latency: LatencyBreakdown = Field(default_factory=LatencyBreakdown)

    # Observability
    retries_by_stage: Dict[str, int] = Field(default_factory=dict, description="Retry counts per stage")
    errors: List[str] = Field(default_factory=list, description="Non-fatal errors encountered during processing")
    stages_trace: List[str] = Field(default_factory=list, description="Ordered list of stages executed")
