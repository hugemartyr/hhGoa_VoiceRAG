"""Adaptive hierarchical chunking engine for MSMARCO-XI passages.

Chunking strategy:
  - Normal passages (≤ max_passage_tokens): indexed as a single retrieval chunk.
  - Long passages (> max_passage_tokens): the original passage becomes a parent
    (not searchable), and sentence-level children become the retrieval units.

Design rationale:
  - MSMARCO passages are already curated semantic units (~50–150 words).
    Splitting short passages would create fragments that lose meaning.
  - Long passages benefit from sentence-level chunks for retrieval precision.
  - The LLM always sees the full parent passage (via context expansion),
    so sentence chunks only need to be good enough for embedding similarity.

Chunk IDs are deterministic (hash-based) so re-ingestion is idempotent.
"""

import hashlib
import uuid
from typing import List

import nltk

from app.models import Chunk, ChunkMetadata

# Ensure punkt tokenizer data is available
try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    nltk.download("punkt", quiet=True)
try:
    nltk.data.find("tokenizers/punkt_tab")
except (LookupError, OSError):
    nltk.download("punkt_tab", quiet=True)


def _make_chunk_id(query_id: int, passage_index: int, chunk_index: int) -> str:
    """Generate a deterministic UUID from (query_id, passage_index, chunk_index).

    Uses MD5 mapped to a UUID string. Collision probability is negligible
    at the scale of millions of chunks, and it satisfies Qdrant's UUID requirement.
    """
    raw = f"{query_id}:{passage_index}:{chunk_index}"
    hex_str = hashlib.md5(raw.encode("utf-8")).hexdigest()
    return str(uuid.UUID(hex=hex_str))


def _count_tokens(text: str) -> int:
    """Approximate token count using whitespace splitting.

    This is a fast heuristic for the chunking threshold decision.
    For English text, word count ≈ token count within ~20%.
    """
    return len(text.split())


def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using NLTK's Punkt tokenizer."""
    sentences = nltk.sent_tokenize(text, language="english")
    # Filter out empty strings that can result from unusual formatting
    return [s.strip() for s in sentences if s.strip()]


def _create_sentence_chunks(
    sentences: List[str],
    sentence_overlap: int = 1,
) -> List[str]:
    """Create sentence-level chunks with configurable overlap.

    Each chunk is a single sentence, optionally prepended with the previous
    sentence(s) for context. The overlap gives the embedding model surrounding
    context without creating full-passage-length chunks.

    With sentence_overlap=1 and sentences [S1, S2, S3, S4]:
      chunk 0: "S1"
      chunk 1: "S1 S2"
      chunk 2: "S2 S3"
      chunk 3: "S3 S4"

    If there are ≤2 sentences, return a single chunk (no benefit to splitting).
    """
    if len(sentences) <= 2:
        return [" ".join(sentences)]

    chunks: List[str] = []
    for i, sentence in enumerate(sentences):
        # Prepend up to `sentence_overlap` preceding sentences for context
        start = max(0, i - sentence_overlap)
        context_sentences = sentences[start : i + 1]
        chunks.append(" ".join(context_sentences))

    return chunks


def chunk_passage(
    passage_text: str,
    query_id: int,
    passage_index: int,
    is_selected: int,
    query_type: str,
    max_passage_tokens: int = 256,
    sentence_overlap: int = 1,
    chunk_index_offset: int = 0,
) -> List[Chunk]:
    """Chunk a single passage using the adaptive hierarchical strategy.

    Args:
        passage_text: The English passage text.
        query_id: MSMARCO query ID (for deterministic chunk IDs).
        passage_index: Index in the English_passages array.
        is_selected: MSMARCO ground truth (1 if passage was selected as relevant).
        query_type: Query type from the MSMARCO row.
        max_passage_tokens: Token threshold for splitting (default 256).
        sentence_overlap: Number of overlapping sentences between children (default 1).
        chunk_index_offset: Starting chunk_index for ID generation (for multi-passage rows).

    Returns:
        List of Chunk objects. For short passages, a single chunk.
        For long passages, a parent chunk + sentence children.
    """
    text = passage_text.strip()
    if not text:
        return []

    token_count = _count_tokens(text)

    # --- Short passage: single retrieval chunk ---
    if token_count <= max_passage_tokens:
        chunk_id = _make_chunk_id(query_id, passage_index, chunk_index_offset)
        return [
            Chunk(
                text=text,
                metadata=ChunkMetadata(
                    chunk_id=chunk_id,
                    chunk_type="passage",
                    is_retrieval_unit=True,
                    parent_id=None,
                    passage_index=passage_index,
                    is_selected=is_selected,
                    query_type=query_type,
                    query_id=query_id,
                    token_count=token_count,
                ),
            )
        ]

    # --- Long passage: parent + sentence children ---
    sentences = _split_into_sentences(text)

    # Edge case: if sentence tokenizer can't split (e.g., single run-on sentence),
    # treat as a single passage-level chunk rather than creating a useless parent.
    if len(sentences) <= 1:
        chunk_id = _make_chunk_id(query_id, passage_index, chunk_index_offset)
        return [
            Chunk(
                text=text,
                metadata=ChunkMetadata(
                    chunk_id=chunk_id,
                    chunk_type="passage",
                    is_retrieval_unit=True,
                    parent_id=None,
                    passage_index=passage_index,
                    is_selected=is_selected,
                    query_type=query_type,
                    query_id=query_id,
                    token_count=token_count,
                ),
            )
        ]

    chunks: List[Chunk] = []

    # Parent: stores full passage text, NOT a retrieval unit
    parent_id = _make_chunk_id(query_id, passage_index, chunk_index_offset)
    parent = Chunk(
        text=text,
        metadata=ChunkMetadata(
            chunk_id=parent_id,
            chunk_type="parent",
            is_retrieval_unit=False,
            parent_id=None,
            passage_index=passage_index,
            is_selected=is_selected,
            query_type=query_type,
            query_id=query_id,
            token_count=token_count,
        ),
    )
    chunks.append(parent)

    # Sentence children: the actual retrieval units
    sentence_chunks = _create_sentence_chunks(sentences, sentence_overlap)
    for i, chunk_text in enumerate(sentence_chunks):
        child_index = chunk_index_offset + 1 + i  # offset past the parent
        child_id = _make_chunk_id(query_id, passage_index, child_index)
        child = Chunk(
            text=chunk_text,
            metadata=ChunkMetadata(
                chunk_id=child_id,
                chunk_type="sentence",
                is_retrieval_unit=True,
                parent_id=parent_id,
                passage_index=passage_index,
                is_selected=is_selected,
                query_type=query_type,
                query_id=query_id,
                token_count=_count_tokens(chunk_text),
            ),
        )
        chunks.append(child)

    return chunks


def chunk_msmarco_row(
    query_id: int,
    english_passages: List[str],
    is_selected_list: List[int],
    query_type: str = "",
    max_passage_tokens: int = 256,
    sentence_overlap: int = 1,
) -> List[Chunk]:
    """Chunk all passages from a single MSMARCO-XI row.

    Each passage is chunked independently. Chunk IDs incorporate the passage_index
    to ensure uniqueness across passages within the same query.

    Args:
        query_id: MSMARCO query ID.
        english_passages: List of English passage texts from the row.
        is_selected_list: Parallel list of is_selected flags (1 = relevant).
        query_type: Query type string from the row.
        max_passage_tokens: Token threshold for splitting.
        sentence_overlap: Sentence overlap for children.

    Returns:
        List of all Chunk objects for this row.
    """
    all_chunks: List[Chunk] = []
    # Use a running chunk_index_offset so IDs are globally unique within this row
    global_chunk_offset = 0

    for p_idx, passage_text in enumerate(english_passages):
        is_selected = is_selected_list[p_idx] if p_idx < len(is_selected_list) else 0

        passage_chunks = chunk_passage(
            passage_text=passage_text,
            query_id=query_id,
            passage_index=p_idx,
            is_selected=is_selected,
            query_type=query_type,
            max_passage_tokens=max_passage_tokens,
            sentence_overlap=sentence_overlap,
            chunk_index_offset=global_chunk_offset,
        )
        all_chunks.extend(passage_chunks)
        global_chunk_offset += len(passage_chunks)

    return all_chunks
