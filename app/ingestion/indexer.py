"""Two-pass ingestion pipeline for Qdrant.

Pass 1: Vocabulary Building
  - Scans the dataset.
  - Generates chunks using the hierarchical chunker.
  - Feeds retrieval-unit chunks into the BM25 vocabulary builder.
  - Persists vocab and IDF.

Pass 2: Indexing
  - Scans the dataset again (or resumes from checkpoint).
  - Generates chunks.
  - Computes dense vectors (SentenceTransformers) for retrieval units.
  - Computes sparse vectors (BM25) for retrieval units.
  - Upserts to Qdrant with named vectors (`dense` and `sparse`).
  - Parent chunks are upserted with empty vectors (context expansion only).
"""

import json
import logging
import os
import resource
import time
from typing import Any, Dict, List, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.ingestion.chunker import chunk_msmarco_row
from app.ingestion.sparse_encoder import BM25SparseEncoder, BM25VocabularyBuilder
from app.models import Chunk

logger = logging.getLogger(__name__)


def _get_memory_usage_mb() -> float:
    """Return the peak resident set size (RSS) in megabytes for the current process."""
    try:
        # ru_maxrss is in KB on Linux, bytes on macOS
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if os.uname().sysname == "Darwin":
            return usage / (1024 * 1024)
        return usage / 1024.0
    except Exception:
        return 0.0


def _extract_row_data(row: Dict[str, Any], row_idx: int) -> Tuple[int, List[str], List[int], str]:
    """Safely extract query_id, English_passages, is_selected, and query_type from various row schemas.

    Supports both top-level keys and nested 'passages' dictionaries.
    """
    # 1. query_id
    raw_qid = row.get("query_id")
    if raw_qid is None:
        query_id = row_idx
    else:
        try:
            query_id = int(raw_qid)
        except (ValueError, TypeError):
            query_id = row_idx

    # 2. English_passages
    english_passages: List[str] = []
    if "English_passages" in row and isinstance(row["English_passages"], (list, tuple)):
        english_passages = [str(p) for p in row["English_passages"] if p]
    elif "passages" in row and isinstance(row["passages"], dict):
        p_dict = row["passages"]
        if "English_passages" in p_dict and isinstance(p_dict["English_passages"], (list, tuple)):
            english_passages = [str(p) for p in p_dict["English_passages"] if p]
    elif "passages" in row and isinstance(row["passages"], (list, tuple)):
        english_passages = [str(p) for p in row["passages"] if p]

    # 3. is_selected
    is_selected: List[int] = []
    if "is_selected" in row and isinstance(row["is_selected"], (list, tuple)):
        is_selected = [int(x) if isinstance(x, (int, bool, str)) and str(x).isdigit() else 0 for x in row["is_selected"]]
    elif "passages" in row and isinstance(row["passages"], dict):
        p_dict = row["passages"]
        if "is_selected" in p_dict and isinstance(p_dict["is_selected"], (list, tuple)):
            is_selected = [int(x) if isinstance(x, (int, bool, str)) and str(x).isdigit() else 0 for x in p_dict["is_selected"]]

    # 4. query_type
    query_type = str(row.get("query_type", "") or "")

    return query_id, english_passages, is_selected, query_type


class QdrantIndexer:
    def __init__(self):
        logger.info(f"Connecting to Qdrant at {settings.qdrant_url} (collection: {settings.qdrant_collection})")
        self.client = QdrantClient(url=settings.qdrant_url, timeout=30.0)
        self.collection_name = settings.qdrant_collection
        self.embedder = None
        self.sparse_encoder = None

    def _setup_collection(self):
        """Create the Qdrant collection if it doesn't exist, configuring named vectors."""
        try:
            exists = self.client.collection_exists(self.collection_name)
        except Exception as e:
            logger.error(f"Failed to check if collection '{self.collection_name}' exists: {e}", exc_info=True)
            raise

        if not exists:
            logger.info(f"Creating Qdrant collection '{self.collection_name}' with dense (384-dim COSINE) and sparse vectors...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": qmodels.VectorParams(
                        size=384,  # all-MiniLM-L6-v2 size
                        distance=qmodels.Distance.COSINE,
                        on_disk=True,  # Move raw vectors to disk to save RAM
                    )
                },
                sparse_vectors_config={
                    "sparse": qmodels.SparseVectorParams()
                },
                quantization_config=qmodels.ScalarQuantization(
                    scalar=qmodels.ScalarQuantizationConfig(
                        type=qmodels.ScalarType.INT8,
                        quantile=0.99,
                        always_ram=True
                    )
                )
            )
            
            # Create payload index for faster filtering during retrieval
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="is_retrieval_unit",
                field_schema=qmodels.PayloadSchemaType.BOOL,
            )
            logger.info(f"Collection '{self.collection_name}' created successfully with payload index.")
        else:
            info = self.client.get_collection(self.collection_name)
            logger.info(f"Collection '{self.collection_name}' already exists ({info.points_count} points, status={info.status})")

    def run_pass_1(self, dataset, max_rows: Optional[int] = None, log_interval: int = 100):
        """Build BM25 vocabulary from the corpus with robust error handling and fine-grained progress logging."""
        logger.info(f"=== Starting Pass 1: BM25 Vocabulary Building (max_rows={max_rows}) ===")
        builder = BM25VocabularyBuilder()
        
        t0 = time.time()
        rows_processed = 0
        total_chunks = 0
        total_retrieval_units = 0
        errors_count = 0

        for i, row in enumerate(dataset):
            if max_rows and i >= max_rows:
                break
                
            try:
                query_id, english_passages, is_selected, query_type = _extract_row_data(row, i)
                if not english_passages:
                    continue

                chunks = chunk_msmarco_row(
                    query_id=query_id,
                    english_passages=english_passages,
                    is_selected_list=is_selected,
                    query_type=query_type,
                    max_passage_tokens=settings.max_passage_tokens,
                    sentence_overlap=settings.sentence_overlap,
                )
                
                total_chunks += len(chunks)
                for c in chunks:
                    if c.metadata.is_retrieval_unit:
                        builder.add_document(c.text)
                        total_retrieval_units += 1

                rows_processed += 1

            except Exception as e:
                errors_count += 1
                if errors_count <= 10 or errors_count % 100 == 0:
                    logger.warning(f"Pass 1: Error processing row {i} (error #{errors_count}): {e}", exc_info=True)
                continue

            if (i + 1) % log_interval == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / max(elapsed, 0.001)
                eta_s = ((max_rows - (i + 1)) / rate) if max_rows and rate > 0 else 0
                mem_mb = _get_memory_usage_mb()
                logger.info(
                    f"Pass 1: [{i + 1}{f'/{max_rows}' if max_rows else ''}] "
                    f"| Rate: {rate:.1f} rows/s "
                    f"| Chunks: {total_chunks} (RU: {total_retrieval_units}) "
                    f"| Vocab: {builder.vocab_size:,} "
                    f"| Memory: {mem_mb:.1f} MB "
                    f"| Errors: {errors_count} "
                    f"| Elapsed: {elapsed:.1f}s"
                    f"{f' | ETA: {eta_s:.1f}s' if max_rows else ''}"
                )

        # Save artifacts to data_dir
        builder.save(settings.data_dir)
        total_time = time.time() - t0
        logger.info(
            f"=== Pass 1 Complete in {total_time:.2f}s ===\n"
            f"  - Rows processed: {rows_processed:,} (Total scanned: {rows_processed + errors_count:,})\n"
            f"  - Total chunks generated: {total_chunks:,}\n"
            f"  - Retrieval units added: {total_retrieval_units:,}\n"
            f"  - Unique vocabulary terms: {builder.vocab_size:,}\n"
            f"  - Average document length: {builder.avg_doc_length:.2f} tokens\n"
            f"  - Errors encountered: {errors_count}\n"
            f"  - Artifacts saved to: {settings.data_dir}/ (vocab.json, idf.json, bm25_params.json)"
        )

    def run_pass_2(self, dataset, max_rows: Optional[int] = None, resume: bool = False, log_interval: int = 100):
        """Index chunks with dense/sparse vectors into Qdrant with detailed progress and error resilience."""
        logger.info(f"=== Starting Pass 2: Indexing to Qdrant (max_rows={max_rows}, resume={resume}) ===")
        self._setup_collection()
        
        if self.embedder is None:
            logger.info(f"Loading dense embedding model: {settings.embedding_model}...")
            self.embedder = SentenceTransformer(settings.embedding_model)
            logger.info("Dense embedding model loaded.")
            
        if self.sparse_encoder is None:
            logger.info(f"Loading BM25 sparse encoder from '{settings.data_dir}'...")
            try:
                self.sparse_encoder = BM25SparseEncoder.load(settings.data_dir)
                logger.info(f"BM25 sparse encoder loaded (vocab: {len(self.sparse_encoder._vocab):,} terms, N={self.sparse_encoder._N:,}).")
            except FileNotFoundError as e:
                logger.error(f"Cannot load BM25 vocabulary from '{settings.data_dir}'. Did Pass 1 run? Error: {e}")
                raise

        start_row = 0
        checkpoint_path = os.path.join(settings.data_dir, "checkpoint.json")
        if resume and os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, "r") as f:
                    ckpt = json.load(f)
                    start_row = ckpt.get("last_processed_row", 0) + 1
                logger.info(f"Resuming Pass 2 from checkpoint row {start_row}")
            except Exception as e:
                logger.warning(f"Could not read checkpoint file '{checkpoint_path}': {e}. Starting from row 0.")

        points_batch = []
        rows_processed = 0
        total_indexed = 0
        batches_upserted = 0
        errors_count = 0
        t0 = time.time()

        for i, row in enumerate(dataset):
            if i < start_row:
                continue
            if max_rows and i >= max_rows:
                break

            try:
                query_id, english_passages, is_selected, query_type = _extract_row_data(row, i)
                if not english_passages:
                    continue

                chunks = chunk_msmarco_row(
                    query_id=query_id,
                    english_passages=english_passages,
                    is_selected_list=is_selected,
                    query_type=query_type,
                    max_passage_tokens=settings.max_passage_tokens,
                    sentence_overlap=settings.sentence_overlap,
                )

                if not chunks:
                    continue

                # Batch encode dense vectors for retrieval units only
                retrieval_chunks = [c for c in chunks if c.metadata.is_retrieval_unit]
                dense_vectors = []
                if retrieval_chunks:
                    texts = [c.text for c in retrieval_chunks]
                    dense_vectors = self.embedder.encode(
                        texts, 
                        convert_to_numpy=True, 
                        show_progress_bar=False,
                        batch_size=len(texts)
                    ).tolist()

                ru_index = 0
                for chunk in chunks:
                    payload = chunk.metadata.model_dump()
                    payload["text"] = chunk.text
                    
                    vectors = {}
                    if chunk.metadata.is_retrieval_unit:
                        # Provide named dense vector
                        vectors["dense"] = dense_vectors[ru_index]
                        
                        # Provide named sparse vector
                        sv = self.sparse_encoder.encode_document(chunk.text)
                        vectors["sparse"] = qmodels.SparseVector(
                            indices=sv.indices, 
                            values=sv.values
                        )
                        ru_index += 1
                    else:
                        # Parent non-searchable chunks get empty dictionaries (valid in Qdrant with named vectors)
                        pass

                    points_batch.append(
                        qmodels.PointStruct(
                            id=chunk.metadata.chunk_id,
                            payload=payload,
                            vector=vectors
                        )
                    )

                rows_processed += 1

            except Exception as e:
                errors_count += 1
                if errors_count <= 10 or errors_count % 100 == 0:
                    logger.warning(f"Pass 2: Error encoding row {i} (error #{errors_count}): {e}", exc_info=True)
                continue

            # Upsert when batch is full
            if len(points_batch) >= settings.batch_size:
                try:
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=points_batch
                    )
                    total_indexed += len(points_batch)
                    batches_upserted += 1
                    points_batch = []
                    
                    # Update checkpoint
                    with open(checkpoint_path, "w") as f:
                        json.dump({"last_processed_row": i, "total_indexed": total_indexed}, f)

                except Exception as e:
                    logger.error(f"Pass 2: Failed to upsert batch at row {i}: {e}", exc_info=True)
                    # Retry once after a short pause
                    time.sleep(1.0)
                    try:
                        self.client.upsert(collection_name=self.collection_name, points=points_batch)
                        total_indexed += len(points_batch)
                        batches_upserted += 1
                        points_batch = []
                    except Exception as retry_err:
                        logger.critical(f"Pass 2: Upsert retry also failed: {retry_err}", exc_info=True)
                        points_batch = []  # avoid runaway batch accumulation

            if (i + 1) % log_interval == 0:
                elapsed = time.time() - t0
                rate = (i + 1 - start_row) / max(elapsed, 0.001)
                eta_s = ((max_rows - (i + 1)) / rate) if max_rows and rate > 0 else 0
                mem_mb = _get_memory_usage_mb()
                logger.info(
                    f"Pass 2: [{i + 1}{f'/{max_rows}' if max_rows else ''}] "
                    f"| Rate: {rate:.1f} rows/s "
                    f"| Indexed: {total_indexed:,} points "
                    f"| Batches: {batches_upserted} "
                    f"| Memory: {mem_mb:.1f} MB "
                    f"| Errors: {errors_count} "
                    f"| Elapsed: {elapsed:.1f}s"
                    f"{f' | ETA: {eta_s:.1f}s' if max_rows else ''}"
                )

        # Final flush of any remaining points in the batch
        if points_batch:
            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points_batch
                )
                total_indexed += len(points_batch)
                batches_upserted += 1
                points_batch = []
            except Exception as e:
                logger.error(f"Pass 2: Failed during final batch upsert: {e}", exc_info=True)

            with open(checkpoint_path, "w") as f:
                json.dump({"last_processed_row": i if 'i' in locals() else 0, "total_indexed": total_indexed}, f)

        total_time = time.time() - t0
        logger.info(
            f"=== Pass 2 Complete in {total_time:.2f}s ===\n"
            f"  - Rows indexed: {rows_processed:,}\n"
            f"  - Total points upserted: {total_indexed:,}\n"
            f"  - Total batches: {batches_upserted}\n"
            f"  - Errors encountered: {errors_count}\n"
            f"  - Checkpoint saved to: {checkpoint_path}"
        )
