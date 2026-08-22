"""Query-level indexing for MSMARCO-XI (Option B fast path).

Indexes one point per query row:
  - Dense vector: embedding of Eng_Query
  - Payload: query_id, eng_query, eng_answer, answer, query_type

At query time the user's English question is matched against Eng_Query
embeddings and the pre-written Eng_Answer is returned directly.
"""

import hashlib
import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer

from app.config import settings

logger = logging.getLogger(__name__)


def _make_query_point_id(query_id: int) -> str:
    """Deterministic UUID for a query-level point."""
    hex_str = hashlib.md5(f"query:{query_id}".encode("utf-8")).hexdigest()
    return str(uuid.UUID(hex=hex_str))


def extract_query_row_data(row: Dict[str, Any], row_idx: int) -> Optional[Tuple[int, str, str, str, str]]:
    """Extract query_id, Eng_Query, Eng_Answer, Answer, query_type from a MSMARCO-XI row.

    Returns None if the row lacks a usable Eng_Query or Eng_Answer.
    """
    raw_qid = row.get("query_id")
    if raw_qid is None:
        query_id = row_idx
    else:
        try:
            query_id = int(raw_qid)
        except (ValueError, TypeError):
            query_id = row_idx

    eng_query = str(row.get("Eng_Query") or row.get("Eng_query") or row.get("eng_query") or "").strip()
    eng_answer = str(row.get("Eng_Answer") or row.get("Eng_answer") or row.get("eng_answer") or "").strip()
    answer = str(row.get("Answer") or row.get("answer") or "").strip()
    query_type = str(row.get("query_type", "") or "")

    if not eng_query or not eng_answer:
        return None

    return query_id, eng_query, eng_answer, answer, query_type


class QueryIndexer:
    """Indexes MSMARCO-XI queries into a dedicated Qdrant collection."""

    def __init__(self):
        logger.info(
            f"Connecting to Qdrant at {settings.qdrant_url} "
            f"(query collection: {settings.qdrant_query_collection})"
        )
        self.client = QdrantClient(url=settings.qdrant_url, timeout=30.0)
        self.collection_name = settings.qdrant_query_collection
        self.embedder: Optional[SentenceTransformer] = None

    def _setup_collection(self):
        """Create the query collection if it does not exist."""
        try:
            exists = self.client.collection_exists(self.collection_name)
        except Exception as e:
            logger.error(f"Failed to check query collection '{self.collection_name}': {e}", exc_info=True)
            raise

        if not exists:
            logger.info(
                f"Creating query collection '{self.collection_name}' "
                f"with dense vectors (384-dim COSINE)..."
            )
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(
                    size=384,
                    distance=qmodels.Distance.COSINE,
                    on_disk=True,
                ),
                quantization_config=qmodels.ScalarQuantization(
                    scalar=qmodels.ScalarQuantizationConfig(
                        type=qmodels.ScalarType.INT8,
                        quantile=0.99,
                        always_ram=True,
                    )
                ),
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="query_id",
                field_schema=qmodels.PayloadSchemaType.INTEGER,
            )
            logger.info(f"Query collection '{self.collection_name}' created.")
        else:
            info = self.client.get_collection(self.collection_name)
            logger.info(
                f"Query collection '{self.collection_name}' exists "
                f"({info.points_count} points, status={info.status})"
            )

    def run(self, dataset, max_rows: Optional[int] = None, resume: bool = False, log_interval: int = 100):
        """Index Eng_Query rows with dense embeddings into Qdrant."""
        logger.info(f"=== Starting Query Indexing (max_rows={max_rows}, resume={resume}) ===")
        self._setup_collection()

        if self.embedder is None:
            logger.info(f"Loading embedding model: {settings.embedding_model}...")
            self.embedder = SentenceTransformer(settings.embedding_model)

        start_row = 0
        checkpoint_path = os.path.join(settings.data_dir, "query_index_checkpoint.json")
        if resume and os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, "r") as f:
                    ckpt = json.load(f)
                    start_row = ckpt.get("last_processed_row", 0) + 1
                logger.info(f"Resuming query indexing from row {start_row}")
            except Exception as e:
                logger.warning(f"Could not read query checkpoint: {e}. Starting from row 0.")

        points_batch: List[qmodels.PointStruct] = []
        rows_indexed = 0
        rows_skipped = 0
        total_upserted = 0
        errors_count = 0
        t0 = time.time()

        for i, row in enumerate(dataset):
            if i < start_row:
                continue
            if max_rows and i >= max_rows:
                break

            try:
                extracted = extract_query_row_data(row, i)
                if extracted is None:
                    rows_skipped += 1
                    continue

                query_id, eng_query, eng_answer, answer, query_type = extracted
                dense_vector = self.embedder.encode(
                    eng_query,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                ).tolist()

                points_batch.append(
                    qmodels.PointStruct(
                        id=_make_query_point_id(query_id),
                        vector=dense_vector,
                        payload={
                            "query_id": query_id,
                            "eng_query": eng_query,
                            "eng_answer": eng_answer,
                            "answer": answer,
                            "query_type": query_type,
                        },
                    )
                )
                rows_indexed += 1

            except Exception as e:
                errors_count += 1
                if errors_count <= 10 or errors_count % 100 == 0:
                    logger.warning(f"Query index: error on row {i} (#{errors_count}): {e}", exc_info=True)
                continue

            if len(points_batch) >= settings.batch_size:
                self._upsert_batch(points_batch, i, total_upserted, checkpoint_path)
                total_upserted += len(points_batch)
                points_batch = []

            if (i + 1) % log_interval == 0:
                elapsed = time.time() - t0
                rate = (i + 1 - start_row) / max(elapsed, 0.001)
                logger.info(
                    f"Query index: [{i + 1}{f'/{max_rows}' if max_rows else ''}] "
                    f"| Rate: {rate:.1f} rows/s "
                    f"| Indexed: {rows_indexed:,} "
                    f"| Skipped: {rows_skipped:,} "
                    f"| Upserted: {total_upserted + len(points_batch):,} "
                    f"| Errors: {errors_count}"
                )

        if points_batch:
            self._upsert_batch(points_batch, i if "i" in locals() else 0, total_upserted, checkpoint_path)
            total_upserted += len(points_batch)

        total_time = time.time() - t0
        logger.info(
            f"=== Query Indexing Complete in {total_time:.2f}s ===\n"
            f"  - Rows indexed: {rows_indexed:,}\n"
            f"  - Rows skipped (missing Eng_Query/Eng_Answer): {rows_skipped:,}\n"
            f"  - Total points upserted: {total_upserted:,}\n"
            f"  - Errors: {errors_count}"
        )

    def _upsert_batch(
        self,
        points: List[qmodels.PointStruct],
        row_idx: int,
        total_upserted: int,
        checkpoint_path: str,
    ):
        self.client.upsert(collection_name=self.collection_name, points=points)
        with open(checkpoint_path, "w") as f:
            json.dump({"last_processed_row": row_idx, "total_indexed": total_upserted + len(points)}, f)
