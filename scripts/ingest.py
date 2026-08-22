"""CLI entry point for running the offline ingestion pipeline.

Supports generating the BM25 vocabulary (Pass 1) and indexing to Qdrant (Pass 2).

Features:
  - Comprehensive logging with timestamps, log levels, memory RSS, and ETA
  - Top-level and per-row error tracking and exception logging
  - Uncaught exception hook and graceful signal handling (SIGINT / SIGTERM)
  - Memory-safe parquet streaming with configurable language and split
  - Pre-flight connection and health checks for Qdrant

Usage:
  python scripts/ingest.py --max-rows 10000
  python scripts/ingest.py --max-rows 5000 --batch-size 500 --split validation --lang hi
  python scripts/ingest.py --mock
  python scripts/ingest.py --resume
"""

import argparse
import logging
import os
import signal
import sys
import time
import traceback
from typing import Any, Callable, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from app.config import settings
from app.ingestion.indexer import QdrantIndexer, _get_memory_usage_mb
from app.ingestion.query_indexer import QueryIndexer

# Configure root logger with detailed formatting
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ingestion")


def _setup_signal_handlers():
    """Register signal handlers to log warnings on termination signals."""
    def handle_signal(sig, frame):
        sig_name = signal.Signals(sig).name
        logger.warning(f"Received signal {sig_name} ({sig}). Shutting down gracefully...")
        sys.exit(128 + sig)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)


def _setup_exception_hook():
    """Log any unhandled exceptions with full traceback."""
    def custom_excepthook(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        logger.critical("UNCAUGHT EXCEPTION IN INGESTION PIPELINE:", exc_info=(exc_type, exc_value, exc_tb))

    sys.excepthook = custom_excepthook


def build_dataset_loader(
    mock: bool = False,
    split: str = "validation",
    lang: str = "hi",
    data_file: Optional[str] = None,
) -> Callable[[], Any]:
    """Return a callable that creates a fresh streaming dataset iterator."""
    if mock:
        logger.info("Configured dataset source: Local mock dataset")
        from scripts.mock_dataset import get_mock_dataset
        return get_mock_dataset

    if data_file:
        logger.info(f"Configured dataset source: Custom file '{data_file}'")
        return lambda: load_dataset("parquet", data_files={"data": data_file}, split="data", streaming=True)

    # 2-letter to 3-letter dataset filename mapping
    lang_file_map = {
        "hi": "hin", "hindi": "hin", "hin": "hin",
        "bn": "ben", "bengali": "ben", "ben": "ben",
        "mr": "mar", "marathi": "mar", "mar": "mar",
        "ta": "tam", "tamil": "tam", "tam": "tam",
        "te": "tel", "telugu": "tel", "tel": "tel",
        "gu": "guj", "gujarati": "guj", "guj": "guj",
        "kn": "kan", "kannada": "kan", "kan": "kan",
        "ml": "mal", "malayalam": "mal", "mal": "mal",
        "pa": "pan", "punjabi": "pan", "pan": "pan",
        "ur": "urd", "urdu": "urd", "urd": "urd",
        "or": "ori", "odia": "ori", "ori": "ori",
        "as": "asm", "assamese": "asm", "asm": "asm",
        "ne": "nep", "nepali": "nep", "nep": "nep",
        "sa": "san", "sanskrit": "san", "san": "san",
    }
    file_prefix = lang_file_map.get(lang.lower(), lang.lower())
    split_suffix = "val" if split.startswith("val") else "train"
    parquet_filename = f"{file_prefix}{split_suffix}.parquet"
    parquet_url = f"https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/{split}/{parquet_filename}"
    logger.info(f"Configured dataset source: MSMARCO-XI [{split}/{lang}] from URL: {parquet_url}")

    def _loader():
        try:
            logger.info(f"Opening parquet stream for {parquet_filename} ({split})...")
            return load_dataset(
                "parquet",
                data_files={split: parquet_url},
                split=split,
                streaming=True,
            )
        except Exception as e:
            logger.error(f"Failed to initialize stream from {parquet_url}: {e}", exc_info=True)
            logger.info("Attempting fallback to standard load_dataset('ai4bharat/MSMARCO-XI', streaming=True)...")
            return load_dataset("ai4bharat/MSMARCO-XI", split=split, streaming=True)

    return _loader


def main():
    _setup_signal_handlers()
    _setup_exception_hook()

    parser = argparse.ArgumentParser(description="Run VoiceRAG ingestion pipeline.")
    parser.add_argument("--max-rows", type=int, default=None, help="Max rows to process (default: settings.max_rows)")
    parser.add_argument("--batch-size", type=int, default=None, help="Qdrant batch size (default: settings.batch_size)")
    parser.add_argument("--split", type=str, default="validation", choices=["validation", "train"], help="Dataset split (default: validation)")
    parser.add_argument("--lang", type=str, default="hi", help="Language subset code (e.g., hi, mr, bn, ta, te, etc.)")
    parser.add_argument("--data-file", type=str, default=None, help="Optional direct path or URL to parquet file")
    parser.add_argument("--log-interval", type=int, default=100, help="Log progress every N rows (default: 100)")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint (Pass 2 only)")
    parser.add_argument("--skip-pass1", action="store_true", help="Skip BM25 vocabulary building (assumes already built)")
    parser.add_argument("--skip-pass2", action="store_true", help="Skip Qdrant passage indexing")
    parser.add_argument("--skip-query-index", action="store_true", help="Skip query-level indexing (Eng_Query → Eng_Answer)")
    parser.add_argument(
        "--queries-only",
        action="store_true",
        help="Only run query-level indexing (skips Pass 1 BM25 and Pass 2 passages)",
    )
    parser.add_argument("--mock", action="store_true", help="Use small mock dataset instead of huggingface streaming")

    args = parser.parse_args()

    max_rows = args.max_rows if args.max_rows is not None else settings.max_rows
    if args.batch_size is not None:
        settings.batch_size = args.batch_size

    logger.info("================================================================")
    logger.info("             VOICERAG INGESTION PIPELINE STARTING               ")
    logger.info("================================================================")
    logger.info(f"Target Max Rows:        {max_rows if max_rows is not None else 'ALL'}")
    logger.info(f"Qdrant Batch Size:      {settings.batch_size}")
    logger.info(f"Qdrant Target URL:      {settings.qdrant_url}")
    logger.info(f"Qdrant Collection:      {settings.qdrant_collection}")
    logger.info(f"Query Collection:       {settings.qdrant_query_collection}")
    logger.info(f"Embedding Model:        {settings.embedding_model}")
    logger.info(f"Data Directory:         {settings.data_dir}")
    logger.info(f"Dataset Split:          {args.split} (language: {args.lang})")
    logger.info(f"Initial Memory RSS:     {_get_memory_usage_mb():.1f} MB")
    logger.info("================================================================")

    # Initialize Indexer and pre-flight check
    try:
        indexer = QdrantIndexer()
        logger.info("Qdrant client initialized successfully.")
    except Exception as e:
        logger.critical(f"FATAL: Could not initialize QdrantIndexer: {e}", exc_info=True)
        sys.exit(1)

    # Dataset loader builder
    get_dataset = build_dataset_loader(
        mock=args.mock,
        split=args.split,
        lang=args.lang,
        data_file=args.data_file,
    )

    pipeline_start_time = time.time()

    run_pass1 = not args.skip_pass1 and not args.queries_only
    run_pass2 = not args.skip_pass2 and not args.queries_only
    run_query_index = not args.skip_query_index

    # --- PASS 1: BM25 Vocabulary Building ---
    if run_pass1:
        logger.info(">>> Starting Pass 1: BM25 Vocabulary Building")
        try:
            pass1_dataset = get_dataset()
            indexer.run_pass_1(pass1_dataset, max_rows=max_rows, log_interval=args.log_interval)
        except Exception as e:
            logger.critical(f"FATAL ERROR in Pass 1: {e}", exc_info=True)
            sys.exit(1)
    else:
        logger.info("Skipping Pass 1 (BM25 Vocabulary Building requested to be skipped).")

    # --- PASS 2: Passage Vector Indexing to Qdrant ---
    if run_pass2:
        logger.info(">>> Starting Pass 2: Passage Indexing to Qdrant")
        try:
            pass2_dataset = get_dataset()
            indexer.run_pass_2(pass2_dataset, max_rows=max_rows, resume=args.resume, log_interval=args.log_interval)
        except Exception as e:
            logger.critical(f"FATAL ERROR in Pass 2: {e}", exc_info=True)
            sys.exit(1)
    else:
        logger.info("Skipping Pass 2 (Passage indexing skipped).")

    # --- PASS 3: Query-Level Indexing (Eng_Query → Eng_Answer) ---
    if run_query_index:
        logger.info(">>> Starting Pass 3: Query-Level Indexing")
        try:
            query_indexer = QueryIndexer()
            query_dataset = get_dataset()
            query_indexer.run(
                query_dataset,
                max_rows=max_rows,
                resume=args.resume,
                log_interval=args.log_interval,
            )
        except Exception as e:
            logger.critical(f"FATAL ERROR in Pass 3 (Query Indexing): {e}", exc_info=True)
            sys.exit(1)
    else:
        logger.info("Skipping Pass 3 (Query-level indexing skipped).")

    total_pipeline_time = time.time() - pipeline_start_time
    logger.info("================================================================")
    logger.info(f"   VOICERAG INGESTION PIPELINE FINISHED IN {total_pipeline_time:.2f}s")
    logger.info(f"   Final Process Memory: {_get_memory_usage_mb():.1f} MB")
    logger.info("================================================================")


if __name__ == "__main__":
    main()
