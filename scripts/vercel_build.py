import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path so `from app...` works when executing
# this script as `python scripts/vercel_build.py` (sys.path[0] is `scripts/`).
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

# Ensure model cache path is set and respected by HuggingFace / Transformers
os.environ["HF_HOME"] = os.environ.get("HF_HOME", os.path.join(os.getcwd(), "model_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.environ["HF_HOME"])

from app.config import settings

def main():
    # Model caching can be large — make it opt-in via the PRECACHE_MODELS env var.
    # By default (PRECACHE_MODELS not set or false) we skip downloading heavy model weights
    # to avoid bundling huge files during platform builds (e.g., Vercel).
    precache = os.environ.get("PRECACHE_MODELS", "false").lower() in ("1", "true", "yes")
    if not precache:
        print("Skipping model pre-cache (PRECACHE_MODELS not enabled).")
        return

    # Import heavy ML libraries lazily only when caching is explicitly requested.
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        print(f"Could not import sentence_transformers for caching: {e}")
        return

    print(f"Caching model to {os.environ['HF_HOME']}")
    SentenceTransformer(settings.embedding_model)
    print("Model cached to model_cache/")


if __name__ == "__main__":
    main()
