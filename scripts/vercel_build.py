import os

# Ensure model cache path is set and respected by HuggingFace / Transformers
os.environ["HF_HOME"] = os.environ.get("HF_HOME", os.path.join(os.getcwd(), "model_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.environ["HF_HOME"])

from sentence_transformers import SentenceTransformer
from app.config import settings

def main():
    print(f"Caching model to {os.environ['HF_HOME']}")
    SentenceTransformer(settings.embedding_model)
    print("Model cached to model_cache/")

if __name__ == "__main__":
    main()
