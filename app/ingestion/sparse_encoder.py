"""BM25-weighted sparse vector encoder for Qdrant sparse vector indexing.

Two-phase design:
  1. OFFLINE (vocabulary building):
     - Scan all retrieval-unit chunk texts
     - Build vocabulary: term → stable integer term_id
     - Compute corpus-level IDF: idf(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
     - Persist vocab, IDF, and corpus stats to disk

  2. ONLINE (query encoding):
     - Load persisted vocab + IDF at startup
     - Tokenize query → look up term_ids → compute BM25 query weights
     - Deterministic: same query always produces same sparse vector

BM25 formula for document term weight:
  w(t, d) = idf(t) * (tf(t,d) * (k1 + 1)) / (tf(t,d) + k1 * (1 - b + b * dl/avgdl))

Default parameters: k1=1.2, b=0.75 (standard BM25 defaults).
"""

import json
import math
import os
import re
import string
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

from app.models import SparseVectorData


# --- Stopwords (small English set, avoids NLTK dependency for this) ---
_STOPWORDS: Set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "don", "now", "and", "but", "or", "if", "while", "that", "this",
    "it", "its", "i", "me", "my", "we", "our", "you", "your", "he",
    "him", "his", "she", "her", "they", "them", "their", "what", "which",
    "who", "whom", "these", "those", "am", "s", "t", "d", "ll", "ve",
    "re", "m",
}

# Regex for splitting text into terms
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase terms, stripping punctuation and stopwords.

    Uses a simple regex-based approach for speed and determinism.
    """
    tokens = _TOKEN_PATTERN.findall(text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


class BM25VocabularyBuilder:
    """Builds vocabulary and IDF statistics from a corpus of documents.

    Usage (offline, first pass over corpus):
        builder = BM25VocabularyBuilder()
        for chunk_text in all_retrieval_chunks:
            builder.add_document(chunk_text)
        builder.save("data/")
    """

    def __init__(self):
        self._doc_count: int = 0
        self._doc_freq: Counter = Counter()  # term → number of docs containing it
        self._total_tokens: int = 0
        self._vocab: Dict[str, int] = {}  # term → term_id
        self._next_id: int = 0

    def add_document(self, text: str) -> None:
        """Process a single document and update statistics."""
        tokens = tokenize(text)
        if not tokens:
            return

        self._doc_count += 1
        self._total_tokens += len(tokens)

        # Count unique terms in this document for DF
        unique_terms = set(tokens)
        for term in unique_terms:
            self._doc_freq[term] += 1
            # Assign term_id if new
            if term not in self._vocab:
                self._vocab[term] = self._next_id
                self._next_id += 1

    @property
    def doc_count(self) -> int:
        return self._doc_count

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    @property
    def avg_doc_length(self) -> float:
        return self._total_tokens / self._doc_count if self._doc_count > 0 else 0.0

    def compute_idf(self) -> Dict[int, float]:
        """Compute IDF for all terms in the vocabulary.

        Uses the BM25 IDF formula:
          idf(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)

        This variant is always non-negative (unlike the classic log(N/df) which
        can be negative for very common terms).
        """
        N = self._doc_count
        idf_map: Dict[int, float] = {}
        for term, df in self._doc_freq.items():
            term_id = self._vocab[term]
            idf_val = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
            idf_map[term_id] = idf_val
        return idf_map

    def save(self, data_dir: str) -> None:
        """Persist vocabulary, IDF values, and BM25 parameters to disk.

        Files created:
          - vocab.json:       {term: term_id}
          - idf.json:         {term_id: idf_value}  (keys are strings due to JSON)
          - bm25_params.json: {N, avgdl, vocab_size}
        """
        os.makedirs(data_dir, exist_ok=True)

        vocab_path = os.path.join(data_dir, "vocab.json")
        idf_path = os.path.join(data_dir, "idf.json")
        params_path = os.path.join(data_dir, "bm25_params.json")

        # Vocabulary
        with open(vocab_path, "w") as f:
            json.dump(self._vocab, f)

        # IDF values (JSON keys must be strings)
        idf_map = self.compute_idf()
        idf_str_keys = {str(k): v for k, v in idf_map.items()}
        with open(idf_path, "w") as f:
            json.dump(idf_str_keys, f)

        # Corpus stats
        params = {
            "N": self._doc_count,
            "avgdl": self.avg_doc_length,
            "vocab_size": self.vocab_size,
            "total_tokens": self._total_tokens,
        }
        with open(params_path, "w") as f:
            json.dump(params, f, indent=2)


class BM25SparseEncoder:
    """Encodes documents and queries into BM25-weighted sparse vectors.

    Loaded from persisted vocabulary + IDF at startup (online).

    Both document encoding (offline indexing) and query encoding (online search)
    use the same vocabulary and IDF values, ensuring deterministic matching.
    """

    def __init__(
        self,
        vocab: Dict[str, int],
        idf: Dict[int, float],
        avgdl: float,
        N: int,
        k1: float = 1.2,
        b: float = 0.75,
    ):
        self._vocab = vocab
        self._idf = idf
        self._avgdl = avgdl
        self._N = N
        self._k1 = k1
        self._b = b

    @classmethod
    def load(cls, data_dir: str, k1: float = 1.2, b: float = 0.75) -> "BM25SparseEncoder":
        """Load a pre-built encoder from persisted files."""
        vocab_path = os.path.join(data_dir, "vocab.json")
        idf_path = os.path.join(data_dir, "idf.json")
        params_path = os.path.join(data_dir, "bm25_params.json")

        with open(vocab_path, "r") as f:
            vocab = json.load(f)

        with open(idf_path, "r") as f:
            idf_str = json.load(f)
            idf = {int(k): v for k, v in idf_str.items()}

        with open(params_path, "r") as f:
            params = json.load(f)

        return cls(
            vocab=vocab,
            idf=idf,
            avgdl=params["avgdl"],
            N=params["N"],
            k1=k1,
            b=b,
        )

    def encode_document(self, text: str) -> SparseVectorData:
        """Encode a document into a BM25-weighted sparse vector.

        Used during offline indexing (pass 2).
        """
        tokens = tokenize(text)
        if not tokens:
            return SparseVectorData(indices=[], values=[])

        dl = len(tokens)
        tf_counts = Counter(tokens)

        indices: List[int] = []
        values: List[float] = []

        for term, tf in tf_counts.items():
            term_id = self._vocab.get(term)
            if term_id is None:
                continue  # Term not in vocabulary (shouldn't happen for indexed docs)

            idf_val = self._idf.get(term_id, 0.0)
            if idf_val <= 0:
                continue  # Skip terms with zero/negative IDF

            # BM25 term weight
            numerator = tf * (self._k1 + 1)
            denominator = tf + self._k1 * (1 - self._b + self._b * dl / self._avgdl)
            weight = idf_val * numerator / denominator

            indices.append(term_id)
            values.append(weight)

        return SparseVectorData(indices=indices, values=values)

    def encode_query(self, text: str) -> SparseVectorData:
        """Encode a query into a BM25-weighted sparse vector.

        Used at query time (online). Deterministic — same text always
        produces the same vector.

        Query encoding uses IDF as the weight (no document-length normalization
        since queries are short and we want IDF-weighted term matching).
        """
        tokens = tokenize(text)
        if not tokens:
            return SparseVectorData(indices=[], values=[])

        tf_counts = Counter(tokens)
        indices: List[int] = []
        values: List[float] = []

        for term, tf in tf_counts.items():
            term_id = self._vocab.get(term)
            if term_id is None:
                continue  # Unknown term — not in corpus vocabulary

            idf_val = self._idf.get(term_id, 0.0)
            if idf_val <= 0:
                continue

            # For queries, weight = idf * tf (simple, since queries are short)
            weight = idf_val * tf
            indices.append(term_id)
            values.append(weight)

        return SparseVectorData(indices=indices, values=values)
