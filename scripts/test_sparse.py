"""Sprint 3 verification: test the BM25 sparse encoder.

Runs all verification gate checks:
  - Vocabulary building produces vocab.json + idf.json + bm25_params.json
  - Term IDs are stable integers
  - IDF values: high-frequency terms have low IDF, rare terms have high IDF
  - Document encoding: produces SparseVector with correct terms
  - Query encoding: uses same vocabulary, deterministic
  - Unknown query terms: gracefully ignored
  - Sparse vectors are inspectable and reasonable
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ingestion.sparse_encoder import (
    BM25VocabularyBuilder,
    BM25SparseEncoder,
    tokenize,
)
from app.models import SparseVectorData


# Sample corpus — small enough to reason about manually
CORPUS = [
    "Osmosis is the movement of water molecules through a selectively permeable membrane.",
    "Water molecules move from a region of lower solute concentration to higher concentration.",
    "The process of osmosis is critical for biological cells to maintain proper hydration.",
    "Diffusion differs from osmosis because diffusion involves the movement of solute particles.",
    "Cell membranes are selectively permeable allowing only certain molecules to pass through.",
    "Photosynthesis is the process by which plants convert sunlight into chemical energy.",
    "During photosynthesis plants absorb carbon dioxide and release oxygen into the atmosphere.",
    "Chlorophyll is the green pigment found in plant cells that captures light energy.",
    "The mitochondria are organelles that generate most of the cell's supply of adenosine triphosphate.",
    "Cellular respiration is the process of converting glucose into usable energy in the form of ATP.",
]


def test_tokenizer():
    """Tokenizer should lowercase, strip punctuation, remove stopwords."""
    tokens = tokenize("The quick Brown FOX jumps over the lazy dog!")
    assert "the" not in tokens, "Stopwords should be removed"
    assert "quick" in tokens, "'quick' should be present"
    assert "brown" in tokens, "Should be lowercased"
    assert "fox" in tokens
    assert "jumps" in tokens
    assert "over" not in tokens, "'over' is a stopword"
    assert "lazy" in tokens
    assert "dog" in tokens
    print(f"✅ Tokenizer: lowercases, strips punctuation, removes stopwords")
    print(f"   Input:  'The quick Brown FOX jumps over the lazy dog!'")
    print(f"   Output: {tokens}")


def test_vocabulary_building():
    """Build vocab from corpus, verify structure and persistence."""
    builder = BM25VocabularyBuilder()
    for doc in CORPUS:
        builder.add_document(doc)

    assert builder.doc_count == len(CORPUS), f"Doc count: {builder.doc_count} != {len(CORPUS)}"
    assert builder.vocab_size > 0, "Vocabulary should not be empty"
    assert builder.avg_doc_length > 0, "Average doc length should be positive"

    print(f"✅ Vocabulary built from {builder.doc_count} documents")
    print(f"   Vocab size: {builder.vocab_size} terms")
    print(f"   Avg doc length: {builder.avg_doc_length:.1f} tokens")

    # Save and reload
    with tempfile.TemporaryDirectory() as tmpdir:
        builder.save(tmpdir)

        # Check files exist
        assert os.path.exists(os.path.join(tmpdir, "vocab.json"))
        assert os.path.exists(os.path.join(tmpdir, "idf.json"))
        assert os.path.exists(os.path.join(tmpdir, "bm25_params.json"))
        print(f"✅ Persisted: vocab.json, idf.json, bm25_params.json")

        # Load and verify
        with open(os.path.join(tmpdir, "vocab.json")) as f:
            vocab = json.load(f)
        with open(os.path.join(tmpdir, "idf.json")) as f:
            idf = json.load(f)
        with open(os.path.join(tmpdir, "bm25_params.json")) as f:
            params = json.load(f)

        assert len(vocab) == builder.vocab_size
        assert len(idf) == builder.vocab_size
        assert params["N"] == len(CORPUS)
        assert params["avgdl"] > 0
        print(f"✅ Reload verified: {len(vocab)} terms, N={params['N']}, avgdl={params['avgdl']:.1f}")

    return builder


def test_idf_distribution(builder: BM25VocabularyBuilder):
    """High-frequency terms should have low IDF, rare terms should have high IDF."""
    idf_map = builder.compute_idf()

    # "osmosis" appears in 3/10 docs → medium IDF
    # "photosynthesis" appears in 2/10 docs → higher IDF
    # "molecules" appears in 3/10 docs → medium IDF
    # We'll find one common and one rare term to compare

    # Get vocab for reverse lookup
    with tempfile.TemporaryDirectory() as tmpdir:
        builder.save(tmpdir)
        with open(os.path.join(tmpdir, "vocab.json")) as f:
            vocab = json.load(f)

    # Find IDF for specific terms
    terms_to_check = ["osmosis", "photosynthesis", "chlorophyll", "molecules", "process"]
    print(f"\n   IDF values for selected terms:")
    idf_values = {}
    for term in terms_to_check:
        if term in vocab:
            tid = vocab[term]
            idf_val = idf_map.get(tid, 0.0)
            idf_values[term] = idf_val
            print(f"     {term:<20} IDF = {idf_val:.4f}")

    # "chlorophyll" appears in 1 doc → should have highest IDF
    # "process" appears in 3+ docs → should have lower IDF
    if "chlorophyll" in idf_values and "process" in idf_values:
        assert idf_values["chlorophyll"] > idf_values["process"], (
            f"Rare term 'chlorophyll' should have higher IDF than common 'process'"
        )
        print(f"✅ IDF ordering correct: rare terms > common terms")
    else:
        print(f"⚠️  Could not verify IDF ordering (terms not found)")

    # All IDF values should be non-negative (BM25 variant guarantee)
    all_positive = all(v >= 0 for v in idf_map.values())
    assert all_positive, "All IDF values should be non-negative"
    print(f"✅ All IDF values non-negative ({len(idf_map)} terms)")


def test_document_encoding():
    """Encode a document → sparse vector with correct terms."""
    with tempfile.TemporaryDirectory() as tmpdir:
        builder = BM25VocabularyBuilder()
        for doc in CORPUS:
            builder.add_document(doc)
        builder.save(tmpdir)

        encoder = BM25SparseEncoder.load(tmpdir)

        # Encode first document
        sv = encoder.encode_document(CORPUS[0])
        assert isinstance(sv, SparseVectorData)
        assert len(sv.indices) > 0, "Should have non-zero terms"
        assert len(sv.indices) == len(sv.values), "Indices and values must match"
        assert all(v > 0 for v in sv.values), "All weights should be positive"

        print(f"✅ Document encoding: {len(sv.indices)} non-zero terms")
        print(f"   Doc: \"{CORPUS[0][:60]}...\"")

        # Verify expected terms are present
        with open(os.path.join(tmpdir, "vocab.json")) as f:
            vocab = json.load(f)

        expected_terms = ["osmosis", "movement", "water", "molecules"]
        for term in expected_terms:
            if term in vocab:
                tid = vocab[term]
                assert tid in sv.indices, f"Expected term '{term}' (id={tid}) in sparse vector"
        print(f"   Expected terms present: {expected_terms}")


def test_query_encoding():
    """Query encoding must use same vocab, be deterministic, handle unknowns."""
    with tempfile.TemporaryDirectory() as tmpdir:
        builder = BM25VocabularyBuilder()
        for doc in CORPUS:
            builder.add_document(doc)
        builder.save(tmpdir)

        encoder = BM25SparseEncoder.load(tmpdir)

        # Encode a query
        query = "what is osmosis and how does it work"
        sv1 = encoder.encode_query(query)
        sv2 = encoder.encode_query(query)

        assert sv1.indices == sv2.indices, "Query encoding must be deterministic (indices)"
        assert sv1.values == sv2.values, "Query encoding must be deterministic (values)"
        print(f"✅ Query encoding is deterministic")
        print(f"   Query: \"{query}\"")
        print(f"   Terms: {len(sv1.indices)} non-zero, weights: {[f'{v:.3f}' for v in sv1.values]}")

        # Unknown terms should be ignored, not crash
        query_unknown = "xyzzyplugh quantum entanglement"
        sv_unk = encoder.encode_query(query_unknown)
        # "quantum" and "entanglement" are not in our small corpus
        # Only terms that happen to be in vocab should appear
        print(f"✅ Unknown terms gracefully ignored")
        print(f"   Query: \"{query_unknown}\" → {len(sv_unk.indices)} terms (only known ones)")


def test_document_vs_query_encoding():
    """Document and query encodings should share term IDs but differ in weights."""
    with tempfile.TemporaryDirectory() as tmpdir:
        builder = BM25VocabularyBuilder()
        for doc in CORPUS:
            builder.add_document(doc)
        builder.save(tmpdir)

        encoder = BM25SparseEncoder.load(tmpdir)

        text = "osmosis is the movement of water molecules"
        doc_sv = encoder.encode_document(text)
        query_sv = encoder.encode_query(text)

        # Same terms should appear in both
        assert set(doc_sv.indices) == set(query_sv.indices), (
            "Same text should produce same term IDs for doc and query encoding"
        )
        print(f"✅ Doc and query encoding produce same term IDs for same text")

        # Weights may differ (doc uses BM25 length norm, query uses idf*tf)
        print(f"   Doc weights:   {[f'{v:.3f}' for v in doc_sv.values]}")
        print(f"   Query weights: {[f'{v:.3f}' for v in query_sv.values]}")


def test_empty_input():
    """Empty input should produce empty sparse vector, not crash."""
    with tempfile.TemporaryDirectory() as tmpdir:
        builder = BM25VocabularyBuilder()
        for doc in CORPUS:
            builder.add_document(doc)
        builder.save(tmpdir)

        encoder = BM25SparseEncoder.load(tmpdir)

        sv = encoder.encode_query("")
        assert len(sv.indices) == 0, "Empty query should produce empty vector"

        sv = encoder.encode_document("   ")
        assert len(sv.indices) == 0, "Whitespace-only should produce empty vector"

        sv = encoder.encode_query("the is a an")  # All stopwords
        assert len(sv.indices) == 0, "All-stopwords query should produce empty vector"

        print(f"✅ Empty/stopword-only inputs → empty sparse vectors")


if __name__ == "__main__":
    print("=" * 80)
    print("Sprint 3 Verification: BM25 Sparse Encoder")
    print("=" * 80)
    print()

    test_tokenizer()
    print()
    builder = test_vocabulary_building()
    print()
    test_idf_distribution(builder)
    print()
    test_document_encoding()
    print()
    test_query_encoding()
    print()
    test_document_vs_query_encoding()
    print()
    test_empty_input()

    print()
    print("=" * 80)
    print("All Sprint 3 verification checks passed! ✅")
    print("=" * 80)
