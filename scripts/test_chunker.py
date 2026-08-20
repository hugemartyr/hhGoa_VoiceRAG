"""Sprint 2 verification: test the chunker against real and synthetic data.

Runs all verification gate checks and prints detailed output for manual inspection.
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ingestion.chunker import chunk_passage, chunk_msmarco_row, _make_chunk_id, _count_tokens


def test_deterministic_ids():
    """chunk_id must be deterministic: same input → same output."""
    id1 = _make_chunk_id(123, 0, 0)
    id2 = _make_chunk_id(123, 0, 0)
    id3 = _make_chunk_id(123, 0, 1)  # different chunk_index → different ID
    assert id1 == id2, f"IDs not deterministic: {id1} != {id2}"
    assert id1 != id3, f"Different inputs should produce different IDs: {id1} == {id3}"
    print("✅ Deterministic chunk IDs: same input → same ID, different input → different ID")


def test_short_passage():
    """Short passage (≤256 tokens) → 1 chunk, type=passage, is_retrieval_unit=true."""
    text = "Osmosis is the movement of water molecules through a selectively permeable membrane from a region of lower solute concentration to a region of higher solute concentration."
    chunks = chunk_passage(text, query_id=1, passage_index=0, is_selected=1, query_type="description")

    assert len(chunks) == 1, f"Expected 1 chunk, got {len(chunks)}"
    c = chunks[0]
    assert c.metadata.chunk_type == "passage", f"Expected 'passage', got '{c.metadata.chunk_type}'"
    assert c.metadata.is_retrieval_unit is True
    assert c.metadata.parent_id is None
    assert c.metadata.passage_index == 0
    assert c.metadata.is_selected == 1
    assert c.metadata.query_type == "description"
    assert c.metadata.query_id == 1
    assert c.metadata.token_count > 0
    assert c.text == text

    print("✅ Short passage → 1 chunk, type='passage', is_retrieval_unit=True, no parent")
    print(f"   text: \"{c.text[:80]}...\"")
    print(f"   token_count: {c.metadata.token_count}")
    print(f"   chunk_id: {c.metadata.chunk_id}")


def test_long_passage():
    """Long passage (>256 tokens) → 1 parent + N sentence children."""
    # Construct a passage that's clearly >256 words
    sentences = [
        "The water cycle, also known as the hydrological cycle, describes the continuous movement of water on, above, and below the surface of the Earth.",
        "The mass of water on Earth remains fairly constant over time but the partitioning of the water into the major reservoirs of ice, fresh water, saline water, and atmospheric water is variable depending on a wide range of climatic variables.",
        "The water moves from one reservoir to another, such as from river to ocean, or from the ocean to the atmosphere, by the physical processes of evaporation, condensation, precipitation, infiltration, surface runoff, and subsurface flow.",
        "In doing so, the water goes through different forms: liquid, solid ice, and vapor.",
        "The water cycle involves the exchange of energy, which leads to temperature changes.",
        "When water evaporates, it takes up energy from its surroundings and cools the environment.",
        "When it condenses, it releases energy and warms the environment.",
        "These heat exchanges influence climate.",
        "The evaporative phase of the cycle purifies water which then replenishes the land with freshwater.",
        "The flow of liquid water and ice transports minerals across the globe.",
        "It is also involved in reshaping the geological features of the Earth, through processes including erosion and sedimentation.",
        "The water cycle is also essential for the maintenance of most life and ecosystems on the planet.",
        "Precipitation is a vital component of how water moves through Earth's water cycle, connecting the ocean, land, and atmosphere.",
        "Knowing where it rains, how much it rains, and the character of the falling rain, snow, or hail allows scientists to better understand precipitation's impact on streams, rivers, surface runoff, and groundwater.",
        "Precipitation affects the distribution and abundance of plants and animals in terrestrial ecosystems.",
    ]
    text = " ".join(sentences)
    token_count = _count_tokens(text)
    assert token_count > 256, f"Test passage too short: {token_count} tokens"

    chunks = chunk_passage(text, query_id=99, passage_index=2, is_selected=0, query_type="description")

    # Should have 1 parent + N children
    parents = [c for c in chunks if c.metadata.chunk_type == "parent"]
    children = [c for c in chunks if c.metadata.chunk_type == "sentence"]
    passages = [c for c in chunks if c.metadata.chunk_type == "passage"]

    assert len(parents) == 1, f"Expected 1 parent, got {len(parents)}"
    assert len(children) > 1, f"Expected multiple children, got {len(children)}"
    assert len(passages) == 0, "Should have no passage-type chunks for long passages"

    parent = parents[0]
    assert parent.metadata.is_retrieval_unit is False, "Parent must NOT be a retrieval unit"
    assert parent.text == text, "Parent must contain the full passage text"
    assert parent.metadata.parent_id is None, "Parent should not have a parent_id"

    for child in children:
        assert child.metadata.is_retrieval_unit is True, "Children must be retrieval units"
        assert child.metadata.parent_id == parent.metadata.chunk_id, (
            f"Child parent_id mismatch: {child.metadata.parent_id} != {parent.metadata.chunk_id}"
        )
        assert child.metadata.chunk_type == "sentence"
        assert child.metadata.passage_index == 2
        assert child.metadata.query_id == 99

    print(f"✅ Long passage ({token_count} tokens) → 1 parent + {len(children)} sentence children")
    print(f"   Parent: is_retrieval_unit=False, chunk_id={parent.metadata.chunk_id}")
    for i, child in enumerate(children):
        print(f"   Child {i}: \"{child.text[:70]}...\" (tokens={child.metadata.token_count})")


def test_no_duplicate_ids():
    """Chunk IDs must be unique across a batch of passages."""
    passages = [
        "This is passage one about science and technology.",
        "This is passage two about history and culture.",
        "This is passage three about mathematics and physics.",
    ]
    is_selected = [1, 0, 1]

    chunks = chunk_msmarco_row(
        query_id=42,
        english_passages=passages,
        is_selected_list=is_selected,
        query_type="factoid",
    )

    chunk_ids = [c.metadata.chunk_id for c in chunks]
    unique_ids = set(chunk_ids)
    assert len(chunk_ids) == len(unique_ids), (
        f"Duplicate IDs found! {len(chunk_ids)} chunks but {len(unique_ids)} unique IDs"
    )
    print(f"✅ No duplicate chunk IDs across {len(chunks)} chunks from {len(passages)} passages")


def test_idempotent_rechunking():
    """Running the chunker twice on the same input produces identical results."""
    passages = ["The quick brown fox jumps over the lazy dog. " * 10]
    is_selected = [1]

    chunks_a = chunk_msmarco_row(query_id=7, english_passages=passages, is_selected_list=is_selected)
    chunks_b = chunk_msmarco_row(query_id=7, english_passages=passages, is_selected_list=is_selected)

    assert len(chunks_a) == len(chunks_b), "Different number of chunks on re-run"
    for a, b in zip(chunks_a, chunks_b):
        assert a.metadata.chunk_id == b.metadata.chunk_id, f"ID mismatch: {a.metadata.chunk_id} != {b.metadata.chunk_id}"
        assert a.text == b.text, "Text mismatch on re-run"

    print(f"✅ Idempotent: rechunking produces identical {len(chunks_a)} chunks")


def test_empty_passage():
    """Empty passage should produce no chunks."""
    chunks = chunk_passage("", query_id=1, passage_index=0, is_selected=0, query_type="")
    assert len(chunks) == 0, f"Expected 0 chunks for empty passage, got {len(chunks)}"

    chunks = chunk_passage("   ", query_id=1, passage_index=0, is_selected=0, query_type="")
    assert len(chunks) == 0, f"Expected 0 chunks for whitespace passage, got {len(chunks)}"

    print("✅ Empty/whitespace passages → 0 chunks")


def test_metadata_completeness():
    """All required metadata fields must be present and correctly typed."""
    text = "Photosynthesis converts sunlight into chemical energy."
    chunks = chunk_passage(text, query_id=555, passage_index=3, is_selected=1, query_type="factoid")

    c = chunks[0]
    meta = c.metadata

    required_fields = {
        "chunk_id": str,
        "chunk_type": str,
        "is_retrieval_unit": bool,
        "passage_index": int,
        "is_selected": int,
        "query_type": str,
        "query_id": int,
        "token_count": int,
    }

    for field, expected_type in required_fields.items():
        value = getattr(meta, field)
        assert isinstance(value, expected_type), (
            f"Field '{field}' has type {type(value).__name__}, expected {expected_type.__name__}"
        )

    print("✅ All metadata fields present with correct types")
    print(f"   Fields: {list(required_fields.keys())}")


def test_full_row_chunking():
    """Chunk an entire MSMARCO-style row with mixed short and long passages."""
    passages = [
        # Short passage (should be a single chunk)
        "The Eiffel Tower is a wrought-iron lattice tower in Paris, France.",
        # Medium passage (still under 256 tokens)
        "The tower was constructed from 1887 to 1889 as the centerpiece of the 1889 World's Fair. It was initially criticized by some of France's leading artists and intellectuals for its design, but it has become a global cultural icon of France.",
        # Long passage (>256 tokens, should split)
        " ".join([
            "The Eiffel Tower is 330 metres tall, about the same height as an 81-storey building.",
            "It is the tallest structure in Paris and the most-visited paid monument in the world.",
            "The tower has three levels for visitors, with restaurants on the first and second levels.",
            "The top level's upper platform is 276 metres above the ground.",
            "Tickets can be purchased to ascend by stairs or lift to the first and second levels.",
            "The climb from ground level to the first level is over 300 steps, as is the climb from the first level to the second.",
            "Although there is a staircase to the top level, it is usually accessible only by lift.",
            "The tower was the world's tallest man-made structure for 41 years until the completion of the Chrysler Building in New York City in 1930.",
            "Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres.",
            "Not including broadcast antennas, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct.",
            "The tower was built as the entrance arch for the World's Fair and many people were against the idea of building it.",
            "The design of the tower was selected through a competition and Gustave Eiffel's company won with their proposal.",
            "Construction of the tower began in January 1887 and was completed in March 1889, taking a total of two years, two months, and five days.",
            "The tower was inaugurated on March 31, 1889, and opened to the public on May 6 of that same year.",
            "During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the entire world.",
        ]),
    ]
    is_selected = [0, 1, 1]

    chunks = chunk_msmarco_row(
        query_id=1001,
        english_passages=passages,
        is_selected_list=is_selected,
        query_type="description",
    )

    print(f"\n📋 Full row chunking: {len(passages)} passages → {len(chunks)} chunks")
    print("-" * 80)

    for i, c in enumerate(chunks):
        meta = c.metadata
        print(
            f"  [{i}] type={meta.chunk_type:<8} retrieval={str(meta.is_retrieval_unit):<5} "
            f"p_idx={meta.passage_index} parent={meta.parent_id or 'None':<16} "
            f"tokens={meta.token_count:<4} id={meta.chunk_id}"
        )
        # Show truncated text for readability
        display_text = c.text[:100].replace("\n", " ")
        if len(c.text) > 100:
            display_text += "..."
        print(f"       \"{display_text}\"")

    # Verify structure
    parent_chunks = [c for c in chunks if c.metadata.chunk_type == "parent"]
    sentence_chunks = [c for c in chunks if c.metadata.chunk_type == "sentence"]
    passage_chunks = [c for c in chunks if c.metadata.chunk_type == "passage"]

    print(f"\n  Summary: {len(passage_chunks)} passage + {len(parent_chunks)} parent + {len(sentence_chunks)} sentence")

    # The first two passages should be passage-type (short)
    assert passage_chunks[0].metadata.passage_index == 0
    assert passage_chunks[1].metadata.passage_index == 1

    # The third passage should have a parent + children
    assert len(parent_chunks) >= 1
    assert all(c.metadata.passage_index == 2 for c in parent_chunks)
    assert all(c.metadata.passage_index == 2 for c in sentence_chunks)

    print("✅ Mixed short/long passages chunked correctly")


if __name__ == "__main__":
    print("=" * 80)
    print("Sprint 2 Verification: Chunking Engine")
    print("=" * 80)
    print()

    test_deterministic_ids()
    print()
    test_short_passage()
    print()
    test_long_passage()
    print()
    test_no_duplicate_ids()
    print()
    test_idempotent_rechunking()
    print()
    test_empty_passage()
    print()
    test_metadata_completeness()
    print()
    test_full_row_chunking()

    print()
    print("=" * 80)
    print("All Sprint 2 verification checks passed! ✅")
    print("=" * 80)
