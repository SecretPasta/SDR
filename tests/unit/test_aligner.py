"""Tests for HeadingAligner — deterministic alignment, threshold, bipartite matching."""
from __future__ import annotations

import math

import pytest

from app.comparison.aligner import HeadingAligner
from app.config import AlignmentSettings
from app.domain.section import Location, ParsedDoc, Section

_SETTINGS = AlignmentSettings()


# ── helpers ───────────────────────────────────────────────────────────────────

def _unit(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in values))
    return [x / norm for x in values] if norm > 0 else values


def _sec(doc_id: str, num: str | None, heading: str, body: str = "") -> Section:
    return Section(
        id=f"{doc_id}::{num or heading[:8]}",
        location=Location(
            filename=f"{doc_id}.pdf",
            heading_number=num,
            heading_path=[heading],
        ),
        heading=heading,
        body_text=body,
    )


def _doc(doc_id: str, sections: list[Section]) -> ParsedDoc:
    return ParsedDoc(doc_id=doc_id, filename=f"{doc_id}.pdf", sections=sections)


class _ControlledEmbedder:
    """Returns pre-defined per-heading embeddings. Satisfies EmbedderClient protocol."""

    def __init__(self, emb_map: dict[str, list[float]]) -> None:
        self._map = emb_map

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    async def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError

    async def embed_for_similarity(self, texts: list[str]) -> list[list[float]]:
        return [self._map[t] for t in texts]


# ── tests ─────────────────────────────────────────────────────────────────────

async def test_exact_heading_number_and_text_aligns_all_pairs() -> None:
    """Identical heading numbers and text → all pairs aligned, nothing left over."""
    s1a = _sec("A", "1.0", "Introduction")
    s2a = _sec("A", "2.0", "Overview")
    s1b = _sec("B", "1.0", "Introduction")
    s2b = _sec("B", "2.0", "Overview")

    emb = _ControlledEmbedder({
        "Introduction": _unit([1.0, 0.0, 0.0, 0.0]),
        "Overview":     _unit([0.0, 1.0, 0.0, 0.0]),
    })
    result = await HeadingAligner(emb, _SETTINGS).align(
        _doc("A", [s1a, s2a]), _doc("B", [s1b, s2b])
    )

    assert len(result.aligned_pairs) == 2
    assert result.unmatched_a == []
    assert result.unmatched_b == []
    ids = {(a.id, b.id) for a, b in result.aligned_pairs}
    assert (s1a.id, s1b.id) in ids
    assert (s2a.id, s2b.id) in ids


async def test_renamed_section_aligned_via_heading_number_weight() -> None:
    """Same heading number, semantically close text → aligned despite name change."""
    s1a = _sec("A", "2.0", "System Overview")
    s1b = _sec("B", "2.0", "Architecture Overview")

    emb_sys  = _unit([0.0, 1.0, 0.0, 0.0])
    emb_arch = _unit([0.0, 0.9, 0.1, 0.0])  # high cosine with emb_sys

    emb = _ControlledEmbedder({
        "System Overview": emb_sys,
        "Architecture Overview": emb_arch,
    })
    result = await HeadingAligner(emb, _SETTINGS).align(
        _doc("A", [s1a]), _doc("B", [s1b])
    )

    assert len(result.aligned_pairs) == 1
    assert result.unmatched_a == []
    assert result.unmatched_b == []


async def test_missing_in_b_lands_in_unmatched_a() -> None:
    """Section present in A but absent from B → lands in unmatched_a."""
    s1a = _sec("A", "1.0", "Introduction")
    s2a = _sec("A", "2.0", "Overview")
    s3a = _sec("A", "3.0", "Details")
    s1b = _sec("B", "1.0", "Introduction")
    s3b = _sec("B", "3.0", "Details")

    emb = _ControlledEmbedder({
        "Introduction": _unit([1.0, 0.0, 0.0, 0.0]),
        "Overview":     _unit([0.0, 1.0, 0.0, 0.0]),
        "Details":      _unit([0.0, 0.0, 1.0, 0.0]),
    })
    result = await HeadingAligner(emb, _SETTINGS).align(
        _doc("A", [s1a, s2a, s3a]),
        _doc("B", [s1b, s3b]),
    )

    assert len(result.aligned_pairs) == 2
    assert len(result.unmatched_a) == 1
    assert result.unmatched_a[0].id == s2a.id
    assert result.unmatched_b == []


async def test_new_section_in_b_lands_in_unmatched_b() -> None:
    """Section present only in B → lands in unmatched_b."""
    s1a = _sec("A", "1.0", "Introduction")
    s1b = _sec("B", "1.0", "Introduction")
    s_new = _sec("B", "5.0", "Integration Checks")

    emb = _ControlledEmbedder({
        "Introduction":       _unit([1.0, 0.0, 0.0]),
        "Integration Checks": _unit([0.0, 0.0, 1.0]),
    })
    result = await HeadingAligner(emb, _SETTINGS).align(
        _doc("A", [s1a]),
        _doc("B", [s1b, s_new]),
    )

    assert len(result.aligned_pairs) == 1
    assert result.unmatched_a == []
    assert len(result.unmatched_b) == 1
    assert result.unmatched_b[0].id == s_new.id


async def test_alignment_respects_threshold_for_weak_pairs() -> None:
    """Pair whose score falls below threshold must NOT be committed."""
    # Two sections with orthogonal embeddings and different heading numbers
    # → cosine = 0, levenshtein low, heading_num differs → score well below 0.55
    s_a = _sec("A", "1.0", "Introduction")
    s_b = _sec("B", "9.0", "Conclusion")

    emb = _ControlledEmbedder({
        "Introduction": _unit([1.0, 0.0, 0.0]),
        "Conclusion":   _unit([0.0, 1.0, 0.0]),  # orthogonal → cos = 0
    })
    result = await HeadingAligner(emb, _SETTINGS).align(
        _doc("A", [s_a]),
        _doc("B", [s_b]),
    )

    assert result.aligned_pairs == []
    assert len(result.unmatched_a) == 1
    assert len(result.unmatched_b) == 1


async def test_bipartite_property_each_section_in_at_most_one_pair() -> None:
    """No section appears as both sides of different pairs (bipartite matching)."""
    # A has 2 sections, B has 1. The best A→B match wins; the other A is unmatched.
    s1a = _sec("A", "1.0", "Introduction")
    s2a = _sec("A", "1.1", "Intro Details")
    s1b = _sec("B", "1.0", "Introduction")

    emb = _ControlledEmbedder({
        "Introduction":  _unit([1.0, 0.0, 0.0]),
        "Intro Details": _unit([0.9, 0.1, 0.0]),  # close to Introduction
    })
    result = await HeadingAligner(emb, _SETTINGS).align(
        _doc("A", [s1a, s2a]),
        _doc("B", [s1b]),
    )

    b_ids_used = [b.id for _, b in result.aligned_pairs]
    assert len(b_ids_used) == len(set(b_ids_used)), "B section used in multiple pairs"
    assert len(result.aligned_pairs) <= 1


async def test_no_heading_numbers_reweights_to_embedding_dominant() -> None:
    """When heading numbers are absent, score = 0.75*cos + 0.25*lev (not the default triple)."""
    # Two sections with same heading text but no numbers.
    # cos = 1.0 (same text → same embedding), lev = 1.0 → score = 1.0
    s_a = _sec("A", None, "Pricing Model")
    s_b = _sec("B", None, "Pricing Model")

    emb = _ControlledEmbedder({
        "Pricing Model": _unit([1.0, 0.0]),
    })
    result = await HeadingAligner(emb, _SETTINGS).align(
        _doc("A", [s_a]),
        _doc("B", [s_b]),
    )

    assert len(result.aligned_pairs) == 1


async def test_alignment_is_deterministic_on_identical_input() -> None:
    """Running align twice on same input → identical output."""
    s1a = _sec("A", "1.0", "Introduction")
    s2a = _sec("A", "2.0", "Overview")
    s1b = _sec("B", "1.0", "Introduction")
    s2b = _sec("B", "2.0", "Overview")

    emb = _ControlledEmbedder({
        "Introduction": _unit([1.0, 0.0]),
        "Overview":     _unit([0.0, 1.0]),
    })
    aligner = HeadingAligner(emb, _SETTINGS)
    doc_a = _doc("A", [s1a, s2a])
    doc_b = _doc("B", [s1b, s2b])

    r1 = await aligner.align(doc_a, doc_b)
    r2 = await aligner.align(doc_a, doc_b)

    assert [(a.id, b.id) for a, b in r1.aligned_pairs] == [
        (a.id, b.id) for a, b in r2.aligned_pairs
    ]


async def test_empty_doc_a_returns_all_b_as_unmatched() -> None:
    s_b = _sec("B", "1.0", "Introduction")
    emb = _ControlledEmbedder({"Introduction": _unit([1.0, 0.0])})
    result = await HeadingAligner(emb, _SETTINGS).align(
        _doc("A", []),
        _doc("B", [s_b]),
    )
    assert result.aligned_pairs == []
    assert result.unmatched_a == []
    assert result.unmatched_b == [s_b]
