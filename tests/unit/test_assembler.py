"""Tests for assemble() — pure mapping from alignment + verdicts → ComparisonResult."""
from __future__ import annotations

import pytest

from app.comparison.assembler import assemble
from app.domain.section import Location, Section
from app.domain.verdict import MissingExplanationBatch, PairwiseVerdict
from app.domain.comparison import MissingEntry


def _sec(doc_id: str, num: str, heading: str, body: str = "body text", filename: str | None = None) -> Section:
    fname = filename or f"{doc_id}.pdf"
    return Section(
        id=f"{doc_id}::{num}",
        location=Location(filename=fname, heading_number=num, heading_path=[num, heading]),
        heading=heading,
        body_text=body,
    )


def _verdict(sec_a: Section, sec_b: Section, verdict: str, reason: str = "") -> PairwiseVerdict:
    return PairwiseVerdict(
        section_id_a=sec_a.id,
        section_id_b=sec_b.id,
        verdict=verdict,  # type: ignore[arg-type]
        reason=reason or None,
        doc_a_text=sec_a.body_text,
        doc_b_text=sec_b.body_text,
        source_a=sec_a.location.cite(),
        source_b=sec_b.location.cite(),
    )


class TestMatchVerdicts:
    def test_match_verdict_produces_match_entry(self) -> None:
        sa = _sec("A", "1.0", "Intro", "Body A")
        sb = _sec("B", "1.0", "Intro", "Body A", filename="doc.docx")
        v = _verdict(sa, sb, "MATCH")

        result = assemble(
            aligned_pairs=[(sa, sb)],
            verdicts={sa.id: v},
            unmatched_a=[],
            unmatched_b=[],
            missing_explanations=MissingExplanationBatch(entries=[]),
        )

        assert len(result.match) == 1
        assert len(result.diff) == 0
        assert result.match[0].textA == "Body A"

    def test_match_entry_source_combines_both_citations(self) -> None:
        sa = _sec("A", "1.0", "Intro")
        sb = _sec("B", "1.0", "Intro", filename="doc.docx")
        v = _verdict(sa, sb, "MATCH")

        result = assemble([(sa, sb)], {sa.id: v}, [], [], MissingExplanationBatch(entries=[]))

        assert "+" in result.match[0].source


class TestDiffVerdicts:
    def test_diff_verdict_produces_diff_entry(self) -> None:
        sa = _sec("A", "2.0", "Pricing", "Old logic")
        sb = _sec("B", "2.0", "Pricing", "New logic", filename="doc.docx")
        v = _verdict(sa, sb, "DIFF", reason="Logic changed")

        result = assemble([(sa, sb)], {sa.id: v}, [], [], MissingExplanationBatch(entries=[]))

        assert len(result.diff) == 1
        assert len(result.match) == 0
        diff = result.diff[0]
        assert diff.docA_text == "Old logic"
        assert diff.docB_text == "New logic"
        assert diff.reason == "Logic changed"

    def test_diff_entry_has_distinct_sourceA_and_sourceB(self) -> None:
        sa = _sec("A", "2.0", "Pricing")
        sb = _sec("B", "2.0", "Pricing", filename="FDS_V5.docx")
        v = _verdict(sa, sb, "DIFF")

        result = assemble([(sa, sb)], {sa.id: v}, [], [], MissingExplanationBatch(entries=[]))

        diff = result.diff[0]
        assert diff.sourceA != diff.sourceB
        assert "A.pdf" in diff.sourceA
        assert "FDS_V5.docx" in diff.sourceB


class TestMissingEntries:
    def test_unmatched_a_produces_missing_entry_with_v0_source_file(self) -> None:
        sa = _sec("A", "3.0", "Legacy", "Old stuff")
        result = assemble([], {}, [sa], [], MissingExplanationBatch(entries=[]))

        assert len(result.missing) == 1
        entry = result.missing[0]
        assert entry.source_file == "A.pdf"
        assert entry.text == "Old stuff"

    def test_unmatched_b_produces_missing_entry_with_v5_source_file(self) -> None:
        sb = _sec("B", "5.0", "New Feature", "New stuff", filename="FDS_V5.docx")
        result = assemble([], {}, [], [sb], MissingExplanationBatch(entries=[]))

        assert len(result.missing) == 1
        assert result.missing[0].source_file == "FDS_V5.docx"

    def test_explanation_populated_from_missing_explanations_batch(self) -> None:
        sa = _sec("A", "3.0", "Legacy")
        cite = sa.location.cite()
        batch = MissingExplanationBatch(entries=[
            MissingEntry(
                text=sa.body_text or sa.heading,
                source_file=sa.location.filename,
                location=cite,
                explanation="Removed during V5 migration.",
            )
        ])

        result = assemble([], {}, [sa], [], batch)

        assert result.missing[0].explanation == "Removed during V5 migration."

    def test_missing_entry_uses_default_explanation_when_not_in_batch(self) -> None:
        sa = _sec("A", "3.0", "Legacy")
        result = assemble([], {}, [sa], [], MissingExplanationBatch(entries=[]))

        assert result.missing[0].explanation != ""


class TestMixedResult:
    def test_mixed_result_populates_all_three_lists(self) -> None:
        sa1 = _sec("A", "1.0", "Intro", "Body A1")
        sb1 = _sec("B", "1.0", "Intro", "Body A1", filename="B.docx")
        sa2 = _sec("A", "2.0", "Pricing", "Old price")
        sb2 = _sec("B", "2.0", "Pricing", "New price", filename="B.docx")
        sa3 = _sec("A", "3.0", "Legacy")

        verdicts = {
            sa1.id: _verdict(sa1, sb1, "MATCH"),
            sa2.id: _verdict(sa2, sb2, "DIFF", reason="changed"),
        }
        result = assemble(
            aligned_pairs=[(sa1, sb1), (sa2, sb2)],
            verdicts=verdicts,
            unmatched_a=[sa3],
            unmatched_b=[],
            missing_explanations=MissingExplanationBatch(entries=[]),
        )

        assert len(result.match) == 1
        assert len(result.diff) == 1
        assert len(result.missing) == 1

    def test_output_validates_against_brief_json_schema(self) -> None:
        result = assemble([], {}, [], [], MissingExplanationBatch(entries=[]))
        d = result.model_dump()
        assert set(d.keys()) == {"missing", "diff", "match"}

    def test_pair_without_verdict_is_silently_skipped(self) -> None:
        sa = _sec("A", "1.0", "Intro")
        sb = _sec("B", "1.0", "Intro", filename="B.docx")
        # No verdict provided for this pair
        result = assemble([(sa, sb)], {}, [], [], MissingExplanationBatch(entries=[]))

        assert result.match == []
        assert result.diff == []
