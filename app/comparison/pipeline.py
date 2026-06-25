from __future__ import annotations

import logging
import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from app.comparison.aligner import HeadingAligner
from app.comparison.assembler import assemble
from app.comparison.explainer import MissingExplainer
from app.comparison.judge import PairwiseJudge
from app.comparison.ranker import Top10Ranker
from app.domain.comparison import ComparisonResult
from app.domain.section import ParsedDoc, Section
from app.domain.summary import ExecutiveSummary
from app.domain.verdict import MissingExplanationBatch, PairwiseVerdict
from app.ports.embedder import EmbedderClient
from app.ports.llm import LLMClient

logger = logging.getLogger(__name__)


# ── state ─────────────────────────────────────────────────────────────────────

class PipelineState(TypedDict):
    doc_a: ParsedDoc
    doc_b: ParsedDoc
    heading_embeddings: dict[str, list[float]]
    aligned_pairs: list[tuple[Section, Section]]
    unmatched_a: list[Section]
    unmatched_b: list[Section]
    verdicts: Annotated[dict[str, PairwiseVerdict], operator.or_]
    missing_explanations: MissingExplanationBatch
    comparison_result: ComparisonResult
    top_10: ExecutiveSummary


# ── pipeline ──────────────────────────────────────────────────────────────────

class ComparisonPipeline:
    def __init__(
        self,
        llm: LLMClient,
        embedder: EmbedderClient,
        aligner: HeadingAligner,
        judge: PairwiseJudge,
        explainer: MissingExplainer,
        ranker: Top10Ranker,
    ) -> None:
        self._llm = llm
        self._embedder = embedder
        self._aligner = aligner
        self._judge = judge
        self._explainer = explainer
        self._ranker = ranker
        self._graph = self._build()

    # ── nodes ─────────────────────────────────────────────────────────────────

    @staticmethod
    async def _load_parsed_docs(state: PipelineState) -> dict[str, Any]:
        logger.info(
            "Pipeline start: doc_a=%s (%d sections), doc_b=%s (%d sections)",
            state["doc_a"].filename, len(state["doc_a"].sections),
            state["doc_b"].filename, len(state["doc_b"].sections),
        )
        return {}

    async def _embed_headings(self, state: PipelineState) -> dict[str, Any]:
        all_secs = state["doc_a"].sections + state["doc_b"].sections
        logger.info("Embedding %d headings", len(all_secs))
        embs = await self._embedder.embed_for_similarity([s.heading for s in all_secs])
        return {"heading_embeddings": {s.id: e for s, e in zip(all_secs, embs)}}

    async def _align_sections(self, state: PipelineState) -> dict[str, Any]:
        logger.info("Aligning sections")
        result = await self._aligner.align(state["doc_a"], state["doc_b"])
        logger.info(
            "Alignment: %d pairs, %d unmatched_a, %d unmatched_b",
            len(result.aligned_pairs), len(result.unmatched_a), len(result.unmatched_b),
        )
        return {
            "aligned_pairs": result.aligned_pairs,
            "unmatched_a": result.unmatched_a,
            "unmatched_b": result.unmatched_b,
        }

    @staticmethod
    def _route_after_alignment(state: PipelineState) -> list[Send]:
        sends: list[Send] = [
            Send("judge_pair", {"section_a": sa, "section_b": sb})
            for sa, sb in state["aligned_pairs"]
        ]
        sends.append(Send("explain_missing", {
            "unmatched_a": state["unmatched_a"],
            "unmatched_b": state["unmatched_b"],
        }))
        return sends

    async def _judge_pair(self, state: dict[str, Any]) -> dict[str, Any]:
        sa: Section = state["section_a"]
        sb: Section = state["section_b"]
        verdict = await self._judge.judge(sa, sb)
        logger.debug("Judged %s ↔ %s → %s", sa.id, sb.id, verdict.verdict)
        return {"verdicts": {sa.id: verdict}}

    async def _explain_missing(self, state: dict[str, Any]) -> dict[str, Any]:
        unmatched_a: list[Section] = state["unmatched_a"]
        unmatched_b: list[Section] = state["unmatched_b"]
        logger.info(
            "Explaining %d missing sections", len(unmatched_a) + len(unmatched_b)
        )
        batch = await self._explainer.explain(unmatched_a, unmatched_b)
        return {"missing_explanations": batch}

    @staticmethod
    async def _assemble_result(state: PipelineState) -> dict[str, Any]:
        logger.info(
            "Assembling: %d verdicts, %d missing entries",
            len(state["verdicts"]), len(state["missing_explanations"].entries),
        )
        result = assemble(
            aligned_pairs=state["aligned_pairs"],
            verdicts=state["verdicts"],
            unmatched_a=state["unmatched_a"],
            unmatched_b=state["unmatched_b"],
            missing_explanations=state["missing_explanations"],
        )
        return {"comparison_result": result}

    async def _rank_top10(self, state: PipelineState) -> dict[str, Any]:
        logger.info("Ranking top-10 changes")
        summary = await self._ranker.rank(state["comparison_result"])
        return {"top_10": summary}

    # ── graph construction ────────────────────────────────────────────────────

    def _build(self) -> Any:
        g: StateGraph = StateGraph(PipelineState)  # type: ignore[arg-type]

        g.add_node("load_parsed_docs", self._load_parsed_docs)   # type: ignore[arg-type]
        g.add_node("embed_headings",   self._embed_headings)     # type: ignore[arg-type]
        g.add_node("align_sections",   self._align_sections)     # type: ignore[arg-type]
        g.add_node("judge_pair",       self._judge_pair)         # type: ignore[arg-type]
        g.add_node("explain_missing",  self._explain_missing)    # type: ignore[arg-type]
        g.add_node("assemble_result",  self._assemble_result)    # type: ignore[arg-type]
        g.add_node("rank_top10",       self._rank_top10)         # type: ignore[arg-type]

        g.add_edge(START,                "load_parsed_docs")
        g.add_edge("load_parsed_docs",   "embed_headings")
        g.add_edge("embed_headings",     "align_sections")
        g.add_conditional_edges("align_sections", self._route_after_alignment)
        g.add_edge("judge_pair",         "assemble_result")
        g.add_edge("explain_missing",    "assemble_result")
        g.add_edge("assemble_result",    "rank_top10")
        g.add_edge("rank_top10",         END)

        return g.compile()

    # ── entry point ───────────────────────────────────────────────────────────

    async def run(
        self, parsed_a: ParsedDoc, parsed_b: ParsedDoc
    ) -> tuple[ComparisonResult, ExecutiveSummary]:
        initial: PipelineState = {
            "doc_a": parsed_a,
            "doc_b": parsed_b,
            "heading_embeddings": {},
            "aligned_pairs": [],
            "unmatched_a": [],
            "unmatched_b": [],
            "verdicts": {},
            "missing_explanations": MissingExplanationBatch(entries=[]),
            "comparison_result": ComparisonResult(),
            "top_10": ExecutiveSummary(
                top_changes=[], total_matches=0, total_diffs=0, total_missing=0
            ),
        }
        final = await self._graph.ainvoke(initial)
        return final["comparison_result"], final["top_10"]