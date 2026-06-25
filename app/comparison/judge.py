from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from app.domain.section import Section
from app.domain.verdict import PairwiseVerdict
from app.ports.llm import LLMClient
from app.prompts.pairwise_judge import build_pairwise_judge_prompt


class _JudgeOutput(BaseModel):
    """Flat schema returned by the LLM via tool-use. No unions — Gemini compatible."""
    verdict: Literal["MATCH", "DIFF"]
    reason: str


class PairwiseJudge:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def judge(self, section_a: Section, section_b: Section) -> PairwiseVerdict:
        prompt = build_pairwise_judge_prompt(section_a, section_b)
        raw = await self._llm.generate(
            [{"role": "user", "content": prompt}],
            response_schema=_JudgeOutput,
        )
        output = _JudgeOutput.model_validate(raw)

        return PairwiseVerdict(
            section_id_a=section_a.id,
            section_id_b=section_b.id,
            verdict=output.verdict,
            reason=output.reason if output.verdict == "DIFF" else None,
            doc_a_text=section_a.body_text,
            doc_b_text=section_b.body_text,
            source_a=section_a.location.cite(),
            source_b=section_b.location.cite(),
        )