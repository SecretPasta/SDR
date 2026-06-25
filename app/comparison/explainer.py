from __future__ import annotations

from pydantic import BaseModel

from app.domain.comparison import MissingEntry
from app.domain.section import Section
from app.domain.verdict import MissingExplanationBatch
from app.ports.llm import LLMClient
from app.prompts.missing_explainer import build_missing_explainer_prompt


class _SectionExplanation(BaseModel):
    section_id: str
    explanation: str

    model_config = {"json_schema_extra": None}


class _ExplainerOutput(BaseModel):
    """Flat list schema — no unions, Gemini compatible.

    explanations defaults to [] so Claude returning {} degrades gracefully
    (each unmatched section falls back to the default explanation string).
    """
    explanations: list[_SectionExplanation] = []


class MissingExplainer:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def explain(
        self,
        unmatched_a: list[Section],
        unmatched_b: list[Section],
    ) -> MissingExplanationBatch:
        all_sections = unmatched_a + unmatched_b
        if not all_sections:
            return MissingExplanationBatch(entries=[])

        prompt = build_missing_explainer_prompt(unmatched_a, unmatched_b)
        raw = await self._llm.generate(
            [{"role": "user", "content": prompt}],
            response_schema=_ExplainerOutput,
        )
        output = _ExplainerOutput.model_validate(raw)
        expl_map = {e.section_id: e.explanation for e in output.explanations}

        entries = [
            MissingEntry(
                text=s.body_text.strip() or s.heading,
                source_file=s.location.filename,
                location=s.location.cite(),
                explanation=expl_map.get(s.id, "No counterpart found in the other version."),
            )
            for s in all_sections
        ]
        return MissingExplanationBatch(entries=entries)