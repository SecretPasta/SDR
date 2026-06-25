from __future__ import annotations

from app.domain.comparison import ComparisonResult

_MAX_TEXT_CHARS = 400


def build_top10_prompt(result: ComparisonResult) -> str:
    blocks: list[str] = [
        "You are ranking the most significant changes between V0 (PDF baseline) and "
        "V5 (DOCX revised) of a Functional Design Specification.\n"
        "Select and rank up to 10 changes by SEMANTIC IMPORTANCE — "
        "not by order of appearance.\n\n"
        "Anchor importance on:\n"
        "- Changes to defined phases, milestones, or workflows\n"
        "- New or removed scope (features, modules, responsibilities)\n"
        "- Modified success criteria, thresholds, or acceptance conditions\n"
        "- Structural reorganisation that changes how requirements are grouped\n"
        "- Changed compliance, regulatory, or integration requirements\n"
    ]

    if result.diff:
        blocks.append(f"## DIFF entries ({len(result.diff)} total)\n")
        for i, e in enumerate(result.diff):
            entry_id = f"diff::{i}"
            blocks.append(
                f"### {entry_id}\n"
                f"V0: {e.docA_text.strip()[:_MAX_TEXT_CHARS]}\n"
                f"V5: {e.docB_text.strip()[:_MAX_TEXT_CHARS]}\n"
                f"Reason: {e.reason}\n"
                f"Sources: {e.sourceA} | {e.sourceB}"
            )

    if result.missing:
        blocks.append(f"## MISSING entries ({len(result.missing)} total)\n")
        for i, e in enumerate(result.missing):
            entry_id = f"missing::{i}"
            blocks.append(
                f"### {entry_id}\n"
                f"File: {e.source_file}\n"
                f"Location: {e.location}\n"
                f"Content: {e.text.strip()[:_MAX_TEXT_CHARS]}\n"
                f"Significance: {e.explanation}"
            )

    blocks.append(
        "---\n"
        "Return up to 10 entries ranked 1 (most important) to 10. "
        "Only include genuinely significant changes — do not pad to 10 if fewer qualify.\n\n"
        "For each entry provide:\n"
        "- rank: integer\n"
        "- change_type: short label (e.g. 'scope removal', 'threshold change', "
        "'new requirement', 'structural reorganisation')\n"
        "- summary: one sentence describing what changed\n"
        "- why_it_matters: 1–2 sentences on the impact to the specification\n"
        "- citations: list of relevant source strings from the entries above\n"
        "- source_entry_id: the ID shown in ### (e.g. 'diff::0', 'missing::3')"
    )

    return "\n\n".join(blocks)