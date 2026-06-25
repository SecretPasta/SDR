from __future__ import annotations

from app.domain.section import Section

_MAX_BODY_CHARS = 600
_MAX_BULLETS = 5


def build_missing_explainer_prompt(
    unmatched_a: list[Section],
    unmatched_b: list[Section],
) -> str:
    blocks: list[str] = [
        "You are analysing sections from two versions of a Functional Design Specification "
        "(FDS) that could not be matched across versions.\n"
    ]

    if unmatched_a:
        blocks.append(
            "**Sections in V0 (PDF) with no equivalent in V5 (DOCX) "
            "— likely removed or significantly restructured:**\n"
        )
        for s in unmatched_a:
            blocks.append(_format_section(s))

    if unmatched_b:
        blocks.append(
            "**Sections in V5 (DOCX) with no equivalent in V0 (PDF) "
            "— likely added or significantly restructured:**\n"
        )
        for s in unmatched_b:
            blocks.append(_format_section(s))

    blocks.append(
        "---\n"
        "For each section above, explain:\n"
        "1. What this section covers and why it matters to the specification.\n"
        "2. What its absence from the other version likely signifies "
        "(scope reduction, new feature, restructuring, compliance change, etc.).\n\n"
        "Be concise — 2–3 sentences per section. "
        "Use the exact section_id shown in brackets as the key in your response."
    )

    return "\n\n".join(blocks)


# ── helpers ───────────────────────────────────────────────────────────────────

def _format_section(section: Section) -> str:
    heading = " > ".join(section.location.heading_path) or section.heading
    header = f"### [{section.id}] {heading}"

    body_parts: list[str] = []
    if section.body_text:
        body_parts.append(section.body_text.strip()[:_MAX_BODY_CHARS])
    for bullet in section.bullets[:_MAX_BULLETS]:
        body_parts.append(f"• {bullet}")
    if section.tables:
        body_parts.append(f"[{len(section.tables)} table(s) not shown]")

    body = "\n".join(body_parts) if body_parts else "(no body content)"
    return f"{header}\n{body}"