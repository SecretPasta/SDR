from __future__ import annotations

from app.domain.section import Section


def build_pairwise_judge_prompt(section_a: Section, section_b: Section) -> str:
    return f"""\
You are a technical reviewer comparing two aligned sections from two versions of a \
Functional Design Specification (FDS).

## V0 (PDF) — {_heading(section_a)}

{_body(section_a)}

## V5 (DOCX) — {_heading(section_b)}

{_body(section_b)}

---

Classify this pair as MATCH or DIFF.

**MATCH** — the sections are semantically equivalent. Paraphrasing, sentence reordering, \
punctuation, capitalisation, or formatting differences alone do not qualify as DIFF.

**DIFF** — there is a meaningful content change: a new or removed requirement, a changed \
rule or threshold, updated table values, added or removed scope, or any factual difference \
that would affect system behaviour or understanding.

Return your verdict and a concise 1–2 sentence reason. \
For DIFF, state specifically what changed and in which version.\
"""


# ── helpers ───────────────────────────────────────────────────────────────────

def _heading(section: Section) -> str:
    return " > ".join(section.location.heading_path) or section.heading


def _body(section: Section) -> str:
    parts: list[str] = []
    if section.body_text:
        parts.append(section.body_text.strip())
    for bullet in section.bullets:
        parts.append(f"• {bullet}")
    for table in section.tables:
        parts.append(table.to_markdown())
    return "\n".join(parts) if parts else "(empty section)"