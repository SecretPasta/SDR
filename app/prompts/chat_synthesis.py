from __future__ import annotations

from app.chat.retriever import RetrievedChunk

_SINGLE_DOC_INSTRUCTIONS = """\
- Answer ONLY from the provided context. Do not use prior knowledge.
- Cite every factual claim inline: [filename · §section · page N].
- In the citations list, provide each citation as a plain string in the exact \
format "filename · §section · page N" (omit " · page N" if the page is unknown).
- If the context is insufficient to answer, set insufficient_context to true \
and explain you couldn't find the information in the document."""

_CROSS_DOC_INSTRUCTIONS = """\
- Answer ONLY from the provided context. Do not use prior knowledge.
- Cite every factual claim inline: [filename · §section · page N].
- In the citations list, provide each citation as a plain string in the exact \
format "filename · §section · page N" (omit " · page N" if the page is unknown).
- When answering comparative questions, attribute each claim to its source \
document (V0 or V5) so the distinction is clear.
- If the context is insufficient to answer, set insufficient_context to true \
and explain you couldn't find the information in either document."""


def build_single_doc_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    return (
        f"## Context\n\n{_format_chunks(chunks)}\n\n"
        f"## Instructions\n{_SINGLE_DOC_INSTRUCTIONS}\n\n"
        f"## Question\n{query}"
    )


def build_cross_doc_prompt(
    query: str,
    chunks_by_doc: dict[str, list[RetrievedChunk]],
) -> str:
    blocks: list[str] = []
    for doc_id, chunks in chunks_by_doc.items():
        label = _doc_label(doc_id, chunks)
        blocks.append(f"## From {label}\n\n{_format_chunks(chunks)}")

    context = "\n\n".join(blocks)
    return (
        f"{context}\n\n"
        f"## Instructions\n{_CROSS_DOC_INSTRUCTIONS}\n\n"
        f"## Question\n{query}"
    )


# ── helpers ───────────────────────────────────────────────────────────────────

def _doc_label(doc_id: str, chunks: list[RetrievedChunk]) -> str:
    """Derive a human-readable label from doc_id or chunk metadata filename."""
    if chunks:
        fname: str = chunks[0].metadata.get("filename", "")
        if fname.lower().endswith(".pdf"):
            return "V0 (PDF)"
        if fname.lower().endswith(".docx"):
            return "V5 (DOCX)"
    return f"doc {doc_id}"


def _format_chunks(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[{_cite(c.metadata)}]\n{c.display_text}" for c in chunks
    )


def _cite(meta: dict) -> str:
    parts: list[str] = [meta.get("filename", "unknown")]
    section = meta.get("heading_number") or (
        meta["heading_path"][-1] if meta.get("heading_path") else ""
    )
    if section:
        parts.append(f"§{section}")
    if (page := meta.get("page_number")) is not None:
        parts.append(f"page {page}")
    return " · ".join(parts)
