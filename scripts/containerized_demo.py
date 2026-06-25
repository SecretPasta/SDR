"""
End-to-end demo: exercises every API endpoint and prints results to stdout.

Usage (server must already be running in the container):
    python scripts/demo.py
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

import httpx

# ── ADJUSTED FOR CONTAINER FILESYSTEM ─────────────────────────────────────────
# The container has these baked into /app/samples/
_CONTAINER_PDF  = "/app/samples/FDS_PriceBook_V0.pdf"
_CONTAINER_DOCX = "/app/samples/FDS_PriceBook_V5.docx"

_WIDTH = 88


# ── formatting helpers ────────────────────────────────────────────────────────

def _header(title: str) -> None:
    print(f"\n{'─' * _WIDTH}")
    print(f"  {title}")
    print(f"{'─' * _WIDTH}")


def _json(data: dict) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))


def _chat(data: dict) -> None:
    print(f"Answer:\n  {textwrap.fill(data['answer'], width=_WIDTH - 2, subsequent_indent='  ')}")
    if data.get("citations"):
        print("Citations:")
        for c in data["citations"]:
            print(f"  • {c}")
    if data.get("insufficient_context"):
        print("  ⚠  insufficient_context = true")


# ── requests ──────────────────────────────────────────────────────────────────

def run_compare(client: httpx.Client, pdf: str, docx: str) -> None:
    _header("POST /compare  —  full comparison pipeline")
    print(f"  Target Container PDF:  {pdf}")
    print(f"  Target Container DOCX: {docx}")
    print("  Running… (this may take 1–3 minutes)")

    resp = client.post(
        "/compare",
        json={"pdf_path": pdf, "docx_path": docx},
        timeout=300,
    )
    resp.raise_for_status()
    data = resp.json()

    result  = data["result"]
    summary = data["summary"]
    print(f"\n  match={len(result['match'])}  diff={len(result['diff'])}  missing={len(result['missing'])}")
    print("  Wrote outputs/comparison.json and outputs/summary.json")
    print(f"\nTop change:")
    if summary["top_changes"]:
        top = summary["top_changes"][0]
        print(f"  [{top['verdict']}] {top['summary']}")
        if top.get("why_it_matters"):
            print(f"  Why it matters: {top['why_it_matters']}")


def run_summary(client: httpx.Client) -> None:
    _header("GET /summary  —  cached executive summary")
    resp = client.get("/summary", timeout=30)
    resp.raise_for_status()
    data = resp.json()["summary"]

    print(f"  matches={data['total_matches']}  diffs={data['total_diffs']}  missing={data['total_missing']}")
    print(f"\nTop-10 changes:")
    for change in data["top_changes"]:
        print(f"  {change['rank']:>2}. [{change['verdict']}] {change['summary']}")


def run_chat_single_a(client: httpx.Client) -> None:
    query = "What are the main processing stages described in the document?"
    _header(f"POST /chat/single  (doc_id=A — PDF)")
    print(f"  Q: {query}")
    resp = client.post(
        "/chat/single",
        json={"query": query, "doc_id": "A"},
        timeout=60,
    )
    resp.raise_for_status()
    print()
    _chat(resp.json())


def run_chat_single_b(client: httpx.Client) -> None:
    query = "How does the Configuration-Driven Price Book Engine work?"
    _header("POST /chat/single  (doc_id=B — DOCX)")
    print(f"  Q: {query}")
    resp = client.post(
        "/chat/single",
        json={"query": query, "doc_id": "B"},
        timeout=60,
    )
    resp.raise_for_status()
    print()
    _chat(resp.json())


def run_chat_cross(client: httpx.Client) -> None:
    query = "How did the pricing calculation logic change between V0 and V5?"
    _header("POST /chat/cross  —  cross-document comparison")
    print(f"  Q: {query}")
    resp = client.post(
        "/chat/cross",
        json={"query": query},
        timeout=60,
    )
    resp.raise_for_status()
    print()
    _chat(resp.json())


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="FDS Reconciler end-to-end demo")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--pdf",  default=_CONTAINER_PDF,  help="Path to V0 PDF inside container")
    parser.add_argument("--docx", default=_CONTAINER_DOCX, help="Path to V5 DOCX inside container")
    parser.add_argument(
        "--skip-compare", action="store_true",
        help="Skip POST /compare (use if pipeline already ran this session)",
    )
    args = parser.parse_args()

    print(f"\nFDS Reconciler demo  →  {args.base_url}")

    with httpx.Client(base_url=args.base_url) as client:
        # Verify server is up
        try:
            client.get("/docs", timeout=5).raise_for_status()
        except Exception as exc:
            print(f"ERROR: server not reachable at {args.base_url} ({exc})", file=sys.stderr)
            sys.exit(1)

        if not args.skip_compare:
            run_compare(client, args.pdf, args.docx)
        else:
            print("\n(Skipping /compare — using cached result)")

        run_summary(client)
        run_chat_single_a(client)
        run_chat_single_b(client)
        run_chat_cross(client)

    print(f"\n{'─' * _WIDTH}")
    print("  Demo complete.")
    print(f"{'─' * _WIDTH}\n")


if __name__ == "__main__":
    main()