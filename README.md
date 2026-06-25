# FDS Reconciler

AI-powered system that compares two versions of a Functional Design Specification (PDF vs DOCX), detects MATCH / DIFF / MISSING sections, ranks the top-10 most significant changes, and answers natural-language questions about each document — including cross-document comparative questions.

---

## Setup

### Native (uv)

```bash
# 1. Install dependencies
uv sync

# 2. Configure secrets
cp .env.example .env
# Fill in ANTHROPIC_API_KEY, GEMINI_API_KEY, PINECONE_API_KEY

# 3. Parse and index both documents into Pinecone (run once per document pair)
python -m scripts.index_docs

# 4. Start the API server
uv run uvicorn app.main:app --reload
```

### Docker

```bash
# Build and start (reads .env automatically)
docker compose up --build
```

The API is available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### Demo script

With the server running, exercise all endpoints end-to-end:

```bash
python scripts/demo.py
```

---

## Endpoints

### `POST /compare` — run the full pipeline

Parses, indexes, aligns, and judges both documents. Blocks until complete (~1–3 min for 50-page docs).

```bash
curl -X POST http://localhost:8000/compare \
  -H "Content-Type: application/json" \
  -d '{
    "pdf_path":  "/app/samples/FDS_PriceBook_V0.pdf",
    "docx_path": "/app/samples/FDS_PriceBook_V5.docx"
  }'
```

### `GET /summary` — cached top-10

Returns the executive summary from the most recent `/compare` run without re-running the pipeline.

```bash
curl http://localhost:8000/summary
```

### `POST /chat/single` — question about one document

```bash
# Ask about the PDF (doc_id="A")
curl -X POST http://localhost:8000/chat/single \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the three processing stages in section 3.1?", "doc_id": "A"}'

# Ask about the DOCX (doc_id="B")
curl -X POST http://localhost:8000/chat/single \
  -H "Content-Type: application/json" \
  -d '{"query": "What new integration checks were introduced in V5?", "doc_id": "B"}'
```

### `POST /chat/cross` — comparative question across both documents

```bash
curl -X POST http://localhost:8000/chat/cross \
  -H "Content-Type: application/json" \
  -d '{"query": "How did the pricing calculation logic change between V0 and V5?"}'
```

**Chat response shape:**

```json
{
  "answer": "V5 adds discount tiers on top of the base price. [FDS_V5.docx · §3.2]",
  "citations": ["FDS_V0.pdf · §3.2 · page 5", "FDS_V5.docx · §3.2"],
  "insufficient_context": false
}
```

---

## Architecture

```
samples/
  FDS_PriceBook_V0.pdf
  FDS_PriceBook_V5.docx
        │
        ▼
  ┌─────────────┐
  │   Parsing   │  pymupdf + pdfplumber (PDF), python-docx (DOCX)
  │             │  → Section(heading, body_text, tables, bullets, location)
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  Indexing   │  chunk_section() → Chunk[]
  │             │  GeminiEmbedder → vectors
  │             │  PineconeStore.upsert()
  └──────┬──────┘
         │
         ├─────────────────────────────────────────────┐
         ▼                                             ▼
  ┌──────────────────┐                      ┌──────────────────┐
  │ Comparison       │                      │ Chat Pipeline    │
  │ Pipeline         │                      │ (LangGraph)      │
  │ (LangGraph)      │                      │                  │
  │                  │                      │ embed_query      │
  │ embed_headings   │                      │   → route_mode   │
  │   → align        │                      │   → retrieve     │
  │   → judge (×N)   │                      │   → sufficiency  │
  │   ‖ explain      │                      │   → synthesize   │
  │   → assemble     │                      └────────┬─────────┘
  │   → rank_top10   │                               │
  └────────┬─────────┘                               │
           │                                         │
           └──────────────────┬──────────────────────┘
                              ▼
                     ┌─────────────────┐
                     │   FastAPI API   │
                     │  /compare       │
                     │  /summary       │
                     │  /chat/single   │
                     │  /chat/cross    │
                     └─────────────────┘
```

`app/deps.py` is the single wiring point — all concrete adapter types are instantiated there via `@lru_cache` singletons and injected via FastAPI `Depends`.

---

## Chunking strategy

Sections are chunked **structure-aware** rather than with a naive sliding window:

| Content type | Chunking rule |
|---|---|
| Tables | Always get their own chunk — never split mid-table or merged with prose |
| Prose | Packed to `target_tokens=600`, split at `max_tokens=900`, with `overlap=80` tokens between chunks |
| Bullets | Grouped if short; split at max if long |
| Cross-section | **Never** — overlap is within a section only |

The **heading breadcrumb** (`"3. Features > 3.1 Process Stages"`) is prepended to the `text` field used for embedding, giving the vector model full context. It is **not** stored in `display_text`, so retrieval results shown to the LLM are clean prose without repeated prefixes.

Chunk IDs are deterministic: `"{doc_id}::{heading_number}::chunk-{seq}"`, making re-indexing idempotent via Pinecone upsert.

---

## Cross-document retrieval strategy

For comparative questions the pipeline performs **dual retrieval** — two parallel Pinecone queries, one per `doc_id` filter — rather than a single unfiltered query.

**Why not single-index-no-filter?**

| Concern | Single unfiltered query | Dual filtered queries |
|---|---|---|
| Coverage guarantee | One doc can dominate top-K if it has higher-scoring matches | Each doc always contributes `top_k` results |
| Citation clarity | Mixed results; must infer source from metadata | Context blocks are pre-labeled `## From V0 (PDF)` / `## From V5 (DOCX)` |
| Hallucination risk | LLM may conflate sources | Source attribution is structural, not inferred |
| Extensibility | Adding a third doc dilutes all results | Add one more filter query; other docs unaffected |

**Fallback path** — if *both* sides return a max relevance score below `CHAT_RELEVANCE_FLOOR` (default `0.5`), the pipeline falls back to a single unfiltered query (`top_k=6`). If that also fails the floor, the response is returned with `insufficient_context: true` rather than hallucinating.

---

## Model choices

### Claude Sonnet 4.6 — pairwise judging, missing-section explanation, top-10 ranking

Claude Sonnet 4.6 scores **Elo 1633 on the GPQA Diamond benchmark**, leading all models on expert-level knowledge work. More practically for this task: it has the strongest structured-output reliability via tool-use — the judge, explainer, and ranker all return strict Pydantic schemas via `tool_choice: {type: "tool"}`, and Claude is the least likely to return `{}` or hallucinate extra fields.

### Gemini 2.5 Flash — chat synthesis

Chat synthesis is a grounded RAG task: the context is provided, the model just needs to follow instructions (cite everything, refuse if insufficient, attribute by source). Gemini 2.5 Flash is cheap, fast, and has a 1M-token context window — well-suited for assembling two labeled context blocks and producing a cited answer. It handles `response_schema` natively without tool-use overhead.

### gemini-embedding-001 — all embeddings (768d)

`gemini-embedding-001` holds **MTEB rank #1** at the time of writing. It supports a `task_type` parameter (`RETRIEVAL_DOCUMENT`, `RETRIEVAL_QUERY`, `SEMANTIC_SIMILARITY`) that lets the model produce task-optimised representations — document and query embeddings are not treated identically, which materially improves retrieval precision. The 768-dimension Matryoshka encoding keeps Pinecone storage and query latency low.

---

## Pinecone — single index with metadata filter

Pinecone is **mandated** for this project. The implementation uses a **single index with `doc_id` metadata filtering** rather than splitting documents into separate namespaces or indexes.

**Rationale:** A namespace split would require two separate query calls anyway (same cost), but would make cross-document unfiltered fallback queries impossible. Keeping both documents in one namespace lets the fallback path do a genuine cross-corpus sweep. Metadata filters in Pinecone Serverless are evaluated at the pod level with negligible latency overhead at ~50-page scale.

---

## Known limitations and what I'd improve

**Retrieval quality**
- **No reranker.** Adding a cross-encoder reranker (e.g. Cohere Rerank) after the initial vector fetch would improve precision, especially for ambiguous queries.
- **Dense-only embeddings.** Sparse (BM25) hybrid retrieval would significantly improve recall for exact-match terms like product codes, CPN codes, or version numbers — query embedding smears these into the dense space where they may not score well.

**Comparison pipeline**
- **Pairwise judge calls are already parallelised** via LangGraph `Send` fan-out, but each call is an independent LLM request. A further optimisation would batch multiple section pairs into a single prompt (reducing API round-trips at the cost of some instruction-following complexity).
- **Alignment quality** degrades on documents without consistent heading numbering, falling back to embedding-only matching. A pre-processing step to infer numbering from indentation/font-size would help.

**Operational**
- **No persistent index across runs.** The Pinecone index is populated fresh each run. Production use would maintain a versioned index keyed by document hash, making re-indexing a no-op when documents haven't changed.
- **In-process comparison cache.** The `/compare` result lives in a module-level variable; a container restart clears it. A Redis or DB-backed store would enable persistence and horizontal scaling.
- **Single document pair only.** The `doc_id` convention of `"A"` and `"B"` is baked into the pipeline. Generalising to N documents would require a registry of doc IDs and dynamic filter construction in the chat retriever.

---

## Tests

```bash
# All unit + adapter tests (~2 s, no API keys needed)
uv run pytest

# Single file
uv run pytest tests/unit/test_aligner.py -v

# Integration tests — requires real .env and sample files
uv run pytest -m integration
```
