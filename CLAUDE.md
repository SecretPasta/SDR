# FDS Reconciler

AI-powered system comparing two versions of a Functional Design Specification (PDF vs DOCX). Detects MATCH / DIFF / MISSING sections, answers questions about each doc, and answers cross-document comparative questions. Take-home for an AI Engineer role.

## Stack
- Python 3.13, FastAPI, Pydantic v2, Pydantic Settings
- LangGraph for orchestration
- Pinecone serverless (vector DB, mandated)
- Claude Sonnet 4.6 (Anthropic API) — comparison judging, missing-explainer, top-10 ranker
- Gemini 2.5/3.5 Flash (AI Studio) — chat synthesis only
- gemini-embedding-001 — all embeddings (768d via Matryoshka)
- pymupdf (PDF), python-docx (DOCX)

## Architecture (hexagonal)
- `app/domain/` — pure Pydantic data models, no I/O
- `app/ports/` — protocol classes (LLMClient, EmbedderClient, VectorStore)
- `app/adapters/` — concrete implementations of each port
- `app/parsing/`, `indexing/`, `comparison/`, `chat/`, `api/` — pipeline modules
- `app/prompts/` — all LLM prompts in one place

## Key design decisions

### Section model (parsing output)
- `id` deterministic: `"A::3.1"`, `"B::8.1::table-0"`
- `heading_path` is full breadcrumb (list of strings)
- `Location`: filename, page_number (None for DOCX), heading_number, heading_path
- `body_text` (prose only), `tables: list[TableData]` (structured), `bullets: list[str]`
- `for_embedding()` method prepends heading breadcrumb

### Chunking (structure-aware)
- One chunk per section (split sub-section if > 900 tokens)
- Tables ALWAYS get own chunks
- Bullets grouped if short, split if long
- Heading breadcrumb prepended to embedding `text` ONLY — `display_text` stays clean
- Target 600, max 900, overlap 80 (WITHIN section only — never cross-section)

### Indexing
- Pinecone single index, single namespace, `doc_id` in metadata
- Embedder: `task_type="RETRIEVAL_DOCUMENT"` for indexing, `"RETRIEVAL_QUERY"` for chat, `"SEMANTIC_SIMILARITY"` for alignment
- Pinecone metadata includes `display_text` so chat doesn't need a second fetch

### Alignment (deterministic, no LLM)
- Score = `0.45 × heading_number + 0.40 × heading_embedding + 0.15 × levenshtein`
- If heading numbers absent on either side, reweight to `0.75 / 0.25`
- Greedy bipartite matching, threshold `0.55`
- Outputs: aligned_pairs, unmatched_a, unmatched_b

### Comparison engine (LangGraph)
- State accumulates: doc_a/b, heading_embeddings, aligned_pairs, unmatched, verdicts (dict reducer), missing_explanations, comparison_result, top_10
- Nodes: load_parsed_docs → embed_headings → align_sections → [judge_pair (Send fan-out) ∥ explain_missing] → assemble_result → rank_top10
- `verdicts` uses `Annotated[dict, operator.or_]` reducer

### Mandated output schema (exact)
```json
{
  "missing": [{"text", "source_file", "location", "explanation"}],
  "diff":    [{"docA_text", "docB_text", "reason", "sourceA", "sourceB"}],
  "match":   [{"textA", "textB", "source"}]
}
```

### Chat
- Single-doc: top_k=6, filter by `doc_id`, Gemini Flash synthesizes with "context only + cite all" constraint
- Cross-doc: DUAL retrieval (top_k=4 per doc, explicit filter), labeled context blocks (`## From V0 (PDF)` / `## From V5 (DOCX)`), Gemini Flash synthesizes
- ChatAnswer has `insufficient_context: bool` for honest refusal
- Citation format: `filename · §section · page N`

### Configuration
- All env via Pydantic Settings with grouped sub-settings
- SecretStr on all API keys
- Every magic number exposed as env var with sensible default

### Dependency injection
- FastAPI Depends() with @lru_cache singleton adapters
- Pipelines take ports (protocols), never concrete types
- `app/deps.py` is the only place where wiring happens

## Code conventions
- Pydantic v2 (`model_config`, not Config class)
- async/await throughout
- Full type hints
- No `utils.py` dumping ground
- `logging`, never `print`
- Tests for deterministic logic (aligner, chunker) — skip LLM-call tests

## Brief constraints
- Vector DB: Pinecone (mandated)
- Hosting: NOT AWS/GCP/Azure
- LLM provider: any (using Claude + Gemini)
- Docker delivery required
- ~50-page scale, two docs