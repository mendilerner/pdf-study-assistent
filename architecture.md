# Academic PDF Study Assistant — Architecture

## Overview
A web app where students upload academic PDF textbooks, search them semantically, and ask AI questions — with the PDF always visible at center.

## Architecture

```
┌─ Frontend (React + TypeScript) ──────────────────────────┐
│                                                          │
│  ┌─────────────────────┐  ┌───────────────────────────┐  │
│  │   PDF Viewer         │  │   Chat Sidebar            │  │
│  │   react-pdf (MIT)    │  │                           │  │
│  │                      │  │   • Semantic search bar   │  │
│  │   • RTL support      │◄─┤   • Q&A chat (streaming)  │  │
│  │   • Auto-scroll      │  │   • Clickable page refs   │  │
│  │     to page          │  │   • Book selector         │  │
│  └─────────────────────┘  └───────────────────────────┘  │
└──────────────────────────┬───────────────────────────────┘
                           │ REST API + SSE
┌──────────────────────────▼───────────────────────────────┐
│  Backend (FastAPI + Python)                              │
│                                                          │
│  POST /ingest          — upload PDF, start bg job        │
│  GET  /ingest/{job_id} — job status + progress           │
│  POST /search          — hybrid search, chunks+pages     │
│  POST /ask             — RAG Q&A (streamed), + sources   │
│  GET  /books           — list indexed books              │
│  GET  /books/{id}/pdf  — serve original PDF              │
└────┬────────────────────────────┬────────────────────────┘
     │                            │
     ▼                            ▼
┌─────────────┐          ┌──────────────┐
│Elasticsearch│          │  Claude API  │
│             │          │  (Anthropic) │
│ • text      │          │              │
│   (Hebrew   │          │  • answer    │
│    analyzer)│          │    generation│
│ • dense_    │          │  • summaries │
│   vector    │          │  • synthesis │
│ • metadata  │          └──────────────┘
└─────────────┘
     ▲
     │ RRF fusion happens in Python (app layer), NOT in Elasticsearch
     └── see "Hybrid Search" below
```

## Stack

| Layer       | Technology                        | Why                                    |
|-------------|-----------------------------------|----------------------------------------|
| Frontend    | React, TypeScript, Tailwind       | Standard, you know it                  |
| PDF viewer  | `react-pdf` (wojtekmaj, MIT)      | Free; pdf.js-based; RTL + page nav     |
| Backend     | FastAPI                           | Async, fast, Python-native             |
| PDF parsing | Marker — **pending Part 0 spike** | Best structure IF Hebrew RTL holds up  |
| Vector DB   | Elasticsearch 8.x + analysis-icu  | BM25 + kNN, Hebrew normalization       |
| Rank fusion | Custom RRF in Python              | ES built-in RRF is a paid feature      |
| Embeddings  | multilingual-e5-large             | Good Hebrew support, 1024 dims         |
| LLM         | Claude API (`claude-opus-5`)      | Strong Hebrew, long context            |

### Licensing notes (verified)

- **Elasticsearch RRF (`rank: {rrf: {}}`) requires a Platinum/Enterprise license.** It is not
  available in the free Basic tier, self-hosted included. We implement RRF in the app layer
  instead. This also lets us weight BM25 vs. vector independently, which the built-in does not.
- **`@react-pdf-viewer` is a commercial product**, not open source. Replaced with `react-pdf`
  (MIT). We hand-roll the page-navigation glue; roughly a day of work.
- **`analysis-icu` is not bundled with Elasticsearch.** It must be installed into the image or
  the index mapping below will fail to create. See the Dockerfile in "Elasticsearch Setup".

## Elasticsearch Setup

### Custom image (required — ICU plugin is not bundled)

```dockerfile
# docker/elasticsearch/Dockerfile
FROM docker.elastic.co/elasticsearch/elasticsearch:8.15.0
RUN bin/elasticsearch-plugin install --batch analysis-icu
```

Local-dev environment settings in `docker-compose.yml`:

```yaml
environment:
  - discovery.type=single-node
  - xpack.security.enabled=false      # ES 8 enables auth + TLS by default
  - ES_JAVA_OPTS=-Xms2g -Xmx2g
```

On Linux the host also needs `sysctl -w vm.max_map_count=262144`, otherwise the container
crashes on startup.

### Index Mapping

```json
{
  "mappings": {
    "properties": {
      "text":         { "type": "text", "analyzer": "hebrew_analyzer" },
      "embedding":    { "type": "dense_vector", "dims": 1024, "similarity": "cosine" },
      "book_id":      { "type": "keyword" },
      "book_title":   { "type": "keyword" },
      "pdf_page":     { "type": "integer" },
      "pdf_page_end": { "type": "integer" },
      "printed_page": { "type": "integer" },
      "chunk_id":     { "type": "keyword" },
      "chunk_index":  { "type": "integer" }
    }
  },
  "settings": {
    "analysis": {
      "analyzer": {
        "hebrew_analyzer": {
          "type": "custom",
          "tokenizer": "icu_tokenizer",
          "filter": ["icu_normalizer", "icu_folding", "hebrew_prefix_strip"]
        }
      },
      "filter": {
        "hebrew_prefix_strip": {
          "type": "pattern_replace",
          "pattern": "^[והבכלמש]{1,2}(?=[א-ת]{3,})",
          "replacement": ""
        }
      }
    }
  }
}
```

## Page Numbers — the two-number rule

**This is the most breakage-prone detail in the project.** A textbook PDF has front matter
(cover, title page, TOC, preface), so the page *printed on the paper* and the page's *position
in the file* are different numbers — often by 10–15.

| Field          | Meaning                            | Used for                                 |
|----------------|------------------------------------|------------------------------------------|
| `pdf_page`     | 1-based position in the PDF file   | **Canonical.** Passed to `jumpToPage()`  |
| `printed_page` | Number printed on the page itself  | **Display only.** The badge label        |

Rules:
1. `pdf_page` is the source of truth. Every citation, every navigation call uses it.
2. `printed_page` is optional and cosmetic. If we can't detect it reliably, leave it `null`
   and display `pdf_page`.
3. Never assign one to the other. If they are ever mixed up, **every citation in the app is
   wrong by a constant offset** and the click-to-scroll feature silently lands on the wrong
   page while appearing to work.

The offset is detected once per book at ingest time (find the first page whose printed number
is 1, store `page_offset` on the book record) and applied only for display.

## Ingestion Pipeline

Ingestion is a **background job**, not a request/response cycle. A 400-page textbook takes
roughly 30–60 minutes on CPU (Marker's ML models plus a 2.2 GB embedding model), which is far
beyond any HTTP timeout.

```
POST /ingest (returns immediately with job_id)
  │
  └─► background worker:
        PDF file
          → Marker (extract text, keep pdf_page for every span)
          → Chunk (~300 tokens, 50 overlap, HARD CAP 512)
              • counted with e5's own tokenizer, not word count
              • may span pages → store pdf_page + pdf_page_end
          → For each chunk:
              → embed with "passage: " prefix, L2-normalized
              → bulk index to Elasticsearch
          → Store original PDF for the frontend viewer
          → Update job progress after each page batch
```

### Two silent failure modes this pipeline guards against

**1. Chunk truncation.** `multilingual-e5-large` has a hard 512-token input limit. Over that,
it does not error — it silently discards the remainder and embeds only the first 512 tokens.
Hebrew consumes noticeably more tokens per word than English in multilingual tokenizers, so a
chunk that "looks like" 500 tokens by word count can easily be 700+ real tokens.

```python
from transformers import AutoTokenizer
TOK = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")

TARGET_TOKENS = 300
MAX_TOKENS = 512

def token_len(text: str) -> int:
    return len(TOK.encode(text, add_special_tokens=True))

# assert in the chunker — fail loudly rather than truncate silently
assert token_len(chunk) <= MAX_TOKENS, f"chunk too long: {token_len(chunk)}"
```

**2. Missing e5 prefixes.** The model was trained with `"query: "` on questions and
`"passage: "` on documents. Omitting them returns a perfectly valid 1024-dim vector while
measurably degrading retrieval — a bug that shape-checking tests cannot catch.

```python
q_vec = model.encode("query: "   + question,   normalize_embeddings=True)
p_vec = model.encode("passage: " + chunk_text, normalize_embeddings=True)
```

## Hybrid Search

Two searches run independently, then are fused in Python.

```
Student query: "what are the 4 resilience factors?"
  │
  ├─ 1. Embed query with "query: " prefix
  │
  ├─ 2. Two independent Elasticsearch requests:
  │     • BM25 on "text"      → ranked list A (catches exact Hebrew terms)
  │     • kNN on "embedding"  → ranked list B (catches semantic matches)
  │
  ├─ 3. RRF fusion in Python → top 5 chunks
  │
  ├─ 4. Send to Claude API (claude-opus-5):
  │     System: "Answer based on these textbook excerpts. Cite page numbers."
  │     User: query + retrieved chunks
  │
  └─ 5. Stream back to frontend:
        { answer: "...", sources: [{ pdf_page: 48, printed_page: 36, book: "dev-psych" }] }
```

### RRF implementation

```python
def rrf(ranked_lists: list[list[str]], weights: list[float] | None = None, k: int = 60):
    """Reciprocal Rank Fusion. Each list is chunk_ids in rank order (best first)."""
    weights = weights or [1.0] * len(ranked_lists)
    scores: dict[str, float] = {}
    for lst, w in zip(ranked_lists, weights):
        for rank, chunk_id in enumerate(lst):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + w / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)

# usage
fused = rrf([bm25_ids, knn_ids], weights=[1.0, 1.0])[:5]
```

`weights` is the knob to tune against the eval set. If Hebrew morphology weakens BM25 (see Key
Decisions), raising the vector weight is a one-line change — something Elasticsearch's built-in
RRF does not permit.

## Project Structure

```
project/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── PdfViewer.tsx        # react-pdf, RTL, jumpToPage
│   │   │   ├── ChatSidebar.tsx      # search + streaming Q&A
│   │   │   ├── SourceBadge.tsx      # clickable page reference
│   │   │   └── IngestProgress.tsx   # job status polling + progress bar
│   │   ├── api/
│   │   │   └── client.ts            # API calls to backend
│   │   └── App.tsx
│   └── package.json
│
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI app
│   │   ├── routers/
│   │   │   ├── ingest.py            # upload + job status
│   │   │   ├── search.py            # hybrid search endpoint
│   │   │   └── ask.py               # RAG Q&A (SSE streaming)
│   │   ├── services/
│   │   │   ├── elastic.py           # ES client, index setup, queries
│   │   │   ├── embeddings.py        # e5 prefixes + normalization
│   │   │   ├── chunker.py           # token-aware chunking
│   │   │   ├── pdf_parser.py        # Marker wrapper, pdf_page tracking
│   │   │   ├── rrf.py               # rank fusion
│   │   │   ├── jobs.py              # background job registry + progress
│   │   │   └── llm.py               # Claude API calls
│   │   └── config.py
│   ├── eval/
│   │   ├── gold_set.py              # 15-20 questions + correct pdf_pages
│   │   └── run_eval.py              # prints recall@5
│   └── requirements.txt
│
├── docker/
│   └── elasticsearch/Dockerfile     # base image + analysis-icu
├── docker-compose.yml
└── README.md
```

## Key Decisions

1. **Hybrid search over pure vector** — Hebrew has many forms for the same concept; BM25 catches
   exact matches that vectors might miss. See caveat #6 below.
2. **RRF in the application layer, not Elasticsearch** — the built-in is license-gated, and our
   own version is ~8 lines and supports per-retriever weighting.
3. **PDF parser is a decision, not an assumption** — Marker is the *candidate* for its structure
   preservation, but Hebrew RTL extraction with reliable page attribution is unproven. Part 0 of
   the roadmap compares Marker against PyMuPDF on a real textbook and picks the winner on
   evidence. Correct plain text beats well-structured garbage.
4. **Claude over OpenAI** — stronger Hebrew comprehension and long context. `claude-opus-5` is
   the current generation (the earlier `claude-sonnet-4-6` still works but is superseded).
5. **`pdf_page` is canonical, `printed_page` is cosmetic** — see the two-number rule above.
6. **The Hebrew analyzer is normalization, not morphology** — `icu_tokenizer` +
   `icu_normalizer` + `icu_folding` clean up unicode and strip niqqud, but understand no Hebrew
   grammar. Hebrew glues prefixes (ו ה ב כ ל מ ש) directly onto words, so `חוסן`, `החוסן`, and
   `וחוסן` are three unrelated tokens to BM25 — which partly undercuts decision #1. Mitigation:
   the `hebrew_prefix_strip` pattern filter in the mapping above. It is imperfect (sometimes ב
   is genuinely part of the word) but cheap. If recall on the eval set is still weak, the
   options are HebMorph (proper morphology, but its releases lag ES versions) or simply raising
   the vector weight in RRF.
7. **Chunks may cross page boundaries** — a definition split across pages 36/37 is exactly what
   study questions ask about. Chunks store `pdf_page` and `pdf_page_end`; citations use
   `pdf_page`.
8. **Ingestion is a background job** — indexing takes tens of minutes; no HTTP request survives
   that. `POST /ingest` returns a `job_id` immediately.
9. **PDF served directly to frontend** — never converted, preserving original layout.
10. **Answers stream** — a 5–10 second blank screen reads as a frozen app. `/ask` streams via
    SSE.
