# Development Roadmap — Build Order

Each part is self-contained and testable before moving to the next.

**Two parts were added and the order changed** after review:
- **Part 0** — validate Hebrew PDF extraction *first*. It is the riskiest unproven assumption
  and everything else is built on top of it.
- **Part 5.5** — build an evaluation set *before* tuning search, so Parts 6 and 7 have a number
  to improve instead of a feeling.

Tests were also rewritten where the original checked *shape* rather than *correctness* — a test
that passes while the bug is present is worse than no test.

---

## Part 0: Hebrew Extraction Spike ⚠️ GO / NO-GO GATE
**Goal:** Prove that we can get correct Hebrew text out of a real textbook *with trustworthy
page numbers* — before building anything on top of it.

**Why this is first:** Marker's layout and OCR models are trained predominantly on left-to-right
Latin script. Hebrew RTL extraction is a known source of scrambled output (reversed letters,
lines out of order, destroyed mixed Hebrew/English lines). Marker also emits Markdown for the
whole document rather than per-page output, so reliable page attribution is extra work, not
free. If extraction is bad, then chunks, embeddings, search, and citations are all built on
garbage — ten parts wasted.

- Install Marker; run it on one real Hebrew textbook
- Run PyMuPDF (`fitz`) on the same file for comparison
- Read the actual output. Judge on: is the Hebrew correct? Are lines in the right order? Are
  headings intact? Do mixed Hebrew/English lines survive?
- Verify `pdf_page` can be attached to every text span, reliably
- Detect `page_offset` — how many PDF pages precede printed page 1

**Decision rule:** correct plain text beats well-structured garbage. If Marker mangles Hebrew
and PyMuPDF doesn't, switch to PyMuPDF and update `architecture.md`.

**Output:** `pdf_parser.py` — input: PDF path → output: `list[{pdf_page: int, text: str}]`,
plus a written decision on which parser we're using and why.

**Test:** Pick 3 pages spread through the book. Print the extracted text next to a screenshot of
the real page. A Hebrew reader should agree they match.

---

## Part 1: Elasticsearch Setup
**Goal:** Running ES instance with a Hebrew-ready index.

- **Custom Dockerfile** — `analysis-icu` is NOT bundled with Elasticsearch; the index mapping
  will fail to create without it:
  ```dockerfile
  FROM docker.elastic.co/elasticsearch/elasticsearch:8.15.0
  RUN bin/elasticsearch-plugin install --batch analysis-icu
  ```
- Docker compose, single node, `xpack.security.enabled=false` for local dev (ES 8 turns on
  auth + TLS by default)
- Linux hosts: `sysctl -w vm.max_map_count=262144` or the container crashes on boot
- Create index with the Hebrew analyzer (ICU + `hebrew_prefix_strip`)
- Mapping: text + dense_vector + `pdf_page` / `pdf_page_end` / `printed_page` metadata

**Output:** `docker-compose.yml`, `docker/elasticsearch/Dockerfile`, `elastic.py`

**Test:** Index a doc containing `החוסן`; search for `חוסן`; it should match. (This is what
verifies the prefix filter is doing its job — plain ICU alone would fail this.)

---

## Part 2: Chunking
**Goal:** Split extracted pages into search-friendly chunks with metadata.

- Target **~300 tokens**, 50 token overlap — *not* 500
- **Count tokens with e5's own tokenizer**, never word count:
  ```python
  from transformers import AutoTokenizer
  TOK = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")
  ```
- **Hard assert every chunk ≤ 512 tokens.** The model silently truncates past 512 with no
  error, and Hebrew uses far more tokens per word than English in multilingual tokenizers — a
  chunk that "looks like" 500 tokens by word count can really be 700+
- Chunks may span pages: store `pdf_page` (start) and `pdf_page_end`
- Each chunk keeps `book_id`, `chunk_index`
- Don't split mid-sentence

**Output:** `chunker.py` — input: page objects → output: chunk objects

**Test:** Chunk a real page. Assert every chunk's real token count is ≤ 512. Verify overlap and
that page metadata survives a chunk that crosses a page boundary.

---

## Part 3: Embedding Generation
**Goal:** Convert text chunks into vectors — correctly.

- Load `multilingual-e5-large` (sentence-transformers)
- **Add the e5 prefixes.** The model was trained with them; omitting them silently degrades
  retrieval while still returning a valid vector:
  - `"query: "` on questions
  - `"passage: "` on chunks
- `normalize_embeddings=True` (cosine expects unit-length vectors)
- Batch for efficiency

**Output:** `embeddings.py` — input: list of texts → output: list of vectors

**Test (rewritten — the original couldn't catch the real bug):** The old test was "embed one
sentence, verify 1024 dims." That passes whether or not the prefixes are present. Instead:
embed one question plus two chunks — one genuinely about the topic, one unrelated — and assert
the relevant chunk scores higher. This test *fails* if the prefixes are missing.

---

## Part 4: Indexing Pipeline
**Goal:** End-to-end ingestion: PDF → chunks → vectors → Elasticsearch.

- Wire together Parts 0 + 2 + 3
- Bulk-index chunks (text + vector + metadata)
- Detect and store `page_offset` for the book
- Index one real textbook

**Output:** `ingest.py` — input: PDF file → side effect: indexed in ES

**Test:** Index a full PDF, check doc count. Then spot-check: pick a chunk from ES, look up its
`pdf_page`, and confirm that text really is on that page of the PDF.

---

## Part 5: Evaluation Set 🆕
**Goal:** Be able to *measure* search quality before trying to improve it.

Without this, tuning is guesswork — you change something, it feels better, you keep it, and
sometimes you make it worse without noticing.

- Write 15–20 real questions you'd actually ask while studying this book
- For each, find and record the `pdf_page`(s) that genuinely contain the answer
  ```python
  GOLD = [
      {"q": "מהם ארבעת גורמי החוסן?", "pages": [48, 49]},
      {"q": "מה ההבדל בין התקשרות בטוחה לנמנעת?", "pages": [100]},
      # ... 15-20 total
  ]
  ```
- Write `run_eval.py`: run each question through search, check whether a correct page appears in
  the top 5, print recall@5

**Output:** `eval/gold_set.py`, `eval/run_eval.py`

**Test:** The script runs and prints a percentage. That number is the baseline for Parts 6–7.

---

## Part 6: Hybrid Search
**Goal:** Query ES with combined BM25 + kNN and get ranked results.

- **Do NOT use Elasticsearch's `rank: {rrf: {}}`** — it requires a Platinum/Enterprise license
  and is unavailable on the free tier. It will fail at runtime with a license error.
- Run two independent queries: BM25 on `text`, kNN on `embedding`
- Fuse the two ranked lists in Python (~8 lines):
  ```python
  def rrf(ranked_lists, weights=None, k=60):
      weights = weights or [1.0] * len(ranked_lists)
      scores = {}
      for lst, w in zip(ranked_lists, weights):
          for rank, cid in enumerate(lst):
              scores[cid] = scores.get(cid, 0.0) + w / (k + rank)
      return sorted(scores, key=scores.get, reverse=True)
  ```
- Return top-k chunks with `pdf_page` and scores
- Bonus: our own version supports **weighting**, which the built-in doesn't. Tune the weights
  against the eval set.

**Output:** `rrf.py` + `search.py` — input: query string → output: ranked chunks with metadata

**Test:** Run `run_eval.py`. Record recall@5. Then tune (chunk size, top-k, RRF weights) and
re-run — you should be able to *see* the number move.

---

## Part 7: RAG Question Answering
**Goal:** Use retrieved chunks as context for Claude to answer questions.

- Build prompt: system instruction + chunks + question
- Call Claude API with **`claude-opus-5`** (the plan's original `claude-sonnet-4-6` still works
  but is superseded; `claude-sonnet-5` is the cheaper current option)
- **Stream the response** — a 5–10s blank screen reads as a frozen app
- Parse response, extract cited pages — cited as `pdf_page`, never `printed_page`

**Output:** `llm.py` — input: query + chunks → output: streamed answer + source pages

**Test:** Ask a question from the gold set. Verify the answer is correct AND that the cited
`pdf_page` matches the gold set's page.

---

## Part 8: Backend API
**Goal:** Expose everything as REST endpoints.

- `POST /ingest` — **starts a background job**, returns `{job_id, status}` immediately.
  Indexing a 400-page textbook takes 30–60 minutes on CPU (Marker's ML models plus a 2.2 GB
  embedding model); no HTTP request survives that. Use FastAPI `BackgroundTasks`.
- `GET /ingest/{job_id}` — `{status, pages_done, total_pages}` for the progress bar
- `POST /search` — hybrid search, return chunks + pages
- `POST /ask` — RAG Q&A, **SSE streamed**, returns answer + sources
- `GET /books` — list indexed books
- `GET /books/{id}/pdf` — serve original PDF file

**Output:** FastAPI app with 6 endpoints, all testable via Swagger UI

**Test:** Upload a real textbook via `/ingest`. Confirm it returns in under a second, and that
`/ingest/{job_id}` shows progress climbing while indexing runs.

---

## Part 9: Frontend — PDF Viewer
**Goal:** Display a PDF with RTL support and programmatic navigation.

- React + **`react-pdf`** (MIT). *Not* `@react-pdf-viewer` — that is a commercial product
  requiring a paid license.
- Trade-off: `react-pdf` has no prebuilt navigation plugin, so we write the page-scroll glue
  ourselves (~1 day)
- RTL theme configuration
- `jumpToPage(n)` exposed to the parent — **takes `pdf_page`**, the position in the file, not
  the printed page number
- Book selector dropdown

**Output:** `PdfViewer.tsx` — renders PDF, accepts page navigation commands

**Test:** Render a Hebrew PDF, call `jumpToPage(48)`, confirm it lands on the 48th page of the
file.

---

## Part 10: Frontend — Chat Sidebar
**Goal:** Search bar + Q&A chat that returns results with clickable page refs.

- Search input → `/search` → results with page badges
- Chat input → `/ask` → **render the streamed answer token by token** as it arrives
- Page badge displays `printed_page` (what the book says) but carries `pdf_page` (where to
  jump). Clicking it calls `jumpToPage(pdf_page)`.
- `IngestProgress.tsx` — polls `/ingest/{job_id}`, shows a progress bar

**Output:** `ChatSidebar.tsx` — search + chat UI with page link callbacks

**Test:** Type a query, see results with page badges, watch the AI answer appear progressively.

---

## Part 11: Integration & Polish
**Goal:** Wire everything together, handle edge cases.

- Connect sidebar page clicks → PDF viewer scroll
- Loading states, error handling
- Multi-book support (switch between indexed books)
- Docker compose for full stack (ES + backend + frontend)

**Output:** Working end-to-end application

---

## Build Order Visualization

```
Part 0   Hebrew Extraction Spike    ← START HERE. Go/no-go gate.
  ↓
Part 1   Elasticsearch Setup         ← needs custom Dockerfile (ICU plugin)
  ↓
Part 2   Chunking                    ← token-aware, 512 hard cap
  ↓
Part 3   Embeddings                  ← e5 prefixes, normalized
  ↓
Part 4   Indexing Pipeline           ← wires 0+2+3 together
  ↓
Part 5   Evaluation Set              ← build the ruler BEFORE measuring
  ↓
Part 6   Hybrid Search               ← RRF in Python; first "wow" moment
  ↓
Part 7   RAG Q&A                     ← claude-opus-5; second "wow" moment
  ↓
Part 8   Backend API                 ← wraps 4+6+7, /ingest as bg job
  ↓
Part 9   PDF Viewer                  ← react-pdf; can parallel with 6-8
  ↓
Part 10  Chat Sidebar
  ↓
Part 11  Integration                 ← done
```

## Testing Strategy

Each part has a standalone test before moving on. Tests marked 🔄 were rewritten because the
original version would pass while the bug was still present.

| Part | Test |
|------|------|
| 0 | Extract 3 pages; a Hebrew reader confirms they match the real pages |
| 1 | Index `החוסן`, search `חוסן`, get a match (proves the prefix filter works) |
| 2 | 🔄 Assert every chunk ≤ 512 real e5 tokens; verify cross-page metadata |
| 3 | 🔄 Relevant chunk outranks irrelevant one (was: "verify 1024 dims" — passes even when broken) |
| 4 | Index full PDF, check doc count, spot-check a chunk's `pdf_page` against the real PDF |
| 5 | `run_eval.py` prints a recall@5 baseline |
| 6 | 🔄 recall@5 improves as you tune (was: one query returns one page) |
| 7 | Gold-set question → correct answer AND correct cited `pdf_page` |
| 8 | `/ingest` returns in <1s; `/ingest/{job_id}` shows climbing progress |
| 9 | Render a Hebrew PDF, `jumpToPage(48)` lands on the file's 48th page |
| 10 | Type query, see page badges, watch the answer stream in |
| 11 | Click page badge → PDF scrolls to that page |
