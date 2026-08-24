# PDF Study Assistant

Academic PDF study assistant — upload Hebrew textbook PDFs, search semantically, ask AI questions.

## Setup

- Python 3.13
- Uses **uv** for dependency management (`pyproject.toml` at project root)
- Install/sync: `uv sync`
- Run scripts: `uv run python <script>`
- Add dependency: `uv add <package>`
- PDF files go in `data/` (gitignored)

## Current Status

Part 0: Hebrew Extraction Spike — **COMPLETE (GO)**.
- Winner: PyMuPDF with RTL-aware span reconstruction
- Marker installed but not needed — PyMuPDF produces correct Hebrew text
- Page offset for test PDF: 24 (printed page 1 = PDF page 25)

Part 1: Elasticsearch Setup — **COMPLETE**.
- Custom Docker image with analysis-icu plugin
- Hebrew analyzer: icu_tokenizer + icu_normalizer + icu_folding + hebrew_prefix_strip
- Index `study_chunks` includes dense_vector field (1024 dims) for future use
- Kibana available at http://localhost:5601
- Start ES + Kibana: `docker compose up -d`
- Verify: `uv run python backend/eval/test_part1.py`

Part 2: Chunking — **COMPLETE**.
- Token-aware chunking using e5's own tokenizer (not word count)
- Target ~300 tokens, ~50-token sentence overlap, hard cap 512 tokens
- Chunks can span page boundaries (tracks pdf_page + pdf_page_end)
- Sentence splitting with Hebrew/English citation handling
- Verify: `PYTHONIOENCODING=utf-8 uv run python backend/eval/test_part2.py`

## Key Technical Decisions

- **PDF parser**: PyMuPDF (`import pymupdf`), NOT `get_text(sort=True)` (reverses RTL word order).
  Uses `get_text("dict")` to extract spans, groups by y-coordinate into visual lines,
  sorts right-to-left within each line for correct Hebrew logical order.
- Two-column detection via text-density histogram finds the column gap.
- Headings detected by font size + bold flags, prefixed with `## `.
- Watermark text filtered by known patterns.

## Project Structure

- `backend/app/services/` — core service modules
- `backend/eval/` — evaluation and spike scripts
- `architecture.md` — full architecture spec
- `roadmap.md` — build order and testing strategy
