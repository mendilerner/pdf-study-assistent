# PDF Study Assistant

Academic PDF study assistant — upload Hebrew textbook PDFs, search semantically, ask AI questions.

## Setup

- Python 3.13
- Virtual environment: `.venv/`
- Activate: `.venv\Scripts\activate` (Windows)
- Install: `pip install -r backend/requirements.txt`
- PDF files go in `data/` (gitignored)

## Current Status

Part 0: Hebrew Extraction Spike — **COMPLETE (GO)**.
- Winner: PyMuPDF with RTL-aware span reconstruction
- Marker installed but not needed — PyMuPDF produces correct Hebrew text
- Page offset for test PDF: 24 (printed page 1 = PDF page 25)

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
