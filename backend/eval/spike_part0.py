"""Part 0: Hebrew Extraction Spike — Marker vs PyMuPDF comparison.

Usage:
    python backend/eval/spike_part0.py <pdf_path> <page1> <page2> [page3] ...

Page numbers are 1-based (matching what a PDF viewer shows).
Outputs comparison files to data/spike_output/.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pymupdf
from app.services.pdf_parser import _extract_page_text, detect_page_offset, has_text_layer


def extract_pymupdf(pdf_path: str, pages: list[int]) -> list[dict]:
    doc = pymupdf.open(pdf_path)
    total_pages = len(doc)
    results = []
    for page_num in pages:
        page_idx = page_num - 1
        if page_idx < 0 or page_idx >= total_pages:
            print(f"WARNING: Page {page_num} out of range (1-{total_pages}), skipping")
            continue
        text = _extract_page_text(doc[page_idx])
        results.append({"pdf_page": page_num, "text": text})
    doc.close()
    return results


def extract_marker(pdf_path: str, pages: list[int]) -> list[dict]:
    try:
        from marker.converters.pdf import PdfConverter
        from marker.config.parser import ConfigParser
        from marker.models import create_model_dict
        from marker.output import text_from_rendered
    except ImportError as e:
        print(f"\nMarker import failed: {e}")
        print("Marker may not be installed or incompatible with this Python version.")
        print("Continuing with PyMuPDF results only.\n")
        return []

    page_indices = [p - 1 for p in pages]
    page_range_str = ",".join(str(i) for i in page_indices)

    try:
        config = {
            "page_range": page_range_str,
            "output_format": "markdown",
        }

        config_parser = ConfigParser(config)
        converter = PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=create_model_dict(),
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
        )

        rendered = converter(pdf_path)
        text, _, _ = text_from_rendered(rendered)

        return [{"pdf_page": pages[0], "text": text, "note": "combined markdown"}]

    except Exception as e:
        print(f"\nMarker extraction failed: {e}")
        print(f"Exception type: {type(e).__name__}")

        try:
            import marker
            print(f"Marker version: {marker.__version__}")
        except Exception:
            pass

        try:
            import inspect
            from marker.converters.pdf import PdfConverter
            print(f"PdfConverter.__init__ signature: {inspect.signature(PdfConverter.__init__)}")
        except Exception:
            pass

        return []


def write_output(output_dir: str, page_num: int, extractor: str, text: str):
    os.makedirs(output_dir, exist_ok=True)
    filename = f"page_{page_num:03d}_{extractor}.txt"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)
    return filepath


def main():
    parser = argparse.ArgumentParser(
        description="Part 0: Compare Hebrew PDF extractors"
    )
    parser.add_argument("pdf_path", help="Path to the Hebrew textbook PDF")
    parser.add_argument(
        "pages", nargs="+", type=int,
        help="1-based page numbers to extract (e.g., 10 100 350)"
    )
    parser.add_argument(
        "--output-dir", default="data/spike_output",
        help="Directory for output files (default: data/spike_output)"
    )
    parser.add_argument(
        "--skip-marker", action="store_true",
        help="Skip Marker extraction (useful if Marker is not installed)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.pdf_path):
        print(f"ERROR: PDF not found: {args.pdf_path}")
        sys.exit(1)

    doc = pymupdf.open(args.pdf_path)
    total_pages = len(doc)
    doc.close()
    print(f"PDF: {args.pdf_path}")
    print(f"Total pages: {total_pages}")
    print(f"Pages to extract: {args.pages}")

    # Check for text layer
    print("\nChecking for text layer...")
    if has_text_layer(args.pdf_path):
        print("Text layer detected — good, extraction should work.")
    else:
        print("WARNING: No text layer detected — PDF may be scanned images.")
        print("PyMuPDF will return empty text. Only Marker with OCR can help.")

    # Detect page offset
    print("\nDetecting page offset...")
    offset = detect_page_offset(args.pdf_path)
    print(f"Page offset: {offset} (printed page 1 = PDF page {offset + 1})")

    # --- PyMuPDF extraction ---
    print("\n" + "=" * 60)
    print("EXTRACTING WITH PyMuPDF (RTL-aware)")
    print("=" * 60)

    start = time.perf_counter()
    pymupdf_results = extract_pymupdf(args.pdf_path, args.pages)
    pymupdf_time = time.perf_counter() - start
    print(f"Time: {pymupdf_time:.3f}s")

    for result in pymupdf_results:
        page_num = result["pdf_page"]
        text = result["text"]
        filepath = write_output(args.output_dir, page_num, "pymupdf", text)

        print(f"\n--- PyMuPDF: Page {page_num} ({len(text)} chars) ---")
        print(f"Saved to: {filepath}")
        preview = text[:2000]
        if len(text) > 2000:
            preview += f"\n... [{len(text) - 2000} more chars]"
        print(preview)

    # --- Marker extraction ---
    marker_results = []
    marker_time = 0.0

    if not args.skip_marker:
        print("\n" + "=" * 60)
        print("EXTRACTING WITH Marker")
        print("=" * 60)

        start = time.perf_counter()
        marker_results = extract_marker(args.pdf_path, args.pages)
        marker_time = time.perf_counter() - start
        print(f"Time: {marker_time:.3f}s")

        for result in marker_results:
            page_num = result["pdf_page"]
            text = result["text"]
            note = result.get("note", "")
            filepath = write_output(args.output_dir, page_num, "marker", text)

            print(f"\n--- Marker: Page {page_num} ({len(text)} chars) {note} ---")
            print(f"Saved to: {filepath}")
            preview = text[:2000]
            if len(text) > 2000:
                preview += f"\n... [{len(text) - 2000} more chars]"
            print(preview)
    else:
        print("\n(Marker extraction skipped)")

    # --- Summary & Decision Checklist ---
    print("\n" + "=" * 60)
    print("TIMING SUMMARY")
    print("=" * 60)
    print(f"PyMuPDF: {pymupdf_time:.3f}s")
    if marker_results:
        print(f"Marker:  {marker_time:.3f}s")

    print(f"\nPage offset detected: {offset}")

    print("\n" + "=" * 60)
    print("PART 0 DECISION: GO / NO-GO")
    print("=" * 60)
    print("""
Review the extracted text above (and the files in {output_dir})
and answer these questions:

1. HEBREW CORRECTNESS
   Are Hebrew characters correct (not reversed/garbled)?
   - PyMuPDF: ___
   - Marker:  ___

2. LINE ORDER
   Are lines in the correct reading order?
   - PyMuPDF: ___
   - Marker:  ___

3. HEADINGS
   Are section headings identifiable and intact?
   - PyMuPDF: ___
   - Marker:  ___

4. MIXED CONTENT
   Do lines with both Hebrew and English text survive intact?
   - PyMuPDF: ___
   - Marker:  ___

5. PAGE ATTRIBUTION
   Does each extracted text map to the correct PDF page?
   - PyMuPDF: native (trivial)
   - Marker:  via page_range config

6. PERFORMANCE
   - PyMuPDF: {pymupdf_time:.3f}s
   - Marker:  {marker_time:.3f}s

DECISION RULE: Correct plain text beats well-structured garbage.
Choose the extractor whose TEXT IS CORRECT, even if less structured.

Winner: _______________
Decision: GO / NO-GO
""".format(
        output_dir=args.output_dir,
        pymupdf_time=pymupdf_time,
        marker_time=marker_time,
    ))

    print(f"Output files are in: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
