"""Part 2 verification: Token-aware chunking with overlap.

Requires the test PDF at data/10493-5127.pdf.

Usage:
    uv run python backend/eval/test_part2.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.pdf_parser import parse_pages, detect_page_offset
from app.services.chunker import chunk, _token_len, MAX_TOKENS, TARGET_TOKENS

PDF_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "10493-5127.pdf")
TEST_PAGES = list(range(50, 56))  # 6 consecutive pages for cross-page testing
EXPECTED_KEYS = {
    "text", "book_id", "book_title", "pdf_page", "pdf_page_end",
    "printed_page", "chunk_id", "chunk_index",
}


def main():
    if not os.path.exists(PDF_PATH):
        print(f"ERROR: PDF not found: {PDF_PATH}")
        print("Place the test PDF at data/10493-5127.pdf")
        sys.exit(1)

    print(f"Parsing pages {TEST_PAGES} from PDF...")
    pages = parse_pages(PDF_PATH, TEST_PAGES)
    print(f"Parsed {len(pages)} pages")

    print("\nDetecting page offset...")
    offset = detect_page_offset(PDF_PATH)
    print(f"Page offset: {offset}")

    print("\nChunking...")
    chunks = chunk(pages, book_id="test-book", book_title="Test Book", page_offset=offset)
    print(f"Produced {len(chunks)} chunks\n")

    if not chunks:
        print("FAIL: No chunks produced")
        sys.exit(1)

    passed = True

    # Test 1: Token cap
    print("--- Test 1: Token cap (every chunk <= 512 tokens) ---")
    for c in chunks:
        tlen = _token_len(c["text"])
        if tlen > MAX_TOKENS:
            print(f"  FAIL: chunk {c['chunk_index']} has {tlen} tokens")
            passed = False
    if all(_token_len(c["text"]) <= MAX_TOKENS for c in chunks):
        print("  PASS")

    # Test 2: Target range
    print("\n--- Test 2: Target range (most chunks 200-400 tokens) ---")
    token_counts = [_token_len(c["text"]) for c in chunks]
    in_range = sum(1 for t in token_counts if 200 <= t <= 400)
    pct = in_range / len(token_counts) * 100
    print(f"  {in_range}/{len(token_counts)} chunks in range ({pct:.0f}%)")
    for i, t in enumerate(token_counts):
        print(f"    chunk {i}: {t} tokens")
    if pct >= 60:
        print("  PASS")
    else:
        print("  FAIL: less than 60% of chunks in target range")
        passed = False

    # Test 3: Overlap
    print("\n--- Test 3: Overlap between consecutive chunks ---")
    overlap_found = False
    for i in range(len(chunks) - 1):
        text_a = chunks[i]["text"]
        text_b = chunks[i + 1]["text"]
        # Find shared suffix/prefix
        words_a = text_a.split()
        words_b = text_b.split()
        max_overlap_words = min(len(words_a), len(words_b), 50)
        shared = 0
        for n in range(1, max_overlap_words + 1):
            suffix = words_a[-n:]
            if words_b[:n] == suffix:
                shared = n
        if shared > 0:
            overlap_text = " ".join(words_b[:shared])
            overlap_tokens = _token_len(overlap_text)
            print(f"  chunks {i}->{i+1}: {shared} shared words ({overlap_tokens} tokens)")
            overlap_found = True
    if overlap_found:
        print("  PASS")
    else:
        print("  FAIL: no overlap found between any consecutive chunks")
        passed = False

    # Test 4: Cross-page metadata
    print("\n--- Test 4: Cross-page chunks ---")
    cross_page = [c for c in chunks if c["pdf_page_end"] > c["pdf_page"]]
    if cross_page:
        for c in cross_page:
            print(f"  chunk {c['chunk_index']}: pages {c['pdf_page']}-{c['pdf_page_end']}")
        print("  PASS")
    else:
        print("  WARNING: no chunks span page boundaries (possible but unlikely)")

    # Test 5: Metadata completeness
    print("\n--- Test 5: Metadata completeness ---")
    all_keys_ok = True
    for c in chunks:
        if set(c.keys()) != EXPECTED_KEYS:
            print(f"  FAIL: chunk {c['chunk_index']} keys: {set(c.keys())}")
            all_keys_ok = False
            passed = False
    if all_keys_ok:
        print("  PASS")

    # Test 6: chunk_id uniqueness
    print("\n--- Test 6: chunk_id uniqueness ---")
    ids = [c["chunk_id"] for c in chunks]
    if len(ids) == len(set(ids)):
        print("  PASS")
    else:
        print("  FAIL: duplicate chunk_ids found")
        passed = False

    # Test 7: printed_page correctness
    print("\n--- Test 7: printed_page = pdf_page - offset ---")
    pp_ok = all(c["printed_page"] == c["pdf_page"] - offset for c in chunks)
    if pp_ok:
        print("  PASS")
    else:
        print("  FAIL: printed_page mismatch")
        passed = False

    print(f"\n{'='*40}")
    if passed:
        print("Part 2 PASSED — chunking works correctly.")
    else:
        print("Part 2 FAILED — see errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
