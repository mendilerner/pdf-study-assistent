"""Part 4 verification: End-to-end indexing pipeline.

Requires:
- Elasticsearch running: docker compose up -d
- Test PDF at data/10493-5127.pdf

Usage:
    PYTHONIOENCODING=utf-8 uv run python backend/eval/test_part4.py
"""

import sys
import os

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.constants import EMBEDDING_DIMS
from app.services.ingest import ingest, BOOKS_DIR
from app.services.elastic import (
    get_client,
    wait_for_ready,
    get_doc_count,
    get_books,
    INDEX_NAME,
    BOOKS_INDEX_NAME,
)
from app.services.pdf_parser import parse_pages

PDF_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "10493-5127.pdf")
TEST_TITLE = "Dev Psychology Test"
# Only ingest 20 pages to keep the test fast on CPU (~2-3 min instead of 30+)
TEST_PAGE_RANGE = (50, 70)


def main():
    if not os.path.exists(PDF_PATH):
        print(f"ERROR: PDF not found: {PDF_PATH}")
        print("Place the test PDF at data/10493-5127.pdf")
        sys.exit(1)

    # Verify ES is running
    print("Connecting to Elasticsearch...")
    client = get_client()
    try:
        wait_for_ready(client, timeout=10)
    except TimeoutError:
        print("ERROR: Elasticsearch not running. Start it with: docker compose up -d")
        sys.exit(1)
    print("ES is ready.\n")

    # Clean up any stale data from previous runs
    print("Cleaning up old test data...")
    client.options(ignore_status=[404]).indices.delete(index=INDEX_NAME)
    client.options(ignore_status=[404]).indices.delete(index=BOOKS_INDEX_NAME)
    print()

    # Run ingest
    print("=" * 50)
    print("INGESTING TEST PDF")
    print("=" * 50)
    result = ingest(PDF_PATH, TEST_TITLE, page_range=TEST_PAGE_RANGE)
    book_id = result["book_id"]
    print()

    passed = True

    # Test 1: Chunks indexed
    print("--- Test 1: Chunks indexed ---")
    count = get_doc_count(client)
    if count > 0:
        print(f"  {count} chunks in ES — PASS")
    else:
        print("  FAIL: no chunks in ES")
        passed = False

    # Test 2: Book registered
    print("\n--- Test 2: Book registered ---")
    books = get_books(client)
    book = next((b for b in books if b["book_id"] == book_id), None)
    if book:
        print(f"  title: {book['title']}")
        print(f"  page_offset: {book['page_offset']}")
        print(f"  page_count: {book['page_count']}")
        print(f"  chunk_count: {book['chunk_count']}")
        if book["title"] == TEST_TITLE and book["chunk_count"] > 0:
            print("  PASS")
        else:
            print("  FAIL: metadata mismatch")
            passed = False
    else:
        print(f"  FAIL: book {book_id} not found in study_books")
        passed = False

    # Test 3: Spot-check page attribution
    print("\n--- Test 3: Spot-check page attribution ---")
    resp = client.search(
        index=INDEX_NAME,
        query={"term": {"book_id": book_id}},
        size=1,
        sort=[{"chunk_index": "asc"}],
        _source=["text", "pdf_page", "chunk_index"],
    )
    if resp["hits"]["hits"]:
        sample = resp["hits"]["hits"][0]["_source"]
        pdf_page = sample["pdf_page"]
        chunk_text_snippet = sample["text"][:80]

        # Extract that page from PDF and check the text is there
        page_data = parse_pages(PDF_PATH, [pdf_page])
        if page_data:
            page_text = page_data[0]["text"]
            # Check if a significant portion of chunk words appear on the page
            chunk_words = set(sample["text"].split()[:10])
            page_words = set(page_text.split())
            overlap = chunk_words & page_words
            pct = len(overlap) / len(chunk_words) * 100 if chunk_words else 0
            print(f"  chunk 0 claims pdf_page={pdf_page}")
            print(f"  chunk starts: \"{chunk_text_snippet}...\"")
            print(f"  word overlap with actual page: {len(overlap)}/{len(chunk_words)} ({pct:.0f}%)")
            if pct >= 50:
                print("  PASS")
            else:
                print("  FAIL: chunk text doesn't match the claimed page")
                passed = False
        else:
            print(f"  FAIL: could not parse page {pdf_page}")
            passed = False
    else:
        print("  FAIL: no chunks found for this book")
        passed = False

    # Test 4: Embeddings present
    print("\n--- Test 4: Embeddings present and correct dims ---")
    resp = client.search(
        index=INDEX_NAME,
        query={"term": {"book_id": book_id}},
        size=5,
        _source=["embedding", "chunk_index"],
    )
    all_ok = True
    for hit in resp["hits"]["hits"]:
        emb = hit["_source"].get("embedding")
        if emb is None:
            print(f"  FAIL: chunk {hit['_source']['chunk_index']} has no embedding")
            all_ok = False
        elif len(emb) != EMBEDDING_DIMS:
            print(f"  FAIL: chunk {hit['_source']['chunk_index']} embedding is {len(emb)} dims")
            all_ok = False
    if all_ok:
        print(f"  checked {len(resp['hits']['hits'])} chunks — all have {EMBEDDING_DIMS}-dim embeddings — PASS")
    else:
        passed = False

    # Test 5: PDF copied
    print("\n--- Test 5: PDF file stored ---")
    stored_path = os.path.join(BOOKS_DIR, f"{book_id}.pdf")
    if os.path.exists(stored_path):
        size_mb = os.path.getsize(stored_path) / (1024 * 1024)
        print(f"  {stored_path}")
        print(f"  size: {size_mb:.1f} MB — PASS")
    else:
        print(f"  FAIL: {stored_path} not found")
        passed = False

    # Test 6: Chunk count matches
    print("\n--- Test 6: Chunk count consistency ---")
    es_count = client.count(index=INDEX_NAME, query={"term": {"book_id": book_id}})["count"]
    meta_count = book["chunk_count"] if book else 0
    if es_count == meta_count and es_count > 0:
        print(f"  ES chunks: {es_count}, book meta: {meta_count} — PASS")
    else:
        print(f"  FAIL: ES has {es_count} but book meta says {meta_count}")
        passed = False

    print(f"\n{'='*50}")
    if passed:
        print(f"Part 4 PASSED — {es_count} chunks indexed, book registered.")
    else:
        print("Part 4 FAILED — see errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
