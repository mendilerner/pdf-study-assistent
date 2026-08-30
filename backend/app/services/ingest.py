"""End-to-end ingestion pipeline: PDF -> parse -> chunk -> embed -> index.

Usage as a script:
    uv run python -m backend.app.services.ingest data/10493-5127.pdf "Dev Psychology"
"""

import os
import shutil
import time
import uuid
from datetime import datetime, timezone

from app.services.logger import get_logger
from app.services.pdf_parser import parse, parse_pages, detect_page_offset, get_page_count
from app.services.chunker import chunk
from app.services.embeddings import embed_passages
from app.services.elastic import (
    get_client,
    wait_for_ready,
    create_index,
    create_books_index,
    bulk_index,
    delete_book_chunks,
    register_book,
    INDEX_NAME,
    BOOKS_INDEX_NAME,
)

logger = get_logger(__name__)

BOOKS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "books")


def ingest(
    pdf_path: str,
    title: str,
    book_id: str | None = None,
    page_range: tuple[int, int] | None = None,
) -> dict:
    """Ingest a PDF: parse, chunk, embed, and index into Elasticsearch.

    Args:
        pdf_path: path to the PDF file
        title: human-readable book title
        book_id: optional UUID; auto-generated if not provided
        page_range: optional (start, end) 1-based page range for partial ingest

    Returns:
        dict with book_id, chunk_count, and elapsed time
    """
    start = time.time()

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    book_id = book_id or str(uuid.uuid4())
    pdf_filename = os.path.basename(pdf_path)

    # Step 1: detect page offset and count
    logger.info("[1/6] Detecting page offset...")
    page_offset = detect_page_offset(pdf_path)
    page_count = get_page_count(pdf_path)
    logger.info("       offset=%d, pages=%d", page_offset, page_count)

    # Step 2: parse pages
    if page_range:
        page_list = list(range(page_range[0], page_range[1] + 1))
        logger.info("[2/6] Parsing pages %d-%d (%d pages)...", page_range[0], page_range[1], len(page_list))
        pages = parse_pages(pdf_path, page_list)
    else:
        logger.info("[2/6] Parsing all %d pages...", page_count)
        pages = parse(pdf_path)
    non_empty = sum(1 for p in pages if p["text"].strip())
    logger.info("       %d pages with text", non_empty)

    # Step 3: chunk
    logger.info("[3/6] Chunking...")
    chunks = chunk(pages, book_id=book_id, book_title=title, page_offset=page_offset)
    logger.info("       %d chunks produced", len(chunks))

    # Step 4: embed (batched with progress)
    logger.info("[4/6] Embedding %d chunks (this is the slow part on CPU)...", len(chunks))
    texts = [c["text"] for c in chunks]
    batch_size = 32
    all_vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        vectors = embed_passages(batch, batch_size=batch_size)
        all_vectors.extend(vectors)
        done = min(i + batch_size, len(texts))
        logger.info("       embedded %d/%d", done, len(texts))

    # Attach vectors to chunks
    assert len(all_vectors) == len(chunks), (
        f"embedding count mismatch: {len(all_vectors)} vectors for {len(chunks)} chunks"
    )
    for c, vec in zip(chunks, all_vectors):
        c["embedding"] = vec

    # Step 5: index into Elasticsearch
    logger.info("[5/6] Indexing into Elasticsearch...")
    client = get_client()
    wait_for_ready(client)
    create_index(client)
    create_books_index(client)

    deleted = delete_book_chunks(client, book_id)
    if deleted:
        logger.info("       removed %d old chunks for this book", deleted)

    indexed = bulk_index(client, chunks)
    client.indices.refresh(index=INDEX_NAME)
    logger.info("       indexed %d chunks", indexed)

    # Step 6: copy PDF and register book
    logger.info("[6/6] Storing PDF and registering book...")
    os.makedirs(BOOKS_DIR, exist_ok=True)
    dest_path = os.path.join(BOOKS_DIR, f"{book_id}.pdf")
    shutil.copy2(pdf_path, dest_path)

    book_meta = {
        "book_id": book_id,
        "title": title,
        "pdf_filename": pdf_filename,
        "page_offset": page_offset,
        "page_count": page_count,
        "chunk_count": len(chunks),
        "indexed_at": datetime.now(timezone.utc).isoformat(),
    }
    register_book(client, book_meta)
    client.indices.refresh(index=BOOKS_INDEX_NAME)

    elapsed = time.time() - start
    logger.info("Done in %.1fs — %d chunks indexed as book_id=%s", elapsed, len(chunks), book_id)

    return {"book_id": book_id, "chunk_count": len(chunks), "elapsed": elapsed}


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: uv run python -m backend.app.services.ingest <pdf_path> <title>")
        sys.exit(1)

    result = ingest(sys.argv[1], sys.argv[2])
