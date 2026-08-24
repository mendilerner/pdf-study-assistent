"""Part 1 verification: Hebrew prefix stripping in Elasticsearch.

Requires a running Elasticsearch instance with the ICU plugin.
Start it with: docker compose up -d

Usage:
    uv run python backend/eval/test_part1.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.elastic import (
    get_client,
    wait_for_ready,
    create_index,
    index_document,
    search_text,
    get_doc_count,
)


def main():
    client = get_client()

    print("Waiting for Elasticsearch...")
    try:
        wait_for_ready(client, timeout=60)
    except TimeoutError:
        print("ERROR: Elasticsearch is not running.")
        print("Start it with: docker compose up -d")
        sys.exit(1)

    info = client.info()
    print(f"Connected to Elasticsearch {info['version']['number']}")

    print("\nCreating index with Hebrew analyzer...")
    create_index(client, delete_existing=True)

    docs = [
        {"text": "החוסן הנפשי הוא גורם מרכזי", "book_id": "test", "pdf_page": 1},
        {"text": "וחוסן חברתי תורם לרווחה", "book_id": "test", "pdf_page": 2},
        {"text": "תיאוריה של התפתחות הילד", "book_id": "test", "pdf_page": 3},
    ]

    print("Indexing 3 test documents...")
    for i, doc in enumerate(docs):
        index_document(client, doc, doc_id=f"test-{i}")

    client.indices.refresh(index="study_chunks")
    count = get_doc_count(client)
    print(f"Documents in index: {count}")

    print("\n--- Test: Hebrew prefix stripping ---")
    print("Searching for: חוסן")
    print("Expected: docs 1 and 2 (החוסן, וחוסן) should match\n")

    results = search_text(client, "חוסן")
    matched_pages = [r["pdf_page"] for r in results]

    passed = True

    if 1 in matched_pages:
        print("  PASS: החוסן (hei prefix) matched")
    else:
        print("  FAIL: החוסן did not match")
        passed = False

    if 2 in matched_pages:
        print("  PASS: וחוסן (vav prefix) matched")
    else:
        print("  FAIL: וחוסן did not match")
        passed = False

    if 3 not in matched_pages:
        print("  PASS: unrelated doc correctly excluded")
    else:
        print("  FAIL: unrelated doc incorrectly matched")
        passed = False

    # Show what the analyzer actually does
    print("\n--- Analyzer debug ---")
    analyze_resp = client.indices.analyze(
        index="study_chunks",
        body={"analyzer": "hebrew_analyzer", "text": "החוסן וחוסן בחוסן חוסן"},
    )
    tokens = [t["token"] for t in analyze_resp["tokens"]]
    print(f"  Input:  החוסן וחוסן בחוסן חוסן")
    print(f"  Tokens: {tokens}")

    print(f"\n{'='*40}")
    if passed:
        print("Part 1 PASSED — Hebrew prefix stripping works.")
    else:
        print("Part 1 FAILED — see errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
