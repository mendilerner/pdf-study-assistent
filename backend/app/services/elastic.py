import os
import time

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from app.services.constants import EMBEDDING_DIMS

INDEX_NAME = "study_chunks"
BOOKS_INDEX_NAME = "study_books"

INDEX_BODY = {
    "settings": {
        "analysis": {
            "analyzer": {
                "hebrew_analyzer": {
                    "type": "custom",
                    "tokenizer": "icu_tokenizer",
                    "filter": [
                        "icu_normalizer",
                        "icu_folding",
                        "hebrew_prefix_strip",
                    ],
                }
            },
            "filter": {
                "hebrew_prefix_strip": {
                    "type": "pattern_replace",
                    "pattern": "^[והבכלמש]{1,2}(?=[א-ת]{3,})",
                    "replacement": "",
                }
            },
        }
    },
    "mappings": {
        "properties": {
            "text": {"type": "text", "analyzer": "hebrew_analyzer"},
            "embedding": {
                "type": "dense_vector",
                "dims": EMBEDDING_DIMS,
                "similarity": "cosine",
            },
            "book_id": {"type": "keyword"},
            "book_title": {"type": "keyword"},
            "pdf_page": {"type": "integer"},
            "pdf_page_end": {"type": "integer"},
            "printed_page": {"type": "integer"},
            "chunk_id": {"type": "keyword"},
            "chunk_index": {"type": "integer"},
        }
    },
}


def get_client(host: str | None = None) -> Elasticsearch:
    host = host or os.environ.get("ES_HOST", "http://localhost:9200")
    return Elasticsearch(host)


def wait_for_ready(client: Elasticsearch, timeout: int = 60) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            health = client.cluster.health(timeout="1s")
            if health["status"] in ("green", "yellow"):
                return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"Elasticsearch not ready after {timeout}s")


def create_index(client: Elasticsearch, delete_existing: bool = False) -> None:
    if delete_existing:
        delete_index(client)
    if not client.indices.exists(index=INDEX_NAME):
        client.indices.create(index=INDEX_NAME, body=INDEX_BODY)


def delete_index(client: Elasticsearch) -> None:
    client.indices.delete(index=INDEX_NAME, ignore=[404])


def index_document(
    client: Elasticsearch, doc: dict, doc_id: str | None = None
) -> None:
    client.index(index=INDEX_NAME, id=doc_id, document=doc)


def bulk_index(client: Elasticsearch, docs: list[dict], id_field: str = "chunk_id") -> int:
    actions = []
    for doc in docs:
        action = {"_index": INDEX_NAME, "_source": doc}
        if id_field and id_field in doc:
            action["_id"] = doc[id_field]
        actions.append(action)
    success, _ = bulk(client, actions)
    return success


def search_text(client: Elasticsearch, query: str, size: int = 10) -> list[dict]:
    resp = client.search(
        index=INDEX_NAME,
        query={"match": {"text": query}},
        size=size,
    )
    return [hit["_source"] for hit in resp["hits"]["hits"]]


def get_doc_count(client: Elasticsearch) -> int:
    return client.count(index=INDEX_NAME)["count"]


# --- Books index ---

BOOKS_INDEX_BODY = {
    "mappings": {
        "properties": {
            "book_id": {"type": "keyword"},
            "title": {"type": "keyword"},
            "pdf_filename": {"type": "keyword"},
            "page_offset": {"type": "integer"},
            "page_count": {"type": "integer"},
            "chunk_count": {"type": "integer"},
            "indexed_at": {"type": "date"},
        }
    },
}


def create_books_index(client: Elasticsearch) -> None:
    if not client.indices.exists(index=BOOKS_INDEX_NAME):
        client.indices.create(index=BOOKS_INDEX_NAME, body=BOOKS_INDEX_BODY)


def register_book(client: Elasticsearch, book_meta: dict) -> None:
    client.index(
        index=BOOKS_INDEX_NAME,
        id=book_meta["book_id"],
        document=book_meta,
    )


def get_books(client: Elasticsearch) -> list[dict]:
    resp = client.search(index=BOOKS_INDEX_NAME, query={"match_all": {}}, size=100)
    return [hit["_source"] for hit in resp["hits"]["hits"]]


def delete_book_chunks(client: Elasticsearch, book_id: str) -> int:
    if not client.indices.exists(index=INDEX_NAME):
        return 0
    resp = client.delete_by_query(
        index=INDEX_NAME,
        query={"term": {"book_id": book_id}},
        refresh=True,
    )
    return resp.get("deleted", 0)
