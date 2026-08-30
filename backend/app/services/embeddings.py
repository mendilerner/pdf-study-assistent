# SentenceTransformer wraps the e5 model and handles encoding text -> vectors
from sentence_transformers import SentenceTransformer

from app.services.constants import MODEL_NAME

# The model is ~2.2GB in memory. We keep one instance alive for the whole
# process lifetime so we don't reload it on every call.
_model = None


def _get_model() -> SentenceTransformer:
    """Lazy singleton: loads the model on first call, reuses it after that."""
    global _model
    if _model is None:
        # device="cpu" because our GPU (Arc 140T) doesn't have enough VRAM.
        # First call takes ~10-20s to load weights; subsequent calls are instant.
        _model = SentenceTransformer(MODEL_NAME, device="cpu")
    return _model


def embed_passages(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Embed document chunks for indexing into Elasticsearch.

    The e5 model was trained with a "passage: " prefix on documents and
    "query: " on questions. Omitting these prefixes silently degrades
    retrieval quality while still returning valid-looking vectors —
    a bug that's impossible to catch by checking vector shape alone.

    normalize_embeddings=True makes every vector unit-length (L2 norm = 1),
    which is required for cosine similarity to work correctly in ES kNN.

    batch_size=32 controls how many texts are encoded at once — keeps
    memory usage reasonable on CPU.
    """
    prefixed = [f"passage: {t}" for t in texts]
    vectors = _get_model().encode(prefixed, normalize_embeddings=True, batch_size=batch_size)
    # .tolist() converts the numpy array to plain Python lists, which is
    # what Elasticsearch expects when indexing dense_vector fields.
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    """Embed a search question. Uses "query: " prefix (not "passage: ").

    Returns a single 768-dim vector (e5-base). At search time this vector
    is compared against all passage vectors in ES via cosine similarity (kNN).
    """
    vector = _get_model().encode(f"query: {text}", normalize_embeddings=True)
    return vector.tolist()
