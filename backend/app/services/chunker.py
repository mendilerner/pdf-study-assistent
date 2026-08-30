import re
from functools import lru_cache

from app.services.constants import MODEL_NAME
from app.services.logger import get_logger

logger = get_logger(__name__)

TARGET_TOKENS = 300
OVERLAP_TOKENS = 50
MAX_TOKENS = 512

_tokenizer = None

_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return _tokenizer


@lru_cache(maxsize=None)
def _token_len(text: str) -> int:
    return len(_get_tokenizer().encode(text, add_special_tokens=True))


def _split_sentences(text: str) -> list[str]:
    raw = _SENTENCE_SPLIT_RE.split(text)
    merged = []
    for s in raw:
        s = s.strip()
        if not s:
            continue
        if merged and len(merged[-1]) < 15:
            merged[-1] = merged[-1] + " " + s
        else:
            merged.append(s)
    return merged


def _build_sentence_stream(pages: list[dict]) -> list[tuple[str, int]]:
    stream = []
    for page in pages:
        text = page["text"].strip()
        if not text:
            continue
        pdf_page = page["pdf_page"]
        for sentence in _split_sentences(text):
            stream.append((sentence, pdf_page))
    return stream


def _force_split_long_sentence(text: str) -> list[str]:
    tokenizer = _get_tokenizer()
    words = text.split()
    pieces = []
    current = []
    current_tokens = 0
    limit = MAX_TOKENS - 10
    for word in words:
        word_tokens = len(tokenizer.encode(word, add_special_tokens=False))
        if current and current_tokens + word_tokens >= limit:
            pieces.append(" ".join(current))
            current = [word]
            current_tokens = word_tokens
        else:
            current.append(word)
            current_tokens += word_tokens
    if current:
        pieces.append(" ".join(current))
    return pieces


def chunk(
    pages: list[dict],
    book_id: str,
    book_title: str,
    page_offset: int = 0,
) -> list[dict]:
    stream = _build_sentence_stream(pages)
    if not stream:
        return []

    # Expand any sentences that exceed MAX_TOKENS
    expanded = []
    for sentence, pdf_page in stream:
        if _token_len(sentence) > MAX_TOKENS:
            logger.warning(
                "sentence on page %d exceeds %d tokens, splitting at word boundary",
                pdf_page, MAX_TOKENS,
            )
            for piece in _force_split_long_sentence(sentence):
                expanded.append((piece, pdf_page))
        else:
            expanded.append((sentence, pdf_page))
    stream = expanded

    results = []
    chunk_index = 0
    i = 0

    while i < len(stream):
        current_sents: list[tuple[str, int]] = []
        current_tokens = 0

        while i < len(stream):
            sent, page = stream[i]
            sent_tokens = _token_len(sent)

            if current_tokens + sent_tokens > TARGET_TOKENS and current_sents:
                break

            current_sents.append((sent, page))
            current_tokens += sent_tokens
            i += 1

        text = " ".join(s for s, _ in current_sents)
        real_tokens = _token_len(text)
        assert real_tokens <= MAX_TOKENS, (
            f"chunk {chunk_index} has {real_tokens} tokens (max {MAX_TOKENS})"
        )

        pdf_page = current_sents[0][1]
        pdf_page_end = current_sents[-1][1]

        results.append({
            "text": text,
            "book_id": book_id,
            "book_title": book_title,
            "pdf_page": pdf_page,
            "pdf_page_end": pdf_page_end,
            "printed_page": pdf_page - page_offset,
            "chunk_id": f"{book_id}__chunk_{chunk_index:05d}",
            "chunk_index": chunk_index,
        })
        chunk_index += 1

        if i < len(stream):
            overlap_tokens = 0
            rewind_count = 0
            max_rewind = len(current_sents) - 1
            for j in range(len(current_sents) - 1, -1, -1):
                if rewind_count >= max_rewind:
                    break
                sent_tokens = _token_len(current_sents[j][0])
                if overlap_tokens + sent_tokens > OVERLAP_TOKENS:
                    break
                overlap_tokens += sent_tokens
                rewind_count += 1
            if rewind_count > 0:
                i -= rewind_count

    _token_len.cache_clear()
    return results
