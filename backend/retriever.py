from __future__ import annotations

import os
import re
from typing import Any, cast

STOP_WORDS = {
    "what", "is", "the", "a", "an", "in", "of", "to", "and", "or", "how", "why", "when", "where",
    "who", "does", "do", "this", "that", "it", "are", "was", "were", "be", "been", "being", "have",
    "has", "had", "will", "would", "could", "should", "may", "might", "can", "about", "tell", "me",
    "explain",
}

try:  # pragma: no cover - optional dependency path
    import numpy as np  # type: ignore[import-not-found]
    import faiss  # type: ignore[import-not-found]
    from sentence_transformers import SentenceTransformer as _SentenceTransformer  # type: ignore[import-not-found]

    EMBEDDING_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency path
    np = None
    faiss = None
    _SentenceTransformer = None
    EMBEDDING_AVAILABLE = False


def _truncate(text: str, max_chars: int) -> str:
    return text[:max_chars].strip()


def _split_chunks(text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    step = max(1, chunk_size - overlap)
    chunks = []
    for start in range(0, len(text), step):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks or [text]


def _keyword_fallback(text: str, question: str, max_chars: int) -> str:
    sentences = [part.strip() for part in re.split(r"\n+|(?<=[.!?])\s+", text) if part.strip()]
    if not sentences:
        return _truncate(text, max_chars)

    question_words = {
        word.lower()
        for word in re.findall(r"[a-zA-Z0-9']+", question)
        if len(word) > 2 and word.lower() not in STOP_WORDS
    }

    if not question_words:
        return _truncate(text, max_chars)

    scored: list[tuple[int, int, str]] = []
    for index, sentence in enumerate(sentences):
        sentence_words = set(re.findall(r"[a-zA-Z0-9']+", sentence.lower()))
        score = len(sentence_words & question_words)
        if score > 0:
            scored.append((score, index, sentence))

    if not scored:
        return _truncate(text, max_chars)

    scored.sort(key=lambda item: (-item[0], item[1]))
    selected = sorted(scored[:15], key=lambda item: item[1])
    combined = " ".join(sentence for _, _, sentence in selected)
    return _truncate(combined, max_chars)


def _embedding_retrieval(text: str, question: str, max_chars: int) -> str:
    if (
        not EMBEDDING_AVAILABLE
        or not os.getenv("OPENROUTER_API_KEY")
        or _SentenceTransformer is None
        or faiss is None
    ):
        return _keyword_fallback(text, question, max_chars)

    chunks = _split_chunks(text)
    if len(chunks) == 1:
        return _truncate(text, max_chars)

    model = _SentenceTransformer("all-MiniLM-L6-v2")
    chunk_embeddings = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    question_embedding = model.encode([question], convert_to_numpy=True, normalize_embeddings=True)

    dim = chunk_embeddings.shape[1]
    faiss_module = cast(Any, faiss)
    index = faiss_module.IndexFlatIP(int(dim))
    index.add(chunk_embeddings.astype("float32"))

    _, indices = index.search(question_embedding.astype("float32"), min(5, len(chunks)))
    retrieved = []
    for idx in indices[0]:
        if idx >= 0:
            retrieved.append(chunks[idx])

    combined = "\n---\n".join(retrieved).strip()
    return _truncate(combined or text, max_chars)


def retrieve_context(text: str, question: str, max_chars: int = 3000) -> str:
    """Returns the most relevant portion of transcript for the question."""
    text = (text or "").strip()
    if not text:
        return ""

    if len(text) <= max_chars:
        return text

    try:
        return _embedding_retrieval(text, question, max_chars)
    except Exception:
        return _keyword_fallback(text, question, max_chars)