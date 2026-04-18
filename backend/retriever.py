from __future__ import annotations

import os
import re
from typing import Any, cast

STOP_WORDS = {
    "what", "is", "the", "a", "an", "in", "of", "to", "and", "or", "how", "why", "when", "where",
    "who", "does", "do", "this", "that", "it", "are", "was", "were", "be", "been", "being", "have",
    "has", "had", "will", "would", "could", "should", "may", "might", "can", "about", "tell", "me",
    "explain", "give", "show", "get", "find", "let", "say", "said", "so", "but", "if", "then", "than",
    "also", "just", "like", "not", "no", "yes", "i", "you", "we", "he", "she", "they", "my", "your",
    "his", "her", "its", "our", "their", "which", "there", "here", "now", "up", "out", "at", "by",
    "for", "with", "on", "from", "into", "through", "during", "more", "very", "well", "back", "even",
    "still", "however", "after", "before", "between", "both", "each", "few", "over", "under", "again",
    "further", "once", "all", "any", "most", "other", "some", "such", "only", "own", "same", "too",
    "make", "use", "made", "call", "called", "go", "come", "know", "think", "need", "want", "look",
    "see", "used", "using", "takes", "take", "put",
}


def extract_keywords(text: str) -> set[str]:
    """Extract meaningful keywords by removing short/common tokens."""
    words = re.findall(r"\b[a-zA-Z]{3,}\b", (text or "").lower())
    return {word for word in words if word not in STOP_WORDS}

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


def retrieve_top_sentences(text: str, question: str, n: int = 3) -> str:
    """Returns the top N question-relevant sentences in original order."""
    sentences = [part.strip() for part in re.split(r"\n+|(?<=[.!?])\s+", text or "") if part.strip()]
    if not sentences:
        return ""

    question_words = extract_keywords(question)
    if not question_words:
        return " ".join(sentences[:n]).strip()

    scored: list[tuple[int, int, str]] = []
    for index, sentence in enumerate(sentences):
        words = extract_keywords(sentence)
        overlap = len(words & question_words)
        if overlap > 0:
            scored.append((index, overlap, sentence))

    if not scored:
        return " ".join(sentences[:n]).strip()

    scored.sort(key=lambda item: (-item[1], item[0]))
    top = sorted(scored[:n], key=lambda item: item[0])
    return " ".join(sentence for _, _, sentence in top).strip()


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

    question_words = extract_keywords(question)

    if not question_words:
        return _truncate(text, max_chars)

    def score_sentence(sentence: str, words_in_question: set[str]) -> float:
        words = extract_keywords(sentence)
        if not words:
            return 0.0
        overlap_words = words & words_in_question
        if not overlap_words:
            return 0.0
        density = len(overlap_words) / max(1, len(words))
        length_bonus = 0.2 if 8 <= len(sentence.split()) <= 35 else 0.0
        return len(overlap_words) + density + length_bonus

    scored: list[tuple[int, float, str]] = []
    for index, sentence in enumerate(sentences):
        score = score_sentence(sentence, question_words)
        if score > 0:
            scored.append((index, score, sentence))

    if not scored:
        return _truncate(text, max_chars)

    scored.sort(key=lambda item: (-item[1], item[0]))
    selected = sorted(scored[:12], key=lambda item: item[0])
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


def retrieve_context(text: str, question: str, max_chars: int = 3500) -> str:
    """Builds anchored context: intro + relevant middle + outro."""
    text = (text or "").strip()
    if not text:
        return ""

    if len(text) <= max_chars:
        return text

    question_keywords = extract_keywords(question)
    if not question_keywords:
        return _truncate(text, max_chars)

    print(f"[DEBUG] Question keywords for retrieval: {sorted(question_keywords)}")

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 10]
    if len(sentences) < 12:
        try:
            return _embedding_retrieval(text, question, max_chars)
        except Exception:
            return _keyword_fallback(text, question, max_chars)

    total = len(sentences)
    anchor_count = max(3, total // 12)
    intro_sents = sentences[:anchor_count]
    outro_sents = sentences[-anchor_count:]
    middle_sents = sentences[anchor_count:-anchor_count]

    def score(sentence: str) -> float:
        words = extract_keywords(sentence)
        overlap_words = words & question_keywords
        if not overlap_words:
            return 0.0
        density = len(overlap_words) / max(1, len(words))
        length_bonus = 0.3 if 8 <= len(sentence.split()) <= 30 else 0.0
        return len(overlap_words) + density + length_bonus

    scored_middle = sorted(enumerate(middle_sents), key=lambda item: score(item[1]), reverse=True)
    top_middle_scored = [(idx, sent) for idx, sent in scored_middle if score(sent) > 0]

    if not top_middle_scored:
        return _truncate(" ".join(intro_sents + outro_sents), max_chars)

    top_middle = sorted(top_middle_scored[:18], key=lambda item: item[0])
    combined = intro_sents + [sentence for _, sentence in top_middle] + outro_sents
    result = " ".join(combined)
    print(f"[DEBUG] Retrieved {len(top_middle)} relevant middle sentences for context")
    return _truncate(result, max_chars)