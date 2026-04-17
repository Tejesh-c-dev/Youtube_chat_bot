from __future__ import annotations

import os
import re

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()

try:
    from .transcript import get_transcript
    from .retriever import retrieve_context
except ImportError:  # pragma: no cover - fallback for direct execution
    from transcript import get_transcript
    from retriever import retrieve_context


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL", "http://localhost:8000")
OPENROUTER_APP_NAME = os.getenv("OPENROUTER_APP_NAME", "YouTube AI Assistant")

transcript_cache: dict[str, dict[str, str]] = {}
metadata_cache: dict[str, dict[str, str]] = {}

MODELS = [
    "meta-llama/llama-3.1-8b-instruct:free",
    "mistralai/mistral-7b-instruct:free",
    "google/gemma-2-9b-it:free",
]


class AskRequest(BaseModel):
    video_id: str = Field(min_length=1)
    question: str = Field(min_length=1, max_length=500)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _keyword_fallback_answer(context: str, question: str, video_title: str) -> str:
    text = _normalize_text(context)
    if not text:
        return "I couldn't find that information in this video."

    question_words = {
        word
        for word in re.findall(r"[a-zA-Z0-9']+", question.lower())
        if len(word) > 2
    }

    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    scored: list[tuple[int, int, str]] = []

    for index, sentence in enumerate(sentences):
        tokens = set(re.findall(r"[a-zA-Z0-9']+", sentence.lower()))
        score = len(tokens & question_words)
        if score > 0:
            scored.append((score, index, sentence.strip()))

    if not scored:
        snippet = text[:800].strip()
        if not snippet:
            return "I couldn't find that information in this video."
        if video_title:
            return f"Based on {video_title}, the transcript mentions: {snippet}"
        return f"Based on the transcript, I found: {snippet}"

    scored.sort(key=lambda item: (-item[0], item[1]))
    selected = sorted(scored[:5], key=lambda item: item[1])
    snippet = " ".join(sentence for _, _, sentence in selected).strip()
    if not snippet:
        return "I couldn't find that information in this video."
    return snippet[:1200]


async def _get_cached_transcript(video_id: str) -> dict[str, str]:
    if video_id in transcript_cache:
        return transcript_cache[video_id]

    transcript_data = await get_transcript(video_id)
    transcript_cache[video_id] = transcript_data
    metadata_cache[video_id] = {
        "title": transcript_data.get("title", "").strip(),
        "description": transcript_data.get("description", "").strip(),
    }
    return transcript_data


async def generate_answer(context: str, question: str, video_title: str) -> tuple[str, str]:
    fallback_answer = _keyword_fallback_answer(context, question, video_title)

    if not OPENROUTER_API_KEY:
        return fallback_answer, "fallback"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": OPENROUTER_SITE_URL,
        "X-Title": OPENROUTER_APP_NAME,
    }

    system_prompt = (
        "You are a helpful assistant that answers questions about YouTube videos. "
        "Answer ONLY based on the provided transcript context. "
        "If the answer is not in the context, say exactly: \"I couldn't find that information in this video.\" "
        "Be concise, clear, and direct. Do not add information not present in the transcript."
    )

    for model in MODELS:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Video title: {video_title or 'Unknown'}\n\n"
                        f"Transcript context:\n{context}\n\n"
                        f"Question: {question}"
                    ),
                },
            ],
            "temperature": 0.2,
        }

        try:
            async with httpx.AsyncClient(timeout=25.0) as client:
                response = await client.post(OPENROUTER_URL, headers=headers, json=payload)

            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"].strip()
            if content:
                return content, "llm"
        except Exception as exc:
            print(f"OpenRouter model failed ({model}): {exc}")

    return fallback_answer, "fallback"


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok", "service": "YouTube AI Assistant"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy", "version": "1.0.0"}


@app.post("/ask")
async def ask(request: AskRequest) -> dict[str, str]:
    video_id = request.video_id.strip()
    question = request.question.strip()

    if not video_id or not question:
        raise HTTPException(status_code=422, detail="video_id and question are required")

    transcript_data = await _get_cached_transcript(video_id)
    transcript_text = transcript_data.get("text", "")
    video_title = transcript_data.get("title", "") or metadata_cache.get(video_id, {}).get("title", "")

    context = retrieve_context(transcript_text, question)
    if not context.strip():
        context = transcript_text[:3000]

    answer, source = await generate_answer(context, question, video_title)
    return {"answer": answer, "video_id": video_id, "source": source}