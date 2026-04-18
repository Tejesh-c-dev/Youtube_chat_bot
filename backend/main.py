from __future__ import annotations

import os
import re
import traceback
from pathlib import Path
from typing import Literal

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Load backend/.env first, then project .env to avoid silent key-missing scenarios.
_BACKEND_DOTENV = Path(__file__).with_name(".env")
_PROJECT_DOTENV = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_BACKEND_DOTENV, override=False)
load_dotenv(dotenv_path=_PROJECT_DOTENV, override=False)
load_dotenv(override=False)

try:
    from .transcript import get_transcript
    from .retriever import extract_keywords, retrieve_context, retrieve_top_sentences
except ImportError:  # pragma: no cover - fallback for direct execution
    from transcript import get_transcript
    from retriever import extract_keywords, retrieve_context, retrieve_top_sentences


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL", "http://localhost:3000")
OPENROUTER_APP_NAME = os.getenv("OPENROUTER_APP_NAME", "YouTube AI Assistant")

transcript_cache: dict[str, dict[str, object]] = {}
metadata_cache: dict[str, dict[str, object]] = {}

FALLBACK_TIERS = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "arcee-ai/trinity-large-preview:free",
    "liquid/lfm-2.5-1.2b-instruct:free",
    "liquid/lfm-2.5-1.2b-thinking:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "google/gemma-3-12b-it:free",
    "openai/gpt-oss-20b:free",
]


INTENT_INSTRUCTIONS = {
    "summarize": "Give a concise, well-structured summary. Cover the main topic, key points, and conclusion. 3-5 sentences.",
    "explain": "Explain clearly using transcript context plus your broader knowledge of the topic. Make it easy to understand.",
    "list": "Format as a clean numbered list. Extract all relevant items from the transcript. Be complete.",
    "timestamp": "Identify which part of the video covers this. Reference early, middle, or late sections if exact timestamps are not available.",
    "opinion": "Give a balanced, reasoned response. Use what the video argues plus general knowledge.",
    "compare": "Structure the comparison clearly. Use what the video says and supplement with general knowledge if needed.",
    "factual": "Answer directly and precisely. If the exact answer is not stated, infer from transcript context.",
}


def detect_intent(question: str) -> str:
    """Classify the question intent to guide answer generation."""
    q = question.lower().strip()

    if any(w in q for w in ["summarize", "summary", "what is this", "what is the video", "about", "overview", "tldr", "tl;dr"]):
        return "summarize"

    if any(w in q for w in ["explain", "what does", "what is", "define", "meaning of", "how does", "why does"]):
        return "explain"

    if any(w in q for w in ["list", "what are", "give me", "mention", "name all", "steps", "points", "tips"]):
        return "list"

    if any(w in q for w in ["timestamp", "timestamps", "time stamp", "time stamps", "time strap", "time straps", "when does", "at what point", "which part", "where in"]):
        return "timestamp"

    if any(w in q for w in ["opinion", "think", "should i", "worth", "recommend", "good", "bad", "better"]):
        return "opinion"

    if any(w in q for w in ["compare", "difference", "vs", "versus", "contrast", "similar"]):
        return "compare"

    return "factual"


def _is_question_unrelated(question: str, context: str, video_title: str, video_description: str) -> bool:
    corpus = _normalize_text(f"{video_title} {video_description} {context[:1200]}").lower()
    generic_words = {
        "what", "about", "video", "this", "that", "with", "from", "explain",
        "tell", "more", "please", "would", "could", "should", "there", "their",
    }
    question_words = {
        word
        for word in re.findall(r"[a-zA-Z0-9']+", question.lower())
        if len(word) > 3 and word not in generic_words
    }
    if not question_words:
        return False
    overlap = len([word for word in question_words if word in corpus])
    return overlap == 0 and len(question_words) >= 2


def _topic_hint(video_title: str, video_description: str) -> str:
    title = (video_title or "").strip()
    if title:
        return title
    desc = _normalize_text(video_description)
    return desc[:80] if desc else "the video's main topic"


def _fallback_unavailable_transcript() -> dict[str, str]:
    return {
        "answer": (
            "I couldn't load the transcript for this video. It may not have captions enabled, "
            "or the video may be private. Try a different video."
        ),
        "source": "transcript_unavailable",
    }


def _build_timestamp_answer(question: str, segments: list[dict[str, str]] | None) -> str:
    """Return transcript-backed timestamp pointers even when the LLM underperforms."""
    if not segments:
        return "I could not find timestamp markers for this video transcript."

    keywords = extract_keywords(question)
    indexed = list(enumerate(segments))

    if keywords:
        ranked: list[tuple[int, int, dict[str, str]]] = []
        for index, segment in indexed:
            overlap = len(extract_keywords(segment.get("text", "")) & keywords)
            if overlap > 0:
                ranked.append((index, overlap, segment))
        if ranked:
            ranked.sort(key=lambda item: (-item[1], item[0]))
            selected = sorted(ranked[:6], key=lambda item: item[0])
            lines = [
                f"{i + 1}. {seg.get('time', '00:00')} - {seg.get('text', '').strip()[:170]}"
                for i, (_, _, seg) in enumerate(selected)
            ]
            return "Here are the most relevant timestamped moments:\n" + "\n".join(lines)

    # No keyword match: provide evenly distributed anchors across the video.
    count = min(6, len(segments))
    step = max(1, len(segments) // count)
    sampled = [segments[i] for i in range(0, len(segments), step)][:count]
    lines = [
        f"{i + 1}. {seg.get('time', '00:00')} - {seg.get('text', '').strip()[:170]}"
        for i, seg in enumerate(sampled)
    ]
    return "Here are useful timestamped points from across the video:\n" + "\n".join(lines)


def _get_openrouter_api_key() -> str:
    """Read API key at call time so runtime env changes are picked up."""
    return os.getenv("OPENROUTER_API_KEY", "").strip()


class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2000)


class AskRequest(BaseModel):
    video_id: str = Field(min_length=1)
    question: str = Field(min_length=1, max_length=500)
    conversation_history: list[ConversationTurn] = Field(default_factory=list)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _keyword_fallback_answer(context: str, question: str, video_title: str) -> str:
    text = _normalize_text(context)
    if not text:
        return "I couldn't find that specific information in this video."

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
            return "I couldn't find that specific information in this video."
        if video_title:
            return f"Based on {video_title}, the transcript mentions: {snippet}"
        return f"Based on the transcript, I found: {snippet}"

    scored.sort(key=lambda item: (-item[0], item[1]))
    selected = sorted(scored[:5], key=lambda item: item[1])
    snippet = " ".join(sentence for _, _, sentence in selected).strip()
    if not snippet:
        return "I couldn't find that specific information in this video."
    return snippet[:1200]


async def _get_cached_transcript(video_id: str) -> dict[str, str]:
    if video_id in transcript_cache:
        return transcript_cache[video_id]

    transcript_data = await get_transcript(video_id)
    transcript_cache[video_id] = transcript_data
    metadata_cache[video_id] = {
        "title": str(transcript_data.get("title", "")).strip(),
        "description": str(transcript_data.get("description", "")).strip(),
        "duration_hint": str(transcript_data.get("duration_hint", "medium")).strip(),
        "segments": transcript_data.get("segments", []),
    }
    return transcript_data


async def generate_answer(
    context: str,
    question: str,
    video_title: str,
    transcript_text: str,
    video_description: str = "",
    duration_hint: str = "medium",
    transcript_segments: list[dict[str, str]] | None = None,
    conversation_history: list[ConversationTurn] | None = None,
    intent: str = "factual",
) -> tuple[str, str]:
    fallback_answer = _keyword_fallback_answer(context, question, video_title)
    api_key = _get_openrouter_api_key()
    debug_key_msg = f"YES (len={len(api_key)})" if api_key else "NO - KEY IS EMPTY"
    print(f"[DEBUG] API key loaded: {debug_key_msg}")

    guarded_intents = {"factual", "opinion", "compare"}
    if intent in guarded_intents and _is_question_unrelated(question, context, video_title, video_description):
        return f"This doesn't seem related to the video. The video covers {_topic_hint(video_title, video_description)}.", "unrelated_guard"

    if intent == "timestamp" and transcript_segments:
        return _build_timestamp_answer(question, transcript_segments), "timestamp_segments"

    if not api_key:
        print("[DEBUG] No API key found. Check backend/.env contains OPENROUTER_API_KEY=...")
        best_sentences = retrieve_top_sentences(transcript_text, question, n=3)
        if best_sentences:
            return (
                f"{best_sentences}\n\n"
                "(Note: AI model is currently unavailable. This answer is pulled directly from the video transcript.)"
            ), "keyword_fallback"
        return fallback_answer, "fallback"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": OPENROUTER_SITE_URL,
        "X-Title": OPENROUTER_APP_NAME,
    }

    system_prompt = f"""You are an expert AI analyst who has fully studied a YouTube video transcript.
You have deep knowledge of the video's topic and can reason intelligently about it.

This is a {duration_hint} video. Adjust your answer depth accordingly.

YOUR CAPABILITIES:
1. DIRECT ANSWERS - Answer questions explicitly covered in the transcript.
2. INFERENCE - If the answer is not stated but can be reasonably inferred from context, infer it and say so.
3. EXPLANATION - Explain concepts mentioned in the transcript using your broader knowledge to make them clearer.
4. ELABORATION - Expand on points the video touches on with relevant background knowledge.
5. ANALYSIS - Identify patterns, themes, implications, or conclusions the video leads to.

RESPONSE RULES:
- If the answer is IN the transcript, answer directly and cite the relevant point.
- If the answer can be INFERRED from the transcript, answer and say "Based on what the video covers..."
- If the answer RELATES to the transcript topic but needs outside knowledge, answer and say "To add context beyond the video..."
- If the question is COMPLETELY UNRELATED to the video topic, say "This doesn't seem related to the video. The video covers [topic]."
- Never say "I don't know" if you can reason about it from the transcript context.
- Never fabricate specific facts, statistics, or quotes not present in the transcript.
- Keep answers focused: 2-4 sentences for simple questions, up to 8 for complex ones.
- Use plain, fluent language. No bullet points unless the question asks for a list.
- Never start with "In this video" or repeat the question.
"""

    intent_instruction = INTENT_INSTRUCTIONS.get(intent, INTENT_INSTRUCTIONS["factual"])
    metadata_block = f"VIDEO TITLE: {video_title or 'Unknown'}\n"
    if video_description and len(video_description) > 20:
        metadata_block += f"VIDEO DESCRIPTION: {video_description[:400].strip()}\n"

    full_context = f"{metadata_block}\nTRANSCRIPT EXCERPT:\n{context}"
    segment_preview = ""
    if transcript_segments:
        preview_lines = [
            f"- {segment.get('time', '00:00')}: {segment.get('text', '')[:110]}"
            for segment in transcript_segments[:10]
        ]
        segment_preview = "\nTIMESTAMPED SEGMENTS:\n" + "\n".join(preview_lines)

    user_message = (
        f"VIDEO TITLE: {video_title or 'Unknown'}\n\n"
        "TRANSCRIPT CONTEXT:\n"
        '"""\n'
        f"{full_context}{segment_preview}\n"
        '"""\n\n'
        f"QUESTION TYPE: {intent}\n"
        f"INSTRUCTION: {intent_instruction}\n\n"
        f"QUESTION: {question}\n\n"
        "Answer now:"
    )

    base_messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for turn in (conversation_history or [])[-8:]:
        base_messages.append({"role": turn.role, "content": turn.content})
    base_messages.append({"role": "user", "content": user_message})

    for model in FALLBACK_TIERS:
        print(f"[DEBUG] Trying model: {model}")
        payload = {
            "model": model,
            "messages": base_messages,
            "temperature": 0.4,
            "max_tokens": 400,
            "top_p": 0.9,
        }

        try:
            async with httpx.AsyncClient(timeout=25.0) as client:
                response = await client.post(OPENROUTER_URL, headers=headers, json=payload)

            print(f"[DEBUG] Response status ({model}): {response.status_code}")
            if response.status_code != 200:
                print(f"[DEBUG] Error body ({model}): {response.text[:800]}")
                continue

            data = response.json()
            if "error" in data:
                print(f"[DEBUG] API error in body ({model}): {data.get('error')}")
                continue
            content = data["choices"][0]["message"]["content"].strip()
            if content:
                print(f"[DEBUG] Got answer ({len(content)} chars) from {model}")
                return content, "llm"
            print(f"[DEBUG] Empty answer content from {model}")
        except httpx.TimeoutException:
            print(f"[DEBUG] Timeout on model: {model}")
        except Exception as exc:
            print(f"[DEBUG] Exception on {model}: {type(exc).__name__}: {exc}")
            traceback.print_exc()

    print("[DEBUG] All models failed. Falling back to transcript matching.")
    best_sentences = retrieve_top_sentences(transcript_text, question, n=3)
    if best_sentences:
        return (
            f"{best_sentences}\n\n"
            "(Note: AI model is currently unavailable. This answer is pulled directly from the video transcript.)"
        ), "keyword_fallback"
    return "I'm having trouble generating an answer right now. Please check that your backend is running and try again.", "error_fallback"


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

    try:
        transcript_data = await _get_cached_transcript(video_id)
    except HTTPException:
        fallback = _fallback_unavailable_transcript()
        return {"answer": fallback["answer"], "video_id": video_id, "source": fallback["source"]}

    transcript_text = str(transcript_data.get("text", ""))
    video_title = str(transcript_data.get("title", "") or metadata_cache.get(video_id, {}).get("title", ""))
    video_description = str(transcript_data.get("description", "") or metadata_cache.get(video_id, {}).get("description", ""))
    duration_hint = str(transcript_data.get("duration_hint", "") or metadata_cache.get(video_id, {}).get("duration_hint", "medium"))
    transcript_segments = transcript_data.get("segments") or metadata_cache.get(video_id, {}).get("segments") or []

    context = retrieve_context(transcript_text, question, max_chars=3500)
    if not context.strip():
        context = transcript_text[:3000]

    intent = detect_intent(question)
    answer, source = await generate_answer(
        context=context,
        question=question,
        video_title=video_title,
        transcript_text=transcript_text,
        video_description=video_description,
        duration_hint=duration_hint,
        transcript_segments=transcript_segments,
        conversation_history=request.conversation_history,
        intent=intent,
    )
    return {"answer": answer, "video_id": video_id, "source": source}