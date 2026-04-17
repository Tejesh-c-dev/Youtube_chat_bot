import sys

sys.dont_write_bytecode = True

import os
import re
import urllib.request
from xml.etree import ElementTree
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from yt_dlp import YoutubeDL

load_dotenv()

app = FastAPI()

# Enable CORS for Chrome extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenRouter API configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Model priority list (free models first, then fallbacks)
FREE_MODELS = [
    "openai/gpt-4o-mini",           # Free tier on OpenRouter
    "google/gemma-2-9b-it:free",    # Free Gemma model
    "meta-llama/llama-3-8b-instruct:free",  # Free Llama 3
    "mistralai/mistral-7b-instruct:free",   # Free Mistral
    "google/gemma-7b-it:free",      # Alternative Gemma
]

TRANSCRIPT_LANGUAGES = ["en", "hi", "auto"]
TRANSCRIPT_CACHE = {}
VIDEO_METADATA_CACHE = {}

class Query(BaseModel):
    video_id: str
    question: str


def _parse_timedtext_xml(xml_text: str) -> str:
    root = ElementTree.fromstring(xml_text)
    parts = []
    for node in root.findall("text"):
        if node.text:
            parts.append(node.text)
    return " ".join(parts).replace("\n", " ").strip()


def _extract_caption_url(info: dict) -> str | None:
    tracks = []

    automatic = info.get("automatic_captions") or {}
    subtitles = info.get("subtitles") or {}

    for lang_key in ("en", "en-US", "en-GB"):
        if lang_key in automatic:
            tracks.extend(automatic[lang_key])
    if not tracks:
        for value in automatic.values():
            tracks.extend(value)

    for lang_key in ("en", "en-US", "en-GB"):
        if lang_key in subtitles:
            tracks.extend(subtitles[lang_key])
    if not tracks:
        for value in subtitles.values():
            tracks.extend(value)

    preferred_ext_order = {"srv3": 0, "srv2": 1, "srv1": 2, "ttml": 3, "vtt": 4, "json3": 5}
    tracks.sort(key=lambda t: preferred_ext_order.get((t.get("ext") or "").lower(), 99))

    for track in tracks:
        if track.get("url"):
            return track["url"]

    return None


def get_transcript_text_via_ytdlp(video_id: str) -> str:
    """Fallback transcript retrieval path using yt-dlp metadata extraction."""
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=False)

    caption_url = _extract_caption_url(info)
    if not caption_url:
        raise RuntimeError("No captions found via yt-dlp")

    with urllib.request.urlopen(caption_url, timeout=20) as response:
        payload = response.read().decode("utf-8", errors="ignore")

    transcript = _parse_timedtext_xml(payload)
    if not transcript:
        raise RuntimeError("Caption payload was empty")

    return transcript


def get_transcript_text(video_id: str) -> str:
    """Fetch transcript text while supporting multiple youtube-transcript-api versions."""
    first_error = None

    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(
            video_id,
            languages=TRANSCRIPT_LANGUAGES,
        )
        transcript = " ".join(item.get("text", "") for item in transcript_list).strip()
        if transcript:
            return transcript
        raise RuntimeError("Primary transcript API returned empty transcript")
    except Exception as err:
        first_error = err

        try:
            api = YouTubeTranscriptApi()
            fetched = api.fetch(video_id, languages=TRANSCRIPT_LANGUAGES)
            if hasattr(fetched, "to_raw_data"):
                raw = fetched.to_raw_data()
                transcript = " ".join(item.get("text", "") for item in raw).strip()
            else:
                transcript = " ".join(getattr(item, "text", "") for item in fetched).strip()
            if transcript:
                return transcript
            raise RuntimeError("Secondary transcript API returned empty transcript")
        except Exception as err:
            second_error = err

    try:
        return get_transcript_text_via_ytdlp(video_id)
    except Exception as fallback_error:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Could not fetch transcript for video '{video_id}'. "
                f"Primary error: {first_error}. Secondary error: {second_error}. "
                f"yt-dlp fallback error: {fallback_error}. "
                "This may be a temporary YouTube/IP block; try again later or from a different network."
            ),
        ) from fallback_error


def get_cached_transcript(video_id: str) -> str:
    """Return cached transcript when available to minimize repeated upstream requests."""
    if video_id in TRANSCRIPT_CACHE:
        return TRANSCRIPT_CACHE[video_id]

    transcript = get_transcript_text(video_id)
    TRANSCRIPT_CACHE[video_id] = transcript
    return transcript


def get_video_metadata(video_id: str) -> tuple[str, str]:
    """Fetch and cache video title/description to improve question-answering context."""
    if video_id in VIDEO_METADATA_CACHE:
        return VIDEO_METADATA_CACHE[video_id]

    video_url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
        title = (info.get("title") or "").strip()
        description = (info.get("description") or "").strip()
    except Exception:
        title = ""
        description = ""

    VIDEO_METADATA_CACHE[video_id] = (title, description)
    return VIDEO_METADATA_CACHE[video_id]


def build_source_text(video_id: str, transcript: str) -> str:
    """Combine metadata and transcript into a single source text for retrieval."""
    title, description = get_video_metadata(video_id)

    parts = []
    if title:
        parts.append(f"Video Title: {title}")
    if description:
        parts.append(f"Video Description:\n{description}")
    parts.append(f"Video Transcript:\n{transcript}")

    return "\n\n".join(parts)


def extract_keyword_context(transcript: str, question: str, max_sentences: int = 12) -> str:
    """Fallback retrieval when vector search is unavailable."""
    question_tokens = {
        token.lower()
        for token in re.findall(r"[a-zA-Z0-9']+", question)
        if len(token) > 2
    }

    if not question_tokens:
        return transcript[:4500]

    sentences = re.split(r"(?<=[.!?])\s+", transcript)
    scored = []
    for sentence in sentences:
        sentence_tokens = set(re.findall(r"[a-zA-Z0-9']+", sentence.lower()))
        score = len(question_tokens & sentence_tokens)
        if score > 0:
            scored.append((score, sentence.strip()))

    scored.sort(key=lambda x: x[0], reverse=True)
    selected = [text for _, text in scored[:max_sentences]]
    if not selected:
        return transcript[:4500]
    return "\n".join(selected)[:4500]


def build_context(transcript: str, question: str) -> str:
    """Try embedding-based retrieval first, then use keyword fallback."""
    if not OPENROUTER_API_KEY:
        return extract_keyword_context(transcript, question)

    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = splitter.create_documents([transcript])

        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=OPENROUTER_API_KEY,
            openai_api_base=OPENROUTER_BASE_URL,
        )
        vector_store = FAISS.from_documents(chunks, embeddings)
        docs = vector_store.as_retriever(search_kwargs={"k": 4}).invoke(question)
        context = "\n\n".join(doc.page_content for doc in docs)
        return context if context.strip() else extract_keyword_context(transcript, question)
    except Exception:
        return extract_keyword_context(transcript, question)

def get_llm_with_fallback():
    """Try to get an LLM with the first available model from the priority list."""
    if not OPENROUTER_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="OPENROUTER_API_KEY not set. Please set the environment variable."
        )

    last_error = None
    for model in FREE_MODELS:
        try:
            llm = ChatOpenAI(
                model=model,
                openai_api_key=OPENROUTER_API_KEY,
                openai_api_base=OPENROUTER_BASE_URL,
                temperature=0.7,
            )
            # Test the model with a simple call
            test_prompt = PromptTemplate.from_template("Say 'OK'")
            test_input = test_prompt.invoke({})
            llm.invoke(test_input)
            print(f"Successfully using model: {model}")
            return llm, model
        except Exception as e:
            last_error = e
            print(f"Model {model} failed: {str(e)}")
            continue

    raise HTTPException(
        status_code=503,
        detail=f"All models failed. Last error: {str(last_error)}"
    )

@app.post("/ask")
def ask_video(query: Query):
    try:
        transcript = get_cached_transcript(query.video_id)
        title, description = get_video_metadata(query.video_id)
        source_text = build_source_text(query.video_id, transcript)
        retrieved_context = build_context(source_text, query.question)

        prompt_context_parts = []
        if title:
            prompt_context_parts.append(f"Video Title: {title}")
        if description:
            prompt_context_parts.append(f"Video Description:\n{description[:2000]}")
        prompt_context_parts.append(f"Relevant Extracted Context:\n{retrieved_context}")
        context = "\n\n".join(prompt_context_parts)

        # 5. Prompt
        prompt = PromptTemplate.from_template("""
        Answer ONLY using the context.

        If the answer is not present, say:
        "Not mentioned in the video transcript."

        Context:
        {context}

        Question: {question}
        """)

        final_prompt = prompt.invoke({
            "context": context,
            "question": query.question
        })

        # 6. LLM with fallback
        llm, used_model = get_llm_with_fallback()
        response = llm.invoke(final_prompt)

        print(f"Used model: {used_model}")
        return {"answer": response.content, "model_used": used_model}

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok", "models_available": len(FREE_MODELS)}


@app.get("/")
def root():
    return {"message": "YouTube AI backend is running", "health": "/health"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
