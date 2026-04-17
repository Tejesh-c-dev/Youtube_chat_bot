from __future__ import annotations

import asyncio
import re
from xml.etree import ElementTree

import httpx
from fastapi import HTTPException
from youtube_transcript_api import YouTubeTranscriptApi
import yt_dlp


LANGUAGES = ["en", "en-US", "hi", "a.en"]


def _extract_metadata(info: dict, video_id: str) -> tuple[str, str]:
    title = (info.get("title") or f"Video {video_id}").strip()
    description = (info.get("description") or "").strip()
    return title, description


def _pick_caption_url(info: dict) -> str | None:
    candidates = []

    for source_key in ("subtitles", "automatic_captions"):
        source = info.get(source_key) or {}
        for language in ("en", "en-US", "en-GB", "hi"):
            candidates.extend(source.get(language) or [])
        for track_list in source.values():
            candidates.extend(track_list or [])

    preferred_order = {"srv3": 0, "srv2": 1, "srv1": 2, "ttml": 3, "vtt": 4, "json3": 5}
    candidates.sort(key=lambda item: preferred_order.get((item.get("ext") or "").lower(), 99))

    for candidate in candidates:
        url = candidate.get("url")
        if url:
            return url
    return None


def _parse_xml_captions(payload: str) -> str:
    root = ElementTree.fromstring(payload)
    parts = []
    for node in root.findall("text"):
        text = "".join(node.itertext()).strip()
        if text:
            parts.append(text)
    return " ".join(parts).strip()


def _parse_text_captions(payload: str) -> str:
    cleaned_lines = []
    for line in payload.splitlines():
        stripped = line.strip()
        if not stripped or stripped.lower().startswith("webvtt"):
            continue
        if re.match(r"^\d+$", stripped):
            continue
        if re.search(r"\d{1,2}:\d{2}:\d{2}[\.,]\d{3}\s+-->\s+\d{1,2}:\d{2}:\d{2}[\.,]\d{3}", stripped):
            continue
        if re.search(r"\d{1,2}:\d{2}\s+-->\s+\d{1,2}:\d{2}", stripped):
            continue
        stripped = re.sub(r"</?c[^>]*>", "", stripped)
        stripped = re.sub(r"<[^>]+>", "", stripped)
        stripped = re.sub(r"\s+", " ", stripped).strip()
        if stripped:
            cleaned_lines.append(stripped)
    return " ".join(cleaned_lines).strip()


def _parse_caption_payload(payload: str) -> str:
    payload = payload.strip()
    if not payload:
        return ""
    if payload.startswith("<") and "<text" in payload:
        try:
            return _parse_xml_captions(payload)
        except Exception:
            pass
    return _parse_text_captions(payload)


def _extract_info(video_url: str) -> dict:
    ydl_opts = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en", "hi"],
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        return ydl.extract_info(video_url, download=False)


async def _fetch_caption_text(caption_url: str) -> str:
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(caption_url)
        response.raise_for_status()
        return response.text


async def _get_transcript_from_youtube_transcript_api(video_id: str) -> str:
    transcript_list = await asyncio.to_thread(
        YouTubeTranscriptApi.get_transcript,
        video_id,
        LANGUAGES,
    )
    return " ".join(item.get("text", "") for item in transcript_list).strip()


async def get_transcript(video_id: str) -> dict:
    """Returns { 'text': str, 'title': str, 'description': str }"""
    video_url = f"https://www.youtube.com/watch?v={video_id}"

    try:
        text = await _get_transcript_from_youtube_transcript_api(video_id)
        if text:
            print(f"Transcript fetched with youtube-transcript-api for {video_id}")
            try:
                info = await asyncio.to_thread(_extract_info, video_url)
                title, description = _extract_metadata(info, video_id)
            except Exception as exc:
                print(f"Metadata fetch failed after transcript API success for {video_id}: {exc}")
                title, description = f"Video {video_id}", ""
            return {"text": text, "title": title, "description": description}
    except Exception as exc:
        print(f"youtube-transcript-api failed for {video_id}: {exc}")

    try:
        info = await asyncio.to_thread(_extract_info, video_url)
        title, description = _extract_metadata(info, video_id)
        caption_url = _pick_caption_url(info)
        if not caption_url:
            raise RuntimeError("No caption URL found")

        payload = await _fetch_caption_text(caption_url)
        text = _parse_caption_payload(payload)
        if text:
            print(f"Transcript fetched with yt-dlp fallback for {video_id}")
            return {"text": text, "title": title, "description": description}

        raise RuntimeError("Caption payload was empty")
    except Exception as exc:
        print(f"yt-dlp fallback failed for {video_id}: {exc}")
        raise HTTPException(status_code=422, detail="Transcript unavailable for this video") from exc