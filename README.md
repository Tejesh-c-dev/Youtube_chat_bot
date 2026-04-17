# YouTube AI Assistant

## Overview
YouTube AI Assistant is a Chrome extension paired with a local FastAPI backend that lets you ask questions about a YouTube video using its transcript. The extension reads the active YouTube tab, sends the video ID and your question to the backend, and displays an AI answer in a compact popup chat UI.

## Features
- Chrome extension popup for asking questions about the current YouTube video.
- Detects video IDs from `youtube.com/watch`, `youtu.be`, and `youtube.com/shorts` URLs.
- Validates that the user is on a YouTube video before sending a request.
- Local FastAPI backend with `/`, `/health`, and `/ask` endpoints.
- Transcript retrieval through `youtube-transcript-api` with a `yt-dlp` fallback.
- In-memory transcript and metadata caching per session.
- Context retrieval using embeddings when available.
- Keyword-based fallback retrieval when embeddings or API access are unavailable.
- OpenRouter-powered answer generation with model fallback.
- Extension-to-backend communication through `fetch`.
- CORS enabled for the Chrome extension.

## Project Structure
```text
yt-ai-assistant/
├── extension/
│   ├── manifest.json
│   ├── popup.html
│   ├── popup.js
│   └── icons/
├── backend/
│   ├── __init__.py
│   ├── main.py
│   ├── transcript.py
│   ├── retriever.py
│   ├── requirements.txt
│   └── .env.example
├── app.py
├── manifest.json
├── popup.html
├── popup.js
├── requirements.txt
└── README.md
```

## Setup — Backend
```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
# add your OpenRouter key to .env
uvicorn main:app --reload --port 8000
```

## Setup — Chrome Extension
1. Go to `chrome://extensions`.
2. Enable Developer Mode.
3. Click `Load unpacked` and select the `extension/` folder.

## Usage
1. Open a YouTube video with captions enabled.
2. Open the extension popup and type a question.
3. Read the answer returned by the backend.

## How It Works
The extension reads the active YouTube tab and sends the video ID plus question to the FastAPI backend. The backend fetches the transcript, retrieves the most relevant context, and sends that context to OpenRouter for an answer. If OpenRouter is unavailable, the backend falls back to keyword-matched transcript extraction.

## Notes
- Works without an OpenRouter key by using keyword fallback.
- Transcripts and metadata are cached in memory for the current session.
- Only works on videos with captions or auto-captions available.