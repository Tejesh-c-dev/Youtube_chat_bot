const state = {
    videoId: null,
    videoLabel: "No video detected",
    messages: [],
    isLoading: false,
    backendOnline: false,
    backendBaseUrl: "http://localhost:8000",
};

const BACKEND_CANDIDATES = ["http://localhost:8000", "http://localhost:8001"];

const questionInput = document.getElementById("question-input");
const sendBtn = document.getElementById("send-btn");
const clearBtn = document.getElementById("clear-btn");
const statusDot = document.getElementById("status-dot");
const statusLabel = document.getElementById("status-label");
const subtitleEl = document.getElementById("subtitle");
const bannerEl = document.getElementById("banner");
const chipsEl = document.getElementById("chips");
const messagesContainer = document.getElementById("messages-container");
const emptyStateEl = document.getElementById("empty-state");
const charCounterEl = document.getElementById("char-counter");

const suggestionChips = Array.from(chipsEl.querySelectorAll("button[data-chip]"));

function decodeHTML(text) {
    const txt = document.createElement("textarea");
    txt.innerHTML = String(text ?? "");
    return txt.value;
}

function escapeHTML(text) {
    return String(text ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
}

function processAIResponse(text) {
    const decoded = decodeHTML(text);
    const escaped = escapeHTML(decoded);
    let processed = escaped.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
    processed = processed.replace(/\*(.*?)\*/g, "<em>$1</em>");
    processed = processed.replace(/(^|<br>)(\d+\.\s)/g, "$1<span class=\"list-num\">$2</span>");
    processed = processed.replace(/\n/g, "<br>");
    processed = processed.replace(/^(<br>)+/, "");
    return processed;
}

function buildConversationHistory(messages) {
    const recent = messages.slice(-8);
    return recent
        .filter((message) => message.role === "user" || message.role === "ai")
        .map((message) => ({
            role: message.role === "user" ? "user" : "assistant",
            content: message.text,
        }));
}

function getVideoId(urlString) {
    try {
        const parsed = new URL(urlString);
        if (parsed.hostname.includes("youtu.be")) {
            return parsed.pathname.slice(1) || null;
        }
        if (parsed.pathname.startsWith("/shorts/")) {
            return parsed.pathname.split("/")[2] || null;
        }
        return parsed.searchParams.get("v");
    } catch {
        return null;
    }
}

function formatTime(date = new Date()) {
    return date.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
}

function scrollToBottom() {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function setBanner(message, type = "notice") {
    bannerEl.textContent = message;
    bannerEl.className = `banner show ${type}`;
}

function clearBanner() {
    bannerEl.textContent = "";
    bannerEl.className = "banner";
}

function updateStatus(online, label) {
    state.backendOnline = online;
    statusDot.classList.remove("online", "offline");
    statusDot.classList.add(online ? "online" : "offline");
    statusLabel.textContent = label;
}

function updateSubtitle() {
    subtitleEl.textContent = state.videoId ? `Video ID: ${state.videoId}` : state.videoLabel;
}

function updateComposerState() {
    const hasText = questionInput.value.trim().length > 0;
    sendBtn.disabled = state.isLoading || !state.videoId || !hasText;
    clearBtn.disabled = state.messages.length === 0 && !questionInput.value.trim();
}

function updateCounter() {
    const length = questionInput.value.length;
    if (length > 200) {
        charCounterEl.textContent = `${length} / 500`;
        charCounterEl.classList.add("visible");
    } else {
        charCounterEl.textContent = "";
        charCounterEl.classList.remove("visible");
    }
}

function autoResizeInput() {
    questionInput.style.height = "auto";
    const computed = window.getComputedStyle(questionInput);
    const lineHeight = Number.parseFloat(computed.lineHeight) || 20;
    const border = Number.parseFloat(computed.borderTopWidth) + Number.parseFloat(computed.borderBottomWidth);
    const maxHeight = lineHeight * 5 + border + 24;
    questionInput.style.height = `${Math.min(questionInput.scrollHeight, maxHeight)}px`;
}

function removeTypingIndicator() {
    const typing = messagesContainer.querySelector("[data-typing='true']");
    if (typing) {
        typing.remove();
    }
}

function renderMessages() {
    const typing = messagesContainer.querySelector("[data-typing='true']");
    messagesContainer.innerHTML = "";

    if (state.messages.length === 0) {
        messagesContainer.appendChild(emptyStateEl);
    }

    for (const message of state.messages) {
        const bubble = document.createElement("div");
        bubble.className = `message ${message.role}${message.isError ? " error" : ""}`;
        if (message.role === "ai" && !message.isError) {
            bubble.innerHTML = processAIResponse(message.text);
        } else {
            bubble.textContent = message.text;
        }

        const meta = document.createElement("span");
        meta.className = "meta";
        meta.textContent = message.time;
        bubble.appendChild(meta);

        messagesContainer.appendChild(bubble);
    }

    if (typing) {
        messagesContainer.appendChild(typing);
    }

    chipsEl.classList.toggle("hidden", state.messages.length > 0 || !state.videoId);
    emptyStateEl.hidden = state.messages.length > 0;
    scrollToBottom();
}

function renderTypingIndicator() {
    removeTypingIndicator();

    const bubble = document.createElement("div");
    bubble.className = "message ai";
    bubble.dataset.typing = "true";

    const typing = document.createElement("div");
    typing.className = "typing";
    typing.innerHTML = "<span></span><span></span><span></span>";
    bubble.appendChild(typing);

    messagesContainer.appendChild(bubble);
    scrollToBottom();
}

function addMessage(role, text, options = {}) {
    const normalizedText = decodeHTML(text);
    state.messages.push({
        role,
        text: normalizedText,
        time: formatTime(),
        isError: Boolean(options.isError),
    });
    renderMessages();
}

function resetChat() {
    state.messages = [];
    questionInput.value = "";
    autoResizeInput();
    updateCounter();
    renderMessages();
    updateComposerState();
}

async function healthCheck() {
    for (const baseUrl of BACKEND_CANDIDATES) {
        try {
            const controller = new AbortController();
            const timeoutId = window.setTimeout(() => controller.abort(), 5000);
            const response = await fetch(`${baseUrl}/health`, { signal: controller.signal });
            window.clearTimeout(timeoutId);

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            state.backendBaseUrl = baseUrl;
            updateStatus(true, `Backend online (${baseUrl.replace("http://", "")})`);
            if (bannerEl.classList.contains("warning")) {
                clearBanner();
            }
            return;
        } catch {
            continue;
        }
    }

    updateStatus(false, "Backend offline");
    setBanner("Backend offline - run uvicorn main:app in /backend", "warning");
}

async function detectActiveVideo() {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

    if (!tab?.url || (!tab.url.includes("youtube.com") && !tab.url.includes("youtu.be"))) {
        state.videoId = null;
        state.videoLabel = "Open a YouTube video to start";
        updateSubtitle();
        setBanner("Open a YouTube video first.", "notice");
        updateComposerState();
        renderMessages();
        return;
    }

    const videoId = getVideoId(tab.url);
    state.videoId = videoId;
    state.videoLabel = videoId ? `Video ID: ${videoId}` : "Could not extract video ID";
    updateSubtitle();

    if (!videoId) {
        setBanner("Could not extract a video ID from the current page.", "notice");
    } else {
        clearBanner();
    }

    updateComposerState();
    renderMessages();
}

async function sendQuestion(questionText) {
    const question = questionText.trim();

    if (!question) {
        setBanner("Enter a question before sending.", "notice");
        return;
    }

    if (!state.videoId) {
        setBanner("Open a YouTube video first.", "notice");
        return;
    }

    const trimmedQuestion = question.slice(0, 500);
    addMessage("user", trimmedQuestion);
    questionInput.value = "";
    autoResizeInput();
    updateCounter();

    state.isLoading = true;
    updateComposerState();
    renderTypingIndicator();

    const controller = new AbortController();
    const timeoutId = window.setTimeout(() => controller.abort(), 30000);
    const conversationHistory = buildConversationHistory(state.messages.slice(0, -1));

    try {
        const response = await fetch(`${state.backendBaseUrl}/ask`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                video_id: state.videoId,
                question: trimmedQuestion,
                conversation_history: conversationHistory,
            }),
            signal: controller.signal,
        });

        window.clearTimeout(timeoutId);

        if (!response.ok) {
            let detail = "Request failed";
            try {
                const payload = await response.json();
                detail = payload?.detail || detail;
            } catch {
                detail = await response.text().catch(() => detail);
            }
            throw new Error(`Error (${response.status}): ${detail}`);
        }

        const data = await response.json();
        removeTypingIndicator();
        addMessage("ai", data.answer || "No answer received.");
    } catch (error) {
        removeTypingIndicator();

        const message = String(error?.message || error || "Unknown error");
        if (controller.signal.aborted) {
            addMessage("ai", "Request timed out.", { isError: true });
        } else if (message.includes("Failed to fetch") || message.includes("NetworkError")) {
            addMessage("ai", `Cannot reach the backend at ${state.backendBaseUrl.replace("http://", "")}.`, { isError: true });
        } else {
            addMessage("ai", message, { isError: true });
        }
    } finally {
        window.clearTimeout(timeoutId);
        state.isLoading = false;
        removeTypingIndicator();
        updateComposerState();
    }
}

function wireEvents() {
    sendBtn.addEventListener("click", () => {
        if (!sendBtn.disabled) {
            sendQuestion(questionInput.value);
        }
    });

    clearBtn.addEventListener("click", resetChat);

    questionInput.addEventListener("input", () => {
        if (questionInput.value.length > 500) {
            questionInput.value = questionInput.value.slice(0, 500);
        }
        autoResizeInput();
        updateCounter();
        updateComposerState();
    });

    questionInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            if (!sendBtn.disabled) {
                sendQuestion(questionInput.value);
            }
        }
    });

    suggestionChips.forEach((chip) => {
        chip.addEventListener("click", () => {
            questionInput.value = chip.dataset.chip || chip.textContent || "";
            autoResizeInput();
            updateCounter();
            updateComposerState();
            sendQuestion(questionInput.value);
        });
    });
}

async function init() {
    wireEvents();
    updateSubtitle();
    updateComposerState();
    autoResizeInput();
    renderMessages();
    await Promise.all([healthCheck(), detectActiveVideo()]);
}

init();