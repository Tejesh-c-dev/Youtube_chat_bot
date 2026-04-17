let isLoading = false;

document.getElementById("askBtn").addEventListener("click", async () => {
    if (isLoading) {
        return;
    }

    const question = document.getElementById("question").value;
    const answerEl = document.getElementById("answer");

    const getVideoId = (urlString) => {
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
    };

    if (!question.trim()) {
        answerEl.innerText = "Please enter a question";
        return;
    }

    try {
        // Get current tab URL
        let [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

        if (!tab.url || (!tab.url.includes("youtube.com") && !tab.url.includes("youtu.be"))) {
            answerEl.innerText = "Please open a YouTube video first";
            return;
        }

        const videoId = getVideoId(tab.url);

        if (!videoId) {
            answerEl.innerText = "Could not extract video ID";
            return;
        }

        isLoading = true;
        answerEl.innerText = "Loading...";

        const response = await fetch("http://localhost:8000/ask", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                video_id: videoId,
                question: question
            })
        });

        if (!response.ok) {
            let detail = "";
            try {
                const payload = await response.json();
                detail = payload.detail ? ` - ${payload.detail}` : "";
            } catch {
                detail = "";
            }
            throw new Error(`Server error: ${response.status}${detail}`);
        }

        const data = await response.json();
        answerEl.innerText = data.answer || "No answer received";
    } catch (error) {
        const message = String(error?.message || "Unknown error");
        if (message.includes("Could not fetch transcript")) {
            answerEl.innerText = "Transcript could not be fetched (YouTube may be blocking your IP). Try another video, network, or retry later.";
            return;
        }

        if (message.includes("Failed to fetch") || message.includes("NetworkError")) {
            answerEl.innerText = "Cannot reach backend at localhost:8000. Start app.py and try again.";
            return;
        }

        answerEl.innerText = `Error: ${message}`;
    } finally {
        isLoading = false;
    }
});