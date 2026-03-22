from __future__ import annotations

import json
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from quiz_engine import generate_quiz_batch_payload, generate_quiz_payload


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
QUIZ_SIZE = 10
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>UPSC GS Quiz Generator</title>
    <link rel="stylesheet" href="/styles.css" />
  </head>
  <body>
    <main class="shell">
      <section class="hero">
        <p class="eyebrow">UPSC General Studies</p>
        <h1>Fresh 10-question quiz session</h1>
        <p class="subhead">
          The app first generates all 10 UPSC-style questions in the background. Once the full
          set is ready, the quiz starts and you can move through the session without waiting
          between questions.
        </p>
        <div class="controls">
          <button id="generateBtn" class="primary">Generate 10-Question Quiz</button>
          <button id="nextBtn" class="secondary" disabled>Next Question</button>
          <button id="revealBtn" class="secondary" disabled>Reveal Answer</button>
        </div>
        <div class="meta">
          <span id="statusPill" class="pill">Idle</span>
          <span id="sourceText">Source: not generated yet</span>
          <span id="scoreLine">Answered: 0 / 10 | Correct: 0</span>
          <span id="queueText">Generation: waiting to start</span>
        </div>
      </section>

      <section class="card" id="introCard">
        <h2>How this works</h2>
        <p>
          Click once to generate a full 10-question UPSC GS quiz. While the questions are being
          prepared, the page shows elapsed time and generation progress. The quiz begins only after
          all 10 questions are ready.
        </p>
      </section>

      <section id="questionCard" class="question-card hidden">
        <div class="question-top">
          <span id="questionNumber"></span>
          <span id="questionSubject"></span>
        </div>
        <h3 id="questionText" class="question-text"></h3>
        <div id="options" class="options"></div>
        <div id="review" class="review hidden">
          <p id="reviewAnswer"></p>
          <p id="reviewExplanation"></p>
          <p id="reviewReference"></p>
        </div>
      </section>

      <section id="resultCard" class="card hidden"></section>
    </main>

    <script src="/app.js"></script>
  </body>
</html>
"""

APP_JS = """const QUIZ_SIZE = 10;
const generateBtn = document.getElementById("generateBtn");
const nextBtn = document.getElementById("nextBtn");
const revealBtn = document.getElementById("revealBtn");
const questionCard = document.getElementById("questionCard");
const resultCard = document.getElementById("resultCard");
const statusPill = document.getElementById("statusPill");
const sourceText = document.getElementById("sourceText");
const introCard = document.getElementById("introCard");
const questionNumber = document.getElementById("questionNumber");
const questionSubject = document.getElementById("questionSubject");
const questionText = document.getElementById("questionText");
const optionsWrap = document.getElementById("options");
const reviewWrap = document.getElementById("review");
const reviewAnswer = document.getElementById("reviewAnswer");
const reviewExplanation = document.getElementById("reviewExplanation");
const reviewReference = document.getElementById("reviewReference");
const scoreLine = document.getElementById("scoreLine");
const queueText = document.getElementById("queueText");

let quizQuestions = [];
let currentQuestion = null;
let currentIndex = 0;
let answered = 0;
let correct = 0;
let revealed = false;
let selectedJobId = null;

function setStatus(text, tone = "neutral") {
  statusPill.textContent = text;
  statusPill.dataset.tone = tone;
}

function formatElapsed(seconds) {
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${mins}:${String(secs).padStart(2, "0")}`;
}

function resetSession() {
  quizQuestions = [];
  currentQuestion = null;
  currentIndex = 0;
  answered = 0;
  correct = 0;
  revealed = false;
  selectedJobId = null;
  questionCard.classList.add("hidden");
  resultCard.classList.add("hidden");
  reviewWrap.classList.add("hidden");
  optionsWrap.innerHTML = "";
  updateScore();
}

function updateScore() {
  scoreLine.textContent = `Answered: ${answered} / ${QUIZ_SIZE} | Correct: ${correct}`;
}

function updateProgress(completed, total, elapsedSeconds) {
  queueText.textContent = `Generation: ${completed} / ${total} ready | Elapsed ${formatElapsed(elapsedSeconds)}`;
}

function renderQuestion(index) {
  currentIndex = index;
  currentQuestion = quizQuestions[index];
  revealed = false;
  optionsWrap.innerHTML = "";
  reviewWrap.classList.add("hidden");
  resultCard.classList.add("hidden");

  questionNumber.textContent = `Question ${index + 1} of ${QUIZ_SIZE}`;
  questionSubject.textContent = currentQuestion.subject;
  questionText.textContent = currentQuestion.question;

  currentQuestion.options.forEach((option, optionIndex) => {
    const label = document.createElement("label");
    label.className = "option";

    const input = document.createElement("input");
    input.type = "radio";
    input.name = "current-question";
    input.value = String(optionIndex);

    const text = document.createElement("span");
    text.textContent = option;

    label.append(input, text);
    optionsWrap.append(label);
  });

  introCard.classList.add("hidden");
  questionCard.classList.remove("hidden");
  revealBtn.disabled = false;
  nextBtn.disabled = true;
  updateScore();
}

function selectedAnswerIndex() {
  const checked = questionCard.querySelector('input[name="current-question"]:checked');
  return checked ? Number(checked.value) : null;
}

function showFinalResult() {
  resultCard.innerHTML = `
    <h2>Quiz Complete</h2>
    <p class="score-line">${correct} / ${answered} correct</p>
    <p>Generate another 10-question quiz whenever you're ready.</p>
  `;
  resultCard.classList.remove("hidden");
}

function revealAnswer() {
  if (!currentQuestion || revealed) {
    return;
  }

  const selectedIndex = selectedAnswerIndex();
  const options = [...questionCard.querySelectorAll(".option")];
  options.forEach((label, optionIndex) => {
    label.querySelector("input").disabled = true;
    label.classList.remove("correct", "wrong", "missed");

    if (optionIndex === currentQuestion.answer_index) {
      label.classList.add("correct");
    }
    if (selectedIndex === optionIndex && selectedIndex !== currentQuestion.answer_index) {
      label.classList.add("wrong");
    }
    if (selectedIndex === null && optionIndex === currentQuestion.answer_index) {
      label.classList.add("missed");
    }
  });

  answered += 1;
  if (selectedIndex === currentQuestion.answer_index) {
    correct += 1;
  }

  const pickedText = selectedIndex === null ? "No answer selected" : currentQuestion.options[selectedIndex];
  reviewAnswer.textContent = `Your answer: ${pickedText} | Correct answer: ${currentQuestion.options[currentQuestion.answer_index]}`;
  reviewExplanation.textContent = `Why: ${currentQuestion.explanation}`;
  reviewReference.textContent = `Reference: ${currentQuestion.reference}`;
  reviewWrap.classList.remove("hidden");

  if (currentIndex === QUIZ_SIZE - 1) {
    showFinalResult();
    nextBtn.disabled = true;
  } else {
    resultCard.innerHTML = `
      <h2>Running Score</h2>
      <p class="score-line">${correct} / ${answered} correct</p>
      <p>Move to the next question when you're ready.</p>
    `;
    resultCard.classList.remove("hidden");
    nextBtn.disabled = false;
  }

  revealBtn.disabled = true;
  revealed = true;
  updateScore();
}

function moveNext() {
  if (!revealed || currentIndex >= QUIZ_SIZE - 1) {
    return;
  }
  renderQuestion(currentIndex + 1);
}

async function generateQuiz() {
  resetSession();
  setStatus("Generating...", "loading");
  sourceText.textContent = "Source: building full 10-question session";
  generateBtn.disabled = true;
  nextBtn.disabled = true;
  revealBtn.disabled = true;

  try {
    const startResponse = await fetch("/api/generate-start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    const startData = await startResponse.json();
    if (!startResponse.ok || !startData.ok) {
      throw new Error(startData.error || "Generation failed");
    }

    selectedJobId = startData.job_id;
    updateProgress(startData.completed ?? 0, startData.total ?? QUIZ_SIZE, startData.elapsed_seconds ?? 0);
    const data = await pollJob(startData.job_id);
    quizQuestions = data.questions || [];
    if (quizQuestions.length !== QUIZ_SIZE) {
      throw new Error("The quiz did not return the full 10 questions.");
    }
    renderQuestion(0);
    setStatus("Ready", "success");
    sourceText.textContent = `Source: ${data.source} (${data.model})`;
    queueText.textContent = `Generation: complete in ${formatElapsed(data.elapsed_seconds ?? 0)}`;
  } catch (error) {
    setStatus("Failed", "error");
    sourceText.textContent = `Source: error - ${error.message}`;
    queueText.textContent = "Generation: failed";
  } finally {
    generateBtn.disabled = false;
    revealBtn.disabled = !currentQuestion || revealed;
    nextBtn.disabled = true;
  }
}

generateBtn.addEventListener("click", generateQuiz);
nextBtn.addEventListener("click", moveNext);
revealBtn.addEventListener("click", revealAnswer);
updateScore();

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function pollJob(jobId) {
  for (let attempt = 0; attempt < 600; attempt += 1) {
    const response = await fetch(`/api/generate-status?id=${encodeURIComponent(jobId)}`);
    const data = await response.json();
    if (!response.ok || !data.ok) {
      throw new Error(data.error || "Generation status failed");
    }
    if (data.status === "ready") {
      return data.payload;
    }
    if (data.status === "error") {
      throw new Error(data.error || "Generation failed");
    }
    updateProgress(data.completed ?? 0, data.total ?? QUIZ_SIZE, data.elapsed_seconds ?? 0);
    sourceText.textContent = `Source: generating question set (${data.completed ?? 0}/${data.total ?? QUIZ_SIZE})`;
    await sleep(1000);
  }
  throw new Error("Generation timed out while waiting for the 10-question quiz.");
}
"""


class QuizHandler(BaseHTTPRequestHandler):
    server_version = "UPSCQuiz/1.0"

    def _send_json(self, payload: dict, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str) -> None:
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, text: str, content_type: str) -> None:
        body = text.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            self._send_text(INDEX_HTML, "text/html; charset=utf-8")
            return
        if parsed.path == "/app.js":
            self._send_text(APP_JS, "application/javascript; charset=utf-8")
            return
        if parsed.path == "/styles.css":
            self._send_file(STATIC_DIR / "styles.css", "text/css; charset=utf-8")
            return
        if parsed.path == "/api/health":
            self._send_json({"ok": True, "quiz_size": QUIZ_SIZE})
            return
        if parsed.path == "/api/generate-status":
            params = parse_qs(parsed.query)
            job_id = params.get("id", [""])[0]
            with JOBS_LOCK:
                job = JOBS.get(job_id)
            if not job:
                self._send_json({"ok": False, "error": "Unknown job id"}, status=HTTPStatus.NOT_FOUND)
                return

            elapsed = int(time.time() - job["started_at"])
            payload = {
                "ok": True,
                "status": job["status"],
                "completed": len(job.get("questions", [])),
                "total": job.get("total", QUIZ_SIZE),
                "elapsed_seconds": elapsed,
            }
            if job["status"] == "ready":
                payload["payload"] = {
                    "source": job["source"],
                    "model": job["model"],
                    "count": len(job["questions"]),
                    "questions": job["questions"],
                    "elapsed_seconds": elapsed,
                }
            if job["status"] == "error":
                payload["error"] = job["error"]
            self._send_json(payload)
            return
        self._send_json({"ok": False, "error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path not in {"/api/generate", "/api/generate-start"}:
            self._send_json({"ok": False, "error": "Not found"}, status=HTTPStatus.NOT_FOUND)
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            content_length = 0

        raw_body = self.rfile.read(content_length) if content_length > 0 else b"{}"
        try:
            data = json.loads(raw_body.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            self._send_json({"ok": False, "error": "Invalid JSON body"}, status=HTTPStatus.BAD_REQUEST)
            return

        if parsed.path == "/api/generate":
            try:
                payload = generate_quiz_payload(force_fallback=bool(data.get("forceFallback", False)))
            except Exception as exc:  # pragma: no cover
                self._send_json(
                    {"ok": False, "error": f"Quiz generation failed: {exc}"},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                return
            self._send_json({"ok": True, **payload})
            return

        job_id = uuid.uuid4().hex
        with JOBS_LOCK:
            JOBS[job_id] = {
                "status": "pending",
                "started_at": time.time(),
                "questions": [],
                "total": QUIZ_SIZE,
                "source": "ollama + vector-rag",
                "model": "pending",
            }
        threading.Thread(target=_generate_quiz_batch, args=(job_id,), daemon=True).start()
        self._send_json(
            {
                "ok": True,
                "job_id": job_id,
                "status": "pending",
                "completed": 0,
                "total": QUIZ_SIZE,
                "elapsed_seconds": 0,
            }
        )

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return


def _generate_quiz_batch(job_id: str) -> None:
    try:
        payload = generate_quiz_batch_payload(QUIZ_SIZE)
    except Exception as exc:  # pragma: no cover
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if job:
                job["status"] = "error"
                job["error"] = f"Quiz generation failed: {exc}"
        return

    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        job["questions"] = list(payload.get("questions", []))
        job["source"] = payload.get("source", "ollama + vector-rag")
        job["model"] = payload.get("model", "unknown")
        job["status"] = "ready"


def main() -> None:
    host = "127.0.0.1"
    port = 8000
    httpd = ThreadingHTTPServer((host, port), QuizHandler)
    print(f"UPSC quiz app running at http://{host}:{port}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
