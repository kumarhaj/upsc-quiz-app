from __future__ import annotations

import json
import random
import re
import threading
import time
import uuid
from collections import Counter
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from quiz_engine import generate_quiz_payload


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
QUIZ_SIZE = 5
CACHE_TARGET = 12
CACHE_FILE = BASE_DIR / "generated_questions_cache.json"
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
QUESTION_CACHE: list[dict] = []
CACHE_LOCK = threading.Lock()
CACHE_WORKER_STARTED = False
INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>General Studies Quiz Generator</title>
    <link rel="stylesheet" href="/styles.css" />
  </head>
  <body>
    <main class="shell">
      <section class="hero">
        <p class="eyebrow">General Studies</p>
        <h1>Fresh 5-question quiz session</h1>
        <p class="subhead">
          Questions are pre-generated in the background and saved into a local cache. When 5 are
          ready, the quiz starts as a full session with no extra wait between questions.
        </p>
        <div class="controls">
          <button id="generateBtn" class="primary">Start 5-Question Quiz</button>
          <button id="nextBtn" class="secondary" disabled>Next Question</button>
          <button id="revealBtn" class="secondary" disabled>Reveal Answer</button>
        </div>
        <div class="meta">
          <span id="statusPill" class="pill">Idle</span>
          <span id="sourceText">Source: cache warming not started yet</span>
          <span id="scoreLine">Answered: 0 / 5 | Correct: 0</span>
          <span id="queueText">Cache: 0 ready</span>
        </div>
      </section>

      <section class="card" id="introCard">
        <h2>How this works</h2>
        <p>
          Click once to start a 5-question General Studies quiz. If the local cache already has enough
          questions, the quiz starts quickly. If not, the page shows cache progress and a timer
          while background generation continues.
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

APP_JS = """const QUIZ_SIZE = 5;
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
  questionCard.classList.add("hidden");
  resultCard.classList.add("hidden");
  reviewWrap.classList.add("hidden");
  optionsWrap.innerHTML = "";
  updateScore();
}

function updateScore() {
  scoreLine.textContent = `Answered: ${answered} / ${QUIZ_SIZE} | Correct: ${correct}`;
}

function updateCacheProgress(cacheReady, elapsedSeconds) {
  queueText.textContent = `Cache: ${cacheReady} ready | Elapsed ${formatElapsed(elapsedSeconds)}`;
}

async function refreshCacheStatus() {
  try {
    const response = await fetch("/api/health");
    const data = await response.json();
    if (!response.ok || !data.ok) {
      return;
    }
    if (!currentQuestion && !quizQuestions.length) {
      queueText.textContent = `Cache: ${data.cache_ready ?? 0} ready`;
      sourceText.textContent = `Source: background cache warming (${data.cache_ready ?? 0} ready)`;
    }
  } catch (_error) {
    // Ignore background status refresh failures.
  }
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
    <p>Start another session whenever you're ready.</p>
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

async function startQuiz() {
  resetSession();
  setStatus("Preparing...", "loading");
  sourceText.textContent = "Source: checking local cache";
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

    updateCacheProgress(startData.cache_ready ?? 0, startData.elapsed_seconds ?? 0);
    const data = await pollJob(startData.job_id);
    quizQuestions = data.questions || [];
    if (quizQuestions.length !== QUIZ_SIZE) {
      throw new Error("The quiz did not return the full 10 questions.");
    }
    renderQuestion(0);
    setStatus("Ready", "success");
    sourceText.textContent = `Source: ${data.source} (${data.model})`;
    queueText.textContent = `Cache: ${data.cache_ready ?? 0} ready | Session ready in ${formatElapsed(data.elapsed_seconds ?? 0)}`;
  } catch (error) {
    setStatus("Failed", "error");
    sourceText.textContent = `Source: error - ${error.message}`;
    queueText.textContent = "Cache: failed to prepare session";
  } finally {
    generateBtn.disabled = false;
    revealBtn.disabled = !currentQuestion || revealed;
    nextBtn.disabled = true;
  }
}

generateBtn.addEventListener("click", startQuiz);
nextBtn.addEventListener("click", moveNext);
revealBtn.addEventListener("click", revealAnswer);
updateScore();
refreshCacheStatus();
setInterval(refreshCacheStatus, 5000);

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function pollJob(jobId) {
  for (let attempt = 0; attempt < 1800; attempt += 1) {
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
    updateCacheProgress(data.cache_ready ?? 0, data.elapsed_seconds ?? 0);
    sourceText.textContent = `Source: warming cache (${data.cache_ready ?? 0} ready / need ${data.required ?? QUIZ_SIZE})`;
    await sleep(1000);
  }
  throw new Error("Generation timed out while waiting for enough cached questions.");
}
"""


def _normalize_for_similarity(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def _is_similar_question(candidate: dict, existing: list[dict]) -> bool:
    candidate_stem = _normalize_for_similarity(candidate["question"])
    candidate_reference = _normalize_for_similarity(candidate.get("reference", ""))
    candidate_tokens = set(candidate_stem.split())
    for item in existing:
        existing_stem = _normalize_for_similarity(item["question"])
        existing_reference = _normalize_for_similarity(item.get("reference", ""))
        if candidate_stem == existing_stem:
            return True
        if candidate_reference and candidate_reference == existing_reference:
            return True
        existing_tokens = set(existing_stem.split())
        if not candidate_tokens or not existing_tokens:
            continue
        overlap = len(candidate_tokens & existing_tokens)
        union = len(candidate_tokens | existing_tokens)
        if union and (overlap / union) >= 0.72:
            return True
    return False


def _save_cache() -> None:
    CACHE_FILE.write_text(json.dumps(QUESTION_CACHE, ensure_ascii=True, indent=2), encoding="utf-8")


def _load_cache() -> None:
    if not CACHE_FILE.exists():
        return
    try:
        cached = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    if not isinstance(cached, list):
        return
    with CACHE_LOCK:
        QUESTION_CACHE.clear()
        QUESTION_CACHE.extend(item for item in cached if isinstance(item, dict))


def _cache_snapshot() -> list[dict]:
    with CACHE_LOCK:
        return list(QUESTION_CACHE)


def _subject_plan(count: int) -> list[str]:
    ordered = ["History", "Polity", "Economy", "Geography", "Environment", "Science", "Current Affairs"]
    plan = list(ordered[: min(count, len(ordered))])
    while len(plan) < count:
        plan.append(random.choice(["Polity", "Economy", "Geography", "Current Affairs", "History"]))
    random.shuffle(plan)
    return plan


def _assemble_session_from_cache(count: int) -> list[dict] | None:
    with CACHE_LOCK:
        if len(QUESTION_CACHE) < count:
            return None
        pool = list(QUESTION_CACHE)

    selected: list[dict] = []
    chosen_indexes: set[int] = set()
    subject_targets = Counter(_subject_plan(count))
    subject_counts: Counter[str] = Counter()

    for subject in subject_targets:
        for idx, question in enumerate(pool):
            if idx in chosen_indexes or question.get("subject") != subject:
                continue
            if _is_similar_question(question, selected):
                continue
            selected.append(question)
            chosen_indexes.add(idx)
            subject_counts[subject] += 1
            break

    for idx, question in enumerate(pool):
        if len(selected) >= count:
            break
        if idx in chosen_indexes:
            continue
        subject = question.get("subject")
        if subject_counts[subject] >= 2:
            continue
        if _is_similar_question(question, selected):
            continue
        selected.append(question)
        chosen_indexes.add(idx)
        subject_counts[subject] += 1

    if len(selected) < count:
        return None

    with CACHE_LOCK:
        QUESTION_CACHE.clear()
        _save_cache()
    return selected


def _cache_worker_loop() -> None:
    global CACHE_WORKER_STARTED
    while True:
        with CACHE_LOCK:
            cache_full = len(QUESTION_CACHE) >= CACHE_TARGET
        if cache_full:
            time.sleep(2)
            continue

        try:
            payload = generate_quiz_payload()
            question = payload["question"]
        except Exception:
            time.sleep(5)
            continue

        with CACHE_LOCK:
            if _is_similar_question(question, QUESTION_CACHE):
                continue
            QUESTION_CACHE.append(question)
            _save_cache()


def _ensure_cache_worker() -> None:
    global CACHE_WORKER_STARTED
    with JOBS_LOCK:
        if CACHE_WORKER_STARTED:
            return
        CACHE_WORKER_STARTED = True
    threading.Thread(target=_cache_worker_loop, daemon=True).start()


def _try_fulfill_job(job_id: str) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job or job.get("status") != "pending":
            return

    questions = _assemble_session_from_cache(QUIZ_SIZE)
    if not questions:
        return

    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job or job.get("status") != "pending":
            return
        job["status"] = "ready"
        job["payload"] = {
            "source": "ollama + vector-rag cache",
            "model": "llama3.1",
            "count": len(questions),
            "questions": questions,
        }


class QuizHandler(BaseHTTPRequestHandler):
    server_version = "GSQuiz/1.0"

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
        _ensure_cache_worker()
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
            self._send_json({"ok": True, "quiz_size": QUIZ_SIZE, "cache_ready": len(_cache_snapshot())})
            return
        if parsed.path == "/api/generate-status":
            params = parse_qs(parsed.query)
            job_id = params.get("id", [""])[0]
            _try_fulfill_job(job_id)
            with JOBS_LOCK:
                job = JOBS.get(job_id)
            if not job:
                self._send_json({"ok": False, "error": "Unknown job id"}, status=HTTPStatus.NOT_FOUND)
                return

            elapsed = int(time.time() - job["started_at"])
            cache_ready = len(_cache_snapshot())
            payload = {
                "ok": True,
                "status": job["status"],
                "cache_ready": cache_ready,
                "required": QUIZ_SIZE,
                "elapsed_seconds": elapsed,
            }
            if job["status"] == "ready":
                ready_payload = dict(job["payload"])
                ready_payload["elapsed_seconds"] = elapsed
                ready_payload["cache_ready"] = cache_ready
                payload["payload"] = ready_payload
            if job["status"] == "error":
                payload["error"] = job["error"]
            self._send_json(payload)
            return
        self._send_json({"ok": False, "error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        _ensure_cache_worker()
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
            }
        _try_fulfill_job(job_id)
        with JOBS_LOCK:
            job = JOBS[job_id]
        cache_ready = len(_cache_snapshot())
        self._send_json(
            {
                "ok": True,
                "job_id": job_id,
                "status": job["status"],
                "cache_ready": cache_ready,
                "required": QUIZ_SIZE,
                "elapsed_seconds": 0,
            }
        )

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return


def main() -> None:
    _load_cache()
    _ensure_cache_worker()
    host = "127.0.0.1"
    port = 8000
    httpd = ThreadingHTTPServer((host, port), QuizHandler)
    print(f"General Studies quiz app running at http://{host}:{port}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
