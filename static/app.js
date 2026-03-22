const generateBtn = document.getElementById("generateBtn");
const fallbackBtn = document.getElementById("fallbackBtn");
const regenerateBtn = document.getElementById("regenerateBtn");
const submitBtn = document.getElementById("submitBtn");
const quizForm = document.getElementById("quizForm");
const actionBar = document.getElementById("actionBar");
const resultCard = document.getElementById("resultCard");
const statusPill = document.getElementById("statusPill");
const sourceText = document.getElementById("sourceText");
const introCard = document.getElementById("introCard");
const template = document.getElementById("questionTemplate");

let currentQuiz = [];
let submitted = false;

function setStatus(text, tone = "neutral") {
  statusPill.textContent = text;
  statusPill.dataset.tone = tone;
}

function setBusy(disabled) {
  generateBtn.disabled = disabled;
  fallbackBtn.disabled = disabled;
  regenerateBtn.disabled = disabled;
  submitBtn.disabled = disabled;
}

function renderQuiz(questions) {
  quizForm.innerHTML = "";
  currentQuiz = questions;
  submitted = false;
  resultCard.classList.add("hidden");

  questions.forEach((item, index) => {
    const fragment = template.content.cloneNode(true);
    const optionsWrap = fragment.querySelector(".options");

    fragment.querySelector(".question-number").textContent = `Q${index + 1}`;
    fragment.querySelector(".question-subject").textContent = item.subject;
    fragment.querySelector(".question-text").textContent = item.question;

    item.options.forEach((option, optionIndex) => {
      const label = document.createElement("label");
      label.className = "option";

      const input = document.createElement("input");
      input.type = "radio";
      input.name = `q-${index}`;
      input.value = String(optionIndex);

      const text = document.createElement("span");
      text.textContent = option;

      label.append(input, text);
      optionsWrap.append(label);
    });

    quizForm.append(fragment);
  });

  introCard.classList.add("hidden");
  quizForm.classList.remove("hidden");
  actionBar.classList.remove("hidden");
}

function collectAnswers() {
  return currentQuiz.map((_, index) => {
    const checked = quizForm.querySelector(`input[name="q-${index}"]:checked`);
    return checked ? Number(checked.value) : null;
  });
}

function reviewQuiz() {
  if (!currentQuiz.length || submitted) {
    return;
  }

  const answers = collectAnswers();
  let score = 0;

  [...quizForm.querySelectorAll(".question-card")].forEach((card, index) => {
    const item = currentQuiz[index];
    const selectedIndex = answers[index];
    const review = card.querySelector(".review");
    const answerText = card.querySelector(".review-answer");
    const explanationText = card.querySelector(".review-explanation");
    const referenceText = card.querySelector(".review-reference");
    const options = [...card.querySelectorAll(".option")];

    options.forEach((label, optionIndex) => {
      label.classList.remove("correct", "wrong", "missed");
      label.querySelector("input").disabled = true;

      if (optionIndex === item.answer_index) {
        label.classList.add("correct");
      }
      if (selectedIndex === optionIndex && selectedIndex !== item.answer_index) {
        label.classList.add("wrong");
      }
      if (selectedIndex === null && optionIndex === item.answer_index) {
        label.classList.add("missed");
      }
    });

    if (selectedIndex === item.answer_index) {
      score += 1;
    }

    const pickedText = selectedIndex === null ? "No answer selected" : item.options[selectedIndex];
    answerText.textContent = `Your answer: ${pickedText} | Correct answer: ${item.options[item.answer_index]}`;
    explanationText.textContent = `Why: ${item.explanation}`;
    referenceText.textContent = `Reference: ${item.reference}`;
    review.classList.remove("hidden");
  });

  const percentage = ((score / currentQuiz.length) * 100).toFixed(1);
  resultCard.innerHTML = `
    <h2>Score</h2>
    <p class="score-line">${score} / ${currentQuiz.length} correct</p>
    <p>You scored ${percentage}% on this generated UPSC GS set.</p>
  `;
  resultCard.classList.remove("hidden");
  submitted = true;
  submitBtn.disabled = true;
}

async function generateQuiz(forceFallback = false) {
  setStatus("Generating...", "loading");
  sourceText.textContent = "Source: working on a fresh quiz";
  setBusy(true);

  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ forceFallback }),
    });
    const data = await response.json();
    if (!response.ok || !data.ok) {
      throw new Error(data.error || "Generation failed");
    }

    renderQuiz(data.questions);
    setStatus("Ready", data.source === "ollama" ? "success" : "fallback");
    sourceText.textContent = `Source: ${data.source} (${data.model})`;
    submitBtn.disabled = false;
  } catch (error) {
    setStatus("Failed", "error");
    sourceText.textContent = `Source: error - ${error.message}`;
  } finally {
    setBusy(false);
    if (!currentQuiz.length) {
      submitBtn.disabled = true;
    }
  }
}

generateBtn.addEventListener("click", () => generateQuiz(false));
fallbackBtn.addEventListener("click", () => generateQuiz(true));
regenerateBtn.addEventListener("click", () => generateQuiz(false));
submitBtn.addEventListener("click", reviewQuiz);
