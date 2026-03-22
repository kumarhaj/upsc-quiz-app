from __future__ import annotations

import json
import os
import random
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable
from urllib.error import URLError
from urllib.request import Request, urlopen

from rag_store import hash_text_to_vector, load_index, search_vector_index


OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "llama2")
OLLAMA_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180"))
QUESTION_RETRY_LIMIT = 2
SUBJECTS = ["History", "Polity", "Economy", "Geography", "Environment", "Science", "Current Affairs"]
BASE_DIR = Path(__file__).resolve().parent
NOTES_PATH = BASE_DIR / "upsc_research_notes.md"
RAG_INDEX_PATH = BASE_DIR / "rag_store" / "index.json"
RAG_VECTOR_INDEX_PATH = BASE_DIR / "rag_store" / "vector_index.json"
CURRENT_RAG_INDEX_PATH = BASE_DIR / "rag_store_current" / "index.json"
CURRENT_RAG_VECTOR_INDEX_PATH = BASE_DIR / "rag_store_current" / "vector_index.json"
SUBJECT_KEYWORDS = {
    "History": ["history", "modern india", "art and culture", "ancient", "bhakti", "ncert history", "heritage"],
    "Polity": ["polity", "constitution", "constitutional", "federalism", "parliament", "reservation act", "criminal laws", "dpdp"],
    "Economy": ["economy", "gst", "ibc", "upi", "inflation", "budget", "green hydrogen", "macro", "digital economy"],
    "Geography": ["geography", "river", "soil", "mapping", "climate", "monsoon", "wetlands"],
    "Environment": ["environment", "biodiversity", "conservation", "wetlands", "climate", "mission life", "green transitions"],
    "Science": ["science", "science-tech", "chandrayaan", "aditya-l1", "indiaai", "semiconductors", "technology", "space"],
    "Current Affairs": ["current", "g20", "dpdp", "indiaai", "green hydrogen", "women's reservation", "criminal laws", "biofuel", "article 370"],
}
MAX_BATCH_ATTEMPTS_MULTIPLIER = 5

def _extract_json_block(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Model response did not contain JSON.")
    return text[start : end + 1]


@lru_cache(maxsize=1)
def _load_research_notes() -> str:
    return NOTES_PATH.read_text(encoding="utf-8")


def _rag_paths(corpus: str) -> tuple[Path, Path]:
    if corpus == "current_affairs":
        return CURRENT_RAG_VECTOR_INDEX_PATH, CURRENT_RAG_INDEX_PATH
    return RAG_VECTOR_INDEX_PATH, RAG_INDEX_PATH


def _select_rag_corpus(subject: str) -> str:
    if subject == "Current Affairs":
        return "current_affairs"
    return "static"


@lru_cache(maxsize=2)
def _load_rag_index(corpus: str = "static") -> dict[str, Any] | None:
    vector_path, index_path = _rag_paths(corpus)
    path = vector_path if vector_path.exists() else index_path
    if not path.exists():
        return None
    index = load_index(path)
    if index.get("kind") == "redirect":
        target = path.parent / index["target"]
        if target.exists():
            index = load_index(target)
        else:
            return None
    if index.get("kind") != "vector":
        return None
    return index


def _extract_markdown_section(notes: str, heading: str, level: int = 2) -> str:
    prefix = "#" * level
    pattern = rf"(?ms)^{re.escape(prefix)} {re.escape(heading)}\n(.*?)(?=^#{{1,{level}}} |\Z)"
    match = re.search(pattern, notes)
    return match.group(1).strip() if match else ""


def _extract_subject_context(subject: str) -> str:
    notes = _load_research_notes()
    current_affairs = _extract_markdown_section(notes, "Last 10 years current-affairs topic map")
    quiz_design = _extract_markdown_section(notes, "How the quiz was designed")

    relevant_lines: list[str] = []
    keywords = SUBJECT_KEYWORDS[subject]
    for line in current_affairs.splitlines():
        lowered = line.lower()
        if any(keyword in lowered for keyword in keywords):
            relevant_lines.append(line.strip())

    relevant_current = "\n".join(relevant_lines[:12]).strip()
    parts = [
        "Relevant current-affairs/topic excerpt:",
        relevant_current or "Use a static-concept-led question if no strong current-affairs match appears.",
        "UPSC pattern excerpt:",
        quiz_design,
        "Keep the question factual, conservative, and suitable for UPSC Prelims General Studies.",
    ]
    return "\n".join(part for part in parts if part).strip()


def _embed_query(text: str) -> list[float]:
    index = _load_rag_index("current_affairs") or _load_rag_index("static")
    embed_model = index.get("embedding_model", OLLAMA_EMBED_MODEL) if index else OLLAMA_EMBED_MODEL
    if isinstance(embed_model, str) and embed_model.startswith("hashed-bow-"):
        return hash_text_to_vector(text)
    payload = {"model": embed_model, "input": [text]}
    req = Request(
        f"{OLLAMA_BASE_URL}/api/embed",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=120) as response:
        data = json.loads(response.read().decode("utf-8"))
    embeddings = data.get("embeddings")
    if not isinstance(embeddings, list) or not embeddings or not isinstance(embeddings[0], list):
        raise ValueError("Embedding query failed.")
    return embeddings[0]


def _build_rag_query(subject: str) -> str:
    query_map = {
        "History": "NCERT history modern india culture themes in history UPSC prelims",
        "Polity": "NCERT political science constitution parliament rights federalism UPSC prelims",
        "Economy": "NCERT indian economic development reforms planning inflation UPSC prelims",
        "Geography": "NCERT geography india physical environment rivers climate monsoon UPSC prelims",
        "Environment": "NCERT geography environment biodiversity conservation climate UPSC prelims",
        "Science": "NCERT science space technology basic science UPSC prelims",
        "Current Affairs": "India recent policy missions laws summits space climate digital governance official updates UPSC prelims",
    }
    return query_map[subject]


def _prefer_official_current_affairs_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def rank_key(chunk: dict[str, Any]) -> tuple[int, float]:
        url = str(chunk.get("url") or "").lower()
        path = chunk.get("path")
        title = str(chunk.get("title") or "").lower()
        is_local_note = bool(path) or "research notes" in title
        is_official = any(domain in url for domain in ("pib.gov.in", "isro.gov.in", "g20.org", "upsc.gov.in"))
        priority = 0 if is_official else 1 if not is_local_note else 2
        return (priority, -float(chunk.get("score") or 0.0))

    return sorted(chunks, key=rank_key)


def _extract_rag_context(subject: str) -> tuple[str, str | None, str | None]:
    corpus = _select_rag_corpus(subject)
    index = _load_rag_index(corpus)
    if not index:
        return "", None, None

    query_embedding = _embed_query(_build_rag_query(subject))
    chunks = search_vector_index(index, query_embedding, subject=subject, top_k=6)
    if not chunks:
        return "", None, None
    if corpus == "current_affairs":
        chunks = _prefer_official_current_affairs_chunks(chunks)

    heading = "Retrieved current-affairs context:" if corpus == "current_affairs" else "Retrieved NCERT context:"
    parts = [heading]
    primary_source = chunks[0]["title"]
    primary_text = chunks[0]["text"]
    primary_chunk = chunks[0]
    parts.append(f'Source: {primary_chunk["title"]}')
    parts.append(primary_chunk["text"][:700].strip())
    return "\n".join(parts).strip(), primary_source, primary_text


def _extract_labeled_field(text: str, label: str) -> str:
    pattern = rf"(?im)^{re.escape(label)}\s*:\s*(.+)$"
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"Missing field: {label}")
    return match.group(1).strip()


def _normalize_for_similarity(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def _question_signature(question: dict[str, Any]) -> tuple[str, str]:
    stem = _normalize_for_similarity(question["question"])
    reference = _normalize_for_similarity(question.get("reference", ""))
    return stem, reference


def _is_similar_question(candidate: dict[str, Any], accepted: list[dict[str, Any]]) -> bool:
    candidate_stem, candidate_reference = _question_signature(candidate)
    candidate_tokens = set(candidate_stem.split())

    for existing in accepted:
        existing_stem, existing_reference = _question_signature(existing)
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


def _build_subject_plan(count: int) -> list[str]:
    ordered_subjects = [
        "History",
        "Polity",
        "Economy",
        "Geography",
        "Environment",
        "Science",
        "Current Affairs",
    ]
    plan = list(ordered_subjects[: min(count, len(ordered_subjects))])
    remaining = count - len(plan)
    if remaining <= 0:
        random.shuffle(plan)
        return plan

    weighted_tail = [
        "Polity",
        "Economy",
        "Geography",
        "Current Affairs",
        "History",
        "Environment",
        "Science",
    ]
    for index in range(remaining):
        plan.append(weighted_tail[index % len(weighted_tail)])
    random.shuffle(plan)
    return plan


def _build_explanation_from_context(subject: str, primary_source: str | None, primary_text: str | None) -> str:
    if not primary_text:
        return f"Grounded in {subject} context prepared for UPSC-style practice."
    sentence = re.split(r"(?<=[.!?])\s+", primary_text.strip())[0]
    sentence = re.sub(r"\s+", " ", sentence).strip()
    if len(sentence) > 240:
        sentence = sentence[:237].rstrip() + "..."
    if primary_source:
        return f"{sentence} Source used: {primary_source}."
    return sentence


def _normalize_structured_question(
    raw_text: str,
    subject: str,
    primary_source: str | None,
    primary_text: str | None,
) -> dict[str, Any]:
    question = _extract_labeled_field(raw_text, "QUESTION")
    correct = _extract_labeled_field(raw_text, "CORRECT")
    wrong1 = _extract_labeled_field(raw_text, "WRONG1")
    wrong2 = _extract_labeled_field(raw_text, "WRONG2")
    wrong3 = _extract_labeled_field(raw_text, "WRONG3")
    explanation = _build_explanation_from_context(subject, primary_source, primary_text)
    reference = primary_source or f"{subject} / retrieved context"

    options = [correct, wrong1, wrong2, wrong3]
    if len({option.strip().lower() for option in options}) != 4:
        raise ValueError("Options were not distinct.")

    rng = random.Random()
    rng.shuffle(options)
    answer_index = options.index(correct)

    return {
        "question": question,
        "options": options,
        "answer_index": answer_index,
        "explanation": explanation,
        "subject": subject,
        "reference": reference,
    }


def _validate_question(item: dict[str, Any]) -> dict[str, Any]:
    required = {"question", "options", "answer_index", "explanation", "subject", "reference"}
    if not required.issubset(item):
        raise ValueError("Question object missing required fields.")
    options = item["options"]
    if not isinstance(options, list) or len(options) != 4 or len(set(options)) != 4:
        raise ValueError("Each question must have exactly 4 distinct options.")
    answer_index = item["answer_index"]
    if not isinstance(answer_index, int) or answer_index not in range(4):
        raise ValueError("answer_index must be 0..3.")
    return {
        "question": str(item["question"]).strip(),
        "options": [str(option).strip() for option in options],
        "answer_index": answer_index,
        "explanation": str(item["explanation"]).strip(),
        "subject": str(item["subject"]).strip(),
        "reference": str(item["reference"]).strip(),
    }


def _call_ollama_raw_question(subject: str) -> str:
    rag_context, primary_source, primary_text = _extract_rag_context(subject)
    subject_context = rag_context or _extract_subject_context(subject)
    prompt = f"""
Create exactly one UPSC-style multiple choice question for UPSC General Studies.

Requirements:
- The subject must be exactly "{subject}".
- Return plain text only in exactly this 5-line format:
QUESTION: ...
CORRECT: ...
WRONG1: ...
WRONG2: ...
WRONG3: ...
- Keep it objective, UPSC-like, and factually grounded.
- Prefer statement-style or elimination-friendly framing when natural.
- Keep all 4 options short and distinct.
- Use the retrieved context below as the main factual grounding when available.
- Use only facts that are supported by the retrieved context below.
- Do not add numbering, bullets, markdown, or any extra text.

Use this subject-specific grounding context:
{subject_context}
""".strip()

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "top_p": 0.9,
            "num_predict": 140,
        },
    }
    req = Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=OLLAMA_TIMEOUT_SECONDS) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data.get("response", "").strip(), primary_source, primary_text


def _call_ollama_question(subject: str | None = None) -> dict[str, Any]:
    last_error: Exception | None = None
    subject = subject or random.choice(SUBJECTS)
    for _attempt in range(QUESTION_RETRY_LIMIT):
        try:
            raw_text, primary_source, primary_text = _call_ollama_raw_question(subject)
            item = _normalize_structured_question(raw_text, subject, primary_source, primary_text)
            return _validate_question(item)
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = exc
    raise ValueError("Model did not return a valid question.") from last_error


def generate_quiz_payload(force_fallback: bool = False) -> dict[str, Any]:
    if force_fallback:
        raise ValueError("Fallback mode is disabled. Ollama generation is required.")

    try:
        question = _call_ollama_question()
    except (URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Unable to generate question with Ollama {OLLAMA_MODEL}. "
            "Please ensure Ollama is running and the model can complete the request."
        ) from exc

    return {
        "source": "ollama + vector-rag",
        "model": OLLAMA_MODEL,
        "count": 1,
        "question": question,
    }


def generate_quiz_batch_payload(
    count: int,
    force_fallback: bool = False,
    progress_callback: Callable[[list[dict[str, Any]]], None] | None = None,
) -> dict[str, Any]:
    if force_fallback:
        raise ValueError("Fallback mode is disabled. Ollama generation is required.")

    accepted: list[dict[str, Any]] = []
    subject_plan = _build_subject_plan(count)
    max_attempts = max(count * MAX_BATCH_ATTEMPTS_MULTIPLIER, count + 3)
    attempts = 0

    while len(accepted) < count and attempts < max_attempts:
        subject = subject_plan[len(accepted)] if len(accepted) < len(subject_plan) else random.choice(SUBJECTS)
        attempts += 1
        try:
            question = _call_ollama_question(subject)
        except (URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError):
            continue
        if _is_similar_question(question, accepted):
            continue
        accepted.append(question)
        if progress_callback:
            progress_callback(list(accepted))

    if len(accepted) != count:
        raise RuntimeError(
            f"Unable to generate a sufficiently diverse {count}-question quiz with Ollama {OLLAMA_MODEL}."
        )

    return {
        "source": "ollama + vector-rag",
        "model": OLLAMA_MODEL,
        "count": count,
        "questions": accepted,
    }
