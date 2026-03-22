from __future__ import annotations

import json
import math
import os
import re
from hashlib import sha1
from collections import Counter
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9\-']+")
DEFAULT_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1800"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))
LOW_VALUE_PATTERNS = [
    "please write to",
    "we value any and every feedback",
    "feedback",
    "coordinator (",
    "national council of educational research and training",
    "publication division",
    "first edition",
    "isbn",
    "printed on",
    "published at the publication division",
    "all rights reserved",
]


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def is_low_value_chunk(text: str) -> bool:
    lowered = text.lower()
    matches = sum(1 for pattern in LOW_VALUE_PATTERNS if pattern in lowered)
    if matches >= 1:
        return True
    if len(tokenize(text)) < 40:
        return True
    return False


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    cleaned = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if not cleaned:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + chunk_size)
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(cleaned):
            break
        start = max(end - overlap, start + 1)
    return chunks


def build_index(documents: list[dict[str, Any]]) -> dict[str, Any]:
    chunk_entries: list[dict[str, Any]] = []
    df: Counter[str] = Counter()

    for document in documents:
        for idx, chunk in enumerate(chunk_text(document["text"])):
            if is_low_value_chunk(chunk):
                continue
            tokens = tokenize(chunk)
            if not tokens:
                continue
            tf = Counter(tokens)
            df.update(set(tf))
            chunk_entries.append(
                {
                    "chunk_id": f'{document["id"]}:{idx}',
                    "source_id": document["id"],
                    "title": document["title"],
                    "subject_tags": document.get("subject_tags", []),
                    "url": document.get("url"),
                    "path": document.get("path"),
                    "text": chunk,
                    "tf": dict(tf),
                }
            )

    total_docs = max(len(chunk_entries), 1)
    idf = {term: math.log((1 + total_docs) / (1 + freq)) + 1.0 for term, freq in df.items()}

    for entry in chunk_entries:
        weights: dict[str, float] = {}
        norm = 0.0
        for term, freq in entry["tf"].items():
            weight = (1.0 + math.log(freq)) * idf.get(term, 0.0)
            weights[term] = weight
            norm += weight * weight
        entry["weights"] = weights
        entry["norm"] = math.sqrt(norm) if norm else 1.0
        del entry["tf"]

    return {"documents": chunk_entries, "idf": idf}


def save_index(index: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index, ensure_ascii=True, indent=2), encoding="utf-8")


def load_index(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def search_index(index: dict[str, Any], query: str, subject: str | None = None, top_k: int = 4) -> list[dict[str, Any]]:
    tokens = tokenize(query)
    if not tokens:
        return []

    tf = Counter(tokens)
    idf = index["idf"]
    query_weights: dict[str, float] = {}
    query_norm = 0.0
    for term, freq in tf.items():
        weight = (1.0 + math.log(freq)) * idf.get(term, 0.0)
        if weight:
            query_weights[term] = weight
            query_norm += weight * weight

    query_norm = math.sqrt(query_norm) if query_norm else 1.0
    scored: list[tuple[float, dict[str, Any]]] = []

    for entry in index["documents"]:
        if subject and entry.get("subject_tags") and subject not in entry["subject_tags"]:
            continue
        dot = 0.0
        for term, q_weight in query_weights.items():
            dot += q_weight * entry["weights"].get(term, 0.0)
        if dot <= 0.0:
            continue
        score = dot / (query_norm * entry["norm"])
        scored.append((score, entry))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [
        {
            "score": round(score, 4),
            "chunk_id": entry["chunk_id"],
            "title": entry["title"],
            "subject_tags": entry.get("subject_tags", []),
            "url": entry.get("url"),
            "path": entry.get("path"),
            "text": entry["text"],
        }
        for score, entry in scored[:top_k]
    ]


def vector_chunk_id(source_id: str, idx: int, text: str) -> str:
    digest = sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"{source_id}:{idx}:{digest}"


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vec_a, vec_b):
        dot += a * b
        norm_a += a * a
        norm_b += b * b
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def hash_text_to_vector(text: str, dimensions: int = 512) -> list[float]:
    vector = [0.0] * dimensions
    tokens = tokenize(text)
    if not tokens:
        return vector

    for token in tokens:
        digest = sha1(token.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0.0:
        return vector
    return [value / norm for value in vector]


def build_vector_documents(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for document in documents:
        for idx, chunk in enumerate(chunk_text(document["text"])):
            if is_low_value_chunk(chunk):
                continue
            entries.append(
                {
                    "chunk_id": vector_chunk_id(document["id"], idx, chunk),
                    "source_id": document["id"],
                    "title": document["title"],
                    "subject_tags": document.get("subject_tags", []),
                    "url": document.get("url"),
                    "path": document.get("path"),
                    "text": chunk,
                }
            )
    return entries


def build_vector_index(documents: list[dict[str, Any]], embeddings: list[list[float]], embedding_model: str) -> dict[str, Any]:
    if len(documents) != len(embeddings):
        raise ValueError("Document and embedding counts do not match.")

    entries: list[dict[str, Any]] = []
    for document, embedding in zip(documents, embeddings):
        entries.append(
            {
                "chunk_id": document["chunk_id"],
                "source_id": document["source_id"],
                "title": document["title"],
                "subject_tags": document.get("subject_tags", []),
                "url": document.get("url"),
                "path": document.get("path"),
                "text": document["text"],
                "embedding": embedding,
            }
        )

    return {
        "kind": "vector",
        "embedding_model": embedding_model,
        "documents": entries,
    }


def search_vector_index(
    index: dict[str, Any],
    query_embedding: list[float],
    subject: str | None = None,
    top_k: int = 4,
) -> list[dict[str, Any]]:
    scored: list[tuple[float, dict[str, Any]]] = []
    for entry in index["documents"]:
        if subject and entry.get("subject_tags") and subject not in entry["subject_tags"]:
            continue
        score = cosine_similarity(query_embedding, entry["embedding"])
        if score <= 0.0:
            continue
        scored.append((score, entry))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [
        {
            "score": round(score, 4),
            "chunk_id": entry["chunk_id"],
            "title": entry["title"],
            "subject_tags": entry.get("subject_tags", []),
            "url": entry.get("url"),
            "path": entry.get("path"),
            "text": entry["text"],
        }
        for score, entry in scored[:top_k]
    ]
