from __future__ import annotations

import json
import os
import re
import ssl
import time
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen

from pypdf import PdfReader

from rag_store import build_vector_documents, build_vector_index, hash_text_to_vector, save_index


BASE_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = Path(os.getenv("RAG_MANIFEST_PATH", str(BASE_DIR / "rag_manifest.json"))).resolve()
RAG_DIR = Path(os.getenv("RAG_DIR", str(BASE_DIR / "rag_store"))).resolve()
RAW_DIR = RAG_DIR / "raw"
TEXT_DIR = RAG_DIR / "text"
INDEX_PATH = RAG_DIR / "index.json"
VECTOR_INDEX_PATH = RAG_DIR / "vector_index.json"
EMBED_CACHE_PATH = RAG_DIR / "embedding_cache.json"
SSL_CONTEXT = ssl._create_unverified_context()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "llama2")
USE_TEXT_CACHE_ONLY = os.getenv("RAG_USE_TEXT_CACHE_ONLY", "1") == "1"
EMBED_BATCH_SIZE = int(os.getenv("RAG_EMBED_BATCH_SIZE", "8"))
VECTOR_BACKEND = os.getenv("RAG_VECTOR_BACKEND", "hashed")
SKIP_FAILED_SOURCES = os.getenv("RAG_SKIP_FAILED_SOURCES", "1") == "1"


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:  # type: ignore[override]
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:  # type: ignore[override]
        if tag in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:  # type: ignore[override]
        if self._skip_depth:
            return
        text = data.strip()
        if text:
            self._parts.append(text)

    def get_text(self) -> str:
        return "\n".join(self._parts)


def html_to_text(html: str) -> str:
    parser = TextExtractor()
    parser.feed(html)
    text = parser.get_text()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fetch_url(url: str) -> str:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            req = Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; GS-RAG-Builder/1.0)",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                },
            )
            with urlopen(req, timeout=45, context=SSL_CONTEXT) as response:
                return response.read().decode("utf-8", errors="ignore")
        except Exception as exc:
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise last_error if last_error else RuntimeError(f"Failed to fetch {url}")


def fetch_bytes(url: str) -> bytes:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            req = Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; GS-RAG-Builder/1.0)",
                    "Accept": "application/pdf,*/*;q=0.8",
                },
            )
            with urlopen(req, timeout=60, context=SSL_CONTEXT) as response:
                return response.read()
        except Exception as exc:
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise last_error if last_error else RuntimeError(f"Failed to fetch {url}")


def ollama_embed(texts: list[str]) -> list[list[float]]:
    payload = {
        "model": OLLAMA_EMBED_MODEL,
        "input": texts,
    }
    req = Request(
        f"{OLLAMA_BASE_URL}/api/embed",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urlopen(req, timeout=1800, context=SSL_CONTEXT) as response:
                data = json.loads(response.read().decode("utf-8"))
            embeddings = data.get("embeddings")
            if not isinstance(embeddings, list) or len(embeddings) != len(texts):
                raise ValueError("Embedding API did not return expected embeddings.")
            return embeddings
        except Exception as exc:
            last_error = exc
            time.sleep(2 * (attempt + 1))
    raise last_error if last_error else RuntimeError("Embedding request failed.")


def load_embedding_cache() -> dict[str, list[float]]:
    if not EMBED_CACHE_PATH.exists():
        return {}
    return json.loads(EMBED_CACHE_PATH.read_text(encoding="utf-8"))


def save_embedding_cache(cache: dict[str, list[float]]) -> None:
    EMBED_CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")


def extract_pdf_text(pdf_bytes: bytes) -> str:
    pdf_path = RAW_DIR / "_temp.pdf"
    pdf_path.write_bytes(pdf_bytes)
    reader = PdfReader(str(pdf_path))
    parts: list[str] = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    pdf_path.unlink(missing_ok=True)
    return "\n".join(parts).strip()


def discover_ncert_chapter_ids(html: str, book_code: str) -> list[str]:
    pattern = rf'textbook\.php\?{re.escape(book_code)}=([a-z0-9]+)-[a-z0-9]+'
    matches = re.findall(pattern, html, flags=re.IGNORECASE)
    chapter_ids: list[str] = []
    for chapter_id in matches:
        if chapter_id == "0":
            continue
        if chapter_id not in chapter_ids:
            chapter_ids.append(chapter_id)
    return chapter_ids


def chapter_range_ids(chapter_count: int) -> list[str]:
    return [str(index) for index in range(1, chapter_count + 1)]


def chapter_id_to_pdf_name(book_code: str, chapter_id: str) -> str:
    if chapter_id.isdigit():
        return f"{book_code}{int(chapter_id):02d}.pdf"
    return f"{book_code}{chapter_id}.pdf"


def load_manifest() -> list[dict]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))["sources"]


def build_documents() -> list[dict]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    documents: list[dict] = []

    for source in load_manifest():
        try:
            if source["type"] == "local_markdown":
                source_path = BASE_DIR / source["path"]
                text = source_path.read_text(encoding="utf-8")
                text_path = TEXT_DIR / f'{source["id"]}.txt'
                text_path.write_text(text, encoding="utf-8")
                documents.append(
                    {
                        "id": source["id"],
                        "title": source["title"],
                        "text": text,
                        "subject_tags": source.get("subject_tags", []),
                        "url": source.get("url"),
                        "path": source.get("path"),
                        "allowed_basis": source.get("allowed_basis"),
                    }
                )
                continue
            elif source["type"] == "html":
                html = fetch_url(source["url"])
                raw_path = RAW_DIR / f'{source["id"]}.html'
                raw_path.write_text(html, encoding="utf-8")
                text = html_to_text(html)
                text_path = TEXT_DIR / f'{source["id"]}.txt'
                text_path.write_text(text, encoding="utf-8")
                documents.append(
                    {
                        "id": source["id"],
                        "title": source["title"],
                        "text": text,
                        "subject_tags": source.get("subject_tags", []),
                        "url": source.get("url"),
                        "path": source.get("path"),
                        "allowed_basis": source.get("allowed_basis"),
                    }
                )
                continue
            elif source["type"] == "ncert_listing":
                html = fetch_url(source["url"])
                raw_path = RAW_DIR / f'{source["id"]}.html'
                raw_path.write_text(html, encoding="utf-8")

                book_code = source["book_code"]
                chapter_ids = chapter_range_ids(source["chapter_count"]) if source.get("chapter_count") else discover_ncert_chapter_ids(html, book_code)
                for chapter_id in chapter_ids:
                    pdf_name = chapter_id_to_pdf_name(book_code, chapter_id)
                    pdf_url = f'https://ncert.nic.in/textbook/pdf/{pdf_name}'
                    pdf_bytes = fetch_bytes(pdf_url)
                    pdf_raw_path = RAW_DIR / pdf_name
                    pdf_raw_path.write_bytes(pdf_bytes)
                    text = extract_pdf_text(pdf_bytes)
                    if not text:
                        continue
                    text_path = TEXT_DIR / f'{source["id"]}_{chapter_id}.txt'
                    text_path.write_text(text, encoding="utf-8")
                    documents.append(
                        {
                            "id": f'{source["id"]}_{chapter_id}',
                            "title": f'{source["title"]} Chapter {chapter_id}',
                            "text": text,
                            "subject_tags": source.get("subject_tags", []),
                            "url": pdf_url,
                            "path": None,
                            "allowed_basis": source.get("allowed_basis"),
                        }
                    )
                continue
            elif source["type"] == "pdf":
                pdf_bytes = fetch_bytes(source["url"])
                pdf_name = source.get("filename") or f'{source["id"]}.pdf'
                pdf_raw_path = RAW_DIR / pdf_name
                pdf_raw_path.write_bytes(pdf_bytes)
                text = extract_pdf_text(pdf_bytes)
                if not text:
                    continue
                text_path = TEXT_DIR / f'{source["id"]}.txt'
                text_path.write_text(text, encoding="utf-8")
                documents.append(
                    {
                        "id": source["id"],
                        "title": source["title"],
                        "text": text,
                        "subject_tags": source.get("subject_tags", []),
                        "url": source.get("url"),
                        "path": None,
                        "allowed_basis": source.get("allowed_basis"),
                    }
                )
                continue
            else:
                raise ValueError(f'Unsupported source type: {source["type"]}')
        except Exception as exc:
            if not SKIP_FAILED_SOURCES:
                raise
            print(f"Skipping source {source['id']}: {exc}")

    return documents


def build_documents_from_text_cache() -> list[dict]:
    if not TEXT_DIR.exists():
        raise FileNotFoundError("No cached text corpus found in rag_store/text.")

    documents: list[dict] = []
    for source in load_manifest():
        source_id = source["id"]
        source_type = source["type"]

        if source_type in {"local_markdown", "html", "pdf"}:
            text_path = TEXT_DIR / f"{source_id}.txt"
            if not text_path.exists():
                raise FileNotFoundError(f"Missing cached text file: {text_path}")
            documents.append(
                {
                    "id": source_id,
                    "title": source["title"],
                    "text": text_path.read_text(encoding="utf-8"),
                    "subject_tags": source.get("subject_tags", []),
                    "url": source.get("url"),
                    "path": source.get("path"),
                    "allowed_basis": source.get("allowed_basis"),
                }
            )
            continue

        if source_type == "ncert_listing":
            pattern = f"{source_id}_*.txt"
            chapter_paths = sorted(
                TEXT_DIR.glob(pattern),
                key=lambda path: int(path.stem.rsplit("_", 1)[-1]),
            )
            if not chapter_paths:
                raise FileNotFoundError(f"No cached chapter texts found for source {source_id}")

            for chapter_path in chapter_paths:
                chapter_id = chapter_path.stem.rsplit("_", 1)[-1]
                pdf_name = chapter_id_to_pdf_name(source["book_code"], chapter_id)
                documents.append(
                    {
                        "id": f"{source_id}_{chapter_id}",
                        "title": f'{source["title"]} Chapter {chapter_id}',
                        "text": chapter_path.read_text(encoding="utf-8"),
                        "subject_tags": source.get("subject_tags", []),
                        "url": f'https://ncert.nic.in/textbook/pdf/{pdf_name}',
                        "path": None,
                        "allowed_basis": source.get("allowed_basis"),
                    }
                )
            continue

        raise ValueError(f'Unsupported source type: {source_type}')

    return documents


def main() -> None:
    documents = build_documents_from_text_cache() if USE_TEXT_CACHE_ONLY else build_documents()
    vector_documents = build_vector_documents(documents)
    if VECTOR_BACKEND == "hashed":
        embeddings = [hash_text_to_vector(entry["text"]) for entry in vector_documents]
        vector_index = build_vector_index(vector_documents, embeddings, "hashed-bow-512")
        save_index(vector_index, VECTOR_INDEX_PATH)
        INDEX_PATH.write_text(json.dumps({"kind": "redirect", "target": "vector_index.json"}, indent=2), encoding="utf-8")
        print(f"Built hashed vector RAG index with {len(vector_index['documents'])} chunks at {VECTOR_INDEX_PATH}")
        return

    cache = load_embedding_cache()
    embeddings: list[list[float]] = []
    for start in range(0, len(vector_documents), EMBED_BATCH_SIZE):
        batch = vector_documents[start : start + EMBED_BATCH_SIZE]
        missing = [entry for entry in batch if entry["chunk_id"] not in cache]
        if missing:
            new_embeddings = ollama_embed([entry["text"] for entry in missing])
            for entry, embedding in zip(missing, new_embeddings):
                cache[entry["chunk_id"]] = embedding
            save_embedding_cache(cache)
            print(f"Embedded {min(start + len(batch), len(vector_documents))}/{len(vector_documents)} chunks")
        embeddings.extend([cache[entry["chunk_id"]] for entry in batch])
    vector_index = build_vector_index(vector_documents, embeddings, OLLAMA_EMBED_MODEL)
    save_index(vector_index, VECTOR_INDEX_PATH)
    INDEX_PATH.write_text(json.dumps({"kind": "redirect", "target": "vector_index.json"}, indent=2), encoding="utf-8")
    print(f"Built vector RAG index with {len(vector_index['documents'])} chunks at {VECTOR_INDEX_PATH}")


if __name__ == "__main__":
    main()
