import argparse
import json
import os
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils import embedding_functions

CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")


def chunk_text(text: str, chunk_size: int = 0, overlap: int = 120) -> list[str]:
    cleaned = " ".join(text.split()).strip()
    if not cleaned:
        return []
    if chunk_size <= 0 or len(cleaned) <= chunk_size:
        return [cleaned]

    step = max(1, chunk_size - overlap)
    chunks: list[str] = []
    for start in range(0, len(cleaned), step):
        part = cleaned[start : start + chunk_size].strip()
        if part:
            chunks.append(part)
        if start + chunk_size >= len(cleaned):
            break
    return chunks


def normalize_records(raw: dict[str, Any], idx: int, default_source: str, chunk_size: int, overlap: int) -> list[dict[str, Any]]:
    text = raw.get("text") or raw.get("content") or raw.get("body") or ""
    parts = chunk_text(str(text), chunk_size=chunk_size, overlap=overlap)
    if not parts:
        return []

    doc_id = str(raw.get("doc_id") or raw.get("id") or f"doc_{idx}").strip()
    title = str(raw.get("title") or raw.get("name") or doc_id).strip()[:200]
    source = str(raw.get("source") or default_source).strip()

    extra_meta: dict[str, Any] = {}
    for extra_key in ("severity", "category", "topic", "tag"):
        if raw.get(extra_key) is not None:
            extra_meta[extra_key] = str(raw[extra_key])

    total_parts = len(parts)
    records: list[dict[str, Any]] = []
    for i, part in enumerate(parts):
        rec_id = f"{source}_{doc_id}" if total_parts == 1 and chunk_size <= 0 else f"{source}_{doc_id}_c{i}"
        meta: dict[str, Any] = {
            "source": source,
            "doc_id": doc_id,
            "title": title,
            "chunk_index": i,
            "chunks_total": total_parts,
            **extra_meta,
        }
        records.append({
            "id": rec_id,
            "text": part,
            "metadata": meta,
        })
    return records


def load_records(input_path: Path, default_source: str, chunk_size: int, overlap: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    if input_path.is_file() and input_path.suffix.lower() == ".json":
        with input_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Expected JSON list of objects.")
        for idx, item in enumerate(data):
            records.extend(normalize_records(item, idx, default_source, chunk_size, overlap))
        return records

    if input_path.is_dir():
        files = sorted(list(input_path.rglob("*.md")) + list(input_path.rglob("*.txt")))
        for idx, file_path in enumerate(files):
            text = file_path.read_text(encoding="utf-8").strip()
            if not text:
                continue
            records.extend(
                normalize_records(
                    {
                        "doc_id": file_path.stem,
                        "title": file_path.stem,
                        "source": default_source,
                        "text": text,
                    },
                    idx,
                    default_source,
                    chunk_size,
                    overlap,
                )
            )
        return records

    raise FileNotFoundError(f"File or directory not found: {input_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--chunk-size", type=int, default=0)
    parser.add_argument("--overlap", type=int, default=120)
    parser.add_argument("--chroma-path", default=CHROMA_PATH)
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()

    docs = load_records(Path(args.input), args.source, args.chunk_size, args.overlap)
    if not docs:
        print("[!] No documents to index.")
        return

    client = chromadb.PersistentClient(path=args.chroma_path)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    collection = client.get_or_create_collection(
        name=args.collection,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    for i in range(0, len(docs), args.batch_size):
        batch = docs[i : i + args.batch_size]
        collection.upsert(
            ids=[d["id"] for d in batch],
            documents=[d["text"] for d in batch],
            metadatas=[d["metadata"] for d in batch],
        )
        if (i + len(batch)) % 500 == 0 or (i + len(batch)) == len(docs):
            print(f"[*] Upserted {i + len(batch)}/{len(docs)} chunks", flush=True)

    print(f"[+] Indexed {len(docs)} chunks into '{args.collection}' (total in collection: {collection.count()})")


if __name__ == "__main__":
    main()