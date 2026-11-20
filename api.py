import os
import re
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions
from fastapi import FastAPI
from pydantic import BaseModel, Field

CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_db")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "secops_knowledge")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
DEFAULT_MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "8000"))

app = FastAPI(title="SecOps Copilot RAG API", version="0.2.0")

client = chromadb.PersistentClient(path=CHROMA_PATH)
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=ef,
    metadata={"hnsw:space": "cosine"},
)

ID_PATTERN = re.compile(r"\b(CVE-\d{4}-\d{4,7}|T\d{4}(?:\.\d{3})?)\b", re.IGNORECASE)





class SearchRequest(BaseModel):
    query: str
    collection: Optional[str] = Field(default=None, description="Имя коллекции в ChromaDB (по умолчанию secops_knowledge)")
    top_k: int = Field(default=3, ge=1, le=20)
    source: Optional[str] = Field(default=None, description="Фильтр по полю metadata.source")
    min_similarity: float = Field(default=0.50, ge=0.0, le=1.0)
    max_context_chars: int = Field(default=DEFAULT_MAX_CONTEXT_CHARS, ge=500, le=60000)


def _apply_char_budget(items: list[dict], max_chars: int) -> tuple[list[dict], int]:
    selected: list[dict] = []
    used_chars = 0

    for item in items:
        text = item.get("text") or ""
        text_len = len(text)

        if selected and (used_chars + text_len > max_chars):
            continue

        if not selected and text_len > max_chars:
            trimmed_text = text[:max_chars]
            selected.append({**item, "text": trimmed_text})
            used_chars = len(trimmed_text)
            break

        selected.append(item)
        used_chars += text_len

    return selected, used_chars


@app.get("/health")
def health():
    return {
        "status": "ok",
        "collection": COLLECTION_NAME,
        "total_docs": collection.count(),
    }




@app.get("/collections")
def list_collections():
    cols = client.list_collections()
    return {
        "collections": [
            {"name": c.name, "count": c.count()}
            for c in cols
        ]
    }


@app.post("/search")
def search_docs(req: SearchRequest):
    items = []
    seen_ids = set()
    where_filter = {"source": req.source} if req.source else None

    target_collection_name = req.collection or COLLECTION_NAME
    active_collection = client.get_or_create_collection(
        name=target_collection_name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    # 1. Точный поиск по упоминаниям MITRE ID / CVE ID в запросе
    raw_ids = ID_PATTERN.findall(req.query)
    candidate_ids = []
    for raw_id in raw_ids:
        norm = raw_id.upper()
        if norm.startswith("T"):
            candidate_ids.extend([f"mitre_{norm}", norm])
        elif norm.startswith("CVE-"):
            candidate_ids.extend([f"nvd_{norm}", f"cve_{norm}", norm])

    if candidate_ids:
        exact_res = collection.get(ids=list(dict.fromkeys(candidate_ids)), where=where_filter)
        for doc_id, meta, text in zip(
            exact_res["ids"], exact_res["metadatas"], exact_res["documents"]
        ):
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                items.append(
                    {
                        "id": doc_id,
                        "title": meta.get("title"),
                        "source": meta.get("source"),
                        "similarity": 1.0,
                        "match_type": "exact_id",
                        "text": text,
                    }
                )

    # 2. Семантический векторный поиск (добираем до top_k с запасом под дедупликацию и порог)
    remaining = req.top_k - len(items)
    total_docs = collection.count()

    if remaining > 0 and total_docs > 0:
        fetch_k = min(req.top_k + len(seen_ids) + 3, total_docs)
        sem_res = collection.query(
            query_texts=[req.query],
            n_results=fetch_k,
            where=where_filter,
        )

        for doc_id, meta, dist, text in zip(
            sem_res["ids"][0],
            sem_res["metadatas"][0],
            sem_res["distances"][0],
            sem_res["documents"][0],
        ):
            if doc_id in seen_ids:
                continue
            sim = round(1 - dist, 4)
            if sim < req.min_similarity:
                continue
            seen_ids.add(doc_id)
            items.append(
                {
                    "id": doc_id,
                    "title": meta.get("title"),
                    "source": meta.get("source"),
                    "similarity": sim,
                    "match_type": "semantic",
                    "text": text,
                }
            )
            if len(items) >= req.top_k:
                break

    bounded_items, context_chars = _apply_char_budget(items[: req.top_k], req.max_context_chars)

    return {
        "query": req.query,
        "source": req.source,
        "count": len(bounded_items),
        "context_chars": context_chars,
        "results": bounded_items,
    }