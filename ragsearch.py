import json
import os
import re
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils import embedding_functions

BASE_DIR = Path(os.getenv("SECOPS_BASE_DIR", Path(__file__).resolve().parent))
CHROMA_PATH = os.getenv("CHROMA_PATH", str(BASE_DIR / "chroma_db"))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "secops_knowledge")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

DEFAULT_SOURCES = [
    {
        "source": "mitre_attack",
        "path": str(BASE_DIR / "data/processed/mitre_docs.json"),
        "id_prefix": "mitre_",
        "default_severity": None,
    },
    {
        "source": "nvd_cve",
        "path": str(BASE_DIR / "data/processed/cve_docs.json"),
        "id_prefix": "cve_",
        "default_severity": "UNKNOWN",
    },
]

ID_PATTERN = re.compile(r"\b(CVE-\d{4}-\d{4,7}|T\d{4}(?:\.\d{3})?)\b", re.IGNORECASE)


def get_sources_config() -> list[dict[str, Any]]:
    raw = os.getenv("SOURCES_CONFIG_JSON")
    if not raw:
        return DEFAULT_SOURCES
    parsed = json.loads(raw)
    if not isinstance(parsed, list):
        raise ValueError("SOURCES_CONFIG_JSON должен быть JSON-списком объектов источников.")
    return parsed


def load_all_documents(sources_config: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    configs = sources_config if sources_config is not None else get_sources_config()
    docs: list[dict[str, Any]] = []

    for cfg in configs:
        path = Path(cfg["path"])
        if not path.is_absolute():
            path = BASE_DIR / path
        if not path.exists():
            continue

        source_name = cfg["source"]
        id_prefix = cfg.get("id_prefix", f"{source_name}_")
        default_severity = cfg.get("default_severity")

        with path.open("r", encoding="utf-8") as f:
            items = json.load(f)

        for item in items:
            raw_doc_id = str(item["doc_id"])
            metadata: dict[str, Any] = {
                "source": source_name,
                "doc_id": raw_doc_id,
                "title": str(item.get("title", raw_doc_id))[:200],
            }
            severity = item.get("severity", default_severity)
            if severity is not None:
                metadata["severity"] = str(severity)

            docs.append({
                "id": f"{id_prefix}{raw_doc_id}",
                "text": str(item["text"]),
                "metadata": metadata,
            })

    return docs


class SecOpsRAG:
    def __init__(
        self,
        chroma_path: str = CHROMA_PATH,
        collection_name: str = COLLECTION_NAME,
        embedding_model: str = EMBEDDING_MODEL,
    ) -> None:
        self.chroma_path = chroma_path
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.client = chromadb.PersistentClient(path=self.chroma_path)
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model
        )
        self.collection = self._init_collection()

    def _init_collection(self):
        collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"},
        )
        if collection.count() == 0:
            self.reindex(only_if_empty=True)
        return collection

    def reindex(self, only_if_empty: bool = False, sources_config: list[dict[str, Any]] | None = None) -> int:
        if only_if_empty and self.collection.count() > 0:
            return self.collection.count()

        docs = load_all_documents(sources_config)
        batch_size = 100
        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            self.collection.upsert(
                ids=[d["id"] for d in batch],
                documents=[d["text"] for d in batch],
                metadatas=[d["metadata"] for d in batch],
            )
        return self.collection.count()

    def _build_where(self, source: str | None) -> dict[str, Any] | None:
        if source:
            return {"source": source}
        return None

    def _exact_id_matches(self, query: str, source: str | None) -> list[dict[str, Any]]:
        matched_ids = {m.upper() for m in ID_PATTERN.findall(query)}
        if not matched_ids:
            return []

        where_filter = self._build_where(source)
        found: list[dict[str, Any]] = []
        seen_doc_ids: set[str] = set()

        for target_id in matched_ids:
            res = self.collection.get(
                where={"$and": [{"doc_id": target_id}, where_filter]} if where_filter else {"doc_id": target_id},
                include=["documents", "metadatas"],
            )
            for doc_id, meta, text in zip(res.get("ids", []), res.get("metadatas", []), res.get("documents", [])):
                if doc_id in seen_doc_ids:
                    continue
                seen_doc_ids.add(doc_id)
                found.append({
                    "id": doc_id,
                    "doc_id": meta.get("doc_id", doc_id),
                    "source": meta.get("source", "unknown"),
                    "title": meta.get("title", ""),
                    "severity": meta.get("severity"),
                    "similarity": 1.0,
                    "match_type": "exact_id",
                    "text": text,
                })
        return found

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.35,
        source: str | None = None,
        max_context_chars: int = 8000,
    ) -> dict[str, Any]:
        where_filter = self._build_where(source)
        exact_hits = self._exact_id_matches(query, source)

        n_results = max(top_k * 2, top_k)
        query_kwargs: dict[str, Any] = {
            "query_texts": [query],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where_filter:
            query_kwargs["where"] = where_filter

        raw = self.collection.query(**query_kwargs)
        ids = raw["ids"][0] if raw.get("ids") else []
        metas = raw["metadatas"][0] if raw.get("metadatas") else []
        dists = raw["distances"][0] if raw.get("distances") else []
        texts = raw["documents"][0] if raw.get("documents") else []

        vector_hits: list[dict[str, Any]] = []
        for doc_id, meta, dist, text in zip(ids, metas, dists, texts):
            similarity = round(1.0 - float(dist), 4)
            if similarity < min_similarity:
                continue
            vector_hits.append({
                "id": doc_id,
                "doc_id": meta.get("doc_id", doc_id),
                "source": meta.get("source", "unknown"),
                "title": meta.get("title", ""),
                "severity": meta.get("severity"),
                "similarity": similarity,
                "match_type": "vector",
                "text": text,
            })

        # Объединяем точные совпадения по ID и векторные результаты без дублей
        merged: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for item in exact_hits + vector_hits:
            if item["id"] in seen_ids:
                continue
            seen_ids.add(item["id"])
            merged.append(item)
            if len(merged) >= top_k:
                break

        # Ограничиваем суммарный объём контекста по символам
        selected: list[dict[str, Any]] = []
        total_chars = 0
        for item in merged:
            text_len = len(item["text"])
            if selected and (total_chars + text_len > max_context_chars):
                continue
            if not selected and text_len > max_context_chars:
                item = {**item, "text": item["text"][:max_context_chars]}
                text_len = len(item["text"])
            selected.append(item)
            total_chars += text_len

        return {
            "query": query,
            "source_filter": source,
            "total_documents_in_collection": self.collection.count(),
            "returned_count": len(selected),
            "context_chars": total_chars,
            "results": selected,
        }


if __name__ == "__main__":
    engine = SecOpsRAG()
    print(f"Документов в коллекции '{engine.collection_name}': {engine.collection.count()}")
    for q, src in [
        ("attacker executes arbitrary commands remotely on server", None),
        ("how to mitigate credential dumping from Windows LSASS memory", "mitre_attack"),
    ]:
        out = engine.search(q, top_k=3, min_similarity=0.25, source=src)
        print(f"\n=== Запрос: '{q}' | source={src} ===")
        for rank, r in enumerate(out["results"], 1):
            print(f"#{rank} [{r['match_type']}] {r['title']} ({r['source']}, {r['doc_id']}) | sim={r['similarity']}")
            print("   " + r["text"][:220].replace("\n", " ") + "...")