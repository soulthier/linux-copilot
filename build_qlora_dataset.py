import argparse
import json
import os
import random
import re
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")

SYSTEM_PROMPT = "\n".join([
    "You are SecOps Copilot. Answer strictly using ONLY the provided Context.",
    "Rules:",
    "1. Rely ONLY on facts explicitly stated in the Context. Do not add external advice, assumptions, or general best practices absent from the Context.",
    "2. Do not repeat points or restate the same claim in different sections.",
    "3. Answer in the exact same language as the user's question (Russian or English). When answering in Russian, keep technical identifiers (IDs, commands, paths, and tool names) in English, and NEVER output Chinese characters.",
    "4. Structure the response clearly when supported by the Context without fabricating sections.",
    "5. If the Context does not contain enough information to answer the question, explicitly state that there is no information in the provided context.",
])


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def build_positive_sample(doc_id: str, meta: dict, text: str, collection_name: str, idx: int) -> dict | None:
    cleaned = clean_text(text)
    if len(cleaned) < 120 or CJK_RE.search(cleaned):
        return None

    title = clean_text(meta.get("title") or doc_id)
    source = clean_text(meta.get("source") or collection_name)
    excerpt = cleaned[:2200]
    context = f"[1] ({source} | {doc_id} | {title})\n{excerpt}"

    if idx % 4 == 0:
        question = f"Какие технические детали и идентификаторы указаны для {title} ({doc_id})?"
        answer = (
            f"На основе предоставленного контекста (`{source}`, `{doc_id}`):\n"
            f"- {excerpt[:1100]}"
        )
    elif idx % 2 == 0:
        question = f"What specific details, commands, or mitigations are stated for {title} ({doc_id})?"
        answer = (
            f"Based on the provided context (`{source}`, `{doc_id}`):\n"
            f"- {excerpt[:1100]}"
        )
    else:
        question = f"Summarize the grounded facts from {title} ({doc_id})."
        answer = (
            f"**Source / ID:** `{source}` / `{doc_id}`\n\n"
            f"**Grounded Facts:**\n"
            f"- {excerpt[:1100]}"
        )

    if CJK_RE.search(answer):
        return None

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"},
            {"role": "assistant", "content": answer.strip()},
        ]
    }


def build_negative_sample(doc_id: str, meta: dict, text: str, collection_name: str, idx: int) -> dict | None:
    cleaned = clean_text(text)
    if len(cleaned) < 120 or CJK_RE.search(cleaned):
        return None

    title = clean_text(meta.get("title") or doc_id)
    source = clean_text(meta.get("source") or collection_name)
    context = f"[1] ({source} | {doc_id} | {title})\n{cleaned[:1600]}"

    if idx % 2 == 0:
        question = "How to configure quantum warp drive calibration on an orbital station?"
        answer = "There is no information in the provided context about quantum warp drive calibration on an orbital station."
    else:
        question = "Как настроить квантовый варп-двигатель на орбитальной станции?"
        answer = "В предоставленном контексте нет информации о настройке квантового варп-двигателя на орбитальной станции."

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"},
            {"role": "assistant", "content": answer},
        ]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collections", nargs="+", default=["secops_knowledge", "linux_ops"])
    parser.add_argument("--max-per-collection", type=int, default=300)
    parser.add_argument("--neg-ratio", type=float, default=0.18)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/processed/qlora_train.jsonl")
    parser.add_argument("--chroma-path", default=CHROMA_PATH)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    client = chromadb.PersistentClient(path=args.chroma_path)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    samples = []
    for col_name in args.collections:
        try:
            col = client.get_collection(name=col_name, embedding_function=ef)
        except Exception:
            continue

        data = col.get(limit=args.max_per_collection, include=["documents", "metadatas"])
        ids = data.get("ids") or []
        docs = data.get("documents") or []
        metas = data.get("metadatas") or []

        col_count = 0
        for i, (doc_id, doc_text, meta) in enumerate(zip(ids, docs, metas)):
            did = str(doc_id)
            m = meta or {}
            txt = str(doc_text or "")

            pos = build_positive_sample(did, m, txt, col_name, i)
            if pos:
                samples.append(pos)
                col_count += 1

            if rng.random() < args.neg_ratio:
                neg = build_negative_sample(did, m, txt, col_name, i)
                if neg:
                    samples.append(neg)
                    col_count += 1

        print(f"[*] Collection '{col_name}': {col_count} samples", flush=True)

    rng.shuffle(samples)
    with out_path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"[+] Wrote {len(samples)} QLoRA chat samples -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
