import argparse
import json
import re
import statistics
import time
from pathlib import Path
import requests

DEFAULT_SEARCH_URL = "http://localhost:8000/search"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen2.5:7b"

SYSTEM_PROMPT = "\n".join([
    "You are SecOps Copilot. Answer strictly using ONLY the provided Context.",
    "Rules:",
    "1. Rely ONLY on facts explicitly stated in the Context. Do not add external advice, assumptions, or general best practices absent from the Context.",
    "2. Do not repeat points or restate the same claim in different sections.",
    "3. Answer in the exact same language as the user's question (Russian or English). When answering in Russian, keep technical identifiers (IDs, commands, paths, and tool names) in English, and NEVER output Chinese characters.",
    "4. Structure the response clearly when supported by the Context without fabricating sections.",
    "5. If the Context does not contain enough information to answer the question, explicitly state that there is no information in the provided context.",
])

CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
REFUSAL_RE = re.compile(r"(no information|not mentioned|cannot answer|insufficient context|нет информации|не содержит достаточно)", re.IGNORECASE)





def evaluate_case(case: dict, search_url: str, ollama_url: str, model: str) -> dict:
    t0 = time.perf_counter()

    payload_search = {
        "query": case["query"],
        "collection": case.get("collection"),
        "source": case.get("source"),
        "top_k": case.get("top_k", 3),
        "min_similarity": case.get("min_similarity", 0.35),
        "max_context_chars": case.get("max_context_chars", 7000),
    }

    s_resp = requests.post(search_url, json=payload_search, timeout=60)
    s_resp.raise_for_status()
    s_data = s_resp.json()
    results = s_data.get("results", [])

    retrieved_ids = [str(r.get("id", "")) for r in results]
    similarities = [float(r["similarity"]) for r in results if isinstance(r.get("similarity"), (int, float))]
    max_sim = max(similarities) if similarities else 0.0
    mean_sim = round(statistics.mean(similarities), 4) if similarities else 0.0

    expected_ids = [str(x) for x in case.get("expected_ids", [])]
    id_hit = any(any(exp_id in rid for rid in retrieved_ids) for exp_id in expected_ids) if expected_ids else None

    context_chunks = []
    total_chars = 0
    max_chars = int(case.get("max_context_chars", 7000))
    for idx, r in enumerate(results, start=1):
        txt = str(r.get("text", "")).strip()
        if not txt:
            continue
        block = f"[{idx}] ({r.get('source', 'unknown')} | {r.get('id', 'N/A')} | sim={r.get('similarity', 'N/A')})\n{txt}"
        if context_chunks and total_chars + len(block) + 6 > max_chars:
            break
        context_chunks.append(block)
        total_chars += len(block) + 6

    formatted_context = "\n\n---\n\n".join(context_chunks) if context_chunks else "No relevant documents found in the knowledge base."

    g_resp = requests.post(
        ollama_url,
        json={
            "model": model,
            "stream": False,
            "options": {"temperature": 0.15, "repeat_penalty": 1.15},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{formatted_context}\n\nQuestion:\n{case['query']}"},
            ],
        },
        timeout=120,
    )
    g_resp.raise_for_status()
    answer = g_resp.json().get("message", {}).get("content", "").strip()
    latency_s = round(time.perf_counter() - t0, 2)

    expected_terms = [str(t).lower() for t in case.get("expected_terms", [])]
    matched_terms = [t for t in expected_terms if t in answer.lower()]
    term_recall = round(len(matched_terms) / len(expected_terms), 4) if expected_terms else None

    has_cjk = bool(CJK_RE.search(answer))
    has_refusal = bool(REFUSAL_RE.search(answer))
    expect_refusal = bool(case.get("expect_empty_or_refusal", False))
    refusal_ok = (has_refusal or len(results) == 0) if expect_refusal else None

    return {
        "id": case.get("id", ""),
        "collection": case.get("collection"),
        "retrieved_count": len(results),
        "context_chars": len(formatted_context),
        "max_similarity": round(max_sim, 4),
        "mean_similarity": mean_sim,
        "id_hit": id_hit,
        "term_recall": term_recall,
        "matched_terms": matched_terms,
        "has_cjk": has_cjk,
        "refusal_ok": refusal_ok,
        "latency_s": latency_s,
        "answer": answer,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="eval_cases.json")
    parser.add_argument("--out", default="eval_report_baseline.json")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--search-url", default=DEFAULT_SEARCH_URL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    args = parser.parse_args()

    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    rows = [evaluate_case(c, args.search_url, args.ollama_url, args.model) for c in cases]

    id_checks = [1.0 if r["id_hit"] else 0.0 for r in rows if r["id_hit"] is not None]
    term_checks = [r["term_recall"] for r in rows if r["term_recall"] is not None]
    refusal_checks = [1.0 if r["refusal_ok"] else 0.0 for r in rows if r["refusal_ok"] is not None]

    summary = {
        "model": args.model,
        "total_cases": len(rows),
        "empty_retrievals": sum(1 for r in rows if r["retrieved_count"] == 0),
        "avg_retrieved_count": round(statistics.mean(r["retrieved_count"] for r in rows), 2) if rows else 0.0,
        "avg_max_similarity": round(statistics.mean(r["max_similarity"] for r in rows), 4) if rows else 0.0,
        "id_hit_rate": round(statistics.mean(id_checks), 4) if id_checks else None,
        "avg_term_recall": round(statistics.mean(term_checks), 4) if term_checks else None,
        "refusal_accuracy": round(statistics.mean(refusal_checks), 4) if refusal_checks else None,
        "cjk_leaks": sum(1 for r in rows if r["has_cjk"]),
        "avg_latency_s": round(statistics.mean(r["latency_s"] for r in rows), 2) if rows else 0.0,
        "results": rows,
    }

    Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
