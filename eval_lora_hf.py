import argparse
import json
import re
import statistics
import time
from pathlib import Path
import requests
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DEFAULT_SEARCH_URL = "http://localhost:8000/search"

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter-dir", default="outputs/qwen25_7b_secops_lora")
    parser.add_argument("--cases", default="eval_cases.json")
    parser.add_argument("--out", default="eval_report_lora.json")
    parser.add_argument("--search-url", default=DEFAULT_SEARCH_URL)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    torch.cuda.set_device(0)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map={"": 0},
        dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()

    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    rows = []

    for case in cases:
        t0 = time.perf_counter()
        s_resp = requests.post(
            args.search_url,
            json={
                "query": case["query"],
                "collection": case.get("collection"),
                "source": case.get("source"),
                "top_k": case.get("top_k", 3),
                "min_similarity": case.get("min_similarity", 0.35),
                "max_context_chars": case.get("max_context_chars", 7000),
            },
            timeout=60,
        )
        s_resp.raise_for_status()
        results = s_resp.json().get("results", [])

        chunks = []
        total = 0
        max_chars = int(case.get("max_context_chars", 7000))
        for idx, r in enumerate(results, start=1):
            txt = str(r.get("text", "")).strip()
            if not txt:
                continue
            block = f"[{idx}] ({r.get('source', 'unknown')} | {r.get('id', 'N/A')} | sim={r.get('similarity', 'N/A')})\n{txt}"
            if chunks and total + len(block) + 6 > max_chars:
                break
            chunks.append(block)
            total += len(block) + 6

        formatted_context = "\n\n---\n\n".join(chunks) if chunks else "No relevant documents found in the knowledge base."

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{formatted_context}\n\nQuestion:\n{case['query']}"},
        ]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda:0")

        with torch.inference_mode():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.15,
                repetition_penalty=1.15,
                pad_token_id=tokenizer.pad_token_id,
            )

        gen_ids = out_ids[0][inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        latency_s = round(time.perf_counter() - t0, 2)

        expected_terms = [str(t).lower() for t in case.get("expected_terms", [])]
        matched_terms = [t for t in expected_terms if t in answer.lower()]
        term_recall = round(len(matched_terms) / len(expected_terms), 4) if expected_terms else None

        has_cjk = bool(CJK_RE.search(answer))
        has_refusal = bool(REFUSAL_RE.search(answer))
        expect_refusal = bool(case.get("expect_empty_or_refusal", False))
        refusal_ok = (has_refusal or len(results) == 0) if expect_refusal else None

        rows.append({
            "id": case.get("id", ""),
            "collection": case.get("collection"),
            "retrieved_count": len(results),
            "context_chars": len(formatted_context),
            "term_recall": term_recall,
            "matched_terms": matched_terms,
            "has_cjk": has_cjk,
            "refusal_ok": refusal_ok,
            "latency_s": latency_s,
            "answer": answer,
        })

    term_checks = [r["term_recall"] for r in rows if r["term_recall"] is not None]
    refusal_checks = [1.0 if r["refusal_ok"] else 0.0 for r in rows if r["refusal_ok"] is not None]

    summary = {
        "model": f"{args.base_model} + {args.adapter_dir}",
        "total_cases": len(rows),
        "empty_retrievals": sum(1 for r in rows if r["retrieved_count"] == 0),
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
