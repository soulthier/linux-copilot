# Linux CLI & Docs Copilot — RAG + QLoRA Fine-Tuning

Grounded QA system for Linux commands and manuals using ChromaDB retrieval, FastAPI, QLoRA-adapted Qwen2.5-7B, and local Ollama inference.
Retrieval-Augmented Generation (RAG) assistant for SecOps, Linux administration, and MITRE ATT&CK queries, paired with a 4-bit **QLoRA** fine-tuning pipeline for **Qwen2.5-7B-Instruct** and an automated evaluation harness.

## Architecture

```text
User Query
   │
   ▼
FastAPI (/search) ──▶ ChromaDB (secops_knowledge, linux_ops)
   │
   ▼
Context Builder (similarity threshold, char budget, source/ID tags)
   │
   ▼
Qwen2.5-7B-Instruct + QLoRA Adapter (v2)
   │
   ▼
Grounded Answer or Explicit Refusal (RU / EN, no CJK leaks)
```

## Tech Stack

- **Retrieval & API:** Python, FastAPI (`uvicorn`), ChromaDB, SentenceTransformers (`BAAI/bge-small-en-v1.5`)
- **Fine-Tuning (QLoRA):** PyTorch, `transformers`, `peft`, `trl` (`SFTTrainer` / `SFTConfig`), `bitsandbytes` (`nf4`, `paged_adamw_8bit`), gradient checkpointing
- **Evaluation:** Custom RAG evaluation suite (`eval_cases.json`) measuring `refusal_accuracy`, `avg_term_recall`, `cjk_leaks`, and `avg_latency_s`

## Project Structure

```text
secops-copilot/
├── api.py                      # FastAPI service exposing /search
├── main.py                     # CLI / interactive entrypoint
├── fetch.py                    # Fetches raw SecOps & Linux documentation sources
├── ing.py                      # Chunks and ingests documents into ChromaDB
├── ragsearch.py                # Retrieval helper & similarity filtering
├── build_qlora_dataset.py      # Builds positive & negative chat samples in JSONL
├── train_qlora.py              # 4-bit QLoRA training script for Qwen2.5-7B-Instruct
├── eval_rag.py                 # Evaluates baseline RAG pipeline
├── eval_lora_hf.py             # Evaluates base + LoRA adapter against /search & eval_cases.json
├── eval_cases.json             # Benchmark prompts (SecOps, Linux, RU/EN, unanswerable)
├── dockerfile                  # Container image definition
├── docker-compose.yml          # Multi-service orchestration
└── data/processed/             # Processed MITRE ATT&CK & CVE corpora
```

## Training & Evaluation Iterations

Local training on an **RTX 3060 (12 GB VRAM)** proved tight for longer sequence lengths without heavy memory trade-offs, so the second training iteration (`v2`) was run in **Google Colab (L4 GPU)** on a balanced dataset with negative refusal examples and bilingual prompts.

| Run | Setup | `refusal_accuracy` | `avg_term_recall` | `cjk_leaks` | `avg_latency_s` | Notes |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Baseline** | `qwen2.5:7b` (Ollama) | `1.0` | `1.0` | `0` | `5.01s` | Strong baseline on short evaluation set |
| **QLoRA v1** | Local QLoRA (positive-only data) | `0.0` | `0.6667` | `0` | `41.28s` | Failed to refuse out-of-context queries; template bias |
| **QLoRA v2** | Colab L4 QLoRA (neg. + RU/EN) | `1.0` | `0.6667` | `0` | `7.33s` | Restored refusals (`1.0`), fixed RU refusal/language behavior |

### Key Findings from Evaluation

- **Refusal behavior (`0.0 -> 1.0`):** Adding negative samples (`no information in the provided context`) eliminated forced answers on out-of-scope prompts such as `unanswerable_01`.
- **Language consistency:** Including Russian and English prompts in `v2` fixed Russian refusal/response behavior while keeping technical identifiers in English and zero CJK leaks.
- **Retrieval vs. generation on `term_recall`:** Remaining `term_recall` misses (such as `sec_01` and `linux_01`) trace back to retrieved chunks not containing the exact expected term/detail rather than ungrounded fabrication by `v2`.

## Quickstart

### 1. Start the Retrieval API

```bash
cd ~/secops-copilot
source .venv/bin/activate
uvicorn api:app --host 0.0.0.0 --port 8000
```

### 2. Build Dataset & Train QLoRA (v2)

```bash
python build_qlora_dataset.py \
  --collections secops_knowledge linux_ops \
  --max-per-collection 300 \
  --neg-ratio 0.18 \
  --out data/processed/qlora_train.jsonl

python train_qlora.py \
  --data data/processed/qlora_train.jsonl \
  --out outputs/qwen25_7b_secops_lora_v2 \
  --max-seq-len 1024 \
  --epochs 1.5 \
  --batch-size 2 \
  --grad-accum 4
```

### 3. Run Evaluation

```bash
python eval_lora_hf.py \
  --adapter-dir outputs/qwen25_7b_secops_lora_v2 \
  --cases eval_cases.json \
  --out eval_report_lora_v2.json
```

## Planned Improvements

- Merge the `v2` LoRA adapter into base weights and export to GGUF (`Q4_K_M`) for faster local inference via Ollama.
- Improve chunking, `top_k`, and similarity filtering in `/search` to boost `term_recall` on multi-fact SecOps and Linux queries.
- Expand `eval_cases.json` with more edge cases and cross-lingual checks.
## Local Inference via Ollama

After merging the QLoRA adapter and quantizing to GGUF (`Q4_K_M`), register the model locally with the included `Modelfile`:

```bash
ollama create secops-copilot:v2 -f Modelfile
ollama run secops-copilot:v2
