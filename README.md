# Linux CLI & Docs Copilot — RAG + QLoRA Fine-Tuning

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-linux--docs--qwen25--7b--gguf-yellow)](https://huggingface.co/soulthier/linux-docs-qwen25-7b-gguf)
[![Base Model](https://img.shields.io/badge/Model-Qwen2.5--7B--Instruct-blue)](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
[![Inference](https://img.shields.io/badge/Inference-Ollama%20%28Q4__K__M%29-black)](https://ollama.com/)
[![Vector DB](https://img.shields.io/badge/Vector%20DB-ChromaDB-orange)](https://www.trychroma.com/)
[![Backend](https://img.shields.io/badge/API-FastAPI-009688)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB)](https://www.python.org/)

Grounded QA assistant for Linux commands, CLI utilities, and manual pages (`man` pages) powered by **ChromaDB** semantic retrieval, **FastAPI**, a **QLoRA fine-tuned `Qwen2.5-7B-Instruct`** model, and local **Ollama** inference.

---

## Overview & Motivation

General-purpose LLMs frequently hallucinate Linux CLI flags, mix up options across GNU/BSD utilities, or guess answers when documentation chunks lack the requested detail. **Linux CLI & Docs Copilot** solves this through two complementary layers:

1. **Retrieval-Augmented Generation (RAG):** Parses, chunks, and indexes Linux manual pages and system documentation into **ChromaDB**, retrieving the most relevant context blocks per query via **FastAPI**.
2. **Task-Specific QLoRA Fine-Tuning (`v1` $\rightarrow$ `v2`):** Adapts `Qwen2.5-7B-Instruct` on curated positive (grounded QA with citations) and negative (unanswerable / out-of-context refusal) examples, enforcing strict context adherence and bilingual consistency (English & Russian).

---

## System Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                        1. DATA & INGESTION PIPELINE                         │
│                                                                             │
│  Linux Man Pages / Docs (fetch.py)                                          │
│          │                                                                  │
│          ▼                                                                  │
│  Text Cleaning & Chunking (ing.py) ──► Embeddings ──► ChromaDB Vector Store │
└─────────────────────────────────────────────────────────────────────────────┘
                                                               │
                                                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        2. ONLINE RAG INFERENCE PIPELINE                     │
│                                                                             │
│  User Query (EN / RU) ──► FastAPI (/search & /ask in main.py)               │
│                                  │                                          │
│                                  ▼                                          │
│                   Top-K Context Retrieval (ragsearch.py)                    │
│                                  │                                          │
│                                  ▼                                          │
│            Strict Grounding System Prompt + Retrieved Context               │
│                                  │                                          │
│                                  ▼                                          │
│         Qwen2.5-7B-Instruct (QLoRA v2, Q4_K_M GGUF via Ollama)              │
│                                  │                                          │
│                    ┌─────────────┴─────────────┐                            │
│                    ▼                           ▼                            │
│     Grounded Answer (Flags & Citations)   Explicit Refusal (No Context)     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Fine-Tuning & Iteration Progression

The model went through iterative dataset refinement and evaluation (`eval_lora_hf.py` & `eval_rag.py`) to balance helpful extraction with strict refusal discipline:

* **Baseline (`Qwen2.5-7B-Instruct` + RAG):** Strong general Linux knowledge, but prone to leaking parametric pretraining facts when the retrieved man-page snippet omits a flag, occasionally mixing languages or producing repetitive explanations.
* **QLoRA `v1` (Initial Grounded SFT):** Introduced strict context-only instructions and structured answers. Significantly reduced external assumptions, though borderline out-of-context queries still triggered occasional partial guesses.
* **QLoRA `v2` (Refusal-Hardened + Bilingual Calibration $\rightarrow$ `Q4_K_M` GGUF):** Expanded negative/unanswerable training pairs, tightened anti-repetition and language-locking rules (preserving English CLI flags/paths inside Russian answers), merged LoRA adapters, and quantized to `Q4_K_M` GGUF (~4.4 GB) for low-latency local execution in Ollama.

---

## Evaluation & Comparison

Evaluated across answerable Linux documentation queries and unanswerable/out-of-context edge cases using `eval_rag.py` and `eval_lora_hf.py`:

| Stage / Configuration | Grounded Answer Accuracy | Unanswerable Refusal Accuracy | Hallucinated Flags / External Leakage | Bilingual & Format Compliance | Deployment Format |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Baseline `Qwen2.5-7B-Instruct` (Zero-Shot RAG)** | _TBD_ | _TBD_ | High (relies on pretraining priors) | Moderate (occasional repetition / drift) | FP16 / Ollama base |
| **QLoRA `v1` (Grounded QA SFT)** | _TBD_ | _TBD_ | Low–Moderate | High | LoRA Adapter (`PEFT`) |
| **QLoRA `v2` (Refusal-Hardened + Bilingual)** | _TBD_ | _TBD_ | **Minimal (strictly context-bound)** | **Strict (EN/RU + exact CLI tokens)** | **`Q4_K_M` GGUF (~4.4 GB)** |

> **Note:** Replace the `_TBD_` cells above with your exact percentages from the latest `eval_rag.py` run output.

### Key Empirical Takeaways
1. **Zero-Guessing Refusal Behavior:** Training explicitly on adversarial/missing-context prompts teaches the model to refuse cleanly rather than invent non-existent CLI flags.
2. **Identifier Preservation in Bilingual Mode:** Russian responses consistently retain exact command names, flags (`-C`, `--exclude`, `-xzvf`), exit codes, and file paths in English without character corruption.
3. **Lightweight Local Execution:** Quantizing the merged `v2` checkpoint to `Q4_K_M` GGUF preserves `v2` grounding quality while running locally via Ollama on a single consumer GPU or CPU.

---

## Pretrained Model Weights (Hugging Face)

The quantized `Q4_K_M` GGUF checkpoint and `Modelfile` are published on Hugging Face Hub:

* **Hugging Face Repository:** [soulthier/linux-docs-qwen25-7b-gguf](https://huggingface.co/soulthier/linux-docs-qwen25-7b-gguf)
* **Artifact:** `linux-docs-qwen25-7b-v2-Q4_K_M.gguf` (~4.4 GB)
* **Base Architecture:** `Qwen/Qwen2.5-7B-Instruct`
* **Inference Runtime:** Ollama (`temperature=0.1`, `top_p=0.9`, `num_ctx=4096`)

---

## Quickstart (Local Setup)

### 1. Clone the repository and set up the environment

```bash
git clone https://github.com/soulthier/linux-copilot.git
cd linux-copilot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download the GGUF weights from Hugging Face and register in Ollama

```bash
mkdir -p outputs
hf download soulthier/linux-docs-qwen25-7b-gguf linux-docs-qwen25-7b-v2-Q4_K_M.gguf --local-dir ./outputs
ollama create linux-copilot -f Modelfile
ollama run linux-copilot
```

### 3. Ingest Linux documentation into ChromaDB

```bash
python fetch.py
python ing.py
```

### 4. Start the FastAPI backend

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive Swagger documentation is available at `http://127.0.0.1:8000/docs`.

---

## API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Service and vector store health check |
| `POST` | `/search` | Retrieves top-k matching Linux documentation chunks from ChromaDB |
| `POST` | `/ask` | Executes full RAG pipeline (retrieval + grounded QLoRA LLM generation) |

---

## Example Outputs

### 1. Grounded Answer (English)
> **Query:** Which flag in `tar` changes to a specific target directory before extracting files?  
> **Response:** Based on the provided `tar` manual context, use the `-C` (or `--directory=DIR`) option to change to the specified directory before performing operations (e.g., `tar -xf archive.tar -C /target/dir`).

### 2. Grounded Answer (Russian with Preserved CLI Tokens)
> **Query:** Какой флаг в `grep` выводит номера строк вместе с найденными совпадениями?  
> **Response:** Согласно предоставленной документации `grep`, для вывода номера строки перед каждой строкой с совпадением используется флаг `-n` (или `--line-number`).

### 3. Explicit Refusal (Missing Context)
> **Query:** What flag enables hardware-accelerated Zstandard GPU compression in `tar`?  
> **Response:** There is no information in the provided context about GPU-accelerated Zstandard compression flags for `tar`.

---

## Repository Structure

```text
linux-copilot/
├── fetch.py                 # Collects and normalizes Linux manual pages & docs
├── ing.py                   # Chunks documents and ingests embeddings into ChromaDB
├── ragsearch.py             # Vector similarity search and context assembly
├── main.py                  # FastAPI server exposing /search and /ask
├── build_qlora_dataset.py   # Constructs grounded QA & refusal SFT dataset
├── train_qlora.py           # QLoRA fine-tuning script for Qwen2.5-7B-Instruct
├── eval_lora_hf.py          # Evaluates Hugging Face LoRA checkpoints
├── eval_rag.py              # End-to-end RAG evaluation (accuracy, refusals, grounding)
├── Modelfile                # Ollama configuration with strict grounding system prompt
├── requirements.txt         # Python dependencies
└── README.md                # Full project documentation
```

---

## License

This project is intended for educational and engineering portfolio purposes. Base model weights are governed by the [Qwen2.5 License](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct).