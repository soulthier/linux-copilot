# Linux CLI & Docs Copilot — RAG + QLoRA Fine-Tuning

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-linux--docs--qwen25--7b--gguf-yellow)](https://huggingface.co/soulthier/linux-docs-qwen25-7b-gguf)
[![Base Model](https://img.shields.io/badge/Model-Qwen2.5--7B--Instruct-blue)](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
[![Inference](https://img.shields.io/badge/Inference-Ollama%20%28Q4__K__M%29-black)](https://ollama.com/)
[![Vector DB](https://img.shields.io/badge/Vector%20DB-ChromaDB-orange)](https://www.trychroma.com/)
[![Backend](https://img.shields.io/badge/API-FastAPI-009688)](https://fastapi.tiangolo.com/)

Grounded QA assistant for Linux commands, CLI utilities, and manual pages using **ChromaDB** retrieval, **FastAPI**, a **QLoRA-adapted `Qwen2.5-7B-Instruct`** model, and local **Ollama** inference.

---

## Architecture

```text
Linux Man Pages / CLI Docs
        │
        ▼
  Chunking & Ingestion (ing.py) ──► ChromaDB Vector Store
                                           │
User Query (EN / RU) ──► FastAPI (/search & /ask)
                                           │
                                           ▼
                      Retrieved Context + Strict Grounding Prompt
                                           │
                                           ▼
                Qwen2.5-7B-Instruct (QLoRA v2, Q4_K_M GGUF via Ollama)
                                           │
                                           ▼
                  Grounded Answer (with citations) OR Explicit Refusal
```

---

## Key Features

- **Strict Context Grounding:** Fine-tuned on both answerable and unanswerable (negative) examples so the model relies strictly on retrieved manual pages and avoids fabricating flags or options.
- **Bilingual Support (EN / RU):** Responds in the exact language of the user's query while keeping Linux commands, CLI flags, file paths, and exit codes in English.
- **Local Quantized Inference:** Merged QLoRA weights quantized to `Q4_K_M` GGUF (~4.4 GB) for fast offline inference via Ollama.
- **Automated RAG Evaluation:** Includes evaluation pipelines (`eval_rag.py`, `eval_lora_hf.py`) to verify citation grounding, language consistency, and refusal accuracy.

---

## Pretrained Model Weights (Hugging Face)

The quantized `Q4_K_M` GGUF weights and `Modelfile` are hosted on Hugging Face Hub:

- **Model Repository:** [soulthier/linux-docs-qwen25-7b-gguf](https://huggingface.co/soulthier/linux-docs-qwen25-7b-gguf)
- **Base Model:** `Qwen/Qwen2.5-7B-Instruct`
- **Quantization:** `Q4_K_M` GGUF (~4.4 GB)

---

## Quickstart (Local Inference via Ollama)

### 1. Clone the repository and install dependencies

```bash
git clone https://github.com/soulthier/linux-copilot.git
cd linux-copilot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download the GGUF model from Hugging Face and register in Ollama

```bash
mkdir -p outputs
hf download soulthier/linux-docs-qwen25-7b-gguf linux-docs-qwen25-7b-v2-Q4_K_M.gguf --local-dir ./outputs
ollama create linux-copilot -f Modelfile
ollama run linux-copilot
```

### 3. Build the vector index and start the FastAPI server

```bash
python ing.py
uvicorn main:app --reload
```

Once running, interactive API docs are available at `http://127.0.0.1:8000/docs`.

---

## Project Structure

```text
linux-copilot/
├── fetch.py                 # Collects and prepares Linux manual pages & documentation
├── ing.py                   # Chunks documents and ingests embeddings into ChromaDB
├── ragsearch.py             # Vector similarity search and context formatting
├── main.py                  # FastAPI application (/search, /ask endpoints)
├── build_qlora_dataset.py   # Builds grounded QA + refusal SFT dataset
├── train_qlora.py           # QLoRA fine-tuning pipeline for Qwen2.5-7B-Instruct
├── eval_lora_hf.py          # Evaluates fine-tuned Hugging Face adapter checkpoints
├── eval_rag.py              # End-to-end RAG evaluation (grounding & refusal metrics)
├── Modelfile                # Ollama model definition with strict grounding system prompt
└── README.md                # Project documentation
```

---

## Example Behavior

### Grounded Answer (Context Present)
> **Query:** Which flag in `tar` extracts an archive into a specific target directory?  
> **Response:** According to the retrieved `tar` documentation, the `-C` (or `--directory=DIR`) option changes to the specified directory before performing operations, allowing extraction into a target directory (`tar -xf archive.tar -C /target/dir`).

### Explicit Refusal (Context Missing)
> **Query:** What flag enables hardware-accelerated Zstandard GPU compression in `tar`?  
> **Response:** There is no information about GPU-accelerated Zstandard compression flags in the provided context.