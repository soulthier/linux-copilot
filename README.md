# Linux CLI & Docs Copilot — RAG + QLoRA Fine-Tuning

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-linux--docs--qwen25--7b--gguf-yellow)](https://huggingface.co/soulthier/linux-docs-qwen25-7b-gguf)
[![Base Model](https://img.shields.io/badge/Model-Qwen2.5--7B--Instruct-blue)](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
[![Inference](https://img.shields.io/badge/Inference-Ollama%20%28Q4__K__M%29-black)](https://ollama.com/)
[![Vector DB](https://img.shields.io/badge/Vector%20DB-ChromaDB-orange)](https://www.trychroma.com/)
[![Backend](https://img.shields.io/badge/API-FastAPI-009688)](https://fastapi.tiangolo.com/)

Grounded QA system for Linux commands and manual pages using **ChromaDB** retrieval, **FastAPI**, a **QLoRA-adapted `Qwen2.5-7B-Instruct`** model, and local **Ollama** inference.

## Architecture

```text
Linux Man Pages / Docs
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
