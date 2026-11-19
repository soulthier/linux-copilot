readme = """# 🛡️ SecOps Copilot: CVE & Threat Intelligence Assistant

An end-to-end AI Engineering system that automates vulnerability triage, structured threat extraction, and mitigation retrieval using **QLoRA Fine-Tuning**, **Hybrid RAG**, and **n8n** workflow orchestration.

## 🏗 Architecture

```mermaid
flowchart LR
    A[NVD CVE API / Raw Incident Text] --> B[n8n Orchestrator]
    B --> C[Fine-Tuned Qwen2.5-3B<br/>Structured JSON Extraction]
    C --> D[Hybrid RAG Engine<br/>Qdrant + BM25 + MITRE ATT&CK]
    D --> E[Mitigation & Actionable Report]
    E --> F[Telegram Bot / Webhook Response]

