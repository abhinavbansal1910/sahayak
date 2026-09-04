# Sahayak — Trust-Aware Legal Document Agent 🇮🇳

> **Sahayak** doesn't summarize contracts. It measures **how lopsided each clause is**, flags the ones that disproportionately favor the other party, and drafts the counter-language to fix them.

Built for Indian gig workers and small business owners who sign contracts they don't fully understand — with no lawyer and no time.

---

## Why this exists

The person handing you a contract had lawyers write it to protect *them*. You sign in a hurry. Six months later a buried clause costs you money. **Sahayak reads the contract like a lawyer on your side** — in seconds, for free.

## How it works (the 5-agent pipeline)

```
[Ingestion] → [Extraction] → [Risk Scoring] ⭐ → [Negotiation] → [Report]
   PDF→text     clause split     RAG-grounded     draft fair       asymmetry
                (OCR fallback)   LLM judge        counter-         report +
                                                  clauses          counter-draft
```

- **Risk Scoring** is a **RAG-grounded LLM judge**: Gemini judges each clause
  (temperature 0, JSON-validated) with the 3 most similar *labeled* clauses
  from our India-specific knowledge base retrieved as grounding.
- Every clause gets an **Asymmetry Score** (−100 favors them … 0 fair … +100 favors you)
  = direction × confidence × 100.
- The Report Agent produces a ranked, quantified breakdown + suggested fixes;
  the Negotiation Agent drafts a balanced rewrite of every lopsided clause.

## Tech stack

| Layer | Tech |
|---|---|
| Agent orchestration | LangGraph |
| Reasoning LLM | Gemini Flash + Groq (free tiers) |
| Risk scorer | LLM + RAG judge over a labeled Indian-clause knowledge base (embeddings + cosine retrieval; pgvector in prod) |
| Comparison track | LoRA fine-tune of InLegalBERT (optional Colab notebook in `model/`) |
| Backend | FastAPI, Pydantic v2, SQLAlchemy 2.0 |
| Database | PostgreSQL + pgvector (Neon in prod) |
| Frontend | Next.js 15, Tailwind, shadcn/ui |
| Infra | Docker, Vercel + Koyeb |

## Project status

🚧 **Sprint 1 — Pipeline skeleton + full Ingestion (complete)**

- [x] Monorepo structure
- [x] Docker setup (FastAPI + Postgres)
- [x] Health endpoint
- [x] LangGraph pipeline skeleton (5 agents, shared State)
- [x] Ingestion agent — born-digital PDFs + OCR fallback for scans (conditional edge)
- [x] Extraction agent — Gemini splits documents into typed, validated clauses
- [x] Risk asymmetry engine — RAG-grounded LLM judge scores every clause −100…+100
- [x] Negotiation agent — balanced counter-drafts for lopsided clauses (Groq fallback)
- [x] Report agent — ranked, quantified report + disclaimer
- [ ] Frontend dashboard

## Getting started (local dev)

```bash
# 1. Copy the example env file and fill in values
cp .env.example .env

# 2. Start the backend + database (needs Docker Desktop running)
docker compose up --build

# 3. Check it's alive → open http://localhost:8000/health
```

For local editing (with IDE autocomplete), create a venv:
```bash
cd apps/api
python -m venv venv
source venv/Scripts/activate    # Git Bash on Windows
pip install -r requirements.txt
```

## Disclaimer

Sahayak is a risk-flagging tool and **not legal advice**. Always consult a qualified lawyer for important decisions.
