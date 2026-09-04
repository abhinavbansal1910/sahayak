# Sahayak — Trust-Aware Legal Document Agent 🇮🇳

> **Sahayak** doesn't summarize contracts. It measures **how lopsided each clause is**, flags the ones that disproportionately favor the other party, and drafts the counter-language to fix them.

Built for Indian gig workers and small business owners who sign contracts they don't fully understand — with no lawyer and no time.

---

## Why this exists

The person handing you a contract had lawyers write it to protect *them*. You sign in a hurry. Six months later a buried clause costs you money. **Sahayak reads the contract like a lawyer on your side** — in seconds, for free.

## How it works (the 5-agent pipeline)

```
[Ingestion] → [Extraction] → [Risk Scoring] ⭐ → [Negotiation] → [Report]
   PDF→text     clause split     fine-tuned       draft fair       asymmetry
                                  classifier       counter-         report +
                                                   clauses          counter-draft
```

- **Risk Scoring** uses a *fine-tuned* LegalBERT classifier — not just a prompt.
- Every clause gets an **Asymmetry Score** (−100 favors them … 0 fair … +100 favors you).
- The Report Agent produces a ranked, quantified breakdown + suggested fixes.

## Tech stack

| Layer | Tech |
|---|---|
| Agent orchestration | LangGraph |
| Reasoning LLM | Gemini Flash + Groq (free tiers) |
| Risk classifier | Fine-tuned LegalBERT (trained on free Colab T4) |
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
- [ ] Fine-tuned risk classifier
- [ ] Risk asymmetry engine
- [ ] Negotiation agent
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
