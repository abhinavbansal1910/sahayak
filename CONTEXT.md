# Sahayak — Project Context (Living Document)

> **Purpose:** This file is the single source of truth for the Sahayak project.
> Any new chat, IDE, or AI model can read this and instantly understand
> the project's goal, current state, architecture, and next steps.
>
> **Updated:** After every completed sprint or major change.
> **Maintained by:** Abhinav (with ZCode).

---

## 1. What is Sahayak? (the one-paragraph pitch)

**Sahayak** is a trust-aware legal document agent for Indian gig workers and
small business owners who sign contracts they don't fully understand.

**The core thesis (what makes it unique):**
> Sahayak does NOT just summarize contracts. It measures **how lopsided each
> clause is** (clause risk *asymmetry*), flags the ones that disproportionately
> favor the other party, scores them quantitatively, and drafts fairer
> counter-clauses to negotiate with.

This "asymmetry score" (−100 favors them … 0 fair … +100 favors you) is the
project's interview hook and differentiator.

---

## 2. Current Status

### ✅ Completed
- **Sprint 0 — Foundation**
  - Monorepo structure (`apps/api`, `apps/web`, `model/`)
  - FastAPI backend scaffolded (`/` and `/health` endpoints)
  - Docker setup (FastAPI + Postgres via `docker compose up`)
  - Local Python venv for editing
  - Git initialized, 5 clean commits, `.env`/`.gitignore` hygiene
  - Verified: `{"status":"healthy","version":"0.1.0"}` returns at localhost:8000

### 🚧 Next Up
- **Sprint 1 — Ingestion + Extraction agents**
  - Build the LangGraph skeleton (the agent orchestration framework)
  - Ingestion Agent: PDF/document → clean text
  - Extraction Agent: text → individual clauses, classified by type

### ⬜ Roadmap
- Sprint 2 — Fine-tuned risk classifier (the heart, LegalBERT + LoRA on Colab)
- Sprint 3 — Risk asymmetry engine + Negotiation agent
- Sprint 4 — Persistence (Postgres models) + Langfuse tracing
- Sprint 5 — Frontend dashboard (Next.js, asymmetry heatmap)
- Sprint 6 — Trust & safety layer (calibration, statute citations, disclaimers)
- Sprint 7 — Polish, demo data, deploy (Vercel + Koyeb)

---

## 3. Architecture (the 5-agent pipeline)

```
[Ingestion] → [Extraction] → [Risk Scoring] ⭐ → [Negotiation] → [Report]
   PDF→text     clause split     fine-tuned        draft fair       asymmetry
                                  classifier        counter-clauses  report
```

| Agent | Job | Built with |
|---|---|---|
| Ingestion | PDF/docx → clean text | pdfplumber + OCR (Tesseract) |
| Extraction | Segment into clauses, classify type | LLM (Gemini) |
| **Risk Scoring** ⭐ | Score each clause's risk + who it favors | **Fine-tuned LegalBERT** (custom model) |
| Negotiation | Draft fairer counter-clauses | LLM (Gemini/Groq) |
| Report | Assemble quantified report + counter-draft | LLM + templates |

These agents are orchestrated by **LangGraph** — a library that lets us define
the flow (nodes + edges) of a multi-agent system in code.

---

## 4. Tech Stack (and WHY each choice)

| Layer | Tech | Why this choice |
|---|---|---|
| Agent orchestration | LangGraph | The standard for building controllable multi-agent systems; main "agentic AI" flex |
| Reasoning LLM | Gemini Flash + Groq (fallback) | Both have free tiers; Gemini = most generous context, Groq = fastest |
| Risk classifier | Fine-tuned LegalBERT | Custom-trained model = differentiator vs "API wrapper" projects |
| Training env | Google Colab (free T4) | Free GPU for LoRA fine-tuning |
| Model hosting | Hugging Face Spaces (ZeroGPU) | Free inference endpoint for the trained model |
| Backend | FastAPI + Pydantic v2 + SQLAlchemy 2.0 | Async, typed, Python standard |
| Database | PostgreSQL + pgvector (Neon in prod) | One DB for data + vector similarity |
| Frontend | Next.js 15 + Tailwind + shadcn/ui | The "Linear/Vercel aesthetic" by default |
| Backend hosting | Koyeb (free, scale-to-zero) | Persistent free tier for the API |
| Frontend hosting | Vercel (free) | Zero-config for Next.js |
| Observability | Langfuse | Traces every agent run — gold for demos |

**Total monthly cost target: $0.**

---

## 5. File Structure

```
Sahayak/
├── CONTEXT.md            ← THIS FILE (living project doc)
├── lessons.md            ← study notes (LOCAL ONLY — gitignored, never on GitHub)
├── .env                  ← REAL secrets (gitignored, NEVER on GitHub)
├── .env.example          ← safe template with fake values (on GitHub)
├── .gitignore            ← tells Git what to skip (venv, .env, etc.)
├── README.md             ← project's front door (recruiters see this)
├── docker-compose.yml    ← boots API + Postgres together (the conductor)
└── apps/api/             ← FastAPI backend
    ├── .dockerignore     ← build-time bouncer for Docker
    ├── Dockerfile        ← recipe to build the backend image
    ├── requirements.txt  ← pinned Python dependencies
    ├── venv/             ← local editing sandbox (gitignored)
    └── app/
        ├── __init__.py   ← marks folder as a Python package
        ├── config.py     ← reads .env → typed Settings (control panel)
        ├── main.py       ← FastAPI app entry point + endpoints
        └── schemas.py    ← Pydantic data shapes (Clause, ClauseType)
```

---

## 6. How to Run (local dev)

### Start the backend + database
```bash
# Docker Desktop must be running first
docker compose up --build          # builds & starts API + Postgres
```
Then open: http://localhost:8000/health

### Edit code locally (with IDE autocomplete)
```bash
cd apps/api
source venv/Scripts/activate       # Git Bash on Windows
pip install -r requirements.txt
```

### Stop everything
```bash
docker compose down                # stops containers (keeps data)
docker compose down -v             # stops + WIPES database data
```

---

## 7. Git Workflow & Commit Conventions

### Daily workflow (after first-time `git push -u origin main`)
```bash
git add .                          # stage changes
git commit -m "type: description"  # save snapshot
git push                           # upload to GitHub (no -u needed again)
```

### Commit message types (Conventional Commits)
- `feat:` — new feature or functionality
- `fix:` — bug fix
- `docs:` — documentation (README, comments, CONTEXT.md)
- `refactor:` — code cleanup, no new behavior
- `chore:` — setup, config, dependencies, tooling
- `style:` — formatting only (whitespace, etc.)
- `test:` — adding or fixing tests

### Rule: commit small & often (atomic commits)
Don't wait for a whole feature. Commit every time ONE small thing works.
This builds a richer commit history and a better story for recruiters.

---

## 8. Key Concepts Learned (interview prep notes)

### What is an "agent"?
An AI program that doesn't just answer one question — it has a GOAL, uses
TOOLS, keeps MEMORY/STATE, and LOOPS until the goal is met. Like hiring an
intern and giving them a task, not asking a friend one question.

### What is LangGraph?
A library that lets you define the flowchart (nodes = agents/functions,
edges = data flow) of a multi-agent system in code. It manages running them
in order, passing state, handling loops, and checkpointing.

### What is fine-tuning?
Taking a pre-trained model (which already "speaks English") and specializing
it for a narrow task by showing it labeled examples. We don't build a brain
from scratch — we specialize an existing one. Sahayak's risk classifier will
be fine-tuned, not just prompted.

### Docker vs venv vs Git
- **venv** — Python-only sandbox for EDITING (gives IDE autocomplete). Lightweight.
- **Docker** — full environment for RUNNING (backend + DB together). Reproducible.
- **Git** — time machine for TRACKING changes & pushing to GitHub. Builds profile.
They solve different problems and coexist perfectly.

### .env vs .env.example
- `.env` — REAL secrets, gitignored, NEVER on GitHub.
- `.env.example` — fake template, on GitHub, shows what vars the app needs.

### What is CORS?
A browser security rule: a site at port 3000 can't call an API at port 8000
unless the API explicitly allows it. We whitelist localhost:3000 for our frontend.

### What is a health check?
A tiny `/health` endpoint that hosting platforms ping to know the app is alive.
If it stops responding, the platform restarts the container. Industry standard.

---

## 9. Key Decisions & Rationale (why we chose X over Y)

1. **Asymmetry score over summarization** — novel, quantitative, defensible.
   Most "legal AI" projects summarize; we measure lopsidedness. This is THE
   differentiator.
2. **Fine-tuned classifier over LLM-only risk scoring** — more accurate, far
   cheaper to run, and proves ML skill (not just API-calling skill).
3. **LangGraph over a plain chain** — recruiters explicitly look for "agentic"
   orchestration; LangGraph is the tool for controllable multi-agent systems.
4. **Postgres + pgvector over separate DB + Pinecone** — one database, simpler,
   fewer moving parts, still free.
5. **Dual LLM provider (Gemini + Groq)** — fallback strategy keeps the app
   working even when one free tier rate-limits.
6. **Docker from day one** — reproducible environment, same image deploys to cloud.

---

## 10. Environment / Machine Notes

- OS: Windows (Git Bash terminal)
- Python: 3.14.5 installed system-wide
  - ⚠️ Host Python 3.14 is too new for some pinned deps (no prebuilt wheels
    yet — pydantic-core tries to compile from Rust and fails). RUN PYTHON
    INSIDE DOCKER instead: `docker compose exec -T api python ...`
    (the container is Python 3.12, per the Dockerfile).
- Git: 2.49.0 installed and authenticated
- Docker Desktop: installed and running
- Working dir: C:\Users\Abhinav\OneDrive\Desktop\Sahayak
- venv location: apps/api/venv (activate with `source venv/Scripts/activate`)

---

## 11. Free API Keys Needed (later sprints)

- [ ] Gemini API key → https://aistudio.google.com/ (free)
- [ ] Groq API key → https://console.groq.com/ (free, optional fallback)
- [ ] Hugging Face token → https://huggingface.co/ (free, for model hosting)
- [ ] Neon database → https://neon.tech/ (free, for production DB)
- [ ] Langfuse → https://langfuse.com/ (free tier, for tracing)

Don't need these yet — only when we reach the relevant sprint.

---

## 12. Change Log

- **2026-07-30** — Sprint 0 complete. Project scaffolded, Docker working,
  FastAPI serving `/health`. 5 commits on `main`.
- **2026-07-31** — Sprint 1 started. Added Pydantic `Clause` / `ClauseType`
  schema (`app/schemas.py`) — the data shape that flows through the pipeline.
  Bumped psycopg pin 3.2.3 → 3.2.13 (old version removed from PyPI). Learned
  host Python 3.14 is too new for pinned wheels → run Python inside Docker.
- **2026-08-01** — Docker deep-dive (Lesson 3) covered. Created `lessons.md`,
  a running study + interview-revision guide that grows each lesson.
