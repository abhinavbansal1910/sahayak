# ──────────────────────────────────────────────────────────────
# Sahayak API — Entry point
# ──────────────────────────────────────────────────────────────
# This is the file Docker runs (`uvicorn app.main:app`).
# It creates the FastAPI app and wires up our routes.
#
# For Sprint 0 it's intentionally tiny — just enough to prove the
# plumbing works. Real endpoints (analyze a contract, stream agent
# progress) get added in later sprints.

import logging

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.agents.graph import run_pipeline
from app.config import settings

# Surface our agents' log lines in the server output so we can SEE the
# pipeline run node-by-node (each agent logs when it starts).
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

app = FastAPI(
    title="Sahayak API",
    description="Trust-aware legal document agent for Indian contracts.",
    version="0.1.0",
)

# ── CORS (Cross-Origin Resource Sharing) ──
# Our Next.js frontend (port 3000) will need to call this API (port 8000).
# Browsers block cross-origin calls by default; CORS whitelists our frontend.
# In production we'll lock this to our real domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # our future frontend
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    """Landing info — answers 'is this the right service?'."""
    return {
        "service": "Sahayak API",
        "status": "running",
        "environment": settings.environment,
        "docs": "/docs",  # FastAPI's auto-generated interactive docs
    }


@app.get("/health")
def health():
    """Health check. Used by Docker / hosting platforms to know we're alive.

    Later we'll also ping the database here. For now: if this returns 200,
    the container booted and FastAPI is serving requests.
    """
    return {"status": "healthy", "version": "0.1.0"}
@app.get("/about")
def about():
    """What this service does, in one line."""
    return {
        "name": "Sahayak",
        "purpose": "A trust-aware legal document agent for Indian contracts.",
        "pipeline": ["ingestion", "ocr (fallback)", "extraction", "risk_scoring", "negotiation", "report"],
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """Run the full agent pipeline on an uploaded document.

    Send a born-digital .pdf (or .txt) as multipart form field `file`.
    Returns scored clauses (−100…+100 asymmetry), counter-drafts for the
    worst ones, and a summary report.
    """
    file_bytes = await file.read()

    try:
        final_state = run_pipeline(
            filename=file.filename or "document",
            file_bytes=file_bytes,
        )
    except ValueError as exc:
        # Bad input (e.g. unsupported file type) → a clean 400,
        # not an ugly 500. Errors at the boundary, in plain language.
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        # Upstream failure (LLM outage / quota) → a clean 502 naming the
        # culprit, never an empty body or a mystery 500.
        logging.error("pipeline failed: %s", str(exc)[:300])
        raise HTTPException(
            status_code=502,
            detail="Analysis pipeline failed — the LLM provider is likely "
                   "unavailable or rate-limited. Try again in a minute.",
        )

    # Never echo `file_bytes` back: raw PDF bytes are BINARY, JSON is
    # text-only — FastAPI would try to UTF-8-decode them and 500 (today's
    # war story). Internal state ≠ API response; return only the analysis.
    final_state.pop("file_bytes", None)

    return final_state