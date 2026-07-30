# ──────────────────────────────────────────────────────────────
# Sahayak API — Entry point
# ──────────────────────────────────────────────────────────────
# This is the file Docker runs (`uvicorn app.main:app`).
# It creates the FastAPI app and wires up our routes.
#
# For Sprint 0 it's intentionally tiny — just enough to prove the
# plumbing works. Real endpoints (analyze a contract, stream agent
# progress) get added in later sprints.

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings

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
