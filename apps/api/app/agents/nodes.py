# ──────────────────────────────────────────────────────────────
# Sahayak — Pipeline nodes (the 5 agents)
# ──────────────────────────────────────────────────────────────
# Each agent is a NODE: a function that takes the shared State, does its
# job, and returns a dict of JUST the keys it changed. LangGraph merges
# that into the running State.
#
# Day 1.2: ingestion is REAL (document → clean text). The other four
# are still stubs — each goes real in its own unit.

import io
import logging
import re

import pdfplumber

from app.agents.state import PipelineState

logger = logging.getLogger(__name__)

# A real contract page holds ~2,000+ characters. Below this total we
# suspect the PDF is a SCAN (image-only, no text layer). Day 1.3 turns
# this warning into a conditional edge that routes to OCR.
MIN_TEXT_CHARS = 50


def _extract_pdf_text(file_bytes: bytes) -> str:
    """Pull the text layer out of a born-digital PDF, page by page."""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        # extract_text() returns None on a page with no text layer —
        # `or ""` keeps the join below from crashing on None.
        pages = [page.extract_text() or "" for page in pdf.pages]
    return "\n".join(pages)


def _normalize_whitespace(text: str) -> str:
    """Clean up PDF extraction junk so the next agent gets tidy text.

    PDFs place glyphs by COORDINATES, not spaces — extraction spits out
    runs of spaces, tabs, and blank lines. We collapse them, because
    the Extraction agent will split THIS text into clauses:
    messy whitespace in → messy clause boundaries out.
    """
    text = re.sub(r"[ \t]+", " ", text)     # many spaces/tabs → one space
    text = re.sub(r"\n{3,}", "\n\n", text)  # 3+ newlines → one blank line
    return text.strip()


def ingestion_node(state: PipelineState) -> dict:
    """Uploaded document → clean text, written into State['raw_text']."""
    filename = state.get("filename", "document")
    file_bytes = state.get("file_bytes", b"")

    logger.info("▶ ingestion_node  | file=%s (%d bytes)", filename, len(file_bytes))

    lower = filename.lower()
    if lower.endswith(".pdf"):
        raw_text = _extract_pdf_text(file_bytes)
    elif lower.endswith(".txt"):
        # Plain text passes straight through (easy for testing).
        raw_text = file_bytes.decode("utf-8", errors="replace")
    else:
        # Fail FAST with a clear reason (the Pydantic lesson, node-level).
        raise ValueError(f"Unsupported file type '{filename}'. Upload a .pdf or .txt file.")

    raw_text = _normalize_whitespace(raw_text)

    # Scanned-PDF detector: near-zero text from a PDF means no text layer.
    if lower.endswith(".pdf") and len(raw_text) < MIN_TEXT_CHARS:
        logger.warning(
            "ingestion: only %d chars from a %d-byte PDF — '%s' looks like a "
            "SCANNED document (no text layer). OCR fallback arrives in Day 1.3.",
            len(raw_text),
            len(file_bytes),
            filename,
        )

    logger.info("ingestion: extracted %d characters of text", len(raw_text))
    return {"raw_text": raw_text}


def extraction_node(state: PipelineState) -> dict:
    """Text -> typed clauses. (Stub; real impl in Day 2.1.)"""
    logger.info("▶ extraction_node | splitting text into clauses")
    return {"clauses": []}


def risk_node(state: PipelineState) -> dict:
    """Score each clause's risk + who it favors. (Stub; Day 3-4.)"""
    logger.info("▶ risk_node      | scoring clauses (the fine-tuned model)")
    return {"scored_clauses": []}


def negotiation_node(state: PipelineState) -> dict:
    """Draft fairer counter-clauses. (Stub; Day 4.)"""
    logger.info("▶ negotiation_node | drafting counter-clauses")
    return {}  # no state keys changed yet


def report_node(state: PipelineState) -> dict:
    """Assemble the final quantified report. (Stub; Day 5.)"""
    logger.info("▶ report_node    | assembling report")
    return {"report": {"status": "stub", "note": "real report lands in Day 5"}}
