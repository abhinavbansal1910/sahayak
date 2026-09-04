# ──────────────────────────────────────────────────────────────
# Sahayak — Pipeline nodes (the agents)
# ──────────────────────────────────────────────────────────────
# Each agent is a NODE: a function that takes the shared State, does its
# job, and returns a dict of JUST the keys it changed. LangGraph merges
# that into the running State.
#
# Day 1.2: ingestion is REAL (document → clean text).
# Day 1.3: OCR fallback is REAL — scanned PDFs detour through ocr_node,
#          routed by our first CONDITIONAL edge (see graph.py).
# The rest are still stubs — each goes real in its own unit.

import io
import logging
import re

import pdfplumber
import pypdfium2 as pdfium
import pytesseract

from app.agents.state import PipelineState

logger = logging.getLogger(__name__)

# A real contract page holds ~2,000+ characters. Below this total from a
# PDF we conclude there is NO text layer (it's a scan) and route to OCR.
MIN_TEXT_CHARS = 50

# PDF pages render natively at 72 DPI; OCR wants ~300. scale=4 upsamples
# each page 4x (288 DPI) — big, crisp glyphs for Tesseract to read.
OCR_RENDER_SCALE = 4


def _extract_pdf_text(file_bytes: bytes) -> str:
    """Pull the text layer out of a born-digital PDF, page by page."""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        # extract_text() returns None on a page with no text layer —
        # `or ""` keeps the join below from crashing on None.
        pages = [page.extract_text() or "" for page in pdf.pages]
    return "\n".join(pages)


def _ocr_pdf_text(file_bytes: bytes) -> str:
    """READ an image-only PDF: render pages to pictures, then OCR them.

    There are no glyphs to extract here — so we RENDER each page into a
    bitmap (pypdfium2 draws it, like a screenshot) and let Tesseract read
    the picture. Slower and less accurate than a text layer, but it's the
    only path that works on scans and phone photos.
    """
    pdf = pdfium.PdfDocument(io.BytesIO(file_bytes))
    try:
        pages = []
        for page in pdf:
            image = page.render(scale=OCR_RENDER_SCALE).to_pil()
            pages.append(pytesseract.image_to_string(image))
            page.close()  # release each page's memory as we go
    finally:
        pdf.close()
    return "\n".join(pages)


def _normalize_whitespace(text: str) -> str:
    """Clean up extraction junk so the next agent gets tidy text.

    PDFs place glyphs by COORDINATES, not spaces — extraction spits out
    runs of spaces, tabs, and blank lines. We collapse them, because
    the Extraction agent will split THIS text into clauses:
    messy whitespace in → messy clause boundaries out.
    """
    text = re.sub(r"[ \t]+", " ", text)     # many spaces/tabs → one space
    text = re.sub(r"\n{3,}", "\n\n", text)  # 3+ newlines → one blank line
    return text.strip()


def ingestion_node(state: PipelineState) -> dict:
    """Uploaded document → best-effort text layer, into State['raw_text'].

    Note: for a scanned PDF this yields (almost) NOTHING — and that's the
    SIGNAL. Detecting the scan and routing to OCR is the router's job
    (route_after_ingestion), not this node's.
    """
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
    logger.info("ingestion: extracted %d characters of text", len(raw_text))
    return {"raw_text": raw_text}


def ocr_node(state: PipelineState) -> dict:
    """OCR fallback agent: image-only PDF → text (the detour path).

    Sits OFF the main line — the conditional edge only routes here when
    ingestion found (almost) no text. Overwrites raw_text with what
    Tesseract managed to read, and flags the run with `used_ocr`.
    """
    filename = state.get("filename", "document")
    file_bytes = state.get("file_bytes", b"")
    logger.info("▶ ocr_node       | Tesseract reading '%s' (image-only)", filename)

    raw_text = _normalize_whitespace(_ocr_pdf_text(file_bytes))
    logger.info("ocr: recovered %d characters via OCR", len(raw_text))
    return {"raw_text": raw_text, "used_ocr": True}


def route_after_ingestion(state: PipelineState) -> str:
    """CONDITIONAL EDGE: inspect the State, return the next node's NAME.

    - 'ocr'        → near-zero text ⇒ scanned PDF, needs Tesseract
    - 'extraction' → real text in hand ⇒ carry on down the main path
    """
    filename = state.get("filename", "").lower()
    raw_text = state.get("raw_text", "")
    if filename.endswith(".pdf") and len(raw_text) < MIN_TEXT_CHARS:
        logger.warning(
            "router: only %d chars from '%s' — looks SCANNED, routing to OCR",
            len(raw_text),
            state.get("filename"),
        )
        return "ocr"
    return "extraction"


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
