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
import json
import logging
import re

import pdfplumber
import pypdfium2 as pdfium
import pytesseract
from google import genai
from google.genai import types as genai_types

from app.agents.state import PipelineState
from app.config import settings
from app.schemas import Clause, ClauseType

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


# ── Extraction (Day 2.1): text → typed clauses via Gemini ──
# The prompt has two jobs: SPLIT (find clause boundaries) and TYPE (label
# each one). We demand VERBATIM text — paraphrasing here would poison
# everything downstream (risk scores and counter-drafts quote this text).

EXTRACTION_PROMPT = """You are a legal-document analyst reviewing Indian \
contracts for a gig worker. Split the document below into its individual \
clauses.

Return ONLY a JSON array. Each element is an object with EXACTLY these keys:
  "text": the clause's FULL original wording, copied VERBATIM (include its \
number/heading)
  "clause_type": one of: {types}

Rules:
- One clause per numbered/heading section. Unnumbered substantive paragraphs \
each count as a clause.
- Never paraphrase, summarize, or invent clauses.
- If the type is unclear, use "other".

DOCUMENT:
{document}"""

# Safety valve: documents longer than this get truncated before the LLM.
# Gemini's context is huge, but latency + token cost are not free.
EXTRACTION_MAX_CHARS = 60_000

# Lazily-created Gemini client (built on first use, then reused).
_gemini_client: genai.Client | None = None


def _get_gemini_client() -> genai.Client:
    """Build the Gemini client once; fail with an ACTIONABLE error if the
    API key is missing (ValueError → clean 400, not a cryptic 500)."""
    global _gemini_client
    if _gemini_client is None:
        if not settings.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Add it to .env (free key: "
                "https://aistudio.google.com/) and restart the API."
            )
        _gemini_client = genai.Client(api_key=settings.gemini_api_key)
    return _gemini_client


def _parse_clauses_json(payload: str) -> list[Clause]:
    """UNTRUSTED LLM output → validated list[Clause].

    The LLM is a probability machine, not a function: it can return
    broken JSON, missing fields, or nonsense types. The boundary policy:
      - unparseable JSON → ValueError (loud, immediate — the Pydantic lesson)
      - malformed item   → dropped with a warning (its text is garbage anyway)
      - unknown type     → coerced to 'other' (keep the text, lose nothing)
      - indexes          → reassigned 0..n-1 by US, never trusted from the LLM
    """
    data = json.loads(payload)  # may raise — a loud failure here is correct
    if not isinstance(data, list):
        raise ValueError(f"LLM returned {type(data).__name__}, expected a JSON array.")

    clauses: list[Clause] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict) or not str(item.get("text", "")).strip():
            logger.warning("extraction: dropping malformed clause #%d: %r", i, item)
            continue
        try:
            clause_type = ClauseType(item.get("clause_type", "other"))
        except ValueError:
            logger.warning(
                "extraction: unknown clause_type %r — coercing to 'other'",
                item.get("clause_type"),
            )
            clause_type = ClauseType.other
        clauses.append(
            Clause(
                text=str(item["text"]).strip(),
                clause_type=clause_type,
                index=len(clauses),
            )
        )
    return clauses


def extraction_node(state: PipelineState) -> dict:
    """Text → typed clauses via Gemini, validated against our Pydantic schema."""
    raw_text = state.get("raw_text", "")
    logger.info("▶ extraction_node | splitting %d chars into typed clauses", len(raw_text))

    if not raw_text.strip():
        logger.warning("extraction: no text to split — returning zero clauses")
        return {"clauses": []}

    document = raw_text[:EXTRACTION_MAX_CHARS]
    if len(raw_text) > EXTRACTION_MAX_CHARS:
        logger.warning(
            "extraction: truncated document to %d chars for the LLM", EXTRACTION_MAX_CHARS
        )

    prompt = EXTRACTION_PROMPT.format(
        types=", ".join(t.value for t in ClauseType),
        document=document,
    )
    response = _get_gemini_client().models.generate_content(
        model=settings.gemini_model,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            response_mime_type="application/json",  # force syntactically-valid JSON
            temperature=0.0,                        # legal parsing wants determinism
        ),
    )

    clauses = _parse_clauses_json(response.text or "[]")
    kinds = sorted({c.clause_type.value for c in clauses})
    logger.info(
        "extraction: %d clauses validated | types: %s",
        len(clauses),
        ", ".join(kinds) or "none",
    )
    return {"clauses": clauses}


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
