# ──────────────────────────────────────────────────────────────
# Sahayak — Pipeline nodes (the agents)
# ──────────────────────────────────────────────────────────────
# Each agent is a NODE: a function that takes the shared State, does its
# job, and returns a dict of JUST the keys it changed. LangGraph merges
# that into the running State.
#
# Day 1.2: ingestion is REAL.  Day 1.3: OCR fallback + conditional edge REAL.
# Day 2.1: extraction REAL (Gemini → Pydantic-validated clauses).
# Day 4:   risk scoring REAL (RAG-grounded LLM judge + asymmetry engine),
#          negotiation REAL (Gemini primary, Groq fallback), report REAL.
# Architecture note (2026-09-05): the scorer is an LLM+RAG judge; the
# fine-tuned InLegalBERT notebook (model/) remains as the comparison track.

import hashlib
import io
import json
import logging
import re
import time
from pathlib import Path

import httpx
import pdfplumber
import pypdfium2 as pdfium
import pytesseract
from google import genai
from google.genai import types as genai_types

from app.agents.state import PipelineState
from app.config import settings
from app.schemas import Clause, ClauseType, RiskDirection, RiskVerdict

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
        # attempts=1 kills the SDK's SILENT internal retries. On the free
        # tier they burst-fire on 503s and trip the 5-attempts/minute
        # limiter, turning one wobbly call into a cascade of 429s. We own
        # retrying: one loop, logged, with real backoff (see _gemini_json).
        _gemini_client = genai.Client(
            api_key=settings.gemini_api_key,
            http_options=genai_types.HttpOptions(
                retry_options=genai_types.HttpRetryOptions(attempts=1),
            ),
        )
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


# ──────────────────────────────────────────────────────────────
# Day 4 — Risk Scoring: RAG-grounded judge + the Asymmetry Engine
# ──────────────────────────────────────────────────────────────
# The "R": our labeled clause KB (apps/api/app/data/labeled_clauses.jsonl)
# is embedded once (disk-cached) and searched by cosine similarity — with
# 73 documents that's a pure-Python brute-force scan; pgvector takes over
# on Day 5.3 when persistence lands.
# The "A": direction sign × confidence × 100 = the asymmetry score.

# KB lives at app/data/ — nodes.py is in app/agents/, so go up one level.
KB_PATH = Path(__file__).parent.parent / "data" / "labeled_clauses.jsonl"
EMBED_CACHE_PATH = KB_PATH.parent / "embeddings_cache.json"

# sign × confidence × 100 → the −100…+100 asymmetry scale
ASYMMETRY_SIGN = {
    RiskDirection.favors_them: -1,
    RiskDirection.balanced: 0,
    RiskDirection.favors_you: 1,
}
# a clause working against the reader at least this hard gets a counter-draft
NEGOTIATION_THRESHOLD = -30

RISK_JUDGE_PROMPT = """You are a contract analyst protecting an Indian gig \
worker or freelancer. Judge how this clause treats THEM (the other party, \
who wrote the contract) versus YOU (our reader).

Similar clauses from our labeled knowledge base (label = who the clause \
favors):

{exemplars}

Now judge the TARGET clause. Respond with ONLY a JSON object:
  {{"direction": "favors_them" | "balanced" | "favors_you",
    "confidence": <0.0-1.0>,
    "reason": "<one sentence, plain English>"}}

Guidance: strongly one-sided terms (no notice, unlimited indemnity, \
perpetual IP assignment, one-way penalties) favor whoever wrote them. \
Mutual notice periods, caps on both sides, shared obligations are balanced. \
Worker-friendly terms (interest on late payment, deemed acceptance, kill \
fees) favor the reader.

TARGET CLAUSE:
{clause}"""

NEGOTIATION_PROMPT = """You are a contract negotiator for an Indian gig \
worker or freelancer. Rewrite this one-sided clause into a BALANCED, \
professional clause both parties could sign — realistic for Indian \
contracts. Respond ONLY with JSON:
  {{"balanced_version": "<the rewritten clause>",
    "explanation": "<one sentence: what changed and why it is fairer>",
    "leverage": "<one sentence: the argument to use when proposing this>"}}

CLAUSE (direction={direction}, asymmetry={asymmetry}):
{clause}"""


def _gemini_json(prompt: str, attempts: int = 3) -> str:
    """One Gemini call forcing JSON-mode output. Shared by judge+negotiator.
    Retries transient blips (503 "high demand") with a short backoff — but
    quota errors (429) fail FAST: sleeping 54s inside a request is worse
    than bubbling up to the caller's fallback (Groq / neutral verdict)."""
    delay = 2.0
    for attempt in range(1, attempts + 1):
        try:
            response = _get_gemini_client().models.generate_content(
                model=settings.gemini_model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0,
                ),
            )
            return response.text or ""
        except Exception as exc:
            exhausted = attempt == attempts
            quota_hit = "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc)
            if exhausted or quota_hit:
                raise
            logger.warning("gemini: attempt %d/%d failed — retrying in %.0fs (%s)",
                           attempt, attempts, delay, str(exc)[:80])
            time.sleep(delay)
            delay *= 2.5


def _embed_texts(texts: list[str], task_type: str) -> list[list[float]]:
    """Embed via Gemini. task_type: RETRIEVAL_DOCUMENT (KB) vs
    RETRIEVAL_QUERY (search) — different types score better together."""
    response = _get_gemini_client().models.embed_content(
        model=settings.gemini_embedding_model,
        contents=texts,
        config=genai_types.EmbedContentConfig(task_type=task_type),
    )
    return [e.values for e in response.embeddings]


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity, pure Python — 73×768 dots are nothing to compute."""
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


_kb: dict | None = None


def _get_clause_kb() -> dict:
    """Embed the labeled KB once per process; cache vectors on disk so
    restarts don't re-embed (or burn free-tier quota)."""
    global _kb
    if _kb is not None:
        return _kb

    rows = [json.loads(line) for line in
            KB_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    model = settings.gemini_embedding_model
    cache: dict = {}
    if EMBED_CACHE_PATH.exists():
        cache = json.loads(EMBED_CACHE_PATH.read_text(encoding="utf-8"))

    keys = [f"{model}:{hashlib.sha1(r['text'].encode()).hexdigest()}" for r in rows]
    missing = [(i, k) for i, k in enumerate(keys) if k not in cache]
    if missing:
        logger.info("rag: embedding %d/%d KB clauses (%s)", len(missing), len(rows), model)
        new_vecs = _embed_texts([rows[i]["text"] for i, _ in missing], "RETRIEVAL_DOCUMENT")
        for (i, k), vec in zip(missing, new_vecs):
            cache[k] = vec
        try:
            EMBED_CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")
        except OSError:
            pass  # read-only FS (prod) → re-embed on next boot, no harm

    _kb = {
        "texts": [r["text"] for r in rows],
        "labels": [r["label"] for r in rows],
        "vectors": [cache[k] for k in keys],
    }
    logger.info("rag: knowledge base ready — %d labeled clauses", len(rows))
    return _kb


def _retrieve_similar_labeled(clause_text: str, k: int) -> list[tuple[str, str]]:
    """Top-k (label, text) exemplars by cosine similarity — the R in RAG."""
    kb = _get_clause_kb()
    query_vec = _embed_texts([clause_text], "RETRIEVAL_QUERY")[0]
    scored = sorted(
        ((_cosine(query_vec, vec), label, text)
         for vec, label, text in zip(kb["vectors"], kb["labels"], kb["texts"])),
        reverse=True,
    )
    return [(label, text) for _, label, text in scored[:k]]


_verdict_cache: dict[str, dict] = {}


def _judge_clause(clause: Clause) -> RiskVerdict:
    """One clause → RiskVerdict. Cached by text hash; degrades to a neutral
    verdict if the judge is unreachable so ONE clause can't kill a run.
    Degraded verdicts are NOT cached — a transient outage shouldn't be
    remembered as the clause's real score on every future run."""
    key = hashlib.sha1(clause.text.encode()).hexdigest()
    if key in _verdict_cache:
        return RiskVerdict.model_validate(_verdict_cache[key])
    try:
        verdict = _judge_clause_uncached(clause)
    except Exception as exc:  # quota, outage, network — stay graceful
        logger.warning("risk: judge unavailable for clause %d (%s) — neutral verdict",
                       clause.index, exc)
        return RiskVerdict(direction=RiskDirection.balanced, confidence=0.2,
                           reason="Risk judge unavailable — treated as neutral.")
    _verdict_cache[key] = verdict.model_dump(mode="json")
    return verdict


def _judge_clause_uncached(clause: Clause) -> RiskVerdict:
    """Retrieve similar labeled clauses, then let Gemini judge with them
    as few-shot grounding. LLM output parsed with the usual coercion."""
    exemplars = _retrieve_similar_labeled(clause.text, settings.rag_top_k)
    exemplar_block = "\n".join(f"- ({label}) {text}" for label, text in exemplars)
    prompt = RISK_JUDGE_PROMPT.format(
        k=len(exemplars), exemplars=exemplar_block, clause=clause.text
    )
    data = json.loads(_gemini_json(prompt) or "{}")
    try:
        direction = RiskDirection(data.get("direction"))
    except ValueError:
        logger.warning("risk: unknown direction %r — treating as balanced",
                       data.get("direction"))
        direction = RiskDirection.balanced
    try:
        confidence = min(1.0, max(0.0, float(data.get("confidence", 0.5))))
    except (TypeError, ValueError):
        confidence = 0.5
    return RiskVerdict(direction=direction, confidence=confidence,
                       reason=str(data.get("reason", ""))[:500])


def risk_node(state: PipelineState) -> dict:
    """Score every clause: RAG-grounded judge → asymmetry = direction ×
    confidence × 100, on the −100…+100 scale. THE differentiator."""
    clauses = state.get("clauses", [])
    logger.info("▶ risk_node      | judging %d clauses (RAG top-%d)",
                len(clauses), settings.rag_top_k)
    scored = []
    for clause in clauses:
        verdict = _judge_clause(clause)
        scored.append({
            **clause.model_dump(),
            "direction": verdict.direction.value,
            "confidence": round(verdict.confidence, 2),
            "asymmetry": round(ASYMMETRY_SIGN[verdict.direction] * verdict.confidence * 100),
            "reason": verdict.reason,
        })
    mean = round(sum(c["asymmetry"] for c in scored) / len(scored)) if scored else 0
    logger.info("risk: %d clauses scored | mean asymmetry %d", len(scored), mean)
    return {"scored_clauses": scored}


def _negotiate_via_groq(prompt: str) -> str:
    """Fallback path: Groq's OpenAI-compatible REST endpoint (plain httpx —
    no SDK needed; httpx is already our dependency)."""
    response = httpx.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {settings.groq_api_key}"},
        json={
            "model": settings.groq_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "response_format": {"type": "json_object"},
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def negotiation_node(state: PipelineState) -> dict:
    """Draft a fairer counter-clause for every clause working against the
    reader harder than NEGOTIATION_THRESHOLD."""
    scored = state.get("scored_clauses", [])
    targets = [c for c in scored if c["asymmetry"] <= NEGOTIATION_THRESHOLD]
    logger.info("▶ negotiation_node | counter-drafts for %d/%d lopsided clauses",
                len(targets), len(scored))
    if not targets:
        return {"negotiated_clauses": []}

    negotiated = []
    for clause in targets:
        prompt = NEGOTIATION_PROMPT.format(
            direction=clause["direction"],
            asymmetry=clause["asymmetry"],
            clause=clause["text"],
        )
        try:
            payload = _gemini_json(prompt)
        except Exception as exc:
            logger.warning("negotiation: Gemini failed (%s) — Groq fallback", exc)
            if not settings.groq_api_key:
                logger.warning("negotiation: no GROQ_API_KEY — skipping clause %d",
                               clause["index"])
                continue
            payload = _negotiate_via_groq(prompt)
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            logger.warning("negotiation: unusable JSON for clause %d — skipped",
                           clause["index"])
            continue
        negotiated.append({
            "index": clause["index"],
            "original_text": clause["text"],
            "asymmetry": clause["asymmetry"],
            "balanced_version": str(data.get("balanced_version", "")).strip(),
            "explanation": str(data.get("explanation", "")).strip(),
            "leverage": str(data.get("leverage", "")).strip(),
        })
    return {"negotiated_clauses": negotiated}


def report_node(state: PipelineState) -> dict:
    """Assemble the final quantified report: ranked clauses + counter-drafts."""
    scored = state.get("scored_clauses", [])
    drafts = {n["index"]: n for n in state.get("negotiated_clauses", [])}
    logger.info("▶ report_node    | assembling report (%d scored, %d drafts)",
                len(scored), len(drafts))

    ranked = sorted(scored, key=lambda c: c["asymmetry"])  # worst first
    counts = {
        d.value: sum(1 for c in scored if c["direction"] == d.value)
        for d in RiskDirection
    }
    mean_asym = round(sum(c["asymmetry"] for c in scored) / len(scored)) if scored else 0
    worst = ranked[0] if ranked else None

    report = {
        "status": "ok",
        "filename": state.get("filename", "document"),
        "summary": {
            "clauses_analyzed": len(scored),
            "direction_counts": counts,
            "mean_asymmetry": mean_asym,
            "most_lopsided": (
                {"index": worst["index"], "asymmetry": worst["asymmetry"],
                 "direction": worst["direction"], "reason": worst["reason"]}
                if worst else None
            ),
        },
        "clauses_ranked": [
            {
                **c,
                "counter_draft": drafts.get(c["index"], {}).get("balanced_version"),
                "negotiation_explanation": drafts.get(c["index"], {}).get("explanation"),
            }
            for c in ranked
        ],
        "disclaimer": "Sahayak flags lopsided clauses — it is NOT legal advice. "
                      "Consult a qualified lawyer for important decisions.",
    }
    return {"report": report}
