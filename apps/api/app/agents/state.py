# ──────────────────────────────────────────────────────────────
# Sahayak — Pipeline State (the shared "memory" of all agents)
# ──────────────────────────────────────────────────────────────
# This is THE object that flows through the whole pipeline. Each agent
# READS what it needs from here and WRITES its own output back into the
# SAME dict. LangGraph threads it from node to node.
#
# This is what makes it an AGENT pipeline rather than 5 isolated functions:
# they all share evolving state, like an intern carrying a notebook through
# a task and adding notes at each step.

from typing import TypedDict

from app.schemas import Clause


class PipelineState(TypedDict, total=False):
    """Everything the pipeline knows, at any point in its run.

    `total=False` means NONE of these keys are required upfront — the
    pipeline fills them in as it goes (ingestion fills raw_text, etc.).
    """

    # ── Input ──
    filename: str  # name of the uploaded document
    file_bytes: bytes  # the raw uploaded bytes (a PDF, or plain text)

    # ── Ingestion output (Day 1.2) ──
    raw_text: str  # the full clean text extracted from the PDF
    used_ocr: bool  # True when the OCR fallback ran (scanned PDF, Day 1.3)

    # ── Extraction output (Day 2.1) ──
    clauses: list[Clause]  # the document, split into typed clauses

    # ── Risk scoring / asymmetry output (Day 4) ──
    scored_clauses: list[dict]  # each clause + its asymmetry score

    # ── Report output (Day 5) ──
    report: dict  # the final assembled report + counter-draft
