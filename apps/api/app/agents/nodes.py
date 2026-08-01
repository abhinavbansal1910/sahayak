# ──────────────────────────────────────────────────────────────
# Sahayak — Pipeline nodes (the 5 agents)
# ──────────────────────────────────────────────────────────────
# Each agent is a NODE: a function that takes the shared State, does its
# job, and returns a dict of JUST the keys it changed. LangGraph merges
# that into the running State.
#
# Today (Day 1.1) these are STUBS — they log + write placeholder values,
# just to prove the graph runs end-to-end. We flesh each one out in the
# coming units (ingestion next).

import logging

from app.agents.state import PipelineState

logger = logging.getLogger(__name__)


def ingestion_node(state: PipelineState) -> dict:
    """PDF/document -> clean text. (Stub; real impl in Day 1.2.)"""
    logger.info("▶ ingestion_node  | file=%s", state.get("filename"))
    return {"raw_text": "[stub] the clean contract text will go here"}


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
