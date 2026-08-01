# ──────────────────────────────────────────────────────────────
# Sahayak — The LangGraph (wiring the 5 agents together)
# ──────────────────────────────────────────────────────────────
# This is the flowchart. We:
#   1. create a graph bound to our PipelineState,
#   2. add each agent as a NODE,
#   3. add EDGES between them (the order data flows),
#   4. compile it into something we can `.invoke()`.
#
# Later we'll add CONDITIONAL edges (e.g. low-text -> OCR fallback,
# risky clause -> negotiate harder). For now it's a straight line.

from langgraph.graph import START, END, StateGraph

from app.agents import nodes
from app.agents.state import PipelineState


def build_pipeline():
    """Wire the 5 agents into a linear pipeline and compile it."""
    graph = StateGraph(PipelineState)

    # ── Nodes: register each agent as a named step ──
    graph.add_node("ingestion", nodes.ingestion_node)
    graph.add_node("extraction", nodes.extraction_node)
    graph.add_node("risk_scoring", nodes.risk_node)
    graph.add_node("negotiation", nodes.negotiation_node)
    graph.add_node("report", nodes.report_node)

    # ── Edges: the order data flows ──
    # START -> ingestion -> extraction -> risk -> negotiation -> report -> END
    graph.add_edge(START, "ingestion")
    graph.add_edge("ingestion", "extraction")
    graph.add_edge("extraction", "risk_scoring")
    graph.add_edge("risk_scoring", "negotiation")
    graph.add_edge("negotiation", "report")
    graph.add_edge("report", END)

    return graph.compile()


# One compiled graph, built once and reused for every request.
pipeline = build_pipeline()


def run_pipeline(filename: str) -> PipelineState:
    """Kick off the whole pipeline. Returns the final shared State."""
    initial_state: PipelineState = {"filename": filename}
    return pipeline.invoke(initial_state)
