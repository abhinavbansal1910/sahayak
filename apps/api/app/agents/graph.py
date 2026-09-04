# ──────────────────────────────────────────────────────────────
# Sahayak — The LangGraph (wiring the agents together)
# ──────────────────────────────────────────────────────────────
# This is the flowchart. We:
#   1. create a graph bound to our PipelineState,
#   2. add each agent as a NODE,
#   3. add EDGES between them (the order data flows) — including our
#      first CONDITIONAL edge (scanned PDF → OCR detour, Day 1.3),
#   4. compile it into something we can `.invoke()`.

from langgraph.graph import START, END, StateGraph

from app.agents import nodes
from app.agents.state import PipelineState


def build_pipeline():
    """Wire the agents into a pipeline (with one branch) and compile it."""
    graph = StateGraph(PipelineState)

    # ── Nodes: register each agent as a named step ──
    graph.add_node("ingestion", nodes.ingestion_node)
    graph.add_node("ocr", nodes.ocr_node)  # the FALLBACK path (Day 1.3)
    graph.add_node("extraction", nodes.extraction_node)
    graph.add_node("risk_scoring", nodes.risk_node)
    graph.add_node("negotiation", nodes.negotiation_node)
    graph.add_node("report", nodes.report_node)

    # ── Edges: the order data flows ──
    graph.add_edge(START, "ingestion")

    # CONDITIONAL edge (Day 1.3): after ingestion, a router function
    # INSPECTS the state and picks the next node:
    #   scanned PDF (no text layer) → 'ocr' detour
    #   real text extracted         → straight to 'extraction'
    graph.add_conditional_edges(
        "ingestion",
        nodes.route_after_ingestion,
        {"ocr": "ocr", "extraction": "extraction"},
    )
    graph.add_edge("ocr", "extraction")  # OCR rejoins the main path

    graph.add_edge("extraction", "risk_scoring")
    graph.add_edge("risk_scoring", "negotiation")
    graph.add_edge("negotiation", "report")
    graph.add_edge("report", END)

    return graph.compile()


# One compiled graph, built once and reused for every request.
pipeline = build_pipeline()


def run_pipeline(filename: str, file_bytes: bytes) -> PipelineState:
    """Kick off the whole pipeline. Returns the final shared State."""
    initial_state: PipelineState = {
        "filename": filename,
        "file_bytes": file_bytes,  # the actual uploaded document
    }
    return pipeline.invoke(initial_state)
