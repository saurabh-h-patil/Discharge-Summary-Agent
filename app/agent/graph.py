"""
LangGraph Agent Graph — defines the state machine for the discharge summary agent.

The graph implements a ReAct-style loop with Part 2 learning:
  ingest → plan → extract → plan → reconcile → assemble → validate → review_and_learn → done
                     ↑__________________________|  (re-plan if needed)
"""

from langgraph.graph import StateGraph, END

from app.agent.state import AgentState
from app.agent.nodes import (
    ingest_node,
    plan_node,
    extract_node,
    reconcile_node,
    assemble_node,
    validate_node,
    review_and_learn_node,
)
from app.core.config import get_settings


def _route_after_plan(state: AgentState) -> str:
    """Conditional edge: decide where to go after planning."""
    status = state.get("agent_status", "")
    iteration = state.get("iteration_count", 0)
    max_iterations = get_settings().max_agent_iterations

    # Hard cap — prevent infinite loops
    if iteration >= max_iterations:
        print(f"  ⚠️ Hit iteration cap ({max_iterations}). Forcing assembly.")
        return "assemble"

    if status == "extracting":
        return "extract"
    elif status == "reconciling":
        return "reconcile"
    elif status == "assembling":
        return "assemble"
    else:
        return "extract"  # Default: keep extracting


def _route_after_extract(state: AgentState) -> str:
    """Conditional edge: decide where to go after extraction."""
    status = state.get("agent_status", "")
    iteration = state.get("iteration_count", 0)
    max_iterations = get_settings().max_agent_iterations

    if iteration >= max_iterations:
        return "assemble"

    if status == "reconciling":
        return "reconcile"
    elif status == "planning":
        return "plan"
    else:
        return "reconcile"  # Default: go to reconciliation


def _route_after_reconcile(state: AgentState) -> str:
    """Always go to assembly after reconciliation."""
    return "assemble"


def build_agent_graph() -> StateGraph:
    """
    Build and compile the LangGraph agent for discharge summary generation.
    
    Graph flow:
        START → ingest → plan → [extract | reconcile | assemble]
                            ↑         |
                            └─────────┘  (re-plan loop)
        ... → assemble → validate → review_and_learn → END
    """
    graph = StateGraph(AgentState)

    # ── Add nodes ─────────────────────────────────────
    graph.add_node("ingest", ingest_node)
    graph.add_node("plan", plan_node)
    graph.add_node("extract", extract_node)
    graph.add_node("reconcile", reconcile_node)
    graph.add_node("assemble", assemble_node)
    graph.add_node("validate", validate_node)
    graph.add_node("review_and_learn", review_and_learn_node)

    # ── Set entry point ───────────────────────────────
    graph.set_entry_point("ingest")

    # ── Add edges ─────────────────────────────────────
    # ingest always goes to plan
    graph.add_edge("ingest", "plan")

    # plan decides where to go (conditional)
    graph.add_conditional_edges(
        "plan",
        _route_after_plan,
        {
            "extract": "extract",
            "reconcile": "reconcile",
            "assemble": "assemble",
        },
    )

    # extract decides whether to re-plan or reconcile
    graph.add_conditional_edges(
        "extract",
        _route_after_extract,
        {
            "plan": "plan",
            "reconcile": "reconcile",
            "assemble": "assemble",
        },
    )

    # reconcile always goes to assemble
    graph.add_conditional_edges(
        "reconcile",
        _route_after_reconcile,
        {
            "assemble": "assemble",
        },
    )

    # assemble → validate → review_and_learn → END
    graph.add_edge("assemble", "validate")
    graph.add_edge("validate", "review_and_learn")
    graph.add_edge("review_and_learn", END)

    return graph.compile()


def run_agent(patient_pdf_paths: list[str]) -> dict:
    """
    Run the discharge summary agent on a set of patient PDFs.
    
    Returns the final state containing:
    - discharge_summary_md: Markdown discharge summary
    - discharge_summary_json: Structured JSON output (includes learning_report)
    - step_trace: Full execution trace
    - clinical_flags: All flags raised
    - conflicts: All conflicts detected
    - learning_report: Part 2 before/after metrics
    """
    print("\n" + "=" * 60)
    print("🏥 DISCHARGE SUMMARY AGENT")
    print("=" * 60)
    print(f"Patient PDFs: {patient_pdf_paths}")
    print(f"Max iterations: {get_settings().max_agent_iterations}")
    print("=" * 60)

    # Initialize state
    initial_state = {
        "patient_pdf_paths": patient_pdf_paths,
        "extracted_pages": [],
        "documents": [],
        "working_memory": {},
        "admission_medications": [],
        "discharge_medications": [],
        "medication_reconciliation": {},
        "conflicts": [],
        "clinical_flags": [],
        "missing_fields": [],
        "messages": [],
        "current_plan": "",
        "iteration_count": 0,
        "agent_status": "ingesting",
        "discharge_summary_md": "",
        "discharge_summary_json": {},
        "step_trace": [],
        # Part 2 fields
        "learning_iterations": [],
        "correction_memory": {},
        "improved_summary_md": "",
        "learning_report": {},
    }

    # Build and run the graph
    graph = build_agent_graph()
    final_state = graph.invoke(initial_state)

    print("\n" + "=" * 60)
    print("✅ AGENT COMPLETE")
    print(f"Total iterations: {final_state.get('iteration_count', 0)}")
    print(f"Flags raised: {len(final_state.get('clinical_flags', []))}")
    print(f"Conflicts found: {len(final_state.get('conflicts', []))}")

    # Part 2 summary
    report = final_state.get("learning_report", {})
    if report:
        print(f"Learning improvement: {report.get('improvement_pct', 0):.1f}%")

    print("=" * 60)

    return final_state
