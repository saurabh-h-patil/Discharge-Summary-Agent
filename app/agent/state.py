"""
Agent State — the typed state definition for the LangGraph agent.
This is the single source of truth the agent accumulates as it works.
"""

from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """State for the discharge summary agent graph."""

    # ── Input ────────────────────────────────────────
    # Path to the patient's source PDFs
    patient_pdf_paths: list[str]

    # ── Extracted Documents ──────────────────────────
    # Raw extracted pages from PDFs
    extracted_pages: list[dict]
    # Classified and grouped documents
    documents: list[dict]

    # ── Working Memory ───────────────────────────────
    # Accumulated clinical findings, keyed by section
    working_memory: dict  # e.g. {"demographics": {...}, "diagnoses": {...}, ...}

    # ── Medications ──────────────────────────────────
    admission_medications: list[dict]
    discharge_medications: list[dict]
    medication_reconciliation: dict

    # ── Conflicts & Flags ────────────────────────────
    conflicts: list[dict]
    clinical_flags: list[dict]
    missing_fields: list[str]

    # ── Agent Control ────────────────────────────────
    messages: Annotated[list, add_messages]  # LLM conversation messages
    current_plan: str  # What the agent plans to do next
    iteration_count: int  # Hard cap counter
    agent_status: str  # "ingesting", "planning", "extracting", "reconciling", "assembling", "validating", "done", "error"

    # ── Output ───────────────────────────────────────
    discharge_summary_md: str  # Final markdown output
    discharge_summary_json: dict  # Structured JSON output
    step_trace: list[dict]  # Observability trace

    # ── Part 2: Learning from Edits ──────────────────
    learning_iterations: list[dict]  # Metrics per learning iteration
    correction_memory: dict  # Persisted correction memory
    improved_summary_md: str  # Summary after learning iterations
    learning_report: dict  # Final before/after report
