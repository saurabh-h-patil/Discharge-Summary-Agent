"""
Agent Nodes — the computational steps in the LangGraph agent graph.
Each node takes the current state and returns state updates.
"""

import json
import time
from datetime import datetime
from typing import Any

from app.agent.state import AgentState
from app.core.config import get_settings
from app.core.llm import get_llm_client
from app.core.prompts import (
    AGENT_SYSTEM_PROMPT,
    PLANNING_PROMPT,
    ASSEMBLY_PROMPT,
)
from app.tools.pdf_ingestion import ingest_pdf, group_pages_into_documents
from app.tools.search import set_searchable_documents
from app.tools.safety import get_all_flags, clear_flags


def _add_trace(state: dict, step: dict) -> list[dict]:
    """Append a trace entry to the step_trace."""
    trace = state.get("step_trace", [])
    step["timestamp"] = datetime.now().isoformat()
    step["step_number"] = len(trace) + 1
    trace.append(step)
    return trace


# ────────────────────────────────────────────────────────────────
# Node 1: INGEST PDFs
# ────────────────────────────────────────────────────────────────
def ingest_node(state: AgentState) -> dict:
    """
    Ingest all patient PDFs: convert to images, extract text via vision, classify documents.
    This is the first node — it populates the extracted_pages and documents in state.
    """
    start = time.time()
    print("\n" + "=" * 60)
    print("🔬 NODE: INGEST PDFs")
    print("=" * 60)

    pdf_paths = state["patient_pdf_paths"]
    all_pages = []
    all_documents = []

    for pdf_path in pdf_paths:
        try:
            pages = ingest_pdf(pdf_path)
            page_dicts = [p.model_dump() for p in pages]
            all_pages.extend(page_dicts)

            documents = group_pages_into_documents(pages)
            doc_dicts = [d.model_dump() for d in documents]
            all_documents.extend(doc_dicts)
        except Exception as e:
            print(f"  ❌ Failed to ingest {pdf_path}: {e}")
            # Don't crash — record the failure
            all_pages.append({
                "page_number": 0,
                "source_file": pdf_path,
                "raw_text": f"[INGESTION FAILED: {str(e)}]",
                "is_blank": False,
                "extraction_confidence": "failed",
            })

    # Register documents for search tool
    searchable = [
        {
            "doc_id": d.get("doc_id", ""),
            "document_type": d.get("document_type", ""),
            "source_file": d.get("source_file", ""),
            "text": d.get("full_text", ""),
        }
        for d in all_documents
    ]
    set_searchable_documents(searchable)

    duration = (time.time() - start) * 1000
    trace = _add_trace(state, {
        "reasoning": f"Ingesting {len(pdf_paths)} PDF(s) using vision extraction",
        "action": "ingest_pdfs",
        "tool_name": "pdf_ingestion",
        "tool_inputs": {"pdf_paths": pdf_paths},
        "result_summary": f"Extracted {len(all_pages)} pages, grouped into {len(all_documents)} documents",
        "next_decision": "Plan extraction strategy based on available documents",
        "duration_ms": duration,
    })

    return {
        "extracted_pages": all_pages,
        "documents": all_documents,
        "agent_status": "planning",
        "iteration_count": state.get("iteration_count", 0) + 1,
        "step_trace": trace,
    }


# ────────────────────────────────────────────────────────────────
# Node 2: PLAN
# ────────────────────────────────────────────────────────────────
def plan_node(state: AgentState) -> dict:
    """
    Analyze what's been extracted and plan next actions.
    This is the decision-making hub — it determines what the agent should do next.
    """
    start = time.time()
    print("\n" + "=" * 60)
    print("🧠 NODE: PLAN")
    print("=" * 60)

    llm = get_llm_client()
    documents = state.get("documents", [])
    working_memory = state.get("working_memory", {})
    missing_fields = state.get("missing_fields", [])

    # Build document summary for the planner
    doc_summary = []
    for doc in documents:
        doc_type = doc.get("document_type", "UNKNOWN")
        pages = doc.get("pages", [])
        page_nums = [p.get("page_number", "?") for p in pages]
        doc_summary.append(f"- {doc.get('doc_id')}: {doc_type} (pages {page_nums})")

    available_docs = "\n".join(doc_summary) if doc_summary else "No documents extracted yet."

    # Determine what's still missing
    required_sections = [
        "demographics", "admission_date", "discharge_date",
        "principal_diagnosis", "secondary_diagnoses", "hospital_course",
        "procedures", "discharge_medications", "admission_medications",
        "allergies", "follow_up", "pending_results", "discharge_condition",
    ]

    current_missing = [s for s in required_sections if s not in working_memory or not working_memory[s]]
    if not missing_fields:
        missing_fields = current_missing

    # Ask LLM to plan
    prompt = PLANNING_PROMPT.format(
        working_memory=json.dumps(working_memory, indent=2, default=str)[:3000],
        available_documents=available_docs,
        missing_fields=json.dumps(current_missing),
    )

    plan_response = llm.chat(
        messages=[
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    try:
        plan = json.loads(plan_response)
    except json.JSONDecodeError:
        plan = {
            "assessment": plan_response,
            "missing_info": current_missing,
            "conflicts_found": [],
            "next_actions": ["Extract information from available documents"],
            "ready_to_assemble": not current_missing,
        }

    print(f"  Assessment: {plan.get('assessment', 'N/A')}")
    print(f"  Missing: {plan.get('missing_info', [])}")
    print(f"  Ready to assemble: {plan.get('ready_to_assemble', False)}")

    # Determine next status
    if plan.get("ready_to_assemble", False):
        next_status = "reconciling"
    else:
        next_status = "extracting"

    duration = (time.time() - start) * 1000
    trace = _add_trace(state, {
        "reasoning": plan.get("assessment", "Planning next steps"),
        "action": "plan",
        "tool_name": None,
        "tool_inputs": None,
        "result_summary": f"Missing: {plan.get('missing_info', [])}. Ready: {plan.get('ready_to_assemble', False)}",
        "next_decision": f"Next actions: {plan.get('next_actions', ['continue extraction'])}",
        "duration_ms": duration,
    })

    return {
        "current_plan": json.dumps(plan, default=str),
        "missing_fields": plan.get("missing_info", current_missing),
        "agent_status": next_status,
        "iteration_count": state.get("iteration_count", 0) + 1,
        "step_trace": trace,
    }


# ────────────────────────────────────────────────────────────────
# Node 3: EXTRACT INFORMATION (Phased, Document-Type-Aware)
# ────────────────────────────────────────────────────────────────
def extract_node(state: AgentState) -> dict:
    """
    Phased extraction: process documents by type with specialized extractors,
    then merge all results into unified working memory.
    
    Processing order (priority):
      1. ADMISSION_NOTE — demographics, dates, diagnoses, allergies, admission meds
      2. DISCHARGE_NOTE — discharge date, condition, follow-up, discharge meds
      3. MEDICATION_RECORD — complete medication lists
      4. LAB_RESULTS — lab values and pending results
      5. IMAGING_REPORT — imaging findings
      6. CONSULTATION — specialist findings and recommendations
      7. NURSING_NOTE — daily progress, vitals
      8. OTHER — catch-all for missed data
    """
    start = time.time()
    print("\n" + "=" * 60)
    print("📝 NODE: EXTRACT INFORMATION")
    print("=" * 60)

    from app.tools.extraction import extract_by_document_type, _safe_parse
    from app.core.prompts import MERGE_EXTRACTIONS_PROMPT

    llm = get_llm_client()
    documents = state.get("documents", [])
    working_memory = state.get("working_memory", {})
    missing_fields = state.get("missing_fields", [])
    conflicts = state.get("conflicts", [])
    flags = state.get("clinical_flags", [])

    if not documents:
        print("  ⚠️ No documents to extract from!")
        trace = _add_trace(state, {
            "reasoning": "No documents available for extraction",
            "action": "extract_information",
            "tool_name": None,
            "tool_inputs": None,
            "result_summary": "No documents available",
            "next_decision": "Flag missing data and proceed to assembly",
            "duration_ms": 0,
        })
        return {
            "agent_status": "assembling",
            "step_trace": trace,
            "iteration_count": state.get("iteration_count", 0) + 1,
        }

    # ── Phase 1: Group documents by type ──────────────────────
    type_priority = [
        "ADMISSION_NOTE", "DISCHARGE_NOTE", "MEDICATION_RECORD",
        "LAB_RESULTS", "IMAGING_REPORT", "CONSULTATION",
        "NURSING_NOTE", "OPERATIVE_NOTE", "PROGRESS_NOTE", "OTHER",
    ]

    docs_by_type = {}
    for doc in documents:
        doc_type = doc.get("document_type", "OTHER")
        if doc_type not in docs_by_type:
            docs_by_type[doc_type] = []
        docs_by_type[doc_type].append(doc)

    print(f"\n  📊 Document breakdown:")
    for dtype in type_priority:
        if dtype in docs_by_type:
            print(f"    {dtype}: {len(docs_by_type[dtype])} document(s)")

    # ── Phase 2: Process each document with type-specific extractors ──
    all_extractions = []
    docs_processed = 0
    docs_failed = 0

    for doc_type in type_priority:
        if doc_type not in docs_by_type:
            continue

        type_docs = docs_by_type[doc_type]
        print(f"\n  ── Processing {doc_type} ({len(type_docs)} docs) ──")

        for doc in type_docs:
            doc_id = doc.get("doc_id", "?")
            full_text = doc.get("full_text", "")

            if not full_text.strip():
                print(f"    ⚠️ {doc_id}: empty text, skipping")
                continue

            try:
                print(f"    📄 {doc_id}: extracting ({len(full_text)} chars)...")
                extraction = extract_by_document_type(full_text, doc_type, doc_id)
                all_extractions.append(extraction)
                docs_processed += 1
                print(f"    ✅ {doc_id}: extracted {len([k for k in extraction if not k.startswith('_')])} sections")
            except Exception as e:
                docs_failed += 1
                print(f"    ❌ {doc_id}: extraction failed: {e}")
                # Don't crash — continue with other documents

    print(f"\n  📊 Extraction complete: {docs_processed} processed, {docs_failed} failed")

    # ── Phase 3: Merge all extractions into unified working memory ──
    if all_extractions:
        print(f"\n  🔀 Merging {len(all_extractions)} extraction results...")

        # Serialize all extractions for the merge prompt
        extractions_text = ""
        for ext in all_extractions:
            doc_id = ext.get("_doc_id", "?")
            doc_type = ext.get("_doc_type", "?")
            # Remove internal keys before serializing
            ext_clean = {k: v for k, v in ext.items() if not k.startswith("_")}
            extractions_text += f"\n\n### Document: {doc_id} (Type: {doc_type})\n"
            extractions_text += json.dumps(ext_clean, indent=2, default=str)

        # Truncate if extremely long (but this is already much better than 30K on raw text)
        extractions_text = extractions_text[:60000]

        merge_prompt = MERGE_EXTRACTIONS_PROMPT.format(
            num_documents=len(all_extractions),
            all_extractions=extractions_text,
        )

        try:
            merge_result = llm.chat(
                messages=[
                    {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                    {"role": "user", "content": merge_prompt},
                ],
                temperature=0.0,
                max_tokens=8000,
                response_format={"type": "json_object"},
            )
            merged = json.loads(merge_result)
        except Exception as e:
            print(f"  ❌ Merge failed: {e}. Falling back to manual merge.")
            merged = _fallback_merge(all_extractions)

        # Update working memory from merged result
        for key, value in merged.items():
            if key in ("conflicts", "sources"):
                continue
            if value and value != "NOT_FOUND":
                working_memory[key] = value
                print(f"  ✅ Merged: {key}")
            else:
                print(f"  ⚠️ Not found across all docs: {key}")

        # Record conflicts from merge
        if "conflicts" in merged and merged["conflicts"]:
            for conflict in merged["conflicts"]:
                if isinstance(conflict, dict):
                    conflicts.append(conflict)
                    flags.append({
                        "flag_type": "CONFLICT",
                        "severity": "HIGH",
                        "description": f"Conflict in {conflict.get('field', 'unknown')}: '{conflict.get('value_1', '')}' vs '{conflict.get('value_2', '')}'",
                        "source_references": [conflict.get("source_1", ""), conflict.get("source_2", "")],
                        "requires_action": True,
                    })
                    print(f"  ⚠️ CONFLICT: {conflict.get('field', 'unknown')}")

        # Store sources for citation
        if "sources" in merged:
            working_memory["_sources"] = merged["sources"]

    # Update missing fields
    required_sections = [
        "demographics", "admission_date", "discharge_date",
        "principal_diagnosis", "secondary_diagnoses", "hospital_course",
        "procedures", "discharge_medications", "admission_medications",
        "allergies", "follow_up", "pending_results", "discharge_condition",
    ]
    updated_missing = [f for f in required_sections if f not in working_memory or not working_memory[f] or working_memory[f] == "NOT_FOUND"]

    duration = (time.time() - start) * 1000
    trace = _add_trace(state, {
        "reasoning": f"Phased extraction from {len(documents)} documents ({docs_processed} processed, {docs_failed} failed)",
        "action": "extract_information",
        "tool_name": "phased_extraction + merge",
        "tool_inputs": {
            "document_count": len(documents),
            "docs_by_type": {k: len(v) for k, v in docs_by_type.items()},
            "docs_processed": docs_processed,
        },
        "result_summary": f"Extracted from {docs_processed} docs, merged into {len(working_memory)} sections, {len(conflicts)} conflicts found",
        "next_decision": "Reconcile medications" if "admission_medications" in working_memory or "discharge_medications" in working_memory else "Assemble with available data",
        "duration_ms": duration,
    })

    return {
        "working_memory": working_memory,
        "admission_medications": working_memory.get("admission_medications", []),
        "discharge_medications": working_memory.get("discharge_medications", []),
        "conflicts": conflicts,
        "clinical_flags": flags,
        "missing_fields": updated_missing,
        "agent_status": "reconciling",
        "iteration_count": state.get("iteration_count", 0) + 1,
        "step_trace": trace,
    }


def _fallback_merge(all_extractions: list[dict]) -> dict:
    """
    Manual fallback merge when LLM merge fails.
    Combines all extractions by taking the first non-empty value for each key.
    """
    merged = {}
    list_keys = {"secondary_diagnoses", "procedures", "admission_medications",
                 "discharge_medications", "pending_results", "lab_results",
                 "imaging_findings", "consultations"}

    for ext in all_extractions:
        for key, value in ext.items():
            if key.startswith("_"):
                continue
            if not value or value == "NOT_FOUND" or value == {}:
                continue

            if key in list_keys:
                # Append to list
                if key not in merged:
                    merged[key] = []
                if isinstance(value, list):
                    merged[key].extend(value)
                elif isinstance(value, dict):
                    # Might be nested — try to get list values from it
                    for v in value.values():
                        if isinstance(v, list):
                            merged[key].extend(v)
            elif key == "demographics" and isinstance(value, dict):
                # Merge demographics dict, preferring non-NOT_FOUND values
                if "demographics" not in merged:
                    merged["demographics"] = {}
                for dk, dv in value.items():
                    if dv and dv != "NOT_FOUND" and (dk not in merged["demographics"] or merged["demographics"][dk] == "NOT_FOUND"):
                        merged["demographics"][dk] = dv
            elif key == "hospital_course":
                # Concatenate hospital course entries
                existing = merged.get("hospital_course", "")
                if isinstance(value, str) and value != "NOT_FOUND":
                    merged["hospital_course"] = (existing + "\n\n" + value).strip() if existing else value
                elif isinstance(value, dict):
                    course_text = value.get("course", value.get("summary", value.get("narrative", str(value))))
                    if course_text and course_text != "NOT_FOUND":
                        merged["hospital_course"] = (existing + "\n\n" + str(course_text)).strip() if existing else str(course_text)
            else:
                # Take first non-empty value
                if key not in merged or merged[key] == "NOT_FOUND":
                    merged[key] = value

    return merged



# ────────────────────────────────────────────────────────────────
# Node 4: RECONCILE MEDICATIONS
# ────────────────────────────────────────────────────────────────
def reconcile_node(state: AgentState) -> dict:
    """
    Compare admission and discharge medications, flag changes and interactions.
    """
    start = time.time()
    print("\n" + "=" * 60)
    print("💊 NODE: RECONCILE MEDICATIONS")
    print("=" * 60)

    llm = get_llm_client()
    admission_meds = state.get("admission_medications", [])
    discharge_meds = state.get("discharge_medications", [])
    working_memory = state.get("working_memory", {})
    flags = state.get("clinical_flags", [])
    reconciliation = {}

    if not admission_meds and not discharge_meds:
        print("  ⚠️ No medication lists available for reconciliation")
        flags.append({
            "flag_type": "MISSING_DATA",
            "severity": "HIGH",
            "description": "Medication lists not found in source documents. Cannot perform reconciliation.",
            "source_references": [],
            "requires_action": True,
        })
        reconciliation = {"status": "UNAVAILABLE", "reason": "No medication lists found"}
    else:
        # Use LLM for reconciliation
        try:
            from app.core.prompts import MEDICATION_RECONCILIATION_PROMPT

            prompt = MEDICATION_RECONCILIATION_PROMPT.format(
                admission_meds=json.dumps(admission_meds, indent=2, default=str),
                discharge_meds=json.dumps(discharge_meds, indent=2, default=str),
            )

            result = llm.chat(
                messages=[
                    {"role": "system", "content": "You are a clinical pharmacist. Compare medication lists and identify ALL changes. Flag anything without a documented reason."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )

            reconciliation = json.loads(result)

            # Flag undocumented changes
            for med_type in ["new", "discontinued", "dose_changed"]:
                for med in reconciliation.get(med_type, []):
                    if isinstance(med, dict):
                        reason = med.get("reason", "")
                        if not reason or "NO DOCUMENTED REASON" in reason.upper():
                            flags.append({
                                "flag_type": "MEDICATION_SAFETY",
                                "severity": "MEDIUM",
                                "description": f"Medication {med_type}: {med.get('name', 'unknown')} — No documented reason for change",
                                "source_references": [],
                                "requires_action": True,
                            })
                            print(f"  🚩 Flagged: {med_type} - {med.get('name', 'unknown')} (no reason)")

            # Check drug interactions on discharge meds
            if discharge_meds:
                med_names = []
                for med in discharge_meds:
                    if isinstance(med, dict):
                        med_names.append(med.get("name", ""))
                    elif isinstance(med, str):
                        med_names.append(med)

                from app.tools.medication import MOCK_DRUG_INTERACTIONS
                med_names_lower = [m.lower() for m in med_names if m]

                for i in range(len(med_names_lower)):
                    for j in range(i + 1, len(med_names_lower)):
                        for (drug_a, drug_b), interaction in MOCK_DRUG_INTERACTIONS.items():
                            if (drug_a in med_names_lower[i] and drug_b in med_names_lower[j]) or \
                               (drug_b in med_names_lower[i] and drug_a in med_names_lower[j]):
                                flags.append({
                                    "flag_type": "DRUG_INTERACTION",
                                    "severity": interaction["severity"],
                                    "description": f"Potential interaction: {med_names[i]} + {med_names[j]} — {interaction['description']}",
                                    "source_references": [],
                                    "requires_action": True,
                                })
                                print(f"  ⚠️ Drug interaction: {med_names[i]} + {med_names[j]}")

            print(f"  ✅ Reconciliation complete")
            print(f"    Continued: {len(reconciliation.get('continued', []))}")
            print(f"    New: {len(reconciliation.get('new', []))}")
            print(f"    Discontinued: {len(reconciliation.get('discontinued', []))}")
            print(f"    Dose changed: {len(reconciliation.get('dose_changed', []))}")

        except Exception as e:
            print(f"  ❌ Reconciliation failed: {e}")
            reconciliation = {"status": "FAILED", "error": str(e)}

    working_memory["medication_reconciliation"] = reconciliation

    duration = (time.time() - start) * 1000
    trace = _add_trace(state, {
        "reasoning": "Reconciling admission vs discharge medications and checking for drug interactions",
        "action": "reconcile_medications",
        "tool_name": "medication_reconciliation",
        "tool_inputs": {
            "admission_med_count": len(admission_meds) if isinstance(admission_meds, list) else 0,
            "discharge_med_count": len(discharge_meds) if isinstance(discharge_meds, list) else 0,
        },
        "result_summary": f"Reconciliation: {json.dumps({k: len(v) if isinstance(v, list) else v for k, v in reconciliation.items()}, default=str)[:500]}",
        "next_decision": "Proceed to assemble the discharge summary draft",
        "duration_ms": duration,
    })

    return {
        "working_memory": working_memory,
        "medication_reconciliation": reconciliation,
        "clinical_flags": flags,
        "agent_status": "assembling",
        "iteration_count": state.get("iteration_count", 0) + 1,
        "step_trace": trace,
    }


# ────────────────────────────────────────────────────────────────
# Node 5: ASSEMBLE DISCHARGE SUMMARY
# ────────────────────────────────────────────────────────────────
def assemble_node(state: AgentState) -> dict:
    """
    Assemble the final structured discharge summary from all gathered information.
    """
    start = time.time()
    print("\n" + "=" * 60)
    print("📋 NODE: ASSEMBLE DISCHARGE SUMMARY")
    print("=" * 60)

    llm = get_llm_client()
    working_memory = state.get("working_memory", {})
    conflicts = state.get("conflicts", [])
    missing_fields = state.get("missing_fields", [])
    flags = state.get("clinical_flags", [])
    reconciliation = state.get("medication_reconciliation", {})

    # Build medication changes summary
    med_changes = []
    for change_type in ["new", "discontinued", "dose_changed"]:
        for med in reconciliation.get(change_type, []):
            if isinstance(med, dict):
                med_changes.append(f"{change_type.upper()}: {med.get('name', 'unknown')} — {med.get('reason', 'NO DOCUMENTED REASON')}")

    prompt = ASSEMBLY_PROMPT.format(
        working_memory=json.dumps(working_memory, indent=2, default=str)[:30000],
        conflicts=json.dumps(conflicts, indent=2, default=str) if conflicts else "None detected",
        missing_fields=json.dumps(missing_fields) if missing_fields else "All fields populated",
        medication_changes="\n".join(med_changes) if med_changes else "No changes detected or medication lists unavailable",
        flags=json.dumps(flags, indent=2, default=str)[:3000] if flags else "No flags raised",
    )

    try:
        summary_md = llm.chat(
            messages=[
                {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=8000,
        )
    except Exception as e:
        summary_md = f"# DISCHARGE SUMMARY — DRAFT\n\n> ⚠️ Assembly failed: {str(e)}\n\nPlease review source documents manually."

    # Build structured JSON output
    summary_json = {
        "is_draft": True,
        "generated_at": datetime.now().isoformat(),
        "working_memory": working_memory,
        "conflicts": conflicts,
        "missing_fields": missing_fields,
        "clinical_flags": flags,
        "medication_reconciliation": reconciliation,
    }

    print(f"  ✅ Discharge summary assembled ({len(summary_md)} characters)")
    print(f"  📊 Flags: {len(flags)}, Conflicts: {len(conflicts)}, Missing: {len(missing_fields)}")

    duration = (time.time() - start) * 1000
    trace = _add_trace(state, {
        "reasoning": "Assembling final discharge summary from all gathered information",
        "action": "assemble_summary",
        "tool_name": "assembly",
        "tool_inputs": {
            "sections_available": list(working_memory.keys()),
            "conflicts_count": len(conflicts),
            "flags_count": len(flags),
        },
        "result_summary": f"Summary assembled: {len(summary_md)} chars, {len(flags)} flags, {len(conflicts)} conflicts",
        "next_decision": "Validate the summary for safety compliance",
        "duration_ms": duration,
    })

    return {
        "discharge_summary_md": summary_md,
        "discharge_summary_json": summary_json,
        "agent_status": "validating",
        "iteration_count": state.get("iteration_count", 0) + 1,
        "step_trace": trace,
    }


# ────────────────────────────────────────────────────────────────
# Node 6: VALIDATE SAFETY
# ────────────────────────────────────────────────────────────────
def validate_node(state: AgentState) -> dict:
    """
    Post-generation safety validation:
    - Check for potential fabrication (unsourced claims)
    - Verify all required sections exist
    - Ensure pending/missing data is properly flagged
    """
    start = time.time()
    print("\n" + "=" * 60)
    print("🛡️ NODE: VALIDATE SAFETY")
    print("=" * 60)

    llm = get_llm_client()
    summary_md = state.get("discharge_summary_md", "")
    flags = state.get("clinical_flags", [])

    # Safety validation prompt
    validation_prompt = f"""Review this AI-generated discharge summary draft for safety issues.

DISCHARGE SUMMARY:
---
{summary_md[:8000]}
---

Check for:
1. FABRICATION: Any clinical facts that appear to be invented (not supported by source citations)
2. MISSING SECTIONS: Any required sections that are absent
3. UNMARKED MISSING DATA: Any data that should be marked "[NOT FOUND]" or "[PENDING]" but isn't
4. CONFLICTS NOT FLAGGED: Any discrepancies that should be highlighted
5. MEDICATION SAFETY: Any medication concerns not flagged

Return a JSON object:
{{
    "is_safe": true/false,
    "fabrication_risks": ["list of suspected fabrications"],
    "missing_sections": ["list of missing required sections"],
    "unmarked_missing_data": ["list of data that should be marked missing"],
    "unflagged_conflicts": ["list of unflagged conflicts"],
    "medication_concerns": ["list of medication safety concerns"],
    "overall_assessment": "summary of safety status"
}}"""

    try:
        result = llm.chat(
            messages=[
                {"role": "system", "content": "You are a clinical safety reviewer. Your job is to catch potential errors and omissions in AI-generated discharge summaries. Be thorough and conservative — it is better to flag a false positive than miss a real issue."},
                {"role": "user", "content": validation_prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        validation = json.loads(result)
    except Exception as e:
        print(f"  ⚠️ Validation failed: {e}")
        validation = {
            "is_safe": False,
            "overall_assessment": f"Validation check failed: {str(e)}. Manual review required.",
        }

    # Add validation-discovered flags
    for risk in validation.get("fabrication_risks", []):
        flags.append({
            "flag_type": "FABRICATION_RISK",
            "severity": "HIGH",
            "description": f"Potential fabrication detected: {risk}",
            "source_references": [],
            "requires_action": True,
        })
        print(f"  🔴 FABRICATION RISK: {risk}")

    for concern in validation.get("medication_concerns", []):
        flags.append({
            "flag_type": "MEDICATION_SAFETY",
            "severity": "HIGH",
            "description": concern,
            "source_references": [],
            "requires_action": True,
        })

    is_safe = validation.get("is_safe", False)
    print(f"  Safety check: {'✅ PASSED' if is_safe else '⚠️ ISSUES FOUND'}")
    print(f"  Assessment: {validation.get('overall_assessment', 'N/A')}")

    duration = (time.time() - start) * 1000
    trace = _add_trace(state, {
        "reasoning": "Running post-generation safety validation on the discharge summary",
        "action": "validate_safety",
        "tool_name": "safety_validator",
        "tool_inputs": {"summary_length": len(summary_md)},
        "result_summary": f"Safety: {'PASSED' if is_safe else 'ISSUES FOUND'}. {validation.get('overall_assessment', '')}",
        "next_decision": "Finalize output" if is_safe else "Finalize with safety warnings",
        "duration_ms": duration,
    })

    return {
        "clinical_flags": flags,
        "agent_status": "learning",
        "iteration_count": state.get("iteration_count", 0) + 1,
        "step_trace": trace,
        "discharge_summary_json": {
            **state.get("discharge_summary_json", {}),
            "safety_validation": validation,
            "clinical_flags": flags,
        },
    }


# ────────────────────────────────────────────────────────────────
# Node 7: REVIEW & LEARN (Part 2)
# ────────────────────────────────────────────────────────────────
def review_and_learn_node(state: AgentState) -> dict:
    """
    Part 2 — Learning from Doctor Edits.
    
    Uses the already-generated discharge summary and runs a learning loop:
    1. Simulated reviewer edits the draft (consistent hidden editing policies)
    2. Compute metrics (edit distance = reward signal)
    3. Extract structured corrections into memory
    4. Regenerate the summary with corrections injected
    5. Repeat for N iterations, measuring improvement
    """
    start = time.time()
    print("\n" + "=" * 60)
    print("📚 NODE: REVIEW & LEARN (Part 2)")
    print("=" * 60)

    from app.part2.simulated_reviewer import review_draft
    from app.part2.metrics import compute_all_metrics
    from app.part2.learning import CorrectionMemory

    llm = get_llm_client()
    summary_md = state.get("discharge_summary_md", "")
    working_memory = state.get("working_memory", {})
    conflicts = state.get("conflicts", [])
    missing_fields = state.get("missing_fields", [])
    flags = state.get("clinical_flags", [])
    reconciliation = state.get("medication_reconciliation", {})

    if not summary_md:
        print("  ⚠️ No summary to review — skipping learning")
        trace = _add_trace(state, {
            "reasoning": "No discharge summary available for review",
            "action": "review_and_learn",
            "tool_name": None,
            "tool_inputs": None,
            "result_summary": "Skipped — no summary",
            "next_decision": "End",
            "duration_ms": 0,
        })
        return {
            "agent_status": "done",
            "step_trace": trace,
            "learning_iterations": [],
            "learning_report": {},
        }

    # Initialize correction memory
    memory = CorrectionMemory()
    num_iterations = 3  # Keep it efficient — 3 iterations shows the pattern
    iteration_results = []

    current_draft = summary_md

    for iteration in range(num_iterations):
        iter_start = time.time()
        print(f"\n  {'─' * 40}")
        print(f"  🔄 Learning Iteration {iteration + 1}/{num_iterations}")

        # Step 1: Simulated reviewer edits the draft
        print(f"    👨‍⚕️ Simulated reviewer editing...")
        try:
            edited = review_draft(current_draft)
        except Exception as e:
            print(f"    ❌ Review failed: {e}")
            break

        # Step 2: Compute metrics
        metrics = compute_all_metrics(current_draft, edited)
        print(f"    📊 Edit burden: {metrics['edit_burden']:.4f}")
        print(f"    📊 Reward:      {metrics['reward']:.4f}")
        print(f"    📊 Accuracy:    {metrics['section_accuracy']['overall_accuracy']:.4f}")

        # Step 3: Extract corrections
        print(f"    🧠 Extracting corrections...")
        try:
            new_corrections = memory.extract_corrections(current_draft, edited)
            print(f"    🧠 Learned {len(new_corrections)} corrections (total: {len(memory.corrections)})")
        except Exception as e:
            print(f"    ⚠️ Correction extraction failed: {e}")

        # Record iteration result
        iter_result = {
            "iteration": iteration + 1,
            "edit_burden": metrics["edit_burden"],
            "reward": metrics["reward"],
            "section_accuracy": metrics["section_accuracy"]["overall_accuracy"],
            "corrections_total": len(memory.corrections),
            "rules_learned": len(memory.rules_learned),
            "duration_s": round(time.time() - iter_start, 1),
        }
        iteration_results.append(iter_result)

        # Step 4: Regenerate draft WITH corrections for next iteration
        if iteration < num_iterations - 1:
            correction_prompt = memory.build_correction_prompt()
            if correction_prompt:
                print(f"    📝 Regenerating with {len(memory.corrections)} learned corrections...")

                med_changes = []
                for change_type in ["new", "discontinued", "dose_changed"]:
                    for med in reconciliation.get(change_type, []):
                        if isinstance(med, dict):
                            med_changes.append(
                                f"{change_type.upper()}: {med.get('name', 'unknown')} — "
                                f"{med.get('reason', 'NO DOCUMENTED REASON')}"
                            )

                prompt = ASSEMBLY_PROMPT.format(
                    working_memory=json.dumps(working_memory, indent=2, default=str)[:30000],
                    conflicts=json.dumps(conflicts, indent=2, default=str) if conflicts else "None detected",
                    missing_fields=json.dumps(missing_fields) if missing_fields else "All fields populated",
                    medication_changes="\n".join(med_changes) if med_changes else "No changes detected",
                    flags=json.dumps(flags, indent=2, default=str)[:3000] if flags else "No flags raised",
                )
                prompt += "\n\n" + correction_prompt

                try:
                    current_draft = llm.chat(
                        messages=[
                            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.1,
                        max_tokens=8000,
                    )
                except Exception as e:
                    print(f"    ❌ Regeneration failed: {e}")
                    break

    # ── Build Learning Report ──────────────────────────────
    if len(iteration_results) >= 2:
        baseline = iteration_results[0]
        final = iteration_results[-1]
        improvement = baseline["edit_burden"] - final["edit_burden"]
        improvement_pct = (improvement / baseline["edit_burden"] * 100) if baseline["edit_burden"] > 0 else 0
    else:
        baseline = iteration_results[0] if iteration_results else {}
        final = baseline
        improvement = 0
        improvement_pct = 0

    learning_report = {
        "num_iterations": len(iteration_results),
        "baseline_edit_burden": baseline.get("edit_burden", 0),
        "final_edit_burden": final.get("edit_burden", 0),
        "improvement": round(improvement, 4),
        "improvement_pct": round(improvement_pct, 1),
        "baseline_accuracy": baseline.get("section_accuracy", 0),
        "final_accuracy": final.get("section_accuracy", 0),
        "total_corrections": len(memory.corrections),
        "rules_learned": memory.rules_learned,
        "correction_memory_stats": memory.stats(),
        "improvement_curve": {
            "iterations": [r["iteration"] for r in iteration_results],
            "edit_burden": [r["edit_burden"] for r in iteration_results],
            "reward": [r["reward"] for r in iteration_results],
        },
    }

    # Print summary
    print(f"\n  {'─' * 40}")
    print(f"  📊 LEARNING RESULTS")
    print(f"    Baseline edit burden: {baseline.get('edit_burden', 0):.4f}")
    print(f"    Final edit burden:    {final.get('edit_burden', 0):.4f}")
    print(f"    Improvement:          {improvement:.4f} ({improvement_pct:.1f}%)")
    print(f"    Corrections learned:  {len(memory.corrections)}")
    print(f"    Rules learned:        {len(memory.rules_learned)}")

    # Print improvement curve
    print(f"\n  📈 IMPROVEMENT CURVE (Edit Burden ↓ = Better)")
    burdens = [r["edit_burden"] for r in iteration_results]
    if burdens:
        max_b = max(burdens)
        for i, b in enumerate(burdens):
            bar_len = int((b / max_b) * 25) if max_b > 0 else 0
            bar = "█" * bar_len + "░" * (25 - bar_len)
            print(f"    Iter {i+1}: {bar} {b:.4f}")

    duration = (time.time() - start) * 1000
    trace = _add_trace(state, {
        "reasoning": f"Part 2: Ran {len(iteration_results)} learning iterations with simulated reviewer",
        "action": "review_and_learn",
        "tool_name": "simulated_reviewer + correction_memory",
        "tool_inputs": {"iterations": len(iteration_results)},
        "result_summary": f"Edit burden: {baseline.get('edit_burden', 0):.4f} → {final.get('edit_burden', 0):.4f} ({improvement_pct:.1f}% improvement). Learned {len(memory.corrections)} corrections.",
        "next_decision": "Agent complete",
        "duration_ms": duration,
    })

    return {
        "agent_status": "done",
        "improved_summary_md": current_draft,
        "learning_iterations": iteration_results,
        "correction_memory": memory.stats(),
        "learning_report": learning_report,
        "step_trace": trace,
        "discharge_summary_json": {
            **state.get("discharge_summary_json", {}),
            "learning_report": learning_report,
        },
    }
