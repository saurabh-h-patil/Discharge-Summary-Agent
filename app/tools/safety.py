"""
Safety & Escalation Tools — mock tools for flagging and escalating concerns.
These simulate the clinical safety workflow the agent must trigger.
"""

import json
from datetime import datetime
from typing import Optional
from langchain_core.tools import tool


# In-memory store for flags (in production, this would go to a database)
_clinical_flags: list[dict] = []


@tool
def flag_for_clinician_review(
    issue_type: str,
    description: str,
    severity: str = "MEDIUM",
    source_references: str = "",
) -> str:
    """Flag an issue for clinician review. Use this when:
    - Required information is missing from the records
    - Conflicting information is found between documents
    - A medication change has no documented reason
    - A potential safety concern is identified
    - Lab results are pending or abnormal
    
    Args:
        issue_type: CONFLICT, MISSING_DATA, MEDICATION_SAFETY, DRUG_INTERACTION, ABNORMAL_LAB, ESCALATION
        description: Clear description of the issue
        severity: HIGH, MEDIUM, or LOW
        source_references: Comma-separated source references (e.g., "Admission Note Page 2, Progress Note Page 5")
    """
    flag = {
        "flag_id": f"FLAG-{len(_clinical_flags) + 1:03d}",
        "timestamp": datetime.now().isoformat(),
        "issue_type": issue_type,
        "severity": severity,
        "description": description,
        "source_references": [s.strip() for s in source_references.split(",") if s.strip()],
        "status": "PENDING_REVIEW",
        "reviewed_by": None,
    }
    _clinical_flags.append(flag)

    return json.dumps({
        "status": "flagged",
        "flag_id": flag["flag_id"],
        "message": f"Issue flagged for clinician review: {description}",
        "severity": severity,
        "note": "MOCK TOOL — In production, this would create an alert in the clinical workflow system.",
    })


@tool
def escalate_to_physician(
    reason: str,
    urgency: str = "ROUTINE",
    patient_context: str = "",
) -> str:
    """Escalate a critical finding to the attending physician. Use this for:
    - Critical drug interactions detected
    - Life-threatening lab values
    - Significant clinical deterioration noted in records
    - Missing critical safety information (allergies not documented)
    
    Args:
        reason: Why escalation is needed
        urgency: STAT, URGENT, or ROUTINE
        patient_context: Brief patient context for the physician
    """
    escalation = {
        "escalation_id": f"ESC-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "timestamp": datetime.now().isoformat(),
        "reason": reason,
        "urgency": urgency,
        "patient_context": patient_context,
        "status": "SENT",
    }

    return json.dumps({
        "status": "escalated",
        "escalation_id": escalation["escalation_id"],
        "message": f"Escalated to attending physician: {reason}",
        "urgency": urgency,
        "note": "MOCK TOOL — In production, this would send a page/notification to the attending physician.",
    })


def get_all_flags() -> list[dict]:
    """Retrieve all clinical flags raised during this session."""
    return _clinical_flags.copy()


def clear_flags():
    """Clear all flags (for testing/reset)."""
    global _clinical_flags
    _clinical_flags = []
