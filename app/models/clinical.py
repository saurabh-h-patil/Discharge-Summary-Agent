"""
Pydantic models for clinical data structures used throughout the agent.
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class DocumentType(str, Enum):
    """Types of clinical documents the agent can encounter."""
    ADMISSION_NOTE = "ADMISSION_NOTE"
    PROGRESS_NOTE = "PROGRESS_NOTE"
    LAB_RESULTS = "LAB_RESULTS"
    MEDICATION_RECORD = "MEDICATION_RECORD"
    DISCHARGE_NOTE = "DISCHARGE_NOTE"
    NURSING_NOTE = "NURSING_NOTE"
    CONSULTATION = "CONSULTATION"
    OPERATIVE_NOTE = "OPERATIVE_NOTE"
    IMAGING_REPORT = "IMAGING_REPORT"
    OTHER = "OTHER"


class ExtractedPage(BaseModel):
    """A single extracted page from a PDF."""
    page_number: int
    source_file: str
    raw_text: str
    document_type: Optional[DocumentType] = None
    is_blank: bool = False
    extraction_confidence: str = "high"  # high, medium, low


class ExtractedDocument(BaseModel):
    """A classified clinical document, possibly spanning multiple pages."""
    doc_id: str
    document_type: DocumentType
    source_file: str
    pages: list[ExtractedPage]
    full_text: str = ""

    def get_full_text(self) -> str:
        """Combine all page texts."""
        if self.full_text:
            return self.full_text
        return "\n\n".join(
            f"[Page {p.page_number}]\n{p.raw_text}"
            for p in self.pages
            if not p.is_blank
        )


class Medication(BaseModel):
    """A single medication entry."""
    name: str
    dose: Optional[str] = None
    route: Optional[str] = None
    frequency: Optional[str] = None
    indication: Optional[str] = None
    source_page: Optional[int] = None


class MedicationChange(BaseModel):
    """A medication change between admission and discharge."""
    medication_name: str
    change_type: str  # CONTINUED, NEW, DISCONTINUED, DOSE_CHANGED
    old_dose: Optional[str] = None
    new_dose: Optional[str] = None
    reason: Optional[str] = None
    flagged: bool = False
    flag_reason: Optional[str] = None


class LabResult(BaseModel):
    """A single laboratory result."""
    test_name: str
    value: Optional[str] = None
    unit: Optional[str] = None
    reference_range: Optional[str] = None
    status: str = "final"  # final, pending, preliminary
    date: Optional[str] = None
    is_abnormal: bool = False
    source_page: Optional[int] = None


class ClinicalFlag(BaseModel):
    """A flag raised for clinician review."""
    flag_type: str  # CONFLICT, MISSING_DATA, MEDICATION_SAFETY, DRUG_INTERACTION, ESCALATION
    severity: str  # HIGH, MEDIUM, LOW
    description: str
    source_references: list[str] = Field(default_factory=list)
    requires_action: bool = True


class ConflictInfo(BaseModel):
    """Information about a conflict between source documents."""
    field: str
    value_1: str
    source_1: str
    value_2: str
    source_2: str
    resolution: Optional[str] = None  # None means unresolved — needs clinician


class StepTrace(BaseModel):
    """A single step in the agent's execution trace."""
    step_number: int
    reasoning: str
    action: str
    tool_name: Optional[str] = None
    tool_inputs: Optional[dict] = None
    result_summary: str
    next_decision: str
    timestamp: str
    duration_ms: Optional[float] = None
