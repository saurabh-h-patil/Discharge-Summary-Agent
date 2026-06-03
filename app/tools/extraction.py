"""
Clinical Data Extraction Tools — extract specific clinical information from document text.
These are registered as LangGraph tools the agent can call.

Each extractor is specialized for a document type and returns parsed JSON dicts
for direct merging into working memory.
"""

import json
from typing import Optional
from langchain_core.tools import tool

from app.core.llm import get_llm_client
from app.core.prompts import EXTRACTION_PROMPT


# ── Helper ─────────────────────────────────────────────────────
def _safe_parse(raw: str) -> dict:
    """Parse LLM JSON response, returning empty dict on failure."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


# ── Extraction Tools ───────────────────────────────────────────

@tool
def extract_demographics(doc_text: str, doc_type: str = "ADMISSION_NOTE") -> str:
    """Extract patient demographics (name, age, sex, MRN, etc.) from clinical document text."""
    llm = get_llm_client()

    prompt = EXTRACTION_PROMPT.format(
        doc_type=doc_type,
        doc_text=doc_text[:12000],
        target_info="Patient demographics: full name, age, date of birth, sex/gender, MRN (medical record number), address, phone, emergency contact, insurance, ward/bed number, attending physician",
    )

    result = llm.chat(
        messages=[
            {"role": "system", "content": "You are a clinical data extractor. Extract ONLY explicitly stated information. Return JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return result


@tool
def extract_diagnoses(doc_text: str, doc_type: str = "ADMISSION_NOTE") -> str:
    """Extract principal and secondary diagnoses from clinical document text."""
    llm = get_llm_client()

    prompt = EXTRACTION_PROMPT.format(
        doc_type=doc_type,
        doc_text=doc_text[:12000],
        target_info="Principal diagnosis, secondary/additional diagnoses, admitting diagnosis, and any ICD codes if present",
    )

    result = llm.chat(
        messages=[
            {"role": "system", "content": "You are a clinical data extractor. Extract ONLY explicitly stated diagnoses. Never infer diagnoses. Return JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return result


@tool
def extract_medications(doc_text: str, doc_type: str = "MEDICATION_RECORD") -> str:
    """Extract medication list (name, dose, route, frequency) from clinical document text."""
    llm = get_llm_client()

    prompt = EXTRACTION_PROMPT.format(
        doc_type=doc_type,
        doc_text=doc_text[:15000],
        target_info="Complete medication list: each medication's name, dose, route, frequency, indication, date/time if available. Note whether this is an admission or discharge medication list. Include ALL medications — oral, IV, injections, PRN.",
    )

    result = llm.chat(
        messages=[
            {"role": "system", "content": "You are a clinical data extractor. Extract EVERY medication mentioned. Include exact doses and frequencies. Return JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return result


@tool
def extract_lab_results(doc_text: str, doc_type: str = "LAB_RESULTS") -> str:
    """Extract laboratory results, marking any pending/incomplete tests."""
    llm = get_llm_client()

    prompt = EXTRACTION_PROMPT.format(
        doc_type=doc_type,
        doc_text=doc_text[:15000],
        target_info="All laboratory results: test name, value, units, reference range, date, and status (final/pending/preliminary). Mark abnormal values. If a result says 'pending', mark it as PENDING.",
    )

    result = llm.chat(
        messages=[
            {"role": "system", "content": "You are a clinical data extractor. Extract lab results exactly as written. If a result is pending, mark it PENDING — do NOT make up a value. Return JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return result


@tool
def extract_hospital_course(doc_text: str, doc_type: str = "PROGRESS_NOTE") -> str:
    """Extract hospital course / clinical narrative from progress notes or discharge notes."""
    llm = get_llm_client()

    prompt = EXTRACTION_PROMPT.format(
        doc_type=doc_type,
        doc_text=doc_text[:15000],
        target_info="Hospital course: timeline of events, treatments given, clinical progress, complications, consultations, response to treatment. Include dates and times when available. Include vital signs, intake/output, and clinical observations.",
    )

    result = llm.chat(
        messages=[
            {"role": "system", "content": "You are a clinical data extractor. Summarize the hospital course based ONLY on what is documented. Do not add interpretation. Return JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return result


@tool
def extract_procedures(doc_text: str, doc_type: str = "OPERATIVE_NOTE") -> str:
    """Extract procedures performed during the hospital stay."""
    llm = get_llm_client()

    prompt = EXTRACTION_PROMPT.format(
        doc_type=doc_type,
        doc_text=doc_text[:12000],
        target_info="All procedures performed: procedure name, date, surgeon/provider, findings, complications. Include both surgical and non-surgical procedures.",
    )

    result = llm.chat(
        messages=[
            {"role": "system", "content": "You are a clinical data extractor. Extract ONLY documented procedures. Return JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return result


@tool
def extract_allergies(doc_text: str, doc_type: str = "ADMISSION_NOTE") -> str:
    """Extract patient allergies and adverse reactions from clinical document text."""
    llm = get_llm_client()

    prompt = EXTRACTION_PROMPT.format(
        doc_type=doc_type,
        doc_text=doc_text[:12000],
        target_info="Patient allergies: allergen name, type of reaction, severity. Include drug allergies, food allergies, and environmental allergies. Note if 'NKDA' (no known drug allergies) is stated.",
    )

    result = llm.chat(
        messages=[
            {"role": "system", "content": "You are a clinical data extractor. Extract ONLY documented allergies. Return JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return result


@tool
def extract_discharge_info(doc_text: str, doc_type: str = "DISCHARGE_NOTE") -> str:
    """Extract discharge-specific information: condition, instructions, follow-up."""
    llm = get_llm_client()

    prompt = EXTRACTION_PROMPT.format(
        doc_type=doc_type,
        doc_text=doc_text[:12000],
        target_info="Discharge information: discharge date, discharge condition (stable/improved/etc.), discharge disposition (home/facility/etc.), follow-up appointments with dates and doctor names, discharge instructions, activity restrictions, diet instructions, return-to-ED criteria, discharge medications.",
    )

    result = llm.chat(
        messages=[
            {"role": "system", "content": "You are a clinical data extractor. Extract ONLY documented discharge information. Return JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return result


@tool
def extract_imaging_report(doc_text: str, doc_type: str = "IMAGING_REPORT") -> str:
    """Extract imaging/radiology findings from imaging reports."""
    llm = get_llm_client()

    prompt = EXTRACTION_PROMPT.format(
        doc_type=doc_type,
        doc_text=doc_text[:12000],
        target_info="Imaging findings: modality (X-ray/CT/MRI/USG/etc.), body part, date, findings (detailed), impression/conclusion, radiologist name. Include ALL findings — normal and abnormal.",
    )

    result = llm.chat(
        messages=[
            {"role": "system", "content": "You are a clinical data extractor. Extract imaging findings exactly as documented. Do NOT interpret beyond what the report states. Return JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return result


@tool
def extract_consultation(doc_text: str, doc_type: str = "CONSULTATION") -> str:
    """Extract specialist consultation findings and recommendations."""
    llm = get_llm_client()

    prompt = EXTRACTION_PROMPT.format(
        doc_type=doc_type,
        doc_text=doc_text[:15000],
        target_info="Consultation details: specialist type, consultant name, date, reason for consultation, findings, diagnosis/impression, recommendations, follow-up plan. Include ALL recommendations made.",
    )

    result = llm.chat(
        messages=[
            {"role": "system", "content": "You are a clinical data extractor. Extract consultation findings exactly as documented. Return JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return result


# ── Dispatcher: route documents to the right extractor ─────────

def extract_by_document_type(doc_text: str, doc_type: str, doc_id: str) -> dict:
    """
    Route a document to the appropriate extraction function(s) based on its type.
    Returns a dict with extracted fields ready for merging into working memory.
    
    Each document type gets processed by the extractors that are most relevant to it.
    """
    result = {"_doc_id": doc_id, "_doc_type": doc_type}

    if doc_type == "ADMISSION_NOTE":
        # Admission notes contain demographics, diagnoses, allergies, admission meds, procedures
        demo_raw = extract_demographics.invoke({"doc_text": doc_text, "doc_type": doc_type})
        result["demographics"] = _safe_parse(demo_raw)

        diag_raw = extract_diagnoses.invoke({"doc_text": doc_text, "doc_type": doc_type})
        result["diagnoses"] = _safe_parse(diag_raw)

        allergy_raw = extract_allergies.invoke({"doc_text": doc_text, "doc_type": doc_type})
        result["allergies"] = _safe_parse(allergy_raw)

        meds_raw = extract_medications.invoke({"doc_text": doc_text, "doc_type": doc_type})
        result["medications"] = _safe_parse(meds_raw)

        course_raw = extract_hospital_course.invoke({"doc_text": doc_text, "doc_type": doc_type})
        result["hospital_course"] = _safe_parse(course_raw)

        proc_raw = extract_procedures.invoke({"doc_text": doc_text, "doc_type": doc_type})
        result["procedures"] = _safe_parse(proc_raw)

    elif doc_type == "DISCHARGE_NOTE":
        # Discharge notes have discharge date, condition, instructions, follow-up, discharge meds
        discharge_raw = extract_discharge_info.invoke({"doc_text": doc_text, "doc_type": doc_type})
        result["discharge_info"] = _safe_parse(discharge_raw)

        diag_raw = extract_diagnoses.invoke({"doc_text": doc_text, "doc_type": doc_type})
        result["diagnoses"] = _safe_parse(diag_raw)

        meds_raw = extract_medications.invoke({"doc_text": doc_text, "doc_type": doc_type})
        result["medications"] = _safe_parse(meds_raw)

    elif doc_type == "MEDICATION_RECORD":
        meds_raw = extract_medications.invoke({"doc_text": doc_text, "doc_type": doc_type})
        result["medications"] = _safe_parse(meds_raw)

    elif doc_type == "LAB_RESULTS":
        lab_raw = extract_lab_results.invoke({"doc_text": doc_text, "doc_type": doc_type})
        result["lab_results"] = _safe_parse(lab_raw)

    elif doc_type == "IMAGING_REPORT":
        img_raw = extract_imaging_report.invoke({"doc_text": doc_text, "doc_type": doc_type})
        result["imaging"] = _safe_parse(img_raw)

    elif doc_type == "CONSULTATION":
        consult_raw = extract_consultation.invoke({"doc_text": doc_text, "doc_type": doc_type})
        result["consultation"] = _safe_parse(consult_raw)

        # Consultations may also mention diagnoses and procedures
        diag_raw = extract_diagnoses.invoke({"doc_text": doc_text, "doc_type": doc_type})
        result["diagnoses"] = _safe_parse(diag_raw)

    elif doc_type == "NURSING_NOTE":
        course_raw = extract_hospital_course.invoke({"doc_text": doc_text, "doc_type": doc_type})
        result["hospital_course"] = _safe_parse(course_raw)

        # Nursing notes may contain vital signs, meds given, procedures
        meds_raw = extract_medications.invoke({"doc_text": doc_text, "doc_type": doc_type})
        result["medications"] = _safe_parse(meds_raw)

    elif doc_type == "OPERATIVE_NOTE":
        proc_raw = extract_procedures.invoke({"doc_text": doc_text, "doc_type": doc_type})
        result["procedures"] = _safe_parse(proc_raw)

    else:  # OTHER
        # Generic extraction — try demographics and diagnoses
        demo_raw = extract_demographics.invoke({"doc_text": doc_text, "doc_type": doc_type})
        result["demographics"] = _safe_parse(demo_raw)

        diag_raw = extract_diagnoses.invoke({"doc_text": doc_text, "doc_type": doc_type})
        result["diagnoses"] = _safe_parse(diag_raw)

    return result
