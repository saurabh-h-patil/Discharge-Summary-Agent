"""
Medication Reconciliation & Drug Interaction Tools.
Drug interaction check is a MOCK tool (as specified in the assignment).
"""

import json
from typing import Optional
from langchain_core.tools import tool

from app.core.llm import get_llm_client
from app.core.prompts import MEDICATION_RECONCILIATION_PROMPT


# ──────────────────────────────────────────────────
# Mock drug interaction database
# ──────────────────────────────────────────────────
MOCK_DRUG_INTERACTIONS = {
    ("warfarin", "aspirin"): {
        "severity": "HIGH",
        "description": "Increased risk of bleeding. Concurrent use requires close INR monitoring.",
        "recommendation": "Monitor INR closely. Consider GI prophylaxis.",
    },
    ("warfarin", "ibuprofen"): {
        "severity": "HIGH",
        "description": "NSAIDs increase anticoagulant effect and risk of GI bleeding.",
        "recommendation": "Avoid combination if possible. Use acetaminophen instead.",
    },
    ("metformin", "contrast dye"): {
        "severity": "HIGH",
        "description": "Risk of lactic acidosis. Hold metformin before/after contrast procedures.",
        "recommendation": "Hold metformin 48h before and after contrast administration.",
    },
    ("lisinopril", "potassium"): {
        "severity": "MEDIUM",
        "description": "ACE inhibitors can increase potassium levels. Supplemental potassium may cause hyperkalemia.",
        "recommendation": "Monitor serum potassium regularly.",
    },
    ("lisinopril", "spironolactone"): {
        "severity": "MEDIUM",
        "description": "Both increase potassium. Risk of hyperkalemia.",
        "recommendation": "Monitor potassium and renal function closely.",
    },
    ("metoprolol", "verapamil"): {
        "severity": "HIGH",
        "description": "Both depress cardiac conduction. Risk of bradycardia and heart block.",
        "recommendation": "Avoid combination. Use alternative agents.",
    },
    ("clopidogrel", "omeprazole"): {
        "severity": "MEDIUM",
        "description": "Omeprazole may reduce antiplatelet effect of clopidogrel via CYP2C19 inhibition.",
        "recommendation": "Consider pantoprazole as alternative PPI.",
    },
    ("ssri", "tramadol"): {
        "severity": "HIGH",
        "description": "Risk of serotonin syndrome.",
        "recommendation": "Avoid combination. Monitor for serotonin syndrome symptoms.",
    },
    ("amlodipine", "simvastatin"): {
        "severity": "MEDIUM",
        "description": "Amlodipine increases simvastatin levels. Risk of myopathy/rhabdomyolysis.",
        "recommendation": "Limit simvastatin to 20mg daily when combined with amlodipine.",
    },
}


@tool
def check_drug_interactions(medications: str) -> str:
    """Check for drug-drug interactions in a medication list. (MOCK TOOL — uses a predefined interaction database)
    
    Input should be a comma-separated list of medication names.
    Returns any interactions found with severity and recommendations.
    """
    med_list = [m.strip().lower() for m in medications.split(",")]
    interactions_found = []

    # Check all pairs
    for i in range(len(med_list)):
        for j in range(i + 1, len(med_list)):
            med_a = med_list[i]
            med_b = med_list[j]

            # Check both orderings
            for pair in [(med_a, med_b), (med_b, med_a)]:
                for (drug_a, drug_b), interaction in MOCK_DRUG_INTERACTIONS.items():
                    if drug_a in pair[0] and drug_b in pair[1]:
                        interactions_found.append({
                            "drug_a": med_list[i],
                            "drug_b": med_list[j],
                            **interaction,
                        })

    if not interactions_found:
        return json.dumps({
            "interactions_found": False,
            "message": "No known drug interactions detected in the provided medication list.",
            "note": "This is a mock tool with a limited interaction database. A real system would use a comprehensive drug interaction API.",
        })

    return json.dumps({
        "interactions_found": True,
        "interactions": interactions_found,
        "note": "MOCK TOOL — interactions found require clinician verification.",
    })


@tool
def reconcile_medications(admission_meds: str, discharge_meds: str) -> str:
    """Compare admission and discharge medication lists to identify changes.
    
    Flags any changes without documented reasons for clinician reconciliation.
    Input: JSON strings of admission and discharge medication lists.
    """
    llm = get_llm_client()

    prompt = MEDICATION_RECONCILIATION_PROMPT.format(
        admission_meds=admission_meds,
        discharge_meds=discharge_meds,
    )

    result = llm.chat(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a clinical pharmacist performing medication reconciliation. "
                    "Compare the two medication lists carefully. Flag ANY change that lacks "
                    "a documented clinical reason. Return JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return result
