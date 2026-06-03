"""
Simulated Reviewer — a stand-in "doctor" that applies a consistent, hidden
editing policy to agent-generated discharge summary drafts.

This produces (draft, edited) pairs used to train the learning mechanism.

The editing policies are intentionally HIDDEN from the agent — the agent must
discover and adapt to them through the correction-memory loop.
"""

import json
import re
from app.core.llm import get_llm_client


# ────────────────────────────────────────────────────────────────
# Editing Policies (hidden from the agent)
# ────────────────────────────────────────────────────────────────
#
# These simulate a real clinician's consistent editing preferences:
#
# 1. FORMAT_MEDS:       Always format medications as "Drug DOSE PO FREQUENCY (indication)"
# 2. ADD_COUNSELING:    Always add patient education / counseling note to follow-up
# 3. CLINICAL_CONTEXT:  Add clinical reasoning to hospital course (connect symptoms to treatment)
# 4. DAMA_EXPLICIT:     If discharged at request, explicitly say "Discharged Against Medical Advice (DAMA)"
# 5. PENDING_URGENCY:   Add urgency level and follow-up plan for every pending result
# 6. VITALS_AT_DISCHARGE: Add "Vitals at discharge: stable" when discharge condition is stated
# 7. DIAGNOSIS_ICD:     Add ICD-10 code placeholders for diagnoses
# 8. SECTION_HEADERS:   Standardize all section headers to uppercase
# ────────────────────────────────────────────────────────────────

REVIEWER_SYSTEM_PROMPT = """You are an experienced attending physician reviewing an AI-generated discharge summary draft.

You have CONSISTENT editing preferences that you ALWAYS apply. Apply ALL of these rules every time:

1. **MEDICATION FORMAT**: Rewrite every medication entry as:
   "- [Drug Name] [Dose] [Route] [Frequency] — [Indication if known, or 'indication to be confirmed']"
   Example: "- Pantoprazole 40mg PO once daily — GI prophylaxis"

2. **PATIENT EDUCATION**: In the Follow-up section, ALWAYS add:
   "Patient and family were counseled regarding diagnosis, treatment plan, warning signs requiring immediate medical attention, and medication compliance."

3. **CLINICAL REASONING**: In the Hospital Course, connect treatments to their clinical rationale.
   Instead of "Treated with IV fluids and antibiotics" write "IV fluid resuscitation was initiated for dehydration correction; empiric antibiotics were started given concern for bacterial gastroenteritis."

4. **DAMA DOCUMENTATION**: If the patient was discharged at their own request or against advice, ALWAYS add a dedicated line:
   "**DISCHARGE TYPE: DAMA (Discharged Against Medical Advice)** — Patient was informed of risks of early discharge including risk of clinical deterioration, and signed DAMA form."

5. **PENDING RESULTS ACTION PLAN**: For every pending result, add an action plan:
   "- [Test Name] [PENDING] — Results to be followed up by [outpatient provider] within [timeframe]. Patient instructed to return if symptoms worsen before results are available."

6. **VITALS AT DISCHARGE**: When discharge condition is mentioned, add:
   "Vitals at discharge: Within normal limits (documented in nursing notes)."

7. **DIAGNOSIS CODING**: Add ICD-10 code placeholder after each diagnosis:
   "Acute Gastroenteritis with Dehydration (ICD-10: ___.___)"

8. **SECTION FORMATTING**: Ensure all section headers are consistently formatted as "## SECTION NAME" in uppercase.

Apply ALL these rules. Do NOT add any clinical facts that are not in the original draft.
Do NOT remove any flags, warnings, or "[REQUIRES CLINICIAN REVIEW]" markers.
Return the COMPLETE edited discharge summary."""


def review_draft(draft_md: str) -> str:
    """
    Apply the simulated doctor's editing policy to a discharge summary draft.
    Returns the edited version.
    """
    llm = get_llm_client()

    result = llm.chat(
        messages=[
            {"role": "system", "content": REVIEWER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Review and edit this discharge summary draft according to your standard preferences:\n\n---\n{draft_md}\n---\n\nReturn the complete edited discharge summary.",
            },
        ],
        temperature=0.1,
        max_tokens=8000,
    )
    return result


def generate_edit_pairs(
    drafts: list[str],
) -> list[dict]:
    """
    Generate (draft, edited) pairs by running the simulated reviewer on each draft.
    Returns list of {draft, edited, sections_changed} dicts.
    """
    pairs = []

    for i, draft in enumerate(drafts):
        print(f"  📝 Reviewing draft {i + 1}/{len(drafts)}...")
        try:
            edited = review_draft(draft)
            pairs.append({
                "draft": draft,
                "edited": edited,
                "review_round": i + 1,
            })
        except Exception as e:
            print(f"  ❌ Review failed for draft {i + 1}: {e}")
            pairs.append({
                "draft": draft,
                "edited": draft,  # fallback: unchanged
                "review_round": i + 1,
                "error": str(e),
            })

    return pairs
