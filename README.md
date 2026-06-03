# 🏥 Discharge Summary Agent

An **agentic AI system** that reads messy, scanned clinical source-note PDFs and produces a **structured, clinically safe discharge summary draft** for clinician review.

Built with **LangGraph** (agent orchestration) and **GPT-4o** (vision + reasoning).

---

## Architecture

```
PDF Input → [PyMuPDF: Page Images] → [GPT-4o Vision: Text Extraction]
         → [Document Classifier] → [LangGraph Agent Loop]
         → [Discharge Summary Draft + Step Trace + Flags]
```

The system uses a **dual-model approach**:
1. **GPT-4o Vision** reads scanned PDF pages as images (no OCR dependency)
2. **GPT-4o Text** powers the agent's reasoning, planning, and extraction

---

## Agent Loop Design

The agent uses a **LangGraph StateGraph** implementing a ReAct-style (Reason-Act) loop:

```
START → INGEST → PLAN → EXTRACT → PLAN → RECONCILE → ASSEMBLE → VALIDATE → REVIEW & LEARN → END
                   ↑        |                                         (Part 1)      (Part 2)
                   └────────┘  (re-plan if more data needed)
```

### Nodes

| Node | Part | Purpose |
|------|------|---------|
| **Ingest** | 1 | Convert PDFs to images, extract text via GPT-4o vision, classify documents |
| **Plan** | 1 | Analyze what's been found, identify gaps, decide next actions |
| **Extract** | 1 | Pull specific clinical data (demographics, diagnoses, meds, labs, etc.) |
| **Reconcile** | 1 | Compare admission vs. discharge medications, check drug interactions |
| **Assemble** | 1 | Build the structured discharge summary from all gathered data |
| **Validate** | 1 | Post-generation safety check for fabrication, missing data, conflicts |
| **Review & Learn** | 2 | Simulated reviewer edits → extract corrections → regenerate improved draft |

### Control Mechanisms
- **Hard iteration cap** (default: 30) prevents infinite loops
- **Conditional edges** allow the agent to re-plan when it discovers new information
- **Status-based routing** determines which node executes next

---

## No-Fabrication Guardrail

This is the **core safety feature**.

1. **System prompt enforcement**: Every LLM call includes explicit instructions to never invent facts
2. **Source tracing**: Every fact in the output must cite its source document and page
3. **Missing data marking**: Any required field not found is marked:
   `[NOT FOUND IN RECORDS — REQUIRES CLINICIAN REVIEW]`
4. **Pending data preservation**: Lab results marked "pending" stay as `[PENDING]`
5. **Post-generation validation**: A separate validation pass scans the draft for potential fabrications
6. **Draft watermark**: The output is always marked as a draft requiring clinician review

---

## Handling Failures & Conflicts

### Conflict Detection
When two notes disagree, **both versions** are documented and flagged — the agent never arbitrarily picks one.

### Medication Reconciliation
- Compares admission vs. discharge medication lists
- Flags any change without a documented reason
- Runs mock drug-interaction checks on discharge medications

### Robust Failure Handling
- **PDF extraction failures**: Caught and recorded, never crash
- **LLM API failures**: Retry with exponential backoff
- **Empty documents**: Detected and marked
- **Tool failures**: Gracefully handled — agent reports and continues

---

## Tools

| Tool | Type | Description |
|------|------|-------------|
| `extract_demographics` | Real | Extract patient demographics |
| `extract_diagnoses` | Real | Extract principal + secondary diagnoses |
| `extract_medications` | Real | Extract medication lists |
| `extract_lab_results` | Real | Extract lab values, flag pending |
| `extract_hospital_course` | Real | Extract clinical narrative |
| `extract_procedures` | Real | Extract procedures performed |
| `extract_allergies` | Real | Extract patient allergies |
| `extract_discharge_info` | Real | Extract discharge condition, follow-up |
| `search_across_notes` | Real | Full-text search across all documents |
| `check_drug_interactions` | **Mock** | Check for drug-drug interactions |
| `flag_for_clinician_review` | **Mock** | Flag an issue for clinician review |
| `escalate_to_physician` | **Mock** | Escalate critical findings |
| `reconcile_medications` | Real | Compare admission vs. discharge meds |

---

## Setup & Run

### Prerequisites
- Python 3.10+
- OpenAI API key with GPT-4o access

### Installation

```bash
pip install -r requirements.txt
```

### Configure

```bash
copy .env.example .env
# Edit .env and set your OPENAI_API_KEY
```

### Run

```bash
# Single command runs Part 1 + Part 2 in one pipeline
python main.py --pdf "patient 2 (1).pdf" --patient-name "patient_2"
```

### Output

All results saved to `output/<patient_name>/`:
- `discharge_summary.md` — Formatted discharge summary draft (Part 1)
- `discharge_summary.json` — Structured data + learning report
- `step_trace.json` — Full execution trace (Part 1 + Part 2)
- `clinical_flags.json` — All flags raised for clinician review
- `learning_report.json` — Before/after metrics + improvement curve (Part 2)
- `improved_summary.md` — Summary after learning iterations (Part 2)

---

## Part 2 — Learning from Doctor Edits

### Design Overview

In production, clinicians edit the agent's draft before finalizing. Those edits are signal. Part 2 implements a **correction-memory** approach that learns from these edits to improve future drafts.

### Approach: Correction-Memory with In-Context Learning

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│ Agent Draft  │ ──→ │  Simulated   │ ──→ │  Compute     │
│ (iteration N)│     │  Reviewer    │     │  Metrics     │
└─────────────┘     └──────────────┘     └──────────────┘
       ↑                    │                     │
       │              (draft, edited)         edit distance
       │                    ↓                     │
       │            ┌──────────────┐              │
       └─────────── │  Correction  │ ←────────────┘
                    │  Memory      │
                    └──────────────┘
```

### 1. Simulated Reviewer

A stand-in "doctor" with 8 consistent, hidden editing policies:
- Standardize medication format (Drug DOSE Route Frequency — Indication)
- Add patient education/counseling notes
- Connect treatments to clinical reasoning
- Document DAMA explicitly
- Add action plans for pending results
- Add vitals at discharge
- Add ICD-10 code placeholders
- Standardize section header formatting

### 2. Reward Signal

- **Edit burden** (primary): Normalized Levenshtein edit distance between draft and doctor-edited version. Range 0.0 (perfect) to 1.0 (complete rewrite).
- **Section-level accuracy**: Per-section edit distance to identify which sections improve most.
- **Reward** = 1.0 − edit_burden (higher = better)

### 3. Learning Mechanism

**Correction-memory with few-shot injection:**
1. After each review, LLM extracts structured corrections (section, before, after, rule)
2. Corrections are stored in a persistent memory
3. Before generating future drafts, relevant corrections are retrieved and injected into the prompt as few-shot examples
4. The agent progressively adapts to the reviewer's preferences

**Why this approach (vs. fine-tuning / DPO / reward model):**
- Works with any LLM — no fine-tuning infrastructure needed
- Cold-start friendly — even 1-2 corrections help immediately
- Safety-preserving — corrections are additive, never remove Part 1 safety guardrails
- Interpretable — every learned rule is human-readable
- No catastrophic forgetting risk

### 4. Limitations & Safety Discussion

**Cold-start problem**: The first iteration has no corrections — edit burden starts high. Improvement is gradual and depends on correction quality. In production, seed the memory with a small set of expert-curated corrections.

**Gaming risk**: An agent could lower edit distance by becoming vaguer or mimicking style rather than improving clinical accuracy. Mitigations:
- Part 1 safety guardrails (no-fabrication, source tracing) remain immutable and are never overridden by learned corrections
- Corrections are additive only — they add formatting/context, never remove safety markers
- Section-level metrics reveal if improvement is real (clinical sections) or superficial (formatting only)
- In production, periodically validate against a held-out set of expert-reviewed summaries

**Learning ceiling**: The correction-memory approach plateaus once all of the reviewer's consistent patterns have been captured. More sophisticated approaches (DPO, reward model) would be needed for continued improvement beyond this ceiling.

---

## Project Structure

```
├── main.py                        # CLI entry point (Part 1 + Part 2)
├── requirements.txt               # Dependencies
├── .env.example                   # Environment template
├── README.md
├── app/
│   ├── core/                      # Infrastructure
│   │   ├── config.py              # Settings
│   │   ├── llm.py                 # OpenAI client (text + vision)
│   │   └── prompts.py             # Safety-engineered prompts
│   ├── models/                    # Data models
│   │   ├── clinical.py            # Clinical Pydantic models
│   │   └── discharge.py           # Discharge summary schema
│   ├── tools/                     # Agent tools
│   │   ├── pdf_ingestion.py       # PDF → images → vision text
│   │   ├── extraction.py          # Clinical data extraction
│   │   ├── medication.py          # Medication reconciliation + interactions (mock)
│   │   ├── safety.py              # Flag & escalate (mock)
│   │   └── search.py              # Search across notes
│   ├── agent/                     # LangGraph agent
│   │   ├── state.py               # Agent state
│   │   ├── nodes.py               # Graph nodes
│   │   └── graph.py               # StateGraph + runner
│   ├── part2/                     # Part 2 — Learning from edits
│   │   ├── simulated_reviewer.py  # Mock doctor with hidden editing policies
│   │   ├── metrics.py             # Edit distance, section accuracy, reward
│   │   ├── learning.py            # Correction-memory mechanism
│   │   └── evaluate.py            # Learning loop runner + reporting
│   └── utils/
│       └── trace.py               # Observability
└── output/                        # Generated results
```

---

## Limitations & Future Work

### Current Limitations
1. **Vision API cost**: Each PDF page requires a GPT-4o vision call (~$0.01-0.03/page)
2. **Mock tools**: Drug interaction and escalation tools use a limited mock database
3. **Context window**: Very large documents may need truncation
4. **Learning ceiling**: Correction-memory plateaus after capturing all reviewer patterns

### With More Time
1. **Multi-pass extraction**: Targeted follow-up queries based on planning gaps
2. **Local OCR fallback**: Tesseract OCR when GPT-4o vision fails
3. **Real drug interaction API**: Integrate with RxNorm or DrugBank
4. **DPO/SFT fine-tuning**: Go beyond in-context learning for deeper model adaptation
5. **Caching**: Cache vision extraction results to avoid re-processing
6. **Confidence scoring**: Per-field confidence scores
7. **Multi-reviewer learning**: Adapt to different clinicians' preferences

---

*Built for the Dscribe (Unriddle Technologies) AI Engineer take-home assignment.*

#   D i s c h a r g e - S u m m a r y - A g e n t  
 