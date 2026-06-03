"""
Learning Mechanism — Correction-Memory approach.

After each simulated doctor review, we extract structured corrections (diffs)
and store them in a correction memory. Before generating future drafts,
relevant past corrections are retrieved and injected into the prompt as
few-shot examples.

This is a form of in-context learning / retrieval-augmented generation:
  - No fine-tuning required
  - Works with any LLM
  - Corrections accumulate over iterations
  - The agent progressively learns the reviewer's preferences

Why this approach:
  - Simple, interpretable, and demonstrably effective
  - Cold-start friendly: even 1-2 corrections help
  - No risk of catastrophic forgetting
  - Safety guardrails from Part 1 remain intact (corrections are additive,
    they never remove safety instructions)
"""

import json
import os
import re
from typing import Optional

from app.core.llm import get_llm_client


class CorrectionMemory:
    """
    Stores structured corrections extracted from (draft, edited) pairs.
    
    Each correction captures:
      - section: which section was changed
      - pattern: what the agent wrote (before)
      - correction: what the doctor changed it to (after)
      - rule: the inferred editing rule
    """

    def __init__(self):
        self.corrections: list[dict] = []
        self.rules_learned: list[str] = []

    def extract_corrections(self, draft: str, edited: str) -> list[dict]:
        """
        Use the LLM to extract structured corrections from a (draft, edited) pair.
        """
        llm = get_llm_client()

        prompt = f"""Compare these two versions of a discharge summary and extract the specific corrections the reviewer made.

ORIGINAL DRAFT:
---
{draft[:5000]}
---

EDITED VERSION:
---
{edited[:5000]}
---

For each change, extract:
1. Which section was changed
2. What the original text said (before)
3. What the reviewer changed it to (after)
4. The underlying rule/pattern the reviewer seems to be following

Return a JSON object:
{{
    "corrections": [
        {{
            "section": "section name",
            "before": "original text snippet",
            "after": "corrected text snippet",
            "rule": "the editing rule being applied"
        }}
    ],
    "general_rules": ["list of general editing preferences observed"]
}}"""

        try:
            result = llm.chat(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at analyzing text differences. Extract precise, structured corrections.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            parsed = json.loads(result)

            corrections = parsed.get("corrections", [])
            general_rules = parsed.get("general_rules", [])

            # Store corrections
            self.corrections.extend(corrections)

            # Store new unique rules
            for rule in general_rules:
                if rule not in self.rules_learned:
                    self.rules_learned.append(rule)

            return corrections

        except Exception as e:
            print(f"  ⚠️ Correction extraction failed: {e}")
            return []

    def get_relevant_corrections(self, section: Optional[str] = None, max_corrections: int = 10) -> list[dict]:
        """
        Retrieve relevant corrections for a given section.
        If section is None, return the most recent corrections across all sections.
        """
        if section:
            relevant = [c for c in self.corrections if section.lower() in c.get("section", "").lower()]
        else:
            relevant = self.corrections

        # Return most recent corrections (up to max)
        return relevant[-max_corrections:]

    def get_learned_rules(self) -> list[str]:
        """Get all learned editing rules."""
        return self.rules_learned

    def build_correction_prompt(self) -> str:
        """
        Build a prompt injection that teaches the agent the reviewer's preferences.
        This is injected into the discharge summary assembly prompt.
        """
        if not self.corrections and not self.rules_learned:
            return ""

        lines = [
            "\n## LEARNED EDITING PREFERENCES",
            "Based on previous clinician reviews, apply these formatting and content rules:",
            "",
        ]

        # Add general rules
        if self.rules_learned:
            lines.append("### General Rules:")
            for i, rule in enumerate(self.rules_learned, 1):
                lines.append(f"{i}. {rule}")
            lines.append("")

        # Add section-specific examples
        if self.corrections:
            lines.append("### Specific Corrections to Apply:")
            # Group by section, show most recent examples
            seen_sections = set()
            for correction in reversed(self.corrections[-15:]):
                section = correction.get("section", "unknown")
                if section in seen_sections:
                    continue
                seen_sections.add(section)

                before = correction.get("before", "")[:150]
                after = correction.get("after", "")[:150]
                rule = correction.get("rule", "")

                lines.append(f"- **{section}**: Instead of writing \"{before}\", write \"{after}\"")
                if rule:
                    lines.append(f"  Reason: {rule}")

        return "\n".join(lines)

    def save(self, path: str):
        """Save correction memory to a JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "corrections": self.corrections,
                "rules_learned": self.rules_learned,
                "total_corrections": len(self.corrections),
            }, f, indent=2)

    def load(self, path: str):
        """Load correction memory from a JSON file."""
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.corrections = data.get("corrections", [])
                self.rules_learned = data.get("rules_learned", [])

    def stats(self) -> dict:
        """Get statistics about the correction memory."""
        sections = {}
        for c in self.corrections:
            s = c.get("section", "unknown")
            sections[s] = sections.get(s, 0) + 1

        return {
            "total_corrections": len(self.corrections),
            "total_rules": len(self.rules_learned),
            "corrections_by_section": sections,
        }
