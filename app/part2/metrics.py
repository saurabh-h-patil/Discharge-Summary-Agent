"""
Metrics — reward / accuracy signals derived from comparing agent drafts to doctor edits.

Implements:
  - Normalized edit distance (character-level Levenshtein)
  - Section-level accuracy
  - Edit burden score (lower = fewer edits needed = better)
"""

import re
from typing import Optional


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    prev_row = range(len(s2) + 1)

    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


def normalized_edit_distance(draft: str, edited: str) -> float:
    """
    Normalized edit distance: 0.0 = identical, 1.0 = completely different.
    
    Normalized by the length of the longer string so scores are comparable
    across drafts of different lengths.
    """
    if not draft and not edited:
        return 0.0
    max_len = max(len(draft), len(edited))
    if max_len == 0:
        return 0.0
    dist = _levenshtein_distance(draft, edited)
    return dist / max_len


def _extract_sections(text: str) -> dict[str, str]:
    """Extract sections from a markdown discharge summary by ## headers."""
    sections = {}
    current_section = "preamble"
    current_content = []

    for line in text.split("\n"):
        header_match = re.match(r"^##\s+(.+)", line)
        if header_match:
            # Save previous section
            if current_content:
                sections[current_section] = "\n".join(current_content).strip()
            current_section = header_match.group(1).strip().lower()
            current_section = re.sub(r"^\d+\.\s*", "", current_section)  # Remove numbering
            current_content = []
        else:
            current_content.append(line)

    # Save last section
    if current_content:
        sections[current_section] = "\n".join(current_content).strip()

    return sections


def section_level_accuracy(draft: str, edited: str) -> dict:
    """
    Compare draft and edited versions section-by-section.
    
    Returns:
        {
            "overall_accuracy": float,  # average across sections
            "sections": {
                "section_name": {
                    "edit_distance": float,
                    "accuracy": float,  # 1.0 - normalized_edit_distance
                    "changed": bool,
                }
            }
        }
    """
    draft_sections = _extract_sections(draft)
    edited_sections = _extract_sections(edited)

    all_section_names = set(list(draft_sections.keys()) + list(edited_sections.keys()))

    section_results = {}
    accuracies = []

    for name in sorted(all_section_names):
        draft_text = draft_sections.get(name, "")
        edited_text = edited_sections.get(name, "")

        if not draft_text and not edited_text:
            continue

        ned = normalized_edit_distance(draft_text, edited_text)
        accuracy = 1.0 - ned

        section_results[name] = {
            "edit_distance": round(ned, 4),
            "accuracy": round(accuracy, 4),
            "changed": ned > 0.01,  # threshold for "changed"
        }
        accuracies.append(accuracy)

    overall = sum(accuracies) / len(accuracies) if accuracies else 0.0

    return {
        "overall_accuracy": round(overall, 4),
        "sections": section_results,
    }


def edit_burden_score(draft: str, edited: str) -> float:
    """
    Edit burden: how much work the doctor had to do.
    
    Score 0.0 = no edits needed (perfect draft)
    Score 1.0 = complete rewrite
    
    This is the primary reward signal — lower is better.
    """
    return normalized_edit_distance(draft, edited)


def compute_all_metrics(draft: str, edited: str) -> dict:
    """Compute all metrics for a (draft, edited) pair."""
    ned = normalized_edit_distance(draft, edited)
    section_acc = section_level_accuracy(draft, edited)

    return {
        "normalized_edit_distance": round(ned, 4),
        "edit_burden": round(ned, 4),
        "reward": round(1.0 - ned, 4),  # higher reward = less editing
        "section_accuracy": section_acc,
    }
