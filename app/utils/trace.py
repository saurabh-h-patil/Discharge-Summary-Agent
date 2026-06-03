"""
Trace & Observability — formats and saves the agent's step trace for review.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box


console = Console()


def format_trace_for_console(step_trace: list[dict]) -> None:
    """Print a formatted step trace to the console using Rich."""

    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]AGENT EXECUTION TRACE[/bold cyan]",
        border_style="cyan",
    ))

    for step in step_trace:
        step_num = step.get("step_number", "?")
        action = step.get("action", "unknown")
        reasoning = step.get("reasoning", "")
        result = step.get("result_summary", "")
        next_dec = step.get("next_decision", "")
        duration = step.get("duration_ms", 0)
        tool_name = step.get("tool_name", None)
        timestamp = step.get("timestamp", "")

        # Color-code by action type
        action_colors = {
            "ingest_pdfs": "blue",
            "plan": "yellow",
            "extract_information": "green",
            "reconcile_medications": "magenta",
            "assemble_summary": "cyan",
            "validate_safety": "red",
            "review_and_learn": "bright_green",
        }
        color = action_colors.get(action, "white")

        console.print(f"\n[bold {color}]━━━ Step {step_num}: {action.upper()} ━━━[/bold {color}]")
        console.print(f"  [dim]Time: {timestamp} | Duration: {duration:.0f}ms[/dim]")
        console.print(f"  [bold]Reasoning:[/bold] {reasoning[:200]}")
        if tool_name:
            console.print(f"  [bold]Tool:[/bold] {tool_name}")
        console.print(f"  [bold]Result:[/bold] {result[:200]}")
        console.print(f"  [bold]Next:[/bold] {next_dec[:200]}")


def save_trace_to_file(step_trace: list[dict], output_path: str) -> str:
    """Save the step trace as a formatted JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "total_steps": len(step_trace),
            "steps": step_trace,
        }, f, indent=2, default=str)

    return output_path


def save_summary_to_file(summary_md: str, output_path: str) -> str:
    """Save the discharge summary markdown to a file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(summary_md)

    return output_path


def save_full_output(
    final_state: dict,
    output_dir: str,
    patient_name: str = "patient",
) -> dict:
    """
    Save all outputs to the output directory:
    - discharge_summary.md — the formatted summary
    - discharge_summary.json — structured data
    - step_trace.json — execution trace
    """
    patient_dir = os.path.join(output_dir, patient_name)
    os.makedirs(patient_dir, exist_ok=True)

    paths = {}

    # Save markdown summary
    summary_md = final_state.get("discharge_summary_md", "")
    if summary_md:
        path = save_summary_to_file(
            summary_md,
            os.path.join(patient_dir, "discharge_summary.md"),
        )
        paths["discharge_summary_md"] = path
        console.print(f"  📄 Summary saved: {path}")

    # Save JSON output
    summary_json = final_state.get("discharge_summary_json", {})
    if summary_json:
        json_path = os.path.join(patient_dir, "discharge_summary.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary_json, f, indent=2, default=str)
        paths["discharge_summary_json"] = json_path
        console.print(f"  📊 JSON saved: {json_path}")

    # Save step trace
    step_trace = final_state.get("step_trace", [])
    if step_trace:
        trace_path = save_trace_to_file(
            step_trace,
            os.path.join(patient_dir, "step_trace.json"),
        )
        paths["step_trace"] = trace_path
        console.print(f"  🔍 Trace saved: {trace_path}")

    # Save clinical flags
    flags = final_state.get("clinical_flags", [])
    if flags:
        flags_path = os.path.join(patient_dir, "clinical_flags.json")
        with open(flags_path, "w", encoding="utf-8") as f:
            json.dump(flags, f, indent=2, default=str)
        paths["clinical_flags"] = flags_path
        console.print(f"  🚩 Flags saved: {flags_path}")

    # Save Part 2: Learning report
    learning_report = final_state.get("learning_report", {})
    if learning_report:
        report_path = os.path.join(patient_dir, "learning_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(learning_report, f, indent=2, default=str)
        paths["learning_report"] = report_path
        console.print(f"  📚 Learning report saved: {report_path}")

    # Save Part 2: Improved summary
    improved_md = final_state.get("improved_summary_md", "")
    if improved_md:
        improved_path = os.path.join(patient_dir, "improved_summary.md")
        with open(improved_path, "w", encoding="utf-8") as f:
            f.write(improved_md)
        paths["improved_summary"] = improved_path
        console.print(f"  ✨ Improved summary saved: {improved_path}")

    return paths


def print_summary_stats(final_state: dict) -> None:
    """Print summary statistics of the agent run."""
    table = Table(title="Agent Run Summary", box=box.ROUNDED)

    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    step_trace = final_state.get("step_trace", [])
    flags = final_state.get("clinical_flags", [])
    conflicts = final_state.get("conflicts", [])
    documents = final_state.get("documents", [])
    missing = final_state.get("missing_fields", [])

    table.add_row("Total Steps", str(len(step_trace)))
    table.add_row("Documents Processed", str(len(documents)))
    table.add_row("Clinical Flags", str(len(flags)))
    table.add_row("Conflicts Detected", str(len(conflicts)))
    table.add_row("Missing Fields", str(len(missing)))

    # Calculate total duration
    total_ms = sum(s.get("duration_ms", 0) for s in step_trace)
    table.add_row("Total Duration", f"{total_ms / 1000:.1f}s")

    # Breakdown by action
    actions = {}
    for step in step_trace:
        action = step.get("action", "unknown")
        actions[action] = actions.get(action, 0) + 1
    table.add_row("Actions", str(actions))

    # Flag breakdown
    if flags:
        flag_types = {}
        for flag in flags:
            ft = flag.get("flag_type", "unknown")
            flag_types[ft] = flag_types.get(ft, 0) + 1
        table.add_row("Flag Types", str(flag_types))

    # Part 2: Learning metrics
    report = final_state.get("learning_report", {})
    if report:
        table.add_row("─── Part 2 ───", "─────")
        table.add_row("Baseline Edit Burden", f"{report.get('baseline_edit_burden', 0):.4f}")
        table.add_row("Final Edit Burden", f"{report.get('final_edit_burden', 0):.4f}")
        table.add_row("Improvement", f"{report.get('improvement_pct', 0):.1f}%")
        table.add_row("Corrections Learned", str(report.get('total_corrections', 0)))

    console.print(table)
