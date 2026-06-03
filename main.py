"""
Discharge Summary Agent — CLI Entry Point.

Usage:
    python main.py --pdf "patient 2 (1).pdf" --patient-name "patient_2"
    python main.py --pdf "admission.pdf" --pdf "labs.pdf" --patient-name "patient_1"
"""

import argparse
import os
import sys

from dotenv import load_dotenv

# Load environment variables before importing app modules
load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="Discharge Summary Agent — Generate structured discharge summaries from clinical source notes",
    )
    parser.add_argument(
        "--pdf",
        action="append",
        required=True,
        help="Path to a patient source-note PDF (can be specified multiple times)",
    )
    parser.add_argument(
        "--patient-name",
        default="patient",
        help="Name/ID for the patient (used for output directory naming)",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory to save output files (default: ./output)",
    )

    args = parser.parse_args()

    # Validate inputs
    for path in args.pdf:
        if not os.path.exists(path):
            print(f"❌ File not found: {path}")
            sys.exit(1)

    # Import after dotenv is loaded
    from app.agent.graph import run_agent
    from app.utils.trace import (
        save_full_output,
        format_trace_for_console,
        print_summary_stats,
    )

    # Run the agent (Part 1 + Part 2 in one pipeline)
    final_state = run_agent(args.pdf)

    # Save outputs
    output_paths = save_full_output(
        final_state,
        output_dir=args.output_dir,
        patient_name=args.patient_name,
    )

    # Print trace and stats
    format_trace_for_console(final_state.get("step_trace", []))
    print_summary_stats(final_state)

    # Print output locations
    print("\n📁 Output files:")
    for key, path in output_paths.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
