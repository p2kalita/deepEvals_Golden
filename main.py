"""
================================================================================
main.py
================================================================================
Command-line entry point. All provider/model selection happens via the
--config flag pointing at a YAML file -- no code changes needed to switch
between Gemini, GPT-4o, Sonnet, Groq, or local Ollama.

USAGE
-----
    # Run all 12 complexity x style combinations:
    python main.py --config "configs/Gemini 2.5 Flash - Simple.yaml"

    # Run just one combination:
    python main.py --config configs/simple-groq-llama.yaml \\
        --tier medium --style technical

ENVIRONMENT VARIABLES (set whichever your chosen YAML's `api` needs)
----------------------------------------------------------------------
    GOOGLE_API_KEY      - for components.api: google   (Gemini)
    OPENAI_API_KEY       - for components.api: openai    (GPT-4o)
    ANTHROPIC_API_KEY    - for components.api: anthropic (Sonnet)
    GROQ_API_KEY          - for components.api: groq      (Groq Llama)
    (none required)       - for components.api: ollama    (local; just run
                             `ollama serve` and `ollama pull gemma2` first)
================================================================================
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from pipeline import GoldenGenerationPipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate DeepEval goldens from a support-ticket CSV."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a provider/model YAML file (e.g. configs/simple-sonnet.yaml)",
    )
    parser.add_argument(
        "--tickets",
        default="tickets.csv",
        help="Path to the tickets CSV (default: tickets.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory to write all generated files into (default: outputs)",
    )
    parser.add_argument(
        "--tier",
        choices=["low", "medium", "high"],
        default=None,
        help="Run only this complexity tier (requires --style too). "
        "Omit both --tier and --style to run all 12 combinations.",
    )
    parser.add_argument(
        "--style",
        choices=["simple", "professional", "technical", "domain"],
        default=None,
        help="Run only this styling variant (requires --tier too).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pipeline = GoldenGenerationPipeline(
        model_config_path=args.config,
        tickets_csv_path=args.tickets,
        output_dir=args.output_dir,
    )

    pipeline.run(tier=args.tier, style=args.style)


if __name__ == "__main__":
    main()
