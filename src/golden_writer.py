"""
================================================================================
golden_writer.py
================================================================================
Saves a batch of generated Goldens in all four formats required by the brief:

  1. DeepEval-native JSON   (synthesizer.save_as)
  2. DeepEval-native CSV    (synthesizer.save_as)
  3. Clean portable JSON    (id, input, expected_output, context,
                             source_file, evolutions, quality_score)
  4. Clean portable CSV     (same fields, flattened for spreadsheet use)

Kept as one focused module so adding a 5th output format later (e.g. JSONL)
touches only this file.
================================================================================
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from deepeval.synthesizer import Synthesizer


def _golden_to_clean_record(index: int, golden: Any) -> Dict[str, Any]:
    """Flatten one DeepEval Golden + its metadata into the required fields."""
    meta = golden.additional_metadata or {}
    return {
        "id": index + 1,
        "input": golden.input,
        "expected_output": golden.expected_output,
        "context": golden.context or [],
        "source_file": meta.get("source_file", ""),
        "evolutions": meta.get("evolutions", []),
        "quality_score": meta.get("synthetic_input_quality", None),
    }


def save_goldens(
    synthesizer: Synthesizer,
    goldens: List[Any],
    output_dir: str | Path,
    file_name: str,
) -> Dict[str, Path]:
    """
    Write all four output formats for one batch of goldens.

    `file_name` is the base name shared across all four files, e.g.
    "goldens_low_simple" produces:
        goldens_low_simple.json          (DeepEval native)
        goldens_low_simple.csv           (DeepEval native)
        goldens_low_simple_clean.json
        goldens_low_simple_clean.csv

    Returns a dict mapping format label -> written path, for logging /
    downstream use.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}

    # ── 1 & 2: DeepEval-native JSON and CSV ─────────────────────────────────
    synthesizer.save_as(
        file_type="json",
        directory=str(output_dir),
        file_name=file_name,
        quiet=True,
    )
    written["deepeval_json"] = output_dir / f"{file_name}.json"

    synthesizer.save_as(
        file_type="csv",
        directory=str(output_dir),
        file_name=file_name,
        quiet=True,
    )
    written["deepeval_csv"] = output_dir / f"{file_name}.csv"

    # ── 3: Clean portable JSON ───────────────────────────────────────────────
    clean_records = [
        _golden_to_clean_record(i, g) for i, g in enumerate(goldens)
    ]
    clean_json_path = output_dir / f"{file_name}_clean.json"
    with open(clean_json_path, "w", encoding="utf-8") as fh:
        json.dump(clean_records, fh, indent=2, ensure_ascii=False)
    written["clean_json"] = clean_json_path

    # ── 4: Clean portable CSV ────────────────────────────────────────────────
    clean_csv_path = output_dir / f"{file_name}_clean.csv"
    fieldnames = [
        "id", "input", "expected_output", "context",
        "source_file", "evolutions", "quality_score",
    ]
    with open(clean_csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rec in clean_records:
            row = dict(rec)
            row["context"] = " | ".join(rec["context"]) if rec["context"] else ""
            row["evolutions"] = " -> ".join(rec["evolutions"]) if rec["evolutions"] else ""
            writer.writerow(row)
    written["clean_csv"] = clean_csv_path

    print(f"  Saved {len(goldens)} goldens -> {output_dir}/{file_name}.* (4 formats)")
    return written
