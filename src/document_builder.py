"""
================================================================================
document_builder.py
================================================================================
Replaces app1.py's three hardcoded sample .txt files (AI/Cloud/Cybersecurity)
with one auto-generated .txt document per row of tickets.csv.

No hardcoded sample content lives here -- every document is derived entirely
from the CSV at runtime, so the same code works unchanged for any support-
ticket CSV with Body / Department / Priority / Tags columns, no matter how
many rows it has.
================================================================================
"""

from __future__ import annotations

import ast
import csv
from pathlib import Path
from typing import Iterator, List


def _normalize_tags(raw_tags: str) -> str:
    """
    The Tags column may already be a clean comma string ("portal,login") or
    a Python-list literal stored as text ("['Account', 'Outage']"), which is
    how tickets.csv actually stores it. Handle both without assuming.
    """
    raw_tags = (raw_tags or "").strip()
    if raw_tags.startswith("[") and raw_tags.endswith("]"):
        try:
            parsed = ast.literal_eval(raw_tags)
            if isinstance(parsed, (list, tuple)):
                return ", ".join(str(t).strip() for t in parsed)
        except (ValueError, SyntaxError):
            pass  # fall through and treat as a plain string
    return ", ".join(part.strip() for part in raw_tags.split(",") if part.strip())


def _row_to_document_text(row: dict) -> str:
    """
    Render one CSV row into the document format specified in the brief:

        -----------------------------------
        Department: IT Support
        Priority: High
        Tags: portal, login, outage

        Ticket:
        Customer reports account portal outage and login failure.
        -----------------------------------
    """
    department = (row.get("Department") or "").strip()
    priority = (row.get("Priority") or "").strip()
    tags = _normalize_tags(row.get("Tags", ""))
    body = (row.get("Body") or "").strip()

    divider = "-" * 35
    return (
        f"{divider}\n"
        f"Department: {department}\n"
        f"Priority: {priority}\n"
        f"Tags: {tags}\n\n"
        f"Ticket:\n"
        f"{body}\n"
        f"{divider}\n"
    )


def iter_ticket_rows(csv_path: str | Path) -> Iterator[dict]:
    """
    Stream rows out of the tickets CSV one at a time, so large datasets don't
    need to be held fully in memory before document generation starts.
    Handles the UTF-8 BOM that pandas/Excel-exported CSVs (like tickets.csv)
    commonly include in the first column name.
    """
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            yield row


def build_documents_from_csv(
    csv_path: str | Path,
    output_dir: str | Path,
    filename_prefix: str = "ticket",
) -> List[str]:
    """
    Read every row of `csv_path` and write one .txt document per row into
    `output_dir`. Returns the list of written file paths (as strings, ready
    to pass straight into Synthesizer.generate_goldens_from_docs()).

    Existing documents from a previous run are left in place if they already
    match expected names; the directory is created if missing. This keeps
    re-runs on large CSVs cheap to resume.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    document_paths: List[str] = []
    for index, row in enumerate(iter_ticket_rows(csv_path)):
        doc_text = _row_to_document_text(row)
        doc_path = output_dir / f"{filename_prefix}_{index:05d}.txt"
        doc_path.write_text(doc_text, encoding="utf-8")
        document_paths.append(str(doc_path))

    if not document_paths:
        raise ValueError(
            f"No rows found in '{csv_path}'. Expected columns: "
            f"Body, Department, Priority, Tags."
        )

    print(f"  Generated {len(document_paths)} ticket documents in {output_dir}/")
    return document_paths
