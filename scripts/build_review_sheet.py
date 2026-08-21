#!/usr/bin/env python3
"""Lay a concept out so a native speaker can judge it without reading JSON.

Every concept in this dataset is drafted before it is approved, and nothing
built on it can claim to be about Arab culture until a native speaker has gone
through the Arabic. That review is the one step in the pipeline a machine
cannot do, so the least this repository can do is make it cheap: put each
concept on one page, in reading order, with the specific question to answer
beside each block, and say exactly which file and field to edit.

The sheet is deliberately not a form to fill in and feed back. A round trip
through a generated document is one more thing to break, and the dataset is
small enough to edit directly. What the sheet does is remove the need to hold
a JSONL file in your head while judging a sentence.

Usage:
    python scripts/build_review_sheet.py
    python scripts/build_review_sheet.py --concepts wasta_001 --output review.md
    python scripts/build_review_sheet.py --pending-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_builder import (  # noqa: E402
    DEFAULT_DATASET_PATH,
    CulturalConcept,
    load_concepts,
    review_summary,
)

DEFAULT_OUTPUT = "docs/review_sheet.md"


def _numbered(items: list[str]) -> list[str]:
    """Render sentences as a numbered Markdown list, or say the list is empty."""
    if not items:
        return ["_(none — this is itself worth flagging)_"]
    return [f"{index}. {text}" for index, text in enumerate(items, start=1)]


def concept_section(concept: CulturalConcept) -> list[str]:
    """Render one concept as a self-contained review page.

    Args:
        concept: The entry to lay out.

    Returns:
        Markdown lines.
    """
    status = "reviewed" if concept.is_reviewed else "**awaiting review**"
    lines = [
        f"## {concept.concept_ar} — `{concept.concept_id}`",
        "",
        f"- **English gloss**: {concept.concept_en}",
        f"- **Category**: {concept.category}  |  **Sentiment**: {concept.sentiment}"
        f"  |  **Dialect**: {concept.dialect}",
        f"- **Status**: {status}",
        "",
        f"> {concept.description}",
        "",
    ]
    if concept.cultural_context:
        lines += [f"_{concept.cultural_context}_", ""]

    lines += [
        "### 1. The Arabic exemplars",
        "",
        "Each should read as a natural sentence a speaker might actually say,",
        "and should express the concept — not merely mention its name.",
        "",
        *_numbered(concept.examples_ar),
        "",
        "### 2. The Arabic contrasts",
        "",
        "These are the negative side of the extraction, and the whole method",
        "rests on them. Each should be as close as possible to the exemplars —",
        "same topic, same register, same kind of event — with the concept",
        "**absent**. A contrast that changes the subject cancels nothing, and",
        "a contrast that still carries the concept poisons the direction.",
        "",
        *_numbered(concept.contrast_ar),
        "",
        "### 3. The framing",
        "",
        f"- Is `{concept.sentiment}` the right valence, or is this contested?",
        f"- Is `{concept.dialect}` right, or is this narrower than it claims?",
        "- Does the one-line description above match how the word is used?",
        "",
        "### 4. English side, for reference only",
        "",
        "Judge these only if something looks wrong; the Arabic is what matters.",
        "",
        "<details><summary>English exemplars and contrasts</summary>",
        "",
        *_numbered(concept.examples_en),
        "",
        *[f"- _{text}_" for text in concept.contrast_en],
        "",
        "</details>",
        "",
        "---",
        "",
    ]
    return lines


def build_sheet(concepts: list[CulturalConcept], dataset_path: Path) -> str:
    """Assemble the full review document.

    Args:
        concepts: Entries to include, in order.
        dataset_path: Path the reviewer should edit.

    Returns:
        The Markdown document.
    """
    counts = review_summary(concepts)
    lines = [
        "# Native-speaker review sheet",
        "",
        f"{counts['reviewed']} of {counts['total']} entries below carry a named review.",
        "",
        "## What this is for",
        "",
        "Every Arabic sentence in this dataset was drafted, not attested. The",
        "measurements built on it are currently claims about a *pipeline* and",
        "about *models* — they cannot be claims about Arab cultural concepts",
        "until the Arabic has been judged by someone who speaks it.",
        "",
        "You are not being asked to approve a method. You are being asked",
        "whether these sentences are ones a speaker would recognise.",
        "",
        "## How to record a verdict",
        "",
        f"Edit `{dataset_path.as_posix()}` — one JSON object per line. For each",
        "concept you approve, fix whatever needs fixing and then set:",
        "",
        "```json",
        '"review_status": "reviewed",',
        '"reviewed_by": "Your Name",',
        '"reviewed_at": "YYYY-MM-DD",',
        '"review_notes": "anything you want on the record"',
        "```",
        "",
        "`reviewed_by` and `reviewed_at` are required: the test suite rejects",
        "an entry that claims review without saying who and when, because a",
        "claim nobody can follow up is not one anybody can correct.",
        "",
        "**Changing a sentence is a normal outcome, not a failure.** So is",
        "leaving a concept unreviewed and saying why in `review_notes`.",
        "",
        "Then run:",
        "",
        "```bash",
        "python scripts/check_dataset.py     # invariants, and pairs that are not minimal",
        "python -m pytest tests/test_dataset_integrity.py",
        "```",
        "",
        "---",
        "",
    ]
    for concept in concepts:
        lines.extend(concept_section(concept))
    return "\n".join(lines).rstrip("\n") + "\n"


def main(argv: list[str] | None = None) -> int:
    """Write the review sheet.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        A process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--concepts", default=None, help="Comma-separated ids.")
    parser.add_argument(
        "--pending-only",
        action="store_true",
        help="Include only entries that do not yet carry a named review.",
    )
    args = parser.parse_args(argv)

    dataset_path = Path(args.dataset)
    concepts = load_concepts(dataset_path)
    if args.concepts:
        wanted = {c.strip() for c in args.concepts.split(",") if c.strip()}
        unknown = wanted - {concept.concept_id for concept in concepts}
        if unknown:
            raise SystemExit(f"Unknown concept ids: {', '.join(sorted(unknown))}")
        concepts = [concept for concept in concepts if concept.concept_id in wanted]
    if args.pending_only:
        concepts = [concept for concept in concepts if not concept.is_reviewed]

    if not concepts:
        print("Nothing to review: every selected entry already carries a named review.")
        return 0

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_sheet(concepts, dataset_path), encoding="utf-8")
    print(f"Wrote {output} covering {len(concepts)} concept(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
