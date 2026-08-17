"""Loading and validation of the Arab cultural concept dataset.

The dataset lives in ``data/datasets/cultural_concepts.jsonl``: one JSON object
per line, each describing a single cultural concept in both Arabic and English.
This module turns those lines into typed :class:`CulturalConcept` records and
provides small helpers used by the extraction scripts and notebooks.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_DATASET_PATH = Path("data/datasets/cultural_concepts.jsonl")

REQUIRED_FIELDS: tuple[str, ...] = (
    "concept_id",
    "concept_ar",
    "concept_en",
    "category",
    "description",
)


def _as_str_list(value: object) -> list[str]:
    """Coerce a raw JSON value into a list of strings.

    Args:
        value: A decoded JSON value, expected to be a list of strings.

    Returns:
        The value as a list of strings. ``None`` and non-list values yield an
        empty list, so a malformed example field degrades gracefully instead of
        crashing the whole load.
    """
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


@dataclass(frozen=True)
class CulturalConcept:
    """A single Arab cultural concept with bilingual examples.

    Attributes:
        concept_id: Stable identifier, for example ``"wasta_001"``.
        concept_ar: The concept name in Arabic.
        concept_en: The concept name transliterated and glossed in English.
        category: Coarse grouping such as ``"social"``, ``"ethical"`` or
            ``"cultural"``.
        description: One-sentence definition in English.
        examples_ar: Arabic sentences that express the concept.
        examples_en: English sentences that express the concept.
        cultural_context: Notes on where and why the concept matters.
        sentiment: Overall valence, one of ``"positive"``, ``"negative"`` or
            ``"mixed"``.
    """

    concept_id: str
    concept_ar: str
    concept_en: str
    category: str
    description: str
    examples_ar: list[str] = field(default_factory=list)
    examples_en: list[str] = field(default_factory=list)
    cultural_context: str = ""
    sentiment: str = "mixed"

    @classmethod
    def from_dict(cls, record: dict[str, object]) -> CulturalConcept:
        """Build a concept from a raw JSON record.

        Args:
            record: Decoded JSON object from the dataset file.

        Returns:
            The corresponding :class:`CulturalConcept`.

        Raises:
            ValueError: If a required field is missing.
        """
        missing = [name for name in REQUIRED_FIELDS if not record.get(name)]
        if missing:
            raise ValueError(f"Concept record is missing required field(s): {', '.join(missing)}")

        return cls(
            concept_id=str(record["concept_id"]),
            concept_ar=str(record["concept_ar"]),
            concept_en=str(record["concept_en"]),
            category=str(record["category"]),
            description=str(record["description"]),
            examples_ar=_as_str_list(record.get("examples_ar")),
            examples_en=_as_str_list(record.get("examples_en")),
            cultural_context=str(record.get("cultural_context", "")),
            sentiment=str(record.get("sentiment", "mixed")),
        )

    @property
    def all_examples(self) -> list[str]:
        """Return the Arabic and English examples concatenated."""
        return [*self.examples_ar, *self.examples_en]


def iter_concepts(path: Path | str = DEFAULT_DATASET_PATH) -> Iterator[CulturalConcept]:
    """Yield concepts from a JSONL dataset file one at a time.

    Args:
        path: Path to the JSONL file.

    Yields:
        Parsed :class:`CulturalConcept` instances.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If a line is not valid JSON or lacks a required field.
    """
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Cultural concept dataset not found at {dataset_path}")

    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {dataset_path}") from exc
            yield CulturalConcept.from_dict(record)


def load_concepts(path: Path | str = DEFAULT_DATASET_PATH) -> list[CulturalConcept]:
    """Load every concept from a JSONL dataset file.

    Args:
        path: Path to the JSONL file.

    Returns:
        All concepts in file order.
    """
    concepts = list(iter_concepts(path))
    logger.info("Loaded %d cultural concepts from %s", len(concepts), path)
    return concepts


def filter_by_category(
    concepts: list[CulturalConcept],
    category: str,
) -> list[CulturalConcept]:
    """Return the concepts belonging to ``category``.

    Args:
        concepts: Concepts to filter.
        category: Category name, compared case-insensitively.

    Returns:
        The matching concepts, in input order.
    """
    wanted = category.strip().lower()
    return [concept for concept in concepts if concept.category.lower() == wanted]
