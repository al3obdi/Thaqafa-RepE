"""Integrity checks for the cultural concepts dataset.

These tests gate the data file itself, so a future contribution that breaks
the schema, duplicates an identifier, or ships an entry too thin to probe
fails CI before any human review. The thresholds mirror CONTRIBUTING.md.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from src.data.dataset_builder import CulturalConcept, load_concepts

DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "datasets" / "cultural_concepts.jsonl"

VALID_SENTIMENTS = {"positive", "negative", "mixed"}
VALID_REVIEW_STATUSES = {"reviewed", "pending_native_review"}
CONCEPT_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*_[0-9]{3}$")

# Thresholds from CONTRIBUTING.md. Raising them is fine; lowering them needs
# a documented reason.
MIN_EXAMPLES_PER_LANGUAGE = 3
MIN_CONTRASTS_PER_LANGUAGE = 2

ARABIC_CHAR = re.compile(r"[؀-ۿ]")


@pytest.fixture(scope="module")
def concepts() -> list[CulturalConcept]:
    """Load the dataset once for the whole module."""
    return load_concepts(DATASET_PATH)


class TestFileFormat:
    """The JSONL container itself."""

    def test_every_line_is_valid_json(self) -> None:
        for number, line in enumerate(
            DATASET_PATH.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:  # pragma: no cover - failure path
                pytest.fail(f"line {number} is not valid JSON: {exc}")

    def test_dataset_is_not_empty(self, concepts: list[CulturalConcept]) -> None:
        assert len(concepts) >= 3


class TestIdentifiers:
    """Concept identifiers."""

    def test_ids_are_unique(self, concepts: list[CulturalConcept]) -> None:
        ids = [c.concept_id for c in concepts]
        assert len(ids) == len(set(ids)), "duplicate concept_id in the dataset"

    def test_ids_match_the_convention(self, concepts: list[CulturalConcept]) -> None:
        for c in concepts:
            assert CONCEPT_ID_PATTERN.match(
                c.concept_id
            ), f"{c.concept_id!r} does not match <transliteration>_<3 digits>"

    def test_seed_concepts_are_still_present(self, concepts: list[CulturalConcept]) -> None:
        # Existing extractions reference these ids; removing one is a breaking
        # change that needs a deliberate migration, not a dataset edit.
        ids = {c.concept_id for c in concepts}
        assert {"wasta_001", "muruah_001", "diyafa_001"} <= ids


class TestEntryQuality:
    """Per-entry content requirements."""

    def test_required_text_fields_are_non_empty(self, concepts: list[CulturalConcept]) -> None:
        for c in concepts:
            assert c.concept_ar.strip(), c.concept_id
            assert c.concept_en.strip(), c.concept_id
            assert c.category.strip(), c.concept_id
            assert c.description.strip(), c.concept_id
            assert c.cultural_context.strip(), c.concept_id

    def test_minimum_example_counts(self, concepts: list[CulturalConcept]) -> None:
        for c in concepts:
            assert len(c.examples_ar) >= MIN_EXAMPLES_PER_LANGUAGE, c.concept_id
            assert len(c.examples_en) >= MIN_EXAMPLES_PER_LANGUAGE, c.concept_id

    def test_minimum_contrast_counts(self, concepts: list[CulturalConcept]) -> None:
        for c in concepts:
            assert len(c.contrast_ar) >= MIN_CONTRASTS_PER_LANGUAGE, c.concept_id
            assert len(c.contrast_en) >= MIN_CONTRASTS_PER_LANGUAGE, c.concept_id

    def test_language_fields_hold_the_right_script(self, concepts: list[CulturalConcept]) -> None:
        for c in concepts:
            assert ARABIC_CHAR.search(c.concept_ar), c.concept_id
            for text in [*c.examples_ar, *c.contrast_ar]:
                assert ARABIC_CHAR.search(text), f"{c.concept_id}: not Arabic: {text!r}"
            for text in [*c.examples_en, *c.contrast_en]:
                assert not ARABIC_CHAR.search(text), f"{c.concept_id}: Arabic in EN: {text!r}"

    def test_no_duplicate_sentences_within_an_entry(self, concepts: list[CulturalConcept]) -> None:
        for c in concepts:
            pool = [*c.all_examples, *c.all_contrasts]
            assert len(pool) == len(set(pool)), f"{c.concept_id}: duplicated sentence"

    def test_contrasts_do_not_repeat_exemplars(self, concepts: list[CulturalConcept]) -> None:
        # A contrast identical to an exemplar would cancel the concept out of
        # its own direction.
        for c in concepts:
            overlap = set(c.all_examples) & set(c.all_contrasts)
            assert not overlap, f"{c.concept_id}: {overlap}"


class TestControlledVocabularies:
    """Enumerated fields."""

    def test_sentiment_is_valid(self, concepts: list[CulturalConcept]) -> None:
        for c in concepts:
            assert c.sentiment in VALID_SENTIMENTS, c.concept_id

    def test_review_status_is_valid(self, concepts: list[CulturalConcept]) -> None:
        for c in concepts:
            assert c.review_status in VALID_REVIEW_STATUSES, c.concept_id

    def test_contested_concepts_stay_mixed(self, concepts: list[CulturalConcept]) -> None:
        # Wasta is normatively contested by the project's own documentation;
        # flattering it into "positive" (or damning it to "negative") is a
        # substantive editorial change that must not happen silently.
        wasta = next(c for c in concepts if c.concept_id == "wasta_001")
        assert wasta.sentiment == "mixed"

    def test_dialect_is_stated(self, concepts: list[CulturalConcept]) -> None:
        for c in concepts:
            assert c.dialect.strip(), c.concept_id
