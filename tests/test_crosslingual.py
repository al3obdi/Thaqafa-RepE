"""Tests for the Arabic-English direction agreement check.

The interesting property here is not that a cosine gets computed, but that the
control does its job: a layer with a dominant axis makes *every* pair of
directions look similar, and the check has to report that as no separation
rather than as cross-lingual agreement. The fake below can be switched between
those two regimes, so both have a correct answer to find.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from src.data.dataset_builder import load_concepts
from src.models.rep_engine import RESID_POST_HOOK, CulturalRepE
from src.utils.crosslingual import (
    ARABIC,
    ENGLISH,
    AlignmentResult,
    alignment,
    cosine,
    language_direction,
    summarize_alignment,
)
from tests.helpers import DATASET_PATH

D_MODEL = 8
N_LAYERS = 4


def _is_arabic(text: str) -> bool:
    """Return whether a prompt is written in Arabic script."""
    return any("؀" <= character <= "ۿ" for character in text)


class LanguageAwareFake:
    """A model whose activations encode a concept, a language and a shared axis.

    Each prompt's activation is built from three parts:

    * a per-concept component, the same for both languages, which is what the
      check is supposed to find;
    * a per-language component, which it must *not* mistake for that;
    * a constant shared component whose weight is tunable, standing in for a
      layer with a dominant axis that drags every cosine upward.

    Concepts are recognised by keyword, because the engine hands this fake real
    prompts from the dataset.
    """

    KEYWORDS: dict[str, tuple[str, ...]] = {
        "diyafa_001": ("ضيف", "ضياف", "guest", "host", "hospital"),
        "karam_001": ("كرم", "كري", "generous", "generosity", "gave", "give"),
        "sabr_001": ("صبر", "صابر", "patien", "endur", "wait"),
    }

    def __init__(self, shared_weight: float = 0.0, concept_weight: float = 1.0) -> None:
        self.cfg = type("Cfg", (), {"n_layers": N_LAYERS, "d_model": D_MODEL})()
        self.tokenizer = None
        self.shared_weight = shared_weight
        self.concept_weight = concept_weight
        self._prompts: list[str] = []
        # A component every exemplar carries and no contrast does. A shared
        # axis present on *both* sides would cancel in the mean difference, so
        # it could not confound anything; this is the version that survives
        # extraction and drags every cosine upward at once.
        self._exemplars = {
            text for entry in load_concepts(DATASET_PATH) for text in entry.all_examples
        }

    def eval(self) -> LanguageAwareFake:
        """Match the ``torch.nn.Module`` interface."""
        return self

    def to_tokens(self, prompts: list[str]) -> torch.Tensor:
        """Record the prompts and return one placeholder token each."""
        self._prompts = list(prompts)
        return torch.ones((len(prompts), 1), dtype=torch.long)

    def _vector(self, prompt: str) -> torch.Tensor:
        """Build the activation for one prompt."""
        vector = torch.zeros(D_MODEL)
        if prompt in self._exemplars:
            vector[0] = self.shared_weight

        lowered = prompt.lower()
        for index, (_concept, keywords) in enumerate(self.KEYWORDS.items()):
            if any(keyword in lowered for keyword in keywords):
                vector[1 + index] = self.concept_weight

        vector[5 if _is_arabic(prompt) else 6] = 0.3
        return vector

    def run_with_cache(
        self,
        tokens: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[None, dict[str, torch.Tensor]]:
        """Return one activation per recorded prompt."""
        name = str(kwargs.get("names_filter") or RESID_POST_HOOK.format(layer=0))
        stacked = torch.stack([self._vector(prompt) for prompt in self._prompts])
        return None, {name: stacked.unsqueeze(1)}


def make_engine(shared_weight: float = 0.0) -> CulturalRepE:
    """Build an engine backed by :class:`LanguageAwareFake`."""
    engine = CulturalRepE(
        model_name="dummy/crosslingual",
        device="cpu",
        dtype="float32",
        dataset_path=DATASET_PATH,
    )
    model = LanguageAwareFake(shared_weight=shared_weight)
    engine.model = model  # type: ignore[assignment]
    engine.tokenizer = model.tokenizer
    return engine


class TestCosine:
    """The similarity itself."""

    def test_identical_vectors_give_one(self) -> None:
        vector = torch.tensor([1.0, 2.0, 3.0])
        assert cosine(vector, vector) == pytest.approx(1.0)

    def test_opposite_vectors_give_minus_one(self) -> None:
        vector = torch.tensor([1.0, 0.0])
        assert cosine(vector, -vector) == pytest.approx(-1.0)

    def test_orthogonal_vectors_give_zero(self) -> None:
        assert cosine(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])) == pytest.approx(0.0)

    def test_magnitude_does_not_matter(self) -> None:
        """Cosine is about direction; a rescaled vector is the same direction."""
        first = torch.tensor([1.0, 2.0])
        assert cosine(first, first * 17.0) == pytest.approx(1.0)

    def test_zero_vector_has_no_angle(self) -> None:
        """Returning 0.0 beats returning NaN into a results table."""
        assert cosine(torch.zeros(3), torch.ones(3)) == 0.0

    def test_shape_mismatch_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="shape mismatch"):
            cosine(torch.ones(3), torch.ones(4))

    def test_two_dimensional_input_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be 1-D"):
            cosine(torch.ones(2, 2), torch.ones(2, 2))


class TestLanguageDirection:
    """Extracting from one language's exemplars only."""

    def test_returns_a_unit_vector_of_model_width(self) -> None:
        engine = make_engine()
        direction = language_direction(engine, "diyafa_001", ARABIC, layer=1)
        assert direction.shape == (D_MODEL,)
        assert torch.linalg.vector_norm(direction) == pytest.approx(1.0, abs=1e-5)

    def test_does_not_cache_over_the_concept_vector(self) -> None:
        """A cross-lingual analysis must not overwrite the run's own vector."""
        engine = make_engine()
        engine.extract_vector("diyafa_001", layer=1)
        before = engine.concept_vectors["diyafa_001"].clone()

        language_direction(engine, "diyafa_001", ARABIC, layer=1)

        assert torch.equal(engine.concept_vectors["diyafa_001"], before)

    def test_both_sides_come_from_the_same_language(self) -> None:
        """Otherwise the direction would just separate Arabic text from English."""
        engine = make_engine()
        arabic = language_direction(engine, "diyafa_001", ARABIC, layer=1)
        # Index 5 is the fake's Arabic marker. Both the positives and the
        # contrasts carry it, so it cancels in the difference.
        assert arabic[5].abs() < 1e-5

    def test_unknown_language_is_rejected(self) -> None:
        engine = make_engine()
        with pytest.raises(ValueError, match="language must be one of"):
            language_direction(engine, "diyafa_001", "fr", layer=1)

    def test_unknown_concept_is_rejected(self) -> None:
        engine = make_engine()
        with pytest.raises(ValueError, match="was not found"):
            language_direction(engine, "not_a_concept_999", ARABIC, layer=1)

    def test_a_model_that_cannot_load_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The engine loads on first use, so a load failure must surface here."""
        engine = CulturalRepE(model_name="dummy/x", device="cpu", dataset_path=DATASET_PATH)

        def refuse() -> None:
            raise RuntimeError("no weights")

        monkeypatch.setattr(engine, "load_model", refuse)

        with pytest.raises(RuntimeError, match="no weights"):
            language_direction(engine, "diyafa_001", ENGLISH, layer=1)


class TestAlignment:
    """The aligned measurement and its control."""

    def test_matched_languages_beat_mismatched_ones(self) -> None:
        """The whole point: same concept across languages is the closer pair."""
        engine = make_engine()
        results = alignment(engine, ["diyafa_001", "karam_001"], layer=1)

        for result in results.values():
            assert result.aligned_cosine > result.mean_mismatched
            assert result.separation > 0.0

    def test_a_dominant_shared_axis_shows_up_as_no_separation(self) -> None:
        """A high aligned cosine is worthless if every pair reaches it too."""
        engine = make_engine(shared_weight=50.0)
        results = alignment(engine, ["diyafa_001", "karam_001"], layer=1)

        for result in results.values():
            assert result.aligned_cosine > 0.9
            assert result.separation < 0.2

    def test_records_the_layer_and_the_exemplar_counts(self) -> None:
        """A cosine without its layer and sample size cannot be interpreted."""
        engine = make_engine()
        result = alignment(engine, ["diyafa_001", "karam_001"], layer=2)["diyafa_001"]

        assert result.layer == 2
        assert result.n_arabic > 0
        assert result.n_english > 0

    def test_every_other_concept_enters_the_control(self) -> None:
        """Dropping one would let a lucky pairing set the baseline."""
        concepts = ["diyafa_001", "karam_001", "sabr_001"]
        results = alignment(make_engine(), concepts, layer=1)

        assert set(results["diyafa_001"].mismatched_cosines) == {"karam_001", "sabr_001"}

    def test_one_layer_is_used_for_every_concept(self) -> None:
        """Cosines from different layers are not comparable."""
        results = alignment(make_engine(), ["diyafa_001", "karam_001"], layer=3)
        assert {r.layer for r in results.values()} == {3}

    def test_empty_concept_list_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one identifier"):
            alignment(make_engine(), [])

    def test_single_concept_warns_that_the_control_is_missing(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Silently reporting separation == aligned_cosine would mislead."""
        with caplog.at_level("WARNING"):
            results = alignment(make_engine(), ["diyafa_001"], layer=1)

        assert "control" in caplog.text
        assert results["diyafa_001"].separation == results["diyafa_001"].aligned_cosine


class TestSummarizeAlignment:
    """Turning results into a table."""

    def test_orders_by_descending_separation(self) -> None:
        """The concepts that cross languages most clearly come first."""
        results = {
            "a": AlignmentResult("a", 1, 0.9, {"b": 0.8}),
            "b": AlignmentResult("b", 1, 0.9, {"a": 0.1}),
        }
        rows = summarize_alignment(results)
        assert [row["concept_id"] for row in rows] == ["b", "a"]

    def test_carries_the_control_alongside_the_headline(self) -> None:
        """A separation without its two inputs cannot be checked."""
        rows = summarize_alignment({"a": AlignmentResult("a", 2, 0.9, {"b": 0.4}, 6, 6)})
        assert rows[0] == {
            "concept_id": "a",
            "layer": 2,
            "aligned_cosine": 0.9,
            "mean_mismatched_cosine": 0.4,
            "separation": pytest.approx(0.5),
            "n_arabic": 6,
            "n_english": 6,
        }

    def test_empty_results_give_no_rows(self) -> None:
        assert summarize_alignment({}) == []


class TestAlignmentResultProperties:
    """The derived numbers."""

    def test_mean_mismatched_averages_the_control(self) -> None:
        result = AlignmentResult("a", 0, 0.9, {"b": 0.2, "c": 0.4})
        assert result.mean_mismatched == pytest.approx(0.3)

    def test_no_control_leaves_separation_equal_to_the_headline(self) -> None:
        """Which is the signal that it must not be read as a separation."""
        result = AlignmentResult("a", 0, 0.75, {})
        assert result.mean_mismatched == 0.0
        assert result.separation == pytest.approx(0.75)
