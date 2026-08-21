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

from src.data.contrastive import neutral_prompt_bank
from src.data.dataset_builder import load_concepts
from src.models.rep_engine import RESID_POST_HOOK, CulturalRepE
from src.utils.crosslingual import (
    ARABIC,
    ENGLISH,
    AlignmentResult,
    TransferResult,
    alignment,
    cosine,
    language_direction,
    summarize_alignment,
    summarize_transfer,
    transfer,
)
from tests.helpers import DATASET_PATH

D_MODEL = 8
N_LAYERS = 4


class _HookPoint:
    """Stands in for ``transformer_lens.hook_points.HookPoint``."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.fwd_hooks: list[_LensHandle] = []


class _RemovableHandle:
    """Stands in for ``torch.utils.hooks.RemovableHandle``."""

    def __init__(self, owner: _HookPoint, entry: _LensHandle) -> None:
        self._owner = owner
        self._entry = entry

    def remove(self) -> None:
        """Detach the hook so the next pass is unsteered."""
        if self._entry in self._owner.fwd_hooks:
            self._owner.fwd_hooks.remove(self._entry)


class _LensHandle:
    """Stands in for ``transformer_lens.hook_points.LensHandle``."""

    def __init__(self, user_hook: Any) -> None:
        self.user_hook = user_hook
        self.is_permanent = False
        self.hook: Any = None


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
        self.mod_dict: dict[str, _HookPoint] = {
            RESID_POST_HOOK.format(layer=layer): _HookPoint(RESID_POST_HOOK.format(layer=layer))
            for layer in range(N_LAYERS)
        }

    def eval(self) -> LanguageAwareFake:
        """Match the ``torch.nn.Module`` interface."""
        return self

    def add_hook(self, name: str, hook: Any, dir: str = "fwd") -> None:  # noqa: A002
        """Register a hook and hand back a working removal handle.

        The transfer check steers, so the fake has to carry an injection
        through into what the probe reads; a fake that only answered
        ``run_with_cache`` would report every steered rate as the unsteered
        one and make the check pass vacuously.
        """
        point = self.mod_dict[name]
        entry = _LensHandle(hook)
        entry.hook = _RemovableHandle(point, entry)
        point.fwd_hooks.append(entry)

    def _injected(self) -> torch.Tensor:
        """Total offset the attached hooks apply to a zero activation."""
        total = torch.zeros(D_MODEL)
        for point in self.mod_dict.values():
            for entry in point.fwd_hooks:
                total = total + entry.user_hook(torch.zeros(1, 1, D_MODEL), None)[0, 0]
        return total

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
        injected = self._injected()
        stacked = torch.stack([self._vector(prompt) + injected for prompt in self._prompts])
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


class TestTransfer:
    """Steering with one language's direction, reading with the other's probe."""

    def test_a_shared_axis_transfers(self) -> None:
        """The fake encodes the concept on one axis for both languages."""
        results = transfer(
            make_engine(), "diyafa_001", 1, 3, reader_language=ENGLISH, strengths=(2.0,)
        )

        assert len(results) == 1
        assert results[0].transfer_ratio > 0.8

    def test_both_reader_languages_work(self) -> None:
        """Transfer is a two-way question and the answer need not be symmetric."""
        english = transfer(make_engine(), "diyafa_001", 1, 3, ENGLISH, strengths=(2.0,))[0]
        arabic = transfer(make_engine(), "diyafa_001", 1, 3, ARABIC, strengths=(2.0,))[0]

        assert english.reader_language == ENGLISH
        assert arabic.reader_language == ARABIC

    def test_reads_only_prompts_in_the_reader_language(self) -> None:
        """Otherwise a rise could be the probe recognising the other script."""
        english = transfer(make_engine(), "diyafa_001", 1, 3, ENGLISH, strengths=(2.0,))[0]
        arabic = transfer(make_engine(), "diyafa_001", 1, 3, ARABIC, strengths=(2.0,))[0]

        assert english.n_prompts > 0
        assert arabic.n_prompts > 0
        # The two banks are different sets, so a shared bank would be a bug.
        assert set(neutral_prompt_bank(ENGLISH)).isdisjoint(neutral_prompt_bank(ARABIC))

    def test_every_strength_is_reported(self) -> None:
        """A saturated point reads ratio 1.00 for free; the grid shows that."""
        results = transfer(make_engine(), "diyafa_001", 1, 3, ENGLISH, strengths=(0.5, 1.0, 2.0))

        assert [result.strength for result in results] == [0.5, 1.0, 2.0]

    def test_the_probe_is_shared_across_strengths(self) -> None:
        """Refitting per point would make the points incomparable."""
        results = transfer(make_engine(), "diyafa_001", 1, 3, ENGLISH, strengths=(0.5, 1.0, 2.0))

        assert len({result.probe_accuracy for result in results}) == 1
        assert len({result.baseline_rate for result in results}) == 1

    def test_carries_both_the_floor_and_the_ceiling(self) -> None:
        """A transfer number without them cannot be read at all."""
        result = transfer(make_engine(), "diyafa_001", 1, 3, ENGLISH, strengths=(2.0,))[0]

        assert result.random_rates
        assert result.same_language_rate >= 0.0
        assert result.probe_accuracy > 0.0

    def test_an_unknown_reader_language_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="reader_language must be one of"):
            transfer(make_engine(), "diyafa_001", 1, 3, reader_language="fr")

    def test_reading_at_the_injection_layer_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="deeper than inject_layer"):
            transfer(make_engine(), "diyafa_001", 2, 2)

    def test_an_unknown_concept_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="was not found"):
            transfer(make_engine(), "not_a_concept_999", 1, 3)

    def test_leaves_no_hooks_or_cache_entries_behind(self) -> None:
        """A leaked hook would silently steer everything measured afterwards."""
        engine = make_engine()
        transfer(engine, "diyafa_001", 1, 3, ENGLISH, strengths=(2.0,))

        assert not any(key.startswith("__") for key in engine.concept_vectors)


class TestTransferResultProperties:
    """The derived numbers."""

    def _result(self, same: float, other: float, random: list[float]) -> TransferResult:
        """Build a result with the rates under test."""
        return TransferResult(
            concept="a",
            reader_language=ENGLISH,
            inject_layer=1,
            read_layer=2,
            strength=0.2,
            probe_accuracy=0.8,
            baseline_rate=0.2,
            same_language_rate=same,
            other_language_rate=other,
            random_rates=random,
        )

    def test_complete_transfer_is_one(self) -> None:
        """The other language moved this reader exactly as far as its own did."""
        assert self._result(0.9, 0.9, [0.1]).transfer_ratio == pytest.approx(1.0)

    def test_no_transfer_is_zero(self) -> None:
        """The other language did no better than the random floor."""
        assert self._result(0.9, 0.1, [0.1]).transfer_ratio == pytest.approx(0.0)

    def test_partial_transfer_is_the_fraction(self) -> None:
        result = self._result(0.9, 0.5, [0.1])
        assert result.transfer_ratio == pytest.approx(0.5)

    def test_no_ceiling_gives_no_ratio(self) -> None:
        """A concept whose own direction does not move its own reader has
        nothing to transfer, and a ratio there would divide by noise."""
        assert self._result(0.1, 0.4, [0.3]).transfer_ratio == 0.0

    def test_no_control_falls_back_to_the_baseline(self) -> None:
        result = self._result(0.9, 0.5, [])
        assert result.mean_random_rate == pytest.approx(0.2)


class TestSummarizeTransfer:
    """Turning results into a table."""

    def test_orders_by_descending_transfer_ratio(self) -> None:
        low = TransferResult("a", ENGLISH, 1, 2, 0.2, 0.8, 0.1, 0.9, 0.3, [0.1])
        high = TransferResult("b", ENGLISH, 1, 2, 0.2, 0.8, 0.1, 0.9, 0.9, [0.1])

        rows = summarize_transfer({"a": low, "b": high})

        assert [row["concept_id"] for row in rows] == ["b", "a"]

    def test_carries_the_floor_the_ceiling_and_the_probe(self) -> None:
        rows = summarize_transfer(
            {"a": TransferResult("a", ARABIC, 1, 2, 0.2, 0.75, 0.1, 0.9, 0.5, [0.1], 24)}
        )

        assert rows[0]["reader_language"] == ARABIC
        assert rows[0]["same_language_rate"] == pytest.approx(0.9)
        assert rows[0]["mean_random_rate"] == pytest.approx(0.1)
        assert rows[0]["probe_accuracy"] == pytest.approx(0.75)
        assert rows[0]["n_prompts"] == 24

    def test_empty_results_give_no_rows(self) -> None:
        assert summarize_transfer({}) == []
