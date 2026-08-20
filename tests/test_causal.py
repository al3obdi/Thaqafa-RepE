"""Tests for the causal read-back check.

The property under test is not "a number comes out" but that the control
works: a fake whose probe reacts to *any* perturbation must show no lift over
random, and a fake that reacts only to the concept direction must show one.
Both regimes are constructed here, so each has a correct answer to find.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from src.models.rep_engine import RESID_POST_HOOK, CulturalRepE
from src.utils.causal import (
    ReadbackResult,
    SuppressionResult,
    random_directions,
    readback,
    summarize_readback,
    summarize_suppression,
    suppression,
)
from tests.helpers import DATASET_PATH

D_MODEL = 16
N_LAYERS = 6
CONCEPT_AXIS = 0
"""The coordinate the fake encodes "concept present" in."""


class ReadbackFake:
    """A model whose probe axis can be made selective or indiscriminate.

    Activations are built per prompt: exemplars of the concept carry a positive
    value on :data:`CONCEPT_AXIS`, everything else carries a negative one. Hooks
    registered by the engine are applied to the activation, so an injected
    direction propagates into what the probe reads - which is the mechanism the
    check relies on.

    ``sensitivity`` decides how the probe axis responds to an injection:

    * ``"selective"`` - only the component along :data:`CONCEPT_AXIS` moves it,
      so a random direction mostly does nothing.
    * ``"indiscriminate"`` - the injection's total magnitude moves it whatever
      its direction, which is the failure mode the random control exists to
      catch.
    """

    def __init__(self, sensitivity: str = "selective") -> None:
        self.cfg = type("Cfg", (), {"n_layers": N_LAYERS, "d_model": D_MODEL})()
        self.tokenizer = None
        self.sensitivity = sensitivity
        self.mod_dict: dict[str, Any] = {
            RESID_POST_HOOK.format(layer=layer): _HookPoint(RESID_POST_HOOK.format(layer=layer))
            for layer in range(N_LAYERS)
        }
        self._prompts: list[str] = []
        self._exemplars = {text for entry in _load_concepts() for text in entry.all_examples}

    def eval(self) -> ReadbackFake:
        """Match the ``torch.nn.Module`` interface."""
        return self

    def add_hook(self, name: str, hook: Any, dir: str = "fwd") -> None:  # noqa: A002
        """Register a hook, as TransformerLens does."""
        self.mod_dict[name].fwd_hooks.append(_LensHandle(hook))

    def to_tokens(self, prompts: list[str]) -> torch.Tensor:
        """Record the prompts and return one placeholder token each."""
        self._prompts = list(prompts)
        return torch.ones((len(prompts), 1), dtype=torch.long)

    def _injected(self) -> torch.Tensor:
        """Total offset the attached hooks apply to a zero activation."""
        total = torch.zeros(D_MODEL)
        for hook_point in self.mod_dict.values():
            for handle in hook_point.fwd_hooks:
                total = total + handle.user_hook(torch.zeros(1, 1, D_MODEL), None)[0, 0]
        return total

    def run_with_cache(
        self,
        tokens: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[None, dict[str, torch.Tensor]]:
        """Return activations that encode the concept, shifted by any injection."""
        name = str(kwargs.get("names_filter") or RESID_POST_HOOK.format(layer=0))
        injected = self._injected()
        if self.sensitivity == "selective":
            shift = float(injected[CONCEPT_AXIS])
        else:
            shift = float(torch.linalg.vector_norm(injected))

        rows = []
        for prompt in self._prompts:
            vector = torch.full((D_MODEL,), 0.1)
            vector[CONCEPT_AXIS] = 1.0 if prompt in self._exemplars else -1.0
            vector[CONCEPT_AXIS] += shift
            rows.append(vector)
        return None, {name: torch.stack(rows).unsqueeze(1)}


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


def _load_concepts() -> Any:
    """Load the shared concept dataset."""
    from src.data.dataset_builder import load_concepts

    return load_concepts(DATASET_PATH)


def make_engine(sensitivity: str = "selective") -> CulturalRepE:
    """Build an engine backed by :class:`ReadbackFake` with a concept vector."""
    engine = CulturalRepE(
        model_name="dummy/readback",
        device="cpu",
        dtype="float32",
        dataset_path=DATASET_PATH,
    )
    model = ReadbackFake(sensitivity=sensitivity)
    engine.model = model  # type: ignore[assignment]
    engine.tokenizer = model.tokenizer

    direction = torch.zeros(D_MODEL)
    direction[CONCEPT_AXIS] = 1.0
    engine.concept_vectors["diyafa_001"] = direction
    engine.extraction_layers["diyafa_001"] = 1
    return engine


@pytest.fixture(autouse=True)
def _wire_hook_removal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the fake's add_hook return a working removal handle."""

    def add_hook(self: ReadbackFake, name: str, hook: Any, dir: str = "fwd") -> None:  # noqa: A002
        point = self.mod_dict[name]
        entry = _LensHandle(hook)
        entry.hook = _RemovableHandle(point, entry)
        point.fwd_hooks.append(entry)

    monkeypatch.setattr(ReadbackFake, "add_hook", add_hook)


class TestRandomDirections:
    """The control directions themselves."""

    def test_are_unit_norm(self) -> None:
        for direction in random_directions(32, 4, seed=0):
            assert torch.linalg.vector_norm(direction) == pytest.approx(1.0, abs=1e-5)

    def test_have_the_requested_shape_and_count(self) -> None:
        directions = random_directions(32, 4, seed=0)
        assert len(directions) == 4
        assert all(d.shape == (32,) for d in directions)

    def test_are_reproducible_for_a_seed(self) -> None:
        """A control nobody can regenerate is not a control."""
        assert torch.equal(random_directions(16, 3, seed=7)[0], random_directions(16, 3, seed=7)[0])

    def test_differ_between_seeds(self) -> None:
        assert not torch.equal(
            random_directions(16, 3, seed=1)[0], random_directions(16, 3, seed=2)[0]
        )

    def test_are_not_all_the_same_direction(self) -> None:
        """A control that draws one direction repeatedly measures nothing."""
        directions = random_directions(64, 5, seed=0)
        assert not torch.allclose(directions[0], directions[1])

    def test_are_spread_over_the_sphere(self) -> None:
        """Uniform on the sphere: in high dimensions, pairs are near-orthogonal."""
        directions = random_directions(512, 8, seed=0)
        cosines = [
            abs(float(torch.dot(directions[i], directions[j])))
            for i in range(len(directions))
            for j in range(i + 1, len(directions))
        ]
        assert max(cosines) < 0.2

    def test_rejects_a_non_positive_width(self) -> None:
        with pytest.raises(ValueError, match="d_model must be positive"):
            random_directions(0, 3)

    def test_rejects_a_non_positive_count(self) -> None:
        with pytest.raises(ValueError, match="count must be positive"):
            random_directions(8, 0)


class TestReadback:
    """Injecting a concept and asking its probe whether it notices."""

    def test_a_selective_model_shows_lift_over_random(self) -> None:
        """When only the concept axis moves the probe, the concept arm wins."""
        result = readback(make_engine("selective"), "diyafa_001", 1, 4, strength=2.0, n_random=4)

        assert result.steered_rate > result.baseline_rate
        assert result.lift_over_random > 0.3

    def test_an_indiscriminate_model_shows_none(self) -> None:
        """Any perturbation moving the probe must not read as a concept effect."""
        result = readback(
            make_engine("indiscriminate"), "diyafa_001", 1, 4, strength=2.0, n_random=4
        )

        assert result.steered_rate > result.baseline_rate
        assert result.lift_over_random == pytest.approx(0.0, abs=1e-6)

    def test_records_both_layers_and_the_probe_quality(self) -> None:
        """A read-back through a probe that cannot read means nothing."""
        result = readback(make_engine(), "diyafa_001", 1, 4, strength=2.0, n_random=2)

        assert result.inject_layer == 1
        assert result.read_layer == 4
        assert result.probe_accuracy > 0.9
        assert result.n_prompts > 0

    def test_runs_one_pass_per_control_direction(self) -> None:
        result = readback(make_engine(), "diyafa_001", 1, 4, strength=2.0, n_random=3)
        assert len(result.random_rates) == 3

    def test_skipping_the_control_leaves_it_empty(self) -> None:
        """No random rates is how "uninterpretable" is signalled."""
        result = readback(make_engine(), "diyafa_001", 1, 4, strength=2.0, n_random=0)
        assert result.random_rates == []

    def test_reading_at_the_injection_layer_is_rejected(self) -> None:
        """That would confirm addition works, not that the concept propagated."""
        with pytest.raises(ValueError, match="deeper than inject_layer"):
            readback(make_engine(), "diyafa_001", 3, 3)

    def test_reading_above_the_injection_layer_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="deeper than inject_layer"):
            readback(make_engine(), "diyafa_001", 4, 2)

    def test_needs_no_prior_extraction(self) -> None:
        """The direction is found at the injection layer, so nothing is cached."""
        engine = make_engine()
        engine.concept_vectors.clear()
        engine.extraction_layers.clear()

        result = readback(engine, "diyafa_001", 1, 4, strength=2.0, n_random=1)

        assert result.n_prompts > 0

    def test_an_unknown_concept_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="was not found"):
            readback(make_engine(), "not_a_concept_999", 1, 4)

    def test_empty_prompts_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one prompt"):
            readback(make_engine(), "diyafa_001", 1, 4, neutral_prompts=[])

    def test_the_control_key_does_not_survive_the_call(self) -> None:
        """A leaked cache entry would show up as a concept in later output."""
        engine = make_engine()
        readback(engine, "diyafa_001", 1, 4, strength=2.0, n_random=2)

        assert not any(key.startswith("__") for key in engine.concept_vectors)
        assert not any(key.startswith("__") for key in engine.extraction_layers)

    def test_no_hooks_are_left_attached(self) -> None:
        """A leaked hook would silently steer everything measured afterwards."""
        engine = make_engine()
        readback(engine, "diyafa_001", 1, 4, strength=2.0, n_random=2)

        model = engine.model
        assert isinstance(model, ReadbackFake)
        attached = sum(len(point.fwd_hooks) for point in model.mod_dict.values())
        assert attached == 0

    def test_is_reproducible_for_a_seed(self) -> None:
        first = readback(make_engine(), "diyafa_001", 1, 4, strength=2.0, n_random=3, seed=5)
        second = readback(make_engine(), "diyafa_001", 1, 4, strength=2.0, n_random=3, seed=5)
        assert first.random_rates == second.random_rates


class TestReadbackResultProperties:
    """The derived numbers."""

    def test_mean_random_rate_averages_the_control(self) -> None:
        result = ReadbackResult("a", 1, 4, 0.2, "relative", 0.2, 0.9, [0.4, 0.6])
        assert result.mean_random_rate == pytest.approx(0.5)

    def test_no_control_falls_back_to_the_baseline(self) -> None:
        """So lift over random degenerates to lift over baseline, not to a win."""
        result = ReadbackResult("a", 1, 4, 0.2, "relative", 0.2, 0.9, [])
        assert result.mean_random_rate == pytest.approx(0.2)
        assert result.lift_over_random == pytest.approx(result.lift_over_baseline)

    def test_lift_over_baseline_is_the_raw_rise(self) -> None:
        result = ReadbackResult("a", 1, 4, 0.2, "relative", 0.25, 0.75, [0.5])
        assert result.lift_over_baseline == pytest.approx(0.5)

    def test_lift_over_random_discounts_the_control(self) -> None:
        result = ReadbackResult("a", 1, 4, 0.2, "relative", 0.25, 0.75, [0.5])
        assert result.lift_over_random == pytest.approx(0.25)


class TestSummarizeReadback:
    """Turning results into a table."""

    def test_orders_by_descending_lift_over_random(self) -> None:
        results = {
            "a": ReadbackResult("a", 1, 4, 0.2, "relative", 0.2, 0.5, [0.45]),
            "b": ReadbackResult("b", 1, 4, 0.2, "relative", 0.2, 0.9, [0.3]),
        }
        assert [row["concept_id"] for row in summarize_readback(results)] == ["b", "a"]

    def test_carries_the_control_and_the_probe_quality(self) -> None:
        rows = summarize_readback(
            {"a": ReadbackResult("a", 1, 4, 0.2, "relative", 0.2, 0.8, [0.4], 0.85, 16)}
        )
        assert rows[0]["mean_random_rate"] == pytest.approx(0.4)
        assert rows[0]["n_random"] == 1
        assert rows[0]["probe_accuracy"] == pytest.approx(0.85)
        assert rows[0]["n_prompts"] == 16

    def test_empty_results_give_no_rows(self) -> None:
        assert summarize_readback({}) == []


def test_numpy_is_used_for_rates_not_python_floats() -> None:
    """Guards the rate helper against returning a numpy scalar into JSON."""
    result = ReadbackResult("a", 1, 4, 0.2, "relative", 0.2, 0.8, [0.4])
    assert isinstance(result.mean_random_rate, float)
    assert not isinstance(result.mean_random_rate, np.floating)


class TestSuppression:
    """Subtracting a concept from its own exemplars."""

    def test_a_selective_model_loses_recognition(self) -> None:
        """When the concept axis is what the probe reads, removing it shows."""
        result = suppression(
            make_engine("selective"), "diyafa_001", 1, 4, strength=-3.0, n_random=3
        )

        assert result.baseline_rate > result.steered_rate
        assert result.drop_beyond_random > 0.3

    def test_an_indiscriminate_model_shows_no_specific_drop(self) -> None:
        """Damage from any large subtraction must not read as removal."""
        result = suppression(
            make_engine("indiscriminate"), "diyafa_001", 1, 4, strength=-3.0, n_random=3
        )

        assert result.drop_beyond_random == pytest.approx(0.0, abs=1e-6)

    def test_scores_only_exemplars_the_probe_never_saw(self) -> None:
        """A probe trained on an exemplar recognises it whatever is injected."""
        result = suppression(make_engine(), "diyafa_001", 1, 4, strength=-3.0, n_random=1)

        assert result.n_folds >= 2
        assert result.n_exemplars > 0

    def test_reports_held_out_balanced_accuracy(self) -> None:
        """A baseline of 1.00 is ambiguous without it: an all-positive probe
        would reach it too, and anything unsettling it would look like removal."""
        result = suppression(make_engine(), "diyafa_001", 1, 4, strength=-3.0, n_random=1)

        assert result.probe_balanced_accuracy > 0.5

    def test_a_positive_strength_is_rejected(self) -> None:
        """That measures amplification, which readback() already does."""
        with pytest.raises(ValueError, match="must be negative to suppress"):
            suppression(make_engine(), "diyafa_001", 1, 4, strength=0.2)

    def test_zero_strength_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be negative to suppress"):
            suppression(make_engine(), "diyafa_001", 1, 4, strength=0.0)

    def test_reading_at_the_injection_layer_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="deeper than inject_layer"):
            suppression(make_engine(), "diyafa_001", 3, 3, strength=-1.0)

    def test_an_unknown_concept_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="was not found"):
            suppression(make_engine(), "not_a_concept_999", 1, 4, strength=-1.0)

    def test_leaves_no_hooks_or_cache_entries_behind(self) -> None:
        """A leaked hook would silently steer everything measured afterwards."""
        engine = make_engine()
        suppression(engine, "diyafa_001", 1, 4, strength=-3.0, n_random=2)

        model = engine.model
        assert isinstance(model, ReadbackFake)
        assert sum(len(point.fwd_hooks) for point in model.mod_dict.values()) == 0
        assert not any(key.startswith("__") for key in engine.concept_vectors)

    def test_is_reproducible_for_a_seed(self) -> None:
        first = suppression(make_engine(), "diyafa_001", 1, 4, strength=-3.0, n_random=2, seed=5)
        again = suppression(make_engine(), "diyafa_001", 1, 4, strength=-3.0, n_random=2, seed=5)
        assert first.random_rates == again.random_rates
        assert first.steered_rate == again.steered_rate

    def test_skipping_the_control_leaves_it_empty(self) -> None:
        result = suppression(make_engine(), "diyafa_001", 1, 4, strength=-3.0, n_random=0)
        assert result.random_rates == []


class TestSuppressionResultProperties:
    """The derived numbers."""

    def test_drop_from_baseline_is_the_raw_fall(self) -> None:
        result = SuppressionResult("a", 1, 4, -0.2, "relative", 1.0, 0.2, [0.9])
        assert result.drop_from_baseline == pytest.approx(0.8)

    def test_drop_beyond_random_discounts_the_control(self) -> None:
        result = SuppressionResult("a", 1, 4, -0.2, "relative", 1.0, 0.2, [0.9])
        assert result.drop_beyond_random == pytest.approx(0.7)

    def test_no_control_falls_back_to_the_baseline(self) -> None:
        """So drop beyond random degenerates to the raw fall, not to a win."""
        result = SuppressionResult("a", 1, 4, -0.2, "relative", 1.0, 0.2, [])
        assert result.mean_random_rate == pytest.approx(1.0)
        assert result.drop_beyond_random == pytest.approx(result.drop_from_baseline)

    def test_a_control_that_drops_as_far_leaves_nothing(self) -> None:
        """Damage from any direction is not evidence about the concept."""
        result = SuppressionResult("a", 1, 4, -0.2, "relative", 1.0, 0.2, [0.2])
        assert result.drop_beyond_random == pytest.approx(0.0)


class TestSummarizeSuppression:
    """Turning results into a table."""

    def test_orders_by_descending_drop_beyond_random(self) -> None:
        results = {
            "a": SuppressionResult("a", 1, 4, -0.2, "relative", 1.0, 0.8, [0.9]),
            "b": SuppressionResult("b", 1, 4, -0.2, "relative", 1.0, 0.1, [0.9]),
        }
        assert [row["concept_id"] for row in summarize_suppression(results)] == ["b", "a"]

    def test_carries_the_control_and_the_probe_quality(self) -> None:
        rows = summarize_suppression(
            {"a": SuppressionResult("a", 1, 4, -0.2, "relative", 1.0, 0.2, [0.9], 12, 5, 0.8)}
        )
        assert rows[0]["mean_random_rate"] == pytest.approx(0.9)
        assert rows[0]["probe_balanced_accuracy"] == pytest.approx(0.8)
        assert rows[0]["n_exemplars"] == 12
        assert rows[0]["n_folds"] == 5

    def test_empty_results_give_no_rows(self) -> None:
        assert summarize_suppression({}) == []
