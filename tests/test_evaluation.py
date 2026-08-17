"""Tests for the layer-set grid and the KL effect measurement.

These run on CPU against :class:`SteerableFakeModel`, whose loss and logits both
rise deterministically with the amount of steering attached. That lets the tests
assert on the *direction* of a trade-off - more layers cost more fluency - rather
than merely that a number came back.
"""

from __future__ import annotations

import pytest

from src.models.rep_engine import CulturalRepE
from src.utils.evaluation import (
    evaluate_layer_sets,
    evaluate_steering,
    measure_steering_effect,
    next_token_log_probs,
    summarize_layer_sets,
)
from tests.helpers import SteerableFakeModel, make_steerable_engine

PROMPTS = ["a short prompt"]


class TestLayerSetSweep:
    """Comparing layer configurations against each other."""

    def test_one_sweep_per_configuration(self) -> None:
        engine = make_steerable_engine()

        grid = evaluate_layer_sets(
            engine, "diyafa", PROMPTS, layer_sets=[[1], [1, 2]], strengths=[1.0]
        )

        assert set(grid) == {(1,), (1, 2)}
        assert list(grid[(1,)]) == [1.0]

    def test_configurations_are_keyed_by_resolved_sorted_layers(self) -> None:
        engine = make_steerable_engine()

        grid = evaluate_layer_sets(
            engine, "diyafa", PROMPTS, layer_sets=[[3, 1, 3]], strengths=[1.0]
        )

        assert set(grid) == {(1, 3)}

    def test_negative_indices_are_resolved_in_the_key(self) -> None:
        engine = make_steerable_engine()

        grid = evaluate_layer_sets(engine, "diyafa", PROMPTS, layer_sets=[[-1]], strengths=[1.0])

        assert set(grid) == {(3,)}

    def test_result_records_which_layers_were_used(self) -> None:
        engine = make_steerable_engine()

        grid = evaluate_layer_sets(engine, "diyafa", PROMPTS, layer_sets=[[0, 3]], strengths=[2.0])

        assert grid[(0, 3)][2.0].layers == (0, 3)

    def test_more_layers_cost_more_fluency(self) -> None:
        # The fake's loss rises with the total attached steering, so a
        # two-layer configuration must score worse than one layer at equal
        # strength. This is the trade-off the grid exists to expose.
        engine = make_steerable_engine()

        grid = evaluate_layer_sets(
            engine, "diyafa", PROMPTS, layer_sets=[[1], [1, 2]], strengths=[1.0]
        )

        assert grid[(1, 2)][1.0].mean_loss > grid[(1,)][1.0].mean_loss

    def test_more_layers_produce_a_larger_effect(self) -> None:
        engine = make_steerable_engine()

        grid = evaluate_layer_sets(
            engine, "diyafa", PROMPTS, layer_sets=[[1], [1, 2]], strengths=[1.0]
        )

        assert grid[(1, 2)][1.0].effect_kl >= grid[(1,)][1.0].effect_kl

    def test_grid_leaves_no_hooks_attached(self) -> None:
        model = SteerableFakeModel()
        engine = make_steerable_engine(model)

        evaluate_layer_sets(engine, "diyafa", PROMPTS, layer_sets=[[0], [1, 2]], strengths=[1.0])

        assert engine.active_hook_names == []
        assert all(not point.fwd_hooks for point in model.mod_dict.values())

    def test_effect_measurement_can_be_disabled(self) -> None:
        engine = make_steerable_engine()

        grid = evaluate_layer_sets(
            engine, "diyafa", PROMPTS, layer_sets=[[1]], strengths=[1.0], measure_effect=False
        )
        effect = grid[(1,)][1.0].effect_kl

        assert effect != effect  # nan

    def test_empty_layer_sets_are_rejected(self) -> None:
        engine = make_steerable_engine()

        with pytest.raises(ValueError, match="at least one configuration"):
            evaluate_layer_sets(engine, "diyafa", PROMPTS, layer_sets=[])

    def test_an_empty_configuration_is_rejected(self) -> None:
        engine = make_steerable_engine()

        with pytest.raises(ValueError, match="at least one layer"):
            evaluate_layer_sets(engine, "diyafa", PROMPTS, layer_sets=[[1], []])

    def test_unknown_concept_is_rejected(self) -> None:
        engine = make_steerable_engine()

        with pytest.raises(KeyError, match="No vector stored"):
            evaluate_layer_sets(engine, "missing", PROMPTS, layer_sets=[[1]])


class TestSummarizeLayerSets:
    """Flattening the grid into table rows."""

    def test_rows_are_sorted_by_configuration_then_strength(self) -> None:
        engine = make_steerable_engine()
        grid = evaluate_layer_sets(
            engine, "diyafa", PROMPTS, layer_sets=[[2], [0, 1]], strengths=[1.0, -1.0]
        )

        rows = summarize_layer_sets(grid)

        assert [row["layers"] for row in rows] == [(0, 1), (0, 1), (2,), (2,)]
        assert [row["strength"] for row in rows] == [-1.0, 1.0, -1.0, 1.0]

    def test_each_row_carries_effect_and_cost(self) -> None:
        engine = make_steerable_engine()
        grid = evaluate_layer_sets(engine, "diyafa", PROMPTS, layer_sets=[[0, 1]], strengths=[1.0])

        row = summarize_layer_sets(grid)[0]

        assert row["n_layers"] == 2
        assert "effect_kl" in row
        assert "mean_loss" in row
        assert "perplexity" in row

    def test_empty_grid_gives_no_rows(self) -> None:
        assert summarize_layer_sets({}) == []


class TestMeasureSteeringEffect:
    """The KL effect size."""

    def test_zero_strength_moves_nothing(self) -> None:
        engine = make_steerable_engine()

        effect = measure_steering_effect(engine, "diyafa", PROMPTS, strength=0.0)

        assert effect == pytest.approx(0.0, abs=1e-6)

    def test_a_real_injection_moves_the_distribution(self) -> None:
        engine = make_steerable_engine()

        effect = measure_steering_effect(engine, "diyafa", PROMPTS, strength=3.0)

        assert effect > 0.0

    def test_effect_is_never_negative(self) -> None:
        engine = make_steerable_engine()

        for strength in (-2.0, -0.5, 0.5, 2.0):
            assert measure_steering_effect(engine, "diyafa", PROMPTS, strength=strength) >= 0.0

    def test_stronger_injection_moves_further(self) -> None:
        engine = make_steerable_engine()

        small = measure_steering_effect(engine, "diyafa", PROMPTS, strength=1.0)
        large = measure_steering_effect(engine, "diyafa", PROMPTS, strength=4.0)

        assert large > small

    def test_measurement_leaves_no_hooks_attached(self) -> None:
        model = SteerableFakeModel()
        engine = make_steerable_engine(model)

        measure_steering_effect(engine, "diyafa", PROMPTS, strength=2.0)

        assert engine.active_hook_names == []
        assert all(not point.fwd_hooks for point in model.mod_dict.values())

    def test_empty_prompts_are_rejected(self) -> None:
        engine = make_steerable_engine()

        with pytest.raises(ValueError, match="at least one text"):
            measure_steering_effect(engine, "diyafa", [], strength=1.0)

    def test_evaluate_steering_can_record_the_effect(self) -> None:
        engine = make_steerable_engine()

        results = evaluate_steering(
            engine, "diyafa", PROMPTS, strengths=[2.0], generate=False, measure_effect=True
        )

        assert results[2.0].effect_kl > 0.0

    def test_effect_is_not_measured_by_default(self) -> None:
        engine = make_steerable_engine()

        results = evaluate_steering(engine, "diyafa", PROMPTS, strengths=[2.0], generate=False)
        effect = results[2.0].effect_kl

        assert effect != effect  # nan


class TestNextTokenLogProbs:
    """The distribution used by the effect measurement."""

    def test_returns_a_normalised_distribution(self) -> None:
        engine = make_steerable_engine()

        log_probs = next_token_log_probs(engine, PROMPTS[0])

        assert log_probs.ndim == 1
        assert log_probs.exp().sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_unloaded_model_is_rejected(self) -> None:
        engine = CulturalRepE(model_name="dummy/model", device="cpu")

        with pytest.raises(RuntimeError, match="not loaded"):
            next_token_log_probs(engine, "a prompt")
