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


class TestNormRelativeStrength:
    """Strength calibrated against the layer's residual norm.

    Residual norms grow steeply with depth - measured at 61 to 396 across
    GPT-2's twelve layers - so an absolute coefficient is a strong
    intervention early and a negligible one late. Relative mode divides that
    scale out.
    """

    def test_absolute_mode_passes_the_coefficient_through(self) -> None:
        engine = make_steerable_engine()

        handle = engine.inject_vector("diyafa", strength=2.0, layers=[1])[0]

        assert handle.strength == pytest.approx(2.0)

    def test_relative_mode_scales_by_the_calibrated_norm(self) -> None:
        engine = make_steerable_engine()
        engine.layer_norms[1] = 50.0

        handle = engine.inject_vector("diyafa", strength=0.1, layers=[1], strength_mode="relative")[
            0
        ]

        # 10% of a residual norm of 50 is an absolute coefficient of 5.
        assert handle.strength == pytest.approx(5.0)

    def test_relative_mode_differs_per_layer(self) -> None:
        # The same requested strength must become a different coefficient at
        # each depth; that is the entire point of the mode.
        engine = make_steerable_engine()
        engine.layer_norms.update({0: 60.0, 3: 400.0})

        handles = engine.inject_vector(
            "diyafa", strength=0.1, layers=[0, 3], strength_mode="relative"
        )

        by_layer = {h.layer: h.strength for h in handles}
        assert by_layer[0] == pytest.approx(6.0)
        assert by_layer[3] == pytest.approx(40.0)

    def test_relative_zero_is_still_a_no_op(self) -> None:
        engine = make_steerable_engine()
        engine.layer_norms[1] = 400.0

        handle = engine.inject_vector("diyafa", strength=0.0, layers=[1], strength_mode="relative")[
            0
        ]

        assert handle.strength == pytest.approx(0.0)

    def test_unknown_mode_is_rejected(self) -> None:
        engine = make_steerable_engine()

        with pytest.raises(ValueError, match="Unknown strength_mode"):
            engine.inject_vector("diyafa", strength=1.0, layers=[1], strength_mode="fractional")

    def test_steering_context_forwards_the_mode(self) -> None:
        engine = make_steerable_engine()
        engine.layer_norms[1] = 20.0

        with engine.steering(
            "diyafa", strength=0.5, layers=[1], strength_mode="relative"
        ) as handles:
            assert handles[0].strength == pytest.approx(10.0)

    def test_evaluate_steering_forwards_the_mode(self) -> None:
        engine = make_steerable_engine()
        engine.layer_norms[1] = 20.0

        results = evaluate_steering(
            engine,
            "diyafa",
            PROMPTS,
            strengths=[0.5],
            layers=[1],
            generate=False,
            strength_mode="relative",
        )

        # The fake's loss rises with the attached steering magnitude, so a
        # 10x-scaled coefficient must cost more than the raw 0.5 would.
        absolute = evaluate_steering(
            engine, "diyafa", PROMPTS, strengths=[0.5], layers=[1], generate=False
        )
        assert results[0.5].mean_loss > absolute[0.5].mean_loss

    def test_layer_set_grid_defaults_to_relative(self) -> None:
        # Comparing layer configurations on an absolute grid is not a fair
        # comparison, so this entry point flips the default.
        import inspect

        signature = inspect.signature(evaluate_layer_sets)
        assert signature.parameters["strength_mode"].default == "relative"


class TestCalibrateLayerNorms:
    """Measuring the residual norm the relative mode divides by."""

    def test_calibration_recovers_the_known_norms(self) -> None:
        # The fake's activations have norm 10*(layer+1) by construction.
        engine = make_steerable_engine()

        norms = engine.calibrate_layer_norms(prompts=["a prompt"], layers=[0, 2])

        assert norms[0] == pytest.approx(10.0)
        assert norms[2] == pytest.approx(30.0)

    def test_calibration_covers_every_layer_by_default(self) -> None:
        engine = make_steerable_engine()

        norms = engine.calibrate_layer_norms(prompts=["a prompt"])

        assert set(norms) == set(range(4))

    def test_results_are_cached_on_the_engine(self) -> None:
        engine = make_steerable_engine()

        engine.calibrate_layer_norms(prompts=["a prompt"], layers=[1])

        assert engine.layer_norms[1] == pytest.approx(20.0)

    def test_relative_injection_calibrates_on_demand(self) -> None:
        # An uncalibrated layer must not silently fall back to absolute.
        engine = make_steerable_engine()
        assert engine.layer_norms == {}

        handle = engine.inject_vector("diyafa", strength=0.1, layers=[3], strength_mode="relative")[
            0
        ]

        assert engine.layer_norms[3] == pytest.approx(40.0)
        assert handle.strength == pytest.approx(4.0)

    def test_default_prompts_come_from_the_dataset(self) -> None:
        engine = make_steerable_engine()

        norms = engine.calibrate_layer_norms(layers=[0])

        assert norms[0] == pytest.approx(10.0)

    def test_empty_prompt_list_is_rejected(self) -> None:
        engine = make_steerable_engine()

        with pytest.raises(ValueError, match="No calibration prompts"):
            engine.calibrate_layer_norms(prompts=[], layers=[0])
