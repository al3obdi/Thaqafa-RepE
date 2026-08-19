"""Tests for the linear probe utilities.

Everything here runs on CPU with no model weights. The probe itself is tested
against synthetic activations where the right answer is known by construction -
perfectly separable data must score 1.0, and label-matched data must score at
chance. The layer sweep is tested against :class:`MarkerTransformer`, whose
activations carry the class signal at two known layers and nowhere else, so the
sweep has a correct answer to find.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.utils.probes import (
    LinearProbe,
    ProbeResult,
    best_layer,
    chance_accuracy,
    probe_layer,
    summarize_probe_sweep,
    sweep_layers_with_probe,
)
from tests.helpers import (
    INFORMATIVE_LAYERS,
    MARKER_N_LAYERS,
    MarkerTransformer,
    make_marker_engine,
    marked_prompts,
)


def separable_data(n_per_class: int = 8) -> tuple[np.ndarray, list[int]]:
    """Build two clearly separated clusters.

    Args:
        n_per_class: Samples per class.

    Returns:
        A ``(features, labels)`` pair.
    """
    negatives = np.column_stack(
        [np.linspace(0.0, 1.0, n_per_class), np.linspace(0.0, 0.5, n_per_class)]
    )
    positives = negatives + 10.0
    features = np.vstack([positives, negatives])
    labels = [1] * n_per_class + [0] * n_per_class
    return features, labels


class TestChanceAccuracy:
    """The majority-class floor."""

    def test_balanced_labels_give_one_half(self) -> None:
        assert chance_accuracy([0, 0, 1, 1]) == pytest.approx(0.5)

    def test_imbalanced_labels_give_the_majority_share(self) -> None:
        # Eight positives to two negatives: always guessing "positive" scores 0.8,
        # so a probe reporting 0.75 is doing worse than nothing.
        assert chance_accuracy([1] * 8 + [0] * 2) == pytest.approx(0.8)

    def test_accepts_a_tensor(self) -> None:
        assert chance_accuracy(torch.tensor([0, 1, 1])) == pytest.approx(2 / 3)

    def test_empty_labels_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one entry"):
            chance_accuracy([])


class TestLinearProbeFit:
    """Fitting and scoring."""

    def test_perfectly_separable_data_scores_one(self) -> None:
        features, labels = separable_data()

        probe = LinearProbe().fit(features, labels)

        assert probe.score(features, labels) == pytest.approx(1.0)

    def test_fit_returns_self_for_chaining(self) -> None:
        features, labels = separable_data()
        probe = LinearProbe()

        assert probe.fit(features, labels) is probe

    def test_accepts_torch_tensors(self) -> None:
        features, labels = separable_data()
        probe = LinearProbe().fit(torch.tensor(features), torch.tensor(labels))

        assert probe.score(torch.tensor(features), torch.tensor(labels)) == pytest.approx(1.0)

    def test_predict_recovers_the_labels(self) -> None:
        features, labels = separable_data()
        probe = LinearProbe().fit(features, labels)

        assert probe.predict(features).tolist() == labels

    def test_direction_has_one_weight_per_feature(self) -> None:
        features, labels = separable_data()
        probe = LinearProbe().fit(features, labels)

        assert probe.direction.shape == (features.shape[1],)

    def test_scoring_before_fitting_is_rejected(self) -> None:
        features, labels = separable_data()

        with pytest.raises(RuntimeError, match="not fitted"):
            LinearProbe().score(features, labels)

    def test_predicting_before_fitting_is_rejected(self) -> None:
        with pytest.raises(RuntimeError, match="not fitted"):
            LinearProbe().predict([[0.0, 1.0]])

    def test_direction_before_fitting_is_rejected(self) -> None:
        with pytest.raises(RuntimeError, match="not fitted"):
            _ = LinearProbe().direction


class TestLinearProbeValidation:
    """Input checking."""

    def test_non_positive_regularisation_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="C must be positive"):
            LinearProbe(C=0.0)

    def test_one_dimensional_activations_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be 2-D"):
            LinearProbe().fit([1.0, 2.0, 3.0], [0, 1, 0])

    def test_mismatched_lengths_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="samples but labels"):
            LinearProbe().fit([[0.0], [1.0]], [0, 1, 0])

    def test_empty_activations_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one sample"):
            LinearProbe().fit(np.zeros((0, 3)), [])

    def test_single_class_labels_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="two distinct classes"):
            LinearProbe().fit([[0.0], [1.0]], [1, 1])


class TestCrossValidation:
    """Held-out scoring, which is the only accuracy worth reporting."""

    def test_separable_data_cross_validates_to_one(self) -> None:
        features, labels = separable_data()

        accuracy, std, folds = LinearProbe().cross_val_accuracy(features, labels)

        assert accuracy == pytest.approx(1.0)
        assert std == pytest.approx(0.0)
        assert len(folds) == 5

    def test_identical_features_cannot_beat_chance(self) -> None:
        # Both classes have exactly the same features, so nothing is learnable.
        # Training accuracy would still be inflated; cross-validation is not.
        features = np.tile(np.linspace(0.0, 1.0, 8).reshape(-1, 1), (2, 1))
        labels = [1] * 8 + [0] * 8

        accuracy, _, _ = LinearProbe().cross_val_accuracy(features, labels)

        assert accuracy <= 0.6

    def test_fold_count_is_capped_by_the_smallest_class(self) -> None:
        features, labels = separable_data(n_per_class=3)

        _, _, folds = LinearProbe().cross_val_accuracy(features, labels, n_splits=5)

        assert len(folds) == 3

    def test_single_sample_class_falls_back_to_training_score(self) -> None:
        features = np.array([[0.0], [10.0], [10.1]])
        labels = [1, 0, 0]

        accuracy, std, folds = LinearProbe().cross_val_accuracy(features, labels)

        assert len(folds) == 1
        assert std == pytest.approx(0.0)
        assert 0.0 <= accuracy <= 1.0

    def test_results_are_reproducible(self) -> None:
        features, labels = separable_data()

        first = LinearProbe(seed=7).cross_val_accuracy(features, labels)
        second = LinearProbe(seed=7).cross_val_accuracy(features, labels)

        assert first == second


class TestProbeLayer:
    """Probing one layer of a model."""

    def test_informative_layer_separates_the_classes(self) -> None:
        engine = make_marker_engine()
        positives, negatives = marked_prompts()

        result = probe_layer(engine, min(INFORMATIVE_LAYERS), positives, negatives)

        assert result.accuracy == pytest.approx(1.0)
        assert result.chance == pytest.approx(0.5)
        assert result.n_samples == 16
        assert result.lift_over_chance == pytest.approx(0.5)

    def test_uninformative_layer_stays_at_chance(self) -> None:
        engine = make_marker_engine()
        positives, negatives = marked_prompts()

        result = probe_layer(engine, 0, positives, negatives)

        assert result.accuracy <= 0.6

    def test_reads_the_requested_layer(self) -> None:
        model = MarkerTransformer()
        engine = make_marker_engine(model)
        positives, negatives = marked_prompts(count=4)

        probe_layer(engine, 4, positives, negatives)

        assert set(model.probed_layers) == {4}

    def test_empty_positive_set_is_rejected(self) -> None:
        engine = make_marker_engine()

        with pytest.raises(ValueError, match="positive_prompts"):
            probe_layer(engine, 0, [], ["a"])

    def test_empty_negative_set_is_rejected(self) -> None:
        engine = make_marker_engine()

        with pytest.raises(ValueError, match="negative_prompts"):
            probe_layer(engine, 0, ["a"], [])


class TestSweepLayersWithProbe:
    """Sweeping every layer to find where the concept lives."""

    def test_sweep_covers_every_layer_by_default(self) -> None:
        engine = make_marker_engine()
        positives, negatives = marked_prompts()

        results = sweep_layers_with_probe(engine, "diyafa", positives, negatives)

        assert set(results) == set(range(MARKER_N_LAYERS))
        assert all(isinstance(result, ProbeResult) for result in results.values())

    def test_sweep_finds_the_informative_layers(self) -> None:
        engine = make_marker_engine()
        positives, negatives = marked_prompts()

        results = sweep_layers_with_probe(engine, "diyafa", positives, negatives)

        for layer in INFORMATIVE_LAYERS:
            assert results[layer].accuracy == pytest.approx(1.0), f"layer {layer}"
        for layer in set(range(MARKER_N_LAYERS)) - INFORMATIVE_LAYERS:
            assert results[layer].accuracy <= 0.6, f"layer {layer}"

    def test_a_subset_of_layers_can_be_requested(self) -> None:
        engine = make_marker_engine()
        positives, negatives = marked_prompts(count=4)

        results = sweep_layers_with_probe(engine, "diyafa", positives, negatives, layers=[0, 3])

        assert set(results) == {0, 3}

    def test_prompts_default_to_the_dataset_and_curated_contrasts(self) -> None:
        engine = make_marker_engine()

        results = sweep_layers_with_probe(engine, "diyafa_001", layers=[0])

        from src.data.dataset_builder import load_concepts

        entry = next(c for c in load_concepts(engine.dataset_path) if c.concept_id == "diyafa_001")
        # Positives come from the entry's exemplars; negatives from its curated
        # minimal-pair contrasts (preferred over the generated neutral bank).
        assert entry.all_contrasts, "diyafa_001 should carry curated contrasts"
        assert results[0].n_samples == len(entry.all_examples) + len(entry.all_contrasts)

    def test_unknown_concept_is_rejected(self) -> None:
        engine = make_marker_engine()

        with pytest.raises(ValueError, match="was not found"):
            sweep_layers_with_probe(engine, "not_a_concept_999", layers=[0])


class TestBestLayer:
    """Choosing a layer from a sweep."""

    def _result(self, layer: int, accuracy: float) -> ProbeResult:
        return ProbeResult(layer=layer, accuracy=accuracy, std=0.0, chance=0.5, n_samples=10)

    def test_picks_the_highest_accuracy(self) -> None:
        results = {0: self._result(0, 0.6), 1: self._result(1, 0.9), 2: self._result(2, 0.7)}

        assert best_layer(results) == 1

    def test_ties_break_toward_the_earlier_layer(self) -> None:
        # An earlier layer leaves more of the network downstream for a steering
        # intervention to act through.
        results = {5: self._result(5, 0.9), 2: self._result(2, 0.9)}

        assert best_layer(results) == 2

    def test_empty_results_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one layer"):
            best_layer({})


class TestSummarizeProbeSweep:
    """Reshaping a sweep for plotting."""

    def test_returns_parallel_lists_sorted_by_layer(self) -> None:
        engine = make_marker_engine()
        positives, negatives = marked_prompts(count=4)
        results = sweep_layers_with_probe(engine, "diyafa", positives, negatives, layers=[3, 0, 1])

        summary = summarize_probe_sweep(results)

        assert summary["layers"] == [0.0, 1.0, 3.0]
        assert len(summary["accuracies"]) == 3
        assert len(summary["stds"]) == 3
        assert summary["chance"] == [0.5, 0.5, 0.5]
