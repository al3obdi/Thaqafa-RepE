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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.probes import (
    DEFAULT_N_PERMUTATIONS,
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

        result = probe_layer(
            engine, min(INFORMATIVE_LAYERS), positives, negatives, n_permutations=0
        )

        assert result.accuracy == pytest.approx(1.0)
        assert result.chance == pytest.approx(0.5)
        assert result.n_samples == 16
        assert result.lift_over_chance == pytest.approx(0.5)

    def test_uninformative_layer_stays_at_chance(self) -> None:
        engine = make_marker_engine()
        positives, negatives = marked_prompts()

        result = probe_layer(engine, 0, positives, negatives, n_permutations=0)

        assert result.accuracy <= 0.6

    def test_reads_the_requested_layer(self) -> None:
        model = MarkerTransformer()
        engine = make_marker_engine(model)
        positives, negatives = marked_prompts(count=4)

        probe_layer(engine, 4, positives, negatives, n_permutations=0)

        assert set(model.probed_layers) == {4}

    def test_empty_positive_set_is_rejected(self) -> None:
        engine = make_marker_engine()

        with pytest.raises(ValueError, match="positive_prompts"):
            probe_layer(engine, 0, [], ["a"], n_permutations=0)

    def test_empty_negative_set_is_rejected(self) -> None:
        engine = make_marker_engine()

        with pytest.raises(ValueError, match="negative_prompts"):
            probe_layer(engine, 0, ["a"], [], n_permutations=0)


class TestSweepLayersWithProbe:
    """Sweeping every layer to find where the concept lives."""

    def test_sweep_covers_every_layer_by_default(self) -> None:
        engine = make_marker_engine()
        positives, negatives = marked_prompts()

        results = sweep_layers_with_probe(engine, "diyafa", positives, negatives, n_permutations=0)

        assert set(results) == set(range(MARKER_N_LAYERS))
        assert all(isinstance(result, ProbeResult) for result in results.values())

    def test_sweep_finds_the_informative_layers(self) -> None:
        engine = make_marker_engine()
        positives, negatives = marked_prompts()

        results = sweep_layers_with_probe(engine, "diyafa", positives, negatives, n_permutations=0)

        for layer in INFORMATIVE_LAYERS:
            assert results[layer].accuracy == pytest.approx(1.0), f"layer {layer}"
        for layer in set(range(MARKER_N_LAYERS)) - INFORMATIVE_LAYERS:
            assert results[layer].accuracy <= 0.6, f"layer {layer}"

    def test_a_subset_of_layers_can_be_requested(self) -> None:
        engine = make_marker_engine()
        positives, negatives = marked_prompts(count=4)

        results = sweep_layers_with_probe(
            engine, "diyafa", positives, negatives, layers=[0, 3], n_permutations=0
        )

        assert set(results) == {0, 3}

    def test_prompts_default_to_the_dataset_and_curated_contrasts(self) -> None:
        engine = make_marker_engine()

        results = sweep_layers_with_probe(engine, "diyafa_001", layers=[0], n_permutations=0)

        from src.data.dataset_builder import load_concepts

        entry = next(c for c in load_concepts(engine.dataset_path) if c.concept_id == "diyafa_001")
        # Positives come from the entry's exemplars; negatives from its curated
        # minimal-pair contrasts (preferred over the generated neutral bank).
        assert entry.all_contrasts, "diyafa_001 should carry curated contrasts"
        assert results[0].n_samples == len(entry.all_examples) + len(entry.all_contrasts)

    def test_unknown_concept_is_rejected(self) -> None:
        engine = make_marker_engine()

        with pytest.raises(ValueError, match="was not found"):
            sweep_layers_with_probe(engine, "not_a_concept_999", layers=[0], n_permutations=0)


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
        results = sweep_layers_with_probe(
            engine, "diyafa", positives, negatives, layers=[3, 0, 1], n_permutations=0
        )

        summary = summarize_probe_sweep(results)

        assert summary["layers"] == [0.0, 1.0, 3.0]
        assert len(summary["accuracies"]) == 3
        assert len(summary["stds"]) == 3
        assert summary["chance"] == [0.5, 0.5, 0.5]


def uninformative_imbalanced_data(
    n_positive: int = 12,
    n_negative: int = 8,
) -> tuple[np.ndarray, list[int]]:
    """Build an uneven split whose features carry no class signal at all.

    Both classes cycle through the same four feature vectors, so every value
    that appears in one class also appears in the other and nothing separates
    them. Under raw accuracy a probe can still reach ``n_positive / n_total``
    by answering "positive" every time; under balanced accuracy it cannot do
    better than 0.5. That gap is exactly what these tests are about.

    Args:
        n_positive: Samples in the larger class.
        n_negative: Samples in the smaller class.

    Returns:
        A ``(features, labels)`` pair.
    """
    block = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [0.25, 0.75]])
    rows = [block[index % len(block)] for index in range(n_positive + n_negative)]
    features = np.vstack(rows)
    labels = [1] * n_positive + [0] * n_negative
    return features, labels


class TestBalancedMetric:
    """The reported number must not be reachable by guessing the larger class."""

    def test_uninformative_imbalanced_data_scores_at_one_half(self) -> None:
        """Raw accuracy would report 0.6 here for a probe that learned nothing."""
        features, labels = uninformative_imbalanced_data()
        probe = LinearProbe(seed=0)

        accuracy, _, _ = probe.cross_val_accuracy(features, labels)

        assert accuracy == pytest.approx(0.5, abs=0.2)
        assert accuracy < chance_accuracy(labels)

    def test_separable_data_still_scores_perfectly(self) -> None:
        """The fix must not cost anything where there is real signal."""
        features, labels = separable_data()
        accuracy, _, _ = LinearProbe(seed=0).cross_val_accuracy(features, labels)
        assert accuracy == pytest.approx(1.0)

    def test_an_unweighted_probe_would_have_taken_the_shortcut(self) -> None:
        """Shows the behaviour the fix removes, so the fix cannot be undone quietly.

        An unweighted logistic regression on this data learns to answer with the
        larger class, which raw accuracy rewards with the class ratio. Both
        halves of the fix matter: the weighting stops the classifier taking the
        shortcut, and the metric stops it paying.
        """
        features, labels = uninformative_imbalanced_data()
        unweighted = Pipeline(
            [("scale", StandardScaler()), ("classify", LogisticRegression(random_state=0))]
        ).fit(features, labels)

        assert set(unweighted.predict(features).tolist()) == {1}
        assert unweighted.score(features, labels) == pytest.approx(chance_accuracy(labels))

        weighted, _, _ = LinearProbe(seed=0).cross_val_accuracy(features, labels)
        assert weighted < chance_accuracy(labels)

    def test_score_is_balanced_too(self) -> None:
        """A training score under a different metric would not be comparable."""
        features, labels = uninformative_imbalanced_data()
        probe = LinearProbe(seed=0).fit(features, labels)
        assert probe.score(features, labels) == pytest.approx(0.5, abs=0.2)

    def test_classifier_is_class_weighted(self) -> None:
        """Scoring alone is not enough; the fit must not chase the larger class."""
        probe = LinearProbe(seed=0).fit(*separable_data())
        assert probe.pipeline is not None
        assert probe.pipeline.named_steps["classify"].class_weight == "balanced"


class TestProbeResultMetadata:
    """A stored result must say which metric produced it."""

    def test_chance_is_one_half_under_the_default_metric(self) -> None:
        """Balanced accuracy has a 0.5 floor whatever the class ratio."""
        engine = make_marker_engine()
        positives, negatives = marked_prompts(6)

        result = probe_layer(engine, 0, positives, negatives[:4], n_permutations=0)

        assert result.chance == 0.5
        assert result.metric == "balanced_accuracy"

    def test_majority_class_rate_describes_the_design(self) -> None:
        """The imbalance is still reported, as a description not a bar."""
        engine = make_marker_engine()
        positives, negatives = marked_prompts(6)

        result = probe_layer(engine, 0, positives, negatives[:4], n_permutations=0)

        assert result.majority_class_rate == pytest.approx(0.6)

    def test_raw_accuracy_restores_the_majority_class_floor(self) -> None:
        """Under raw accuracy the floor is the class ratio, and must say so."""
        engine = make_marker_engine()
        positives, negatives = marked_prompts(6)

        result = probe_layer(
            engine, 0, positives, negatives[:4], scoring="accuracy", n_permutations=0
        )

        assert result.chance == pytest.approx(0.6)
        assert result.metric == "accuracy"

    def test_lift_is_measured_against_the_matching_floor(self) -> None:
        """Comparing a balanced score to a raw floor would understate every lift."""
        engine = make_marker_engine()
        positives, negatives = marked_prompts(6)

        result = probe_layer(
            engine, next(iter(INFORMATIVE_LAYERS)), positives, negatives[:4], n_permutations=0
        )

        assert result.lift_over_chance == pytest.approx(result.accuracy - 0.5)

    def test_sweep_applies_one_metric_to_every_layer(self) -> None:
        """A metric that varied by layer would make the sweep incomparable."""
        engine = make_marker_engine()
        results = sweep_layers_with_probe(engine, "wasta_001", scoring="accuracy", n_permutations=0)
        assert {r.metric for r in results.values()} == {"accuracy"}
        assert len(results) == MARKER_N_LAYERS


class TestPermutationTest:
    """A high score on twenty prompts needs more than a chance floor to read."""

    def test_real_structure_is_unlikely_under_shuffled_labels(self) -> None:
        """Separable data must come out significant."""
        features, labels = separable_data()
        p_value = LinearProbe(seed=0).permutation_p_value(features, labels, n_permutations=50)
        assert p_value is not None
        assert p_value < 0.05

    def test_no_structure_is_not_significant(self) -> None:
        """Uninformative data must not sneak past as a finding."""
        features, labels = uninformative_imbalanced_data()
        p_value = LinearProbe(seed=0).permutation_p_value(features, labels, n_permutations=50)
        assert p_value is not None
        assert p_value > 0.05

    def test_the_floor_is_one_over_permutations_plus_one(self) -> None:
        """So a reader can tell a strong result from the limit of the test."""
        features, labels = separable_data()
        p_value = LinearProbe(seed=0).permutation_p_value(features, labels, n_permutations=20)
        assert p_value == pytest.approx(1 / 21)

    def test_too_few_samples_gives_no_p_value_rather_than_a_wrong_one(self) -> None:
        """With one sample in a class there is nothing to cross-validate."""
        features = np.array([[0.0], [10.0], [10.1]])
        assert LinearProbe(seed=0).permutation_p_value(features, [1, 0, 0]) is None

    def test_non_positive_permutation_count_is_rejected(self) -> None:
        features, labels = separable_data()
        with pytest.raises(ValueError, match="n_permutations must be positive"):
            LinearProbe().permutation_p_value(features, labels, n_permutations=0)

    def test_probe_layer_records_the_p_value_and_its_resolution(self) -> None:
        """A p-value without its permutation count cannot be interpreted."""
        engine = make_marker_engine()
        positives, negatives = marked_prompts(6)

        result = probe_layer(
            engine, min(INFORMATIVE_LAYERS), positives, negatives, n_permutations=30
        )

        assert result.p_value is not None
        assert result.p_value < 0.05
        assert result.n_permutations == 30

    def test_skipping_the_test_leaves_no_p_value_behind(self) -> None:
        """Absent must read as absent, never as "not significant"."""
        engine = make_marker_engine()
        positives, negatives = marked_prompts(6)

        result = probe_layer(engine, 0, positives, negatives, n_permutations=0)

        assert result.p_value is None
        assert result.n_permutations == 0

    def test_the_default_is_large_enough_to_reach_one_percent(self) -> None:
        """A floor above 0.01 could not support any claim of significance."""
        assert 1 / (DEFAULT_N_PERMUTATIONS + 1) < 0.01
