"""Tests for the plotting helpers.

Matplotlib runs on the Agg backend, so these tests exercise the full drawing
path without a display. Assertions target the drawn content (line data, tick
labels, axis titles), not just "no exception".
"""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.utils.visualization import (
    plot_concept_similarity,
    plot_layer_scores,
    plot_steering_sweep,
)


@pytest.fixture(autouse=True)
def close_figures() -> Iterator[None]:
    """Close every figure after each test to keep memory flat."""
    yield
    plt.close("all")


class TestPlotLayerScores:
    """The per-layer separability plot."""

    def test_plots_layers_in_sorted_order(self) -> None:
        scores = {8: 0.9, 0: 0.5, 4: 0.7}

        ax = plot_layer_scores(scores)

        line = ax.lines[0]
        assert np.asarray(line.get_xdata()).tolist() == [0, 4, 8]
        assert np.asarray(line.get_ydata()).tolist() == [0.5, 0.7, 0.9]

    def test_labels_and_title(self) -> None:
        ax = plot_layer_scores({0: 0.5}, title="Sweep title")

        assert ax.get_xlabel() == "Layer"
        assert ax.get_ylabel() == "Score"
        assert ax.get_title() == "Sweep title"

    def test_draws_on_a_provided_axes(self) -> None:
        _, ax = plt.subplots()

        returned = plot_layer_scores({0: 0.5, 1: 0.6}, ax=ax)

        assert returned is ax
        assert len(ax.lines) == 1

    def test_empty_scores_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one layer"):
            plot_layer_scores({})


class TestPlotConceptSimilarity:
    """The pairwise cosine-similarity heatmap."""

    def test_heatmap_carries_the_matrix_and_labels(self) -> None:
        similarity = np.array([[1.0, 0.2], [0.2, 1.0]])

        ax = plot_concept_similarity(similarity, ["wasta", "diyafa"])

        image = ax.images[0]
        assert np.allclose(np.asarray(image.get_array()), similarity)
        assert [t.get_text() for t in ax.get_xticklabels()] == ["wasta", "diyafa"]
        assert [t.get_text() for t in ax.get_yticklabels()] == ["wasta", "diyafa"]

    def test_color_scale_is_fixed_to_cosine_range(self) -> None:
        ax = plot_concept_similarity(np.eye(2), ["a", "b"])

        assert ax.images[0].get_clim() == (-1.0, 1.0)

    def test_non_square_matrix_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="square"):
            plot_concept_similarity(np.zeros((2, 3)), ["a", "b"])

    def test_label_count_must_match(self) -> None:
        with pytest.raises(ValueError, match="one entry per row"):
            plot_concept_similarity(np.eye(3), ["a", "b"])


class TestPlotSteeringSweep:
    """The strength-versus-metric plot."""

    def test_plots_the_given_series(self) -> None:
        strengths = [-1.0, 0.0, 1.0]
        values = [2.2, 2.0, 2.1]

        ax = plot_steering_sweep(strengths, values, metric_name="Loss")

        line = ax.lines[0]
        assert np.asarray(line.get_xdata()).tolist() == strengths
        assert np.asarray(line.get_ydata()).tolist() == values
        assert ax.get_ylabel() == "Loss"

    def test_marks_the_unsteered_point_with_a_vertical_line(self) -> None:
        ax = plot_steering_sweep([-1.0, 1.0], [2.0, 2.1])

        vlines = [ln for ln in ax.lines if np.asarray(ln.get_xdata()).tolist() == [0.0, 0.0]]
        assert vlines, "expected a vertical reference line at strength 0"

    def test_mismatched_lengths_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            plot_steering_sweep([1.0, 2.0], [0.5])
