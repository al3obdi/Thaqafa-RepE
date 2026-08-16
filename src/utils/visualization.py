"""Plotting helpers for concept vectors and steering experiments.

All functions return a Matplotlib :class:`~matplotlib.axes.Axes` so that
callers keep full control over figure size, saving and composition inside
notebooks.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from matplotlib.axes import Axes

logger = logging.getLogger(__name__)

DEFAULT_FIGSIZE: tuple[float, float] = (8.0, 5.0)


def plot_layer_scores(
    scores: dict[int, float],
    title: str = "Concept separability per layer",
    ax: Axes | None = None,
) -> Axes:
    """Plot how well a concept is linearly separable at each layer.

    Args:
        scores: Mapping from layer index to a separability score, typically the
            accuracy of a linear probe trained on that layer.
        title: Plot title.
        ax: Existing axes to draw on. A new figure is created when omitted.

    Returns:
        The axes the plot was drawn on.

    Raises:
        ValueError: If ``scores`` is empty.
    """
    if not scores:
        raise ValueError("scores must contain at least one layer")

    if ax is None:
        _, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    layers = sorted(scores)
    values = [scores[layer] for layer in layers]

    ax.plot(layers, values, marker="o")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return ax


def plot_concept_similarity(
    similarity: np.ndarray,
    labels: list[str],
    title: str = "Concept vector cosine similarity",
    ax: Axes | None = None,
) -> Axes:
    """Draw a heatmap of pairwise cosine similarities between concept vectors.

    Args:
        similarity: Square matrix of shape ``(n_concepts, n_concepts)``.
        labels: Concept names, one per row of ``similarity``.
        title: Plot title.
        ax: Existing axes to draw on. A new figure is created when omitted.

    Returns:
        The axes the plot was drawn on.

    Raises:
        ValueError: If ``similarity`` is not square or does not match ``labels``.
    """
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError("similarity must be a square 2-D matrix")
    if len(labels) != similarity.shape[0]:
        raise ValueError("labels must contain one entry per row of similarity")

    if ax is None:
        _, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    image = ax.imshow(similarity, cmap="viridis", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title(title)
    ax.figure.colorbar(image, ax=ax, label="Cosine similarity")
    return ax


def plot_steering_sweep(
    strengths: list[float],
    metric_values: list[float],
    metric_name: str = "Cultural alignment score",
    ax: Axes | None = None,
) -> Axes:
    """Plot an evaluation metric against the injection strength.

    Args:
        strengths: Injection coefficients that were swept over.
        metric_values: Metric value measured at each strength.
        metric_name: Label for the y axis.
        ax: Existing axes to draw on. A new figure is created when omitted.

    Returns:
        The axes the plot was drawn on.

    Raises:
        ValueError: If the two input sequences have different lengths.
    """
    if len(strengths) != len(metric_values):
        raise ValueError("strengths and metric_values must have the same length")

    if ax is None:
        _, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    ax.plot(strengths, metric_values, marker="s")
    ax.axvline(0.0, color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel("Injection strength")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} vs. injection strength")
    ax.grid(True, alpha=0.3)
    return ax
