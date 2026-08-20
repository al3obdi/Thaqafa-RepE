"""Linear probes for locating where a concept becomes readable.

Phase 2 extracts a concept vector at the middle layer because mid-stack is where
semantic features *tend* to be most linearly separable. That is a prior, not a
measurement. A linear probe turns it into one: train a logistic regression to
tell concept prompts from neutral prompts using nothing but the residual stream
at layer :math:`\\ell`, and its held-out accuracy says whether the concept is
linearly readable there at all.

Two things make this easy to get wrong on a dataset this small.

**Training accuracy is meaningless.** With a handful of prompts and a
``d_model``-dimensional input, a linear classifier can separate almost any
labelling perfectly - including random labels. Every accuracy reported here is
therefore cross-validated, and :func:`chance_accuracy` gives the majority-class
floor to compare against. A layer scoring at chance is not evidence of absence;
it is evidence that the probe had nothing to learn from.

**Scale differs across layers.** Residual stream norms grow with depth, so an
unstandardised probe would find later layers "easier" for reasons that have
nothing to do with the concept. Features are standardised before fitting.

**Classes are rarely the same size.** A concept with twelve exemplars and eight
curated contrasts gives raw accuracy a floor of 0.6, and a probe can reach it by
answering "positive" every time. Everything here therefore reports *balanced*
accuracy - the mean of the per-class recalls - whose chance level is exactly 0.5
for any class ratio, and fits with ``class_weight="balanced"`` so the classifier
is not rewarded for the same shortcut. The majority-class rate is still reported
alongside, as a description of the design rather than as the bar to clear.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from src.models.rep_engine import CulturalRepE

logger = logging.getLogger(__name__)

DEFAULT_C = 1.0
"""Inverse regularisation strength. Low by default: the datasets are small."""

DEFAULT_MAX_ITER = 2000
DEFAULT_SEED = 0
DEFAULT_N_SPLITS = 5
DEFAULT_SCORING = "balanced_accuracy"
"""Scoring rule for cross-validation. See the module docstring."""

CHANCE_BALANCED = 0.5
"""Chance level for balanced accuracy, whatever the class ratio."""

ArrayLike = "np.ndarray | torch.Tensor | list[list[float]]"


def _as_numpy(values: Any) -> np.ndarray:
    """Convert activations or labels into a plain float/int numpy array.

    Args:
        values: A torch tensor, numpy array or nested sequence.

    Returns:
        The same data as a numpy array, detached from any autograd graph.
    """
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def chance_accuracy(labels: Any) -> float:
    """Return the majority-class accuracy for ``labels``.

    This is the floor a probe has to beat to mean anything. For a balanced
    two-class split it is 0.5; for 8 positives and 2 negatives it is 0.8, and a
    probe scoring 0.75 there is doing *worse* than always guessing "positive".

    Args:
        labels: Class labels.

    Returns:
        The proportion of the most common label.

    Raises:
        ValueError: If ``labels`` is empty.
    """
    flat = _as_numpy(labels).ravel().tolist()
    if not flat:
        raise ValueError("labels must contain at least one entry")

    counts = Counter(flat)
    return counts.most_common(1)[0][1] / len(flat)


@dataclass
class ProbeResult:
    """Cross-validated performance of a probe at one layer.

    Attributes:
        layer: The layer the probe was trained on.
        accuracy: Mean score across cross-validation folds, under
            :attr:`metric`. With the default metric this is balanced accuracy,
            the mean of the per-class recalls.
        std: Standard deviation across folds. Large values on a small dataset
            mean the estimate is not trustworthy.
        chance: The floor to compare :attr:`accuracy` against. For balanced
            accuracy this is 0.5 whatever the class ratio.
        n_samples: How many prompts the probe saw in total.
        fold_scores: The per-fold scores behind :attr:`accuracy`.
        majority_class_rate: What a classifier answering with the larger class
            every time would score under *raw* accuracy. Reported as a
            description of how balanced the design is, not as a bar to clear -
            balanced accuracy already discounts that strategy to 0.5.
        metric: Name of the scoring rule, so a stored result cannot be read
            under the wrong one.
    """

    layer: int
    accuracy: float
    std: float
    chance: float
    n_samples: int
    fold_scores: list[float] = field(default_factory=list)
    majority_class_rate: float = CHANCE_BALANCED
    metric: str = DEFAULT_SCORING

    @property
    def lift_over_chance(self) -> float:
        """How far above the metric's chance level the probe scored."""
        return self.accuracy - self.chance


class LinearProbe:
    """A standardised logistic regression over residual stream activations.

    Args:
        C: Inverse regularisation strength. Smaller values regularise harder,
            which matters when there are more dimensions than prompts.
        max_iter: Maximum solver iterations.
        seed: Random seed, for reproducible fits and folds.

    Attributes:
        pipeline: The fitted scikit-learn pipeline, or ``None`` before
            :meth:`fit`.

    Example:
        >>> probe = LinearProbe()
        >>> activations = [[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0]]
        >>> _ = probe.fit(activations, [0, 0, 1, 1])
        >>> probe.score(activations, [0, 0, 1, 1])
        1.0
    """

    def __init__(
        self,
        C: float = DEFAULT_C,  # noqa: N803 - matches the scikit-learn spelling
        max_iter: int = DEFAULT_MAX_ITER,
        seed: int = DEFAULT_SEED,
    ) -> None:
        if C <= 0:
            raise ValueError(f"C must be positive, got {C}")

        self.C = C
        self.max_iter = max_iter
        self.seed = seed
        self.pipeline: Pipeline | None = None

    def _build_pipeline(self) -> Pipeline:
        """Return an unfitted standardise-then-classify pipeline.

        Returns:
            A scikit-learn pipeline. Standardisation is part of the pipeline
            rather than a preprocessing step so that cross-validation refits it
            per fold, instead of leaking test-fold statistics into training.
        """
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classify",
                    LogisticRegression(
                        C=self.C,
                        max_iter=self.max_iter,
                        random_state=self.seed,
                        # Pairs with balanced-accuracy scoring: without it the
                        # fit itself is pulled toward the larger class, and the
                        # metric would only be measuring that pull.
                        class_weight="balanced",
                    ),
                ),
            ]
        )

    def fit(self, activations: Any, labels: Any) -> LinearProbe:
        """Fit the probe on ``activations``.

        Args:
            activations: Shape ``(n_samples, d_model)``, as a tensor, array or
                nested sequence.
            labels: Class label per sample, shape ``(n_samples,)``.

        Returns:
            The fitted probe, for chaining.

        Raises:
            ValueError: If the shapes disagree, the activations are not 2-D, or
                fewer than two classes are present.
        """
        features, targets = self._validate(activations, labels)

        self.pipeline = self._build_pipeline()
        self.pipeline.fit(features, targets)
        return self

    def score(self, activations: Any, labels: Any) -> float:
        """Return balanced accuracy on ``activations``.

        Balanced rather than raw, for the reason in the module docstring: on an
        uneven split, raw accuracy rewards answering with the larger class.

        Scoring on the data the probe was fitted to reports training
        performance, which on a small dataset is close to meaningless. Prefer
        :meth:`cross_val_accuracy`.

        Args:
            activations: Shape ``(n_samples, d_model)``.
            labels: Class label per sample.

        Returns:
            Balanced accuracy in ``[0, 1]``.

        Raises:
            RuntimeError: If the probe has not been fitted.
            ValueError: If the inputs are malformed.
        """
        if self.pipeline is None:
            raise RuntimeError("Probe is not fitted. Call fit() first.")

        features, targets = self._validate(activations, labels, require_two_classes=False)
        return float(balanced_accuracy_score(targets, self.pipeline.predict(features)))

    def predict(self, activations: Any) -> np.ndarray:
        """Return the predicted label for each row of ``activations``.

        Args:
            activations: Shape ``(n_samples, d_model)``.

        Returns:
            Predicted labels, shape ``(n_samples,)``.

        Raises:
            RuntimeError: If the probe has not been fitted.
        """
        if self.pipeline is None:
            raise RuntimeError("Probe is not fitted. Call fit() first.")

        return np.asarray(self.pipeline.predict(_as_numpy(activations)))

    @property
    def direction(self) -> np.ndarray:
        """The probe's weight vector, in standardised feature space.

        Comparing this against the contrastive vector from Phase 2 is a useful
        cross-check: if the two point in similar directions, two independent
        methods have found the same axis.

        Returns:
            Shape ``(d_model,)`` for a binary probe.

        Raises:
            RuntimeError: If the probe has not been fitted.
        """
        if self.pipeline is None:
            raise RuntimeError("Probe is not fitted. Call fit() first.")

        classifier = self.pipeline.named_steps["classify"]
        return np.asarray(classifier.coef_).reshape(-1)

    def cross_val_accuracy(
        self,
        activations: Any,
        labels: Any,
        n_splits: int = DEFAULT_N_SPLITS,
        scoring: str = DEFAULT_SCORING,
    ) -> tuple[float, float, list[float]]:
        """Return the cross-validated score, its spread, and the fold scores.

        Folds are stratified so each holds both classes. ``n_splits`` is capped
        at the size of the smallest class, because a stratified fold cannot
        contain fewer than one sample of each; with a single sample per class,
        cross-validation is impossible and the training score is returned with
        a warning rather than an exception.

        Args:
            activations: Shape ``(n_samples, d_model)``.
            labels: Class label per sample.
            n_splits: Requested number of folds.
            scoring: A scikit-learn scoring name. The default discounts the
                answer-with-the-larger-class strategy to 0.5; switching to
                ``"accuracy"`` reintroduces the class-ratio floor, so anything
                reported under it needs that floor quoted next to it.

        Returns:
            A tuple of ``(mean_score, std_score, fold_scores)``.

        Raises:
            ValueError: If the inputs are malformed or single-class.
        """
        features, targets = self._validate(activations, labels)

        smallest_class = min(Counter(targets.tolist()).values())
        effective_splits = min(n_splits, smallest_class)

        if effective_splits < 2:
            logger.warning(
                "Only %d sample(s) in the smallest class; cross-validation is not possible. "
                "Reporting the training score, which overstates performance.",
                smallest_class,
            )
            training_score = self.fit(features, targets).score(features, targets)
            return training_score, 0.0, [training_score]

        splitter = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=self.seed)
        scores = cross_val_score(
            self._build_pipeline(), features, targets, cv=splitter, scoring=scoring
        )
        fold_scores = [float(score) for score in scores]
        return float(np.mean(fold_scores)), float(np.std(fold_scores)), fold_scores

    @staticmethod
    def _validate(
        activations: Any,
        labels: Any,
        require_two_classes: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Coerce and check a probe's inputs.

        Args:
            activations: Candidate feature matrix.
            labels: Candidate label vector.
            require_two_classes: Whether to reject single-class label sets.

        Returns:
            The validated ``(features, targets)`` pair as numpy arrays.

        Raises:
            ValueError: If the features are not 2-D, the lengths disagree, the
                inputs are empty, or a single class is present when two are
                required.
        """
        features = _as_numpy(activations).astype(np.float64)
        targets = _as_numpy(labels).ravel()

        if features.ndim != 2:
            raise ValueError(
                f"activations must be 2-D of shape (n_samples, d_model), "
                f"got shape {features.shape}"
            )
        if features.shape[0] == 0:
            raise ValueError("activations must contain at least one sample")
        if features.shape[0] != targets.shape[0]:
            raise ValueError(
                f"activations has {features.shape[0]} samples but labels has {targets.shape[0]}"
            )
        if require_two_classes and len(set(targets.tolist())) < 2:
            raise ValueError("labels must contain at least two distinct classes")

        return features, targets


def probe_layer(
    engine: CulturalRepE,
    layer: int,
    positive_prompts: list[str],
    negative_prompts: list[str],
    n_splits: int = DEFAULT_N_SPLITS,
    seed: int = DEFAULT_SEED,
    scoring: str = DEFAULT_SCORING,
) -> ProbeResult:
    """Train and cross-validate a probe on one layer.

    Args:
        engine: Engine with a loaded model.
        layer: Block to read activations from.
        positive_prompts: Prompts that express the concept.
        negative_prompts: Prompts that do not.
        n_splits: Requested cross-validation folds.
        seed: Random seed for the probe and the folds.
        scoring: scikit-learn scoring name. The result records which one was
            used and sets its chance floor to match.

    Returns:
        The cross-validated result for this layer.

    Raises:
        ValueError: If either prompt set is empty.
        IndexError: If ``layer`` is out of range.
        RuntimeError: If the model has not been loaded.
    """
    if not positive_prompts:
        raise ValueError("positive_prompts must contain at least one prompt")
    if not negative_prompts:
        raise ValueError("negative_prompts must contain at least one prompt")

    positive = engine.collect_activations(positive_prompts, layer)
    negative = engine.collect_activations(negative_prompts, layer)

    features = torch.cat([positive, negative], dim=0)
    labels = [1] * len(positive_prompts) + [0] * len(negative_prompts)

    probe = LinearProbe(seed=seed)
    accuracy, std, fold_scores = probe.cross_val_accuracy(
        features, labels, n_splits=n_splits, scoring=scoring
    )

    return ProbeResult(
        layer=engine._resolve_layer(layer),
        accuracy=accuracy,
        std=std,
        chance=CHANCE_BALANCED if scoring == DEFAULT_SCORING else chance_accuracy(labels),
        n_samples=len(labels),
        fold_scores=fold_scores,
        majority_class_rate=chance_accuracy(labels),
        metric=scoring,
    )


def sweep_layers_with_probe(
    engine: CulturalRepE,
    concept: str,
    positive_prompts: list[str] | None = None,
    negative_prompts: list[str] | None = None,
    layers: list[int] | None = None,
    n_splits: int = DEFAULT_N_SPLITS,
    seed: int = DEFAULT_SEED,
    scoring: str = DEFAULT_SCORING,
) -> dict[int, ProbeResult]:
    """Probe every layer and report where the concept is linearly readable.

    This is the empirical replacement for defaulting to the middle layer: run
    it once per model, then extract and inject at the layer that actually scores
    highest rather than at the one theory suggests.

    Args:
        engine: Engine with a loaded model.
        concept: Concept identifier. Used to resolve prompts from the dataset
            when ``positive_prompts`` is omitted.
        positive_prompts: Prompts expressing the concept. Omit to load the
            concept's examples from :attr:`~CulturalRepE.dataset_path`.
        negative_prompts: Prompts that do not express it. Omit to use the
            deterministic neutral bank, balanced to the positive count.
        layers: Layers to probe. Defaults to every block in the model.
        n_splits: Requested cross-validation folds.
        seed: Random seed for the probes and the folds.
        scoring: scikit-learn scoring name, applied at every layer so the
            comparison across layers is like for like.

    Returns:
        A mapping from layer index to :class:`ProbeResult`, ordered by layer.

    Raises:
        ValueError: If the concept cannot be resolved or a prompt set is empty.
        RuntimeError: If the model has not been loaded.
    """
    from src.data.contrastive import build_contrast_examples

    engine._require_model()

    positives, curated_contrasts = engine._resolve_examples(concept, positive_prompts)
    # Same negative-side priority as extraction: explicit argument, then the
    # dataset entry's curated minimal pairs, then the generated neutral bank.
    negatives = build_contrast_examples(positives, negative_prompts or curated_contrasts)
    target_layers = list(range(engine.n_layers)) if layers is None else sorted(set(layers))

    logger.info(
        "Probing %d layer(s) for %s with %d positive and %d negative prompts",
        len(target_layers),
        concept,
        len(positives),
        len(negatives),
    )

    results: dict[int, ProbeResult] = {}
    for layer in target_layers:
        result = probe_layer(
            engine,
            layer=layer,
            positive_prompts=positives,
            negative_prompts=negatives,
            n_splits=n_splits,
            seed=seed,
            scoring=scoring,
        )
        results[result.layer] = result
        logger.info(
            "layer %2d: %.3f +/- %.3f (chance %.3f)",
            result.layer,
            result.accuracy,
            result.std,
            result.chance,
        )

    return results


def best_layer(results: dict[int, ProbeResult]) -> int:
    """Return the layer with the highest probe accuracy.

    Ties are broken toward the earlier layer: if two layers read the concept
    equally well, the earlier one leaves more of the network downstream for a
    steering intervention to act through.

    Args:
        results: Output of :func:`sweep_layers_with_probe`.

    Returns:
        The best-scoring layer index.

    Raises:
        ValueError: If ``results`` is empty.
    """
    if not results:
        raise ValueError("results must contain at least one layer")

    return min(results, key=lambda layer: (-results[layer].accuracy, layer))


def summarize_probe_sweep(results: dict[int, ProbeResult]) -> dict[str, list[float]]:
    """Reshape a probe sweep into parallel lists for plotting.

    Args:
        results: Output of :func:`sweep_layers_with_probe`.

    Returns:
        A mapping with ``"layers"``, ``"accuracies"``, ``"stds"`` and
        ``"chance"``, each sorted by ascending layer.
    """
    ordered = sorted(results)
    return {
        "layers": [float(layer) for layer in ordered],
        "accuracies": [results[layer].accuracy for layer in ordered],
        "stds": [results[layer].std for layer in ordered],
        "chance": [results[layer].chance for layer in ordered],
    }
