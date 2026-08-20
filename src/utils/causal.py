"""Does steering write the concept the probe reads?

Everything measured so far about steering is a magnitude. KL divergence says
the next-token distribution moved; cross-entropy says fluency got worse. A
random vector of the same norm injected at the same layer moves the
distribution and hurts fluency too, so neither number distinguishes steering
with a concept direction from steering with noise.

This module closes the loop between the two halves of the method. A probe is
trained to *read* a concept off the residual stream at some layer. Steering
*writes* a direction into an earlier layer. If the write carries the concept,
then injecting it into otherwise neutral prompts should make the probe fire on
them - and a random direction of the same norm, injected at the same place,
should not.

The random arm is the point. Reporting only that the steered rate rose would
leave open that any perturbation large enough moves a probe trained on twenty
prompts, which on twenty prompts is entirely plausible.

Two constraints keep the measurement honest.

The injection layer must sit strictly below the read layer. Injecting and
reading at the same layer measures that addition works, since the probe would
be looking directly at the vector that was just added.

The read layer should be one where the probe can actually read the concept.
A probe scoring at chance still has a decision boundary, and pushing
activations across an arbitrary hyperplane produces a lift that means nothing.
The probe's own cross-validated accuracy therefore travels with every result,
and a lift measured through a weak probe should be discarded rather than
explained.

The injected direction is extracted at the injection layer, not carried down
from wherever the concept vector was cached. Residual bases differ between
layers, so a direction found at one layer is not the concept's direction at
another, and injecting it there would be measuring an arbitrary vector.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from src.data.contrastive import build_contrast_examples, neutral_prompt_bank
from src.utils.probes import DEFAULT_N_SPLITS, DEFAULT_SEED, LinearProbe

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from src.models.rep_engine import CulturalRepE

logger = logging.getLogger(__name__)

DEFAULT_N_RANDOM = 5
"""Random control directions per concept. Each is a full steered pass."""

DEFAULT_RELATIVE_STRENGTH = 0.2
"""Injection coefficient, as a fraction of the injection layer's residual norm."""

_CONTROL_KEY = "__causal_control__"
"""Cache key the random control vectors are parked under, then removed."""


@dataclass(frozen=True)
class ReadbackResult:
    """Whether injecting a concept makes that concept's probe fire.

    Attributes:
        concept: Concept identifier.
        inject_layer: Layer the direction was added to.
        read_layer: Layer the probe read from, strictly deeper.
        strength: Injection coefficient, in the mode named by
            :attr:`strength_mode`.
        strength_mode: ``"relative"`` or ``"absolute"``.
        baseline_rate: Share of neutral prompts the probe called positive with
            no steering at all.
        steered_rate: The same share with the concept direction injected.
        random_rates: The same share for each random control direction, matched
            in norm and injected at the same layer.
        probe_accuracy: Cross-validated balanced accuracy of the probe at
            :attr:`read_layer`. A read-back through a probe that cannot read
            the concept in the first place means nothing, so it travels with
            the result.
        n_prompts: Neutral prompts behind every rate.
    """

    concept: str
    inject_layer: int
    read_layer: int
    strength: float
    strength_mode: str
    baseline_rate: float
    steered_rate: float
    random_rates: list[float] = field(default_factory=list)
    probe_accuracy: float = 0.0
    n_prompts: int = 0

    @property
    def mean_random_rate(self) -> float:
        """Average positive rate across the random control directions."""
        if not self.random_rates:
            return self.baseline_rate
        return float(np.mean(self.random_rates))

    @property
    def lift_over_baseline(self) -> float:
        """How much steering raised the probe's positive rate."""
        return self.steered_rate - self.baseline_rate

    @property
    def lift_over_random(self) -> float:
        """How much of that survives a matched-norm random direction.

        This is the number worth reading. A lift over baseline that a random
        direction also produces is evidence about the probe's fragility, not
        about the concept.
        """
        return self.steered_rate - self.mean_random_rate


def random_directions(
    d_model: int,
    count: int,
    seed: int = DEFAULT_SEED,
) -> list[torch.Tensor]:
    """Draw unit-norm directions uniformly from the sphere.

    Gaussian coordinates normalised to unit length are uniform on the sphere,
    which is what a control needs: any other construction would privilege some
    region of the space and could accidentally land near the concept direction
    more or less often than chance.

    Args:
        d_model: Width of the residual stream.
        count: How many directions to draw.
        seed: Seed for the generator, so a control is reproducible.

    Returns:
        ``count`` tensors of shape ``(d_model,)``, each of unit L2 norm.

    Raises:
        ValueError: If ``d_model`` or ``count`` is not positive.
    """
    if d_model < 1:
        raise ValueError(f"d_model must be positive, got {d_model}")
    if count < 1:
        raise ValueError(f"count must be positive, got {count}")

    generator = torch.Generator().manual_seed(seed)
    raw = torch.randn(count, d_model, generator=generator)
    return list(raw / torch.linalg.vector_norm(raw, dim=1, keepdim=True))


def _positive_rate(probe: LinearProbe, activations: torch.Tensor) -> float:
    """Share of rows the probe assigns to the positive class.

    Args:
        probe: A fitted probe.
        activations: Shape ``(n_prompts, d_model)``.

    Returns:
        A rate in ``[0, 1]``.
    """
    predictions = probe.predict(activations)
    return float(np.mean(np.asarray(predictions) == 1))


def _rate_under(
    engine: CulturalRepE,
    direction: torch.Tensor,
    probe: LinearProbe,
    prompts: list[str],
    *,
    inject_layer: int,
    read_layer: int,
    strength: float,
    strength_mode: str,
) -> float:
    """Positive rate with *direction* injected, then cleanly removed.

    The concept arm and the random arm both go through here, so the two cannot
    differ by anything except the direction that was passed in. The direction
    is parked in the engine's cache only for the duration of the steered pass,
    and removed in a ``finally`` so a failure part-way cannot leave the model
    steered or the cache polluted.

    Args:
        engine: Engine with a loaded model.
        direction: Unit-norm direction to inject.
        probe: A fitted probe to read with.
        prompts: Texts to run through the steered model.
        inject_layer: Layer to add the direction to.
        read_layer: Layer the probe reads from.
        strength: Injection coefficient. Negative suppresses.
        strength_mode: ``"relative"`` or ``"absolute"``.

    Returns:
        The share of *prompts* the probe calls positive.
    """
    engine.concept_vectors[_CONTROL_KEY] = direction
    engine.extraction_layers[_CONTROL_KEY] = inject_layer
    try:
        with engine.steering(
            _CONTROL_KEY,
            strength=strength,
            layers=[inject_layer],
            strength_mode=strength_mode,
        ):
            return _positive_rate(probe, engine.collect_activations(prompts, read_layer))
    finally:
        engine.concept_vectors.pop(_CONTROL_KEY, None)
        engine.extraction_layers.pop(_CONTROL_KEY, None)


def readback(
    engine: CulturalRepE,
    concept: str,
    inject_layer: int,
    read_layer: int,
    neutral_prompts: list[str] | None = None,
    strength: float = DEFAULT_RELATIVE_STRENGTH,
    strength_mode: str = "relative",
    n_random: int = DEFAULT_N_RANDOM,
    seed: int = DEFAULT_SEED,
) -> ReadbackResult:
    """Inject a concept into neutral prompts and see whether its probe notices.

    Args:
        engine: Engine with a loaded model. The injected direction is derived
            here from the concept's exemplars, so nothing needs extracting
            first.
        inject_layer: Layer to add the direction to.
        read_layer: Layer to probe, strictly deeper than ``inject_layer``.
        concept: Concept identifier.
        neutral_prompts: Prompts to steer. Defaults to the deterministic
            neutral bank, which is what the concept was contrasted against, so
            a rise in the positive rate is a rise on the very prompts the
            concept was defined as absent from.
        strength: Injection coefficient.
        strength_mode: ``"relative"`` or ``"absolute"``.
        n_random: Matched-norm random control directions. Zero skips the
            control, which leaves the result uninterpretable and says so by
            reporting no random rates.
        seed: Seed for the probe, its folds, and the control directions.

    Returns:
        The rates, with the control alongside.

    Raises:
        ValueError: If the layers are not ordered, the concept cannot be
            resolved from the dataset, or the prompt set is empty.
        RuntimeError: If the model has not been loaded.
    """
    if read_layer <= inject_layer:
        raise ValueError(
            f"read_layer must be deeper than inject_layer, got "
            f"read={read_layer} inject={inject_layer}. Reading at the injection "
            "layer would only confirm that addition works."
        )
    engine._require_model()
    positives, curated = engine._resolve_examples(concept, None)
    negatives = build_contrast_examples(positives, curated)

    prompts = list(neutral_prompts) if neutral_prompts is not None else neutral_prompt_bank()
    if not prompts:
        raise ValueError("neutral_prompts must contain at least one prompt")

    # Fit the reader at the layer it will read from, on the same exemplars the
    # written direction came from.
    features = torch.cat(
        [
            engine.collect_activations(positives, read_layer),
            engine.collect_activations(negatives, read_layer),
        ],
        dim=0,
    )
    labels = [1] * len(positives) + [0] * len(negatives)
    probe = LinearProbe(seed=seed)
    accuracy, _, _ = probe.cross_val_accuracy(features, labels)
    probe.fit(features, labels)

    baseline_rate = _positive_rate(probe, engine.collect_activations(prompts, read_layer))

    # The direction is found at the layer it will be added to. A direction
    # extracted elsewhere is not this layer's concept direction.
    concept_direction = engine.contrast_direction(
        positives, negatives, layer=inject_layer, label=f"{concept}@L{inject_layer}"
    )

    def rate_under(direction: torch.Tensor) -> float:
        """Positive rate with *direction* injected, then cleanly removed."""
        return _rate_under(
            engine,
            direction,
            probe,
            prompts,
            inject_layer=inject_layer,
            read_layer=read_layer,
            strength=strength,
            strength_mode=strength_mode,
        )

    # Both arms go through one code path, so they cannot differ by anything
    # except the direction that was injected.
    steered_rate = rate_under(concept_direction)

    random_rates: list[float] = []
    if n_random > 0:
        d_model = int(engine.model.cfg.d_model)  # type: ignore[union-attr]
        for index, direction in enumerate(random_directions(d_model, n_random, seed)):
            random_rates.append(rate_under(direction))
            logger.debug("control %d/%d: rate %.3f", index + 1, n_random, random_rates[-1])

    result = ReadbackResult(
        concept=concept,
        inject_layer=inject_layer,
        read_layer=read_layer,
        strength=strength,
        strength_mode=strength_mode,
        baseline_rate=baseline_rate,
        steered_rate=steered_rate,
        random_rates=random_rates,
        probe_accuracy=accuracy,
        n_prompts=len(prompts),
    )
    logger.info(
        "%s: inject L%d -> read L%d | baseline %.2f steered %.2f random %.2f "
        "(lift over random %+.2f, probe %.3f)",
        concept,
        inject_layer,
        read_layer,
        result.baseline_rate,
        result.steered_rate,
        result.mean_random_rate,
        result.lift_over_random,
        result.probe_accuracy,
    )
    return result


def summarize_readback(results: dict[str, ReadbackResult]) -> list[dict[str, float | int | str]]:
    """Flatten read-back results into rows for a table or CSV.

    Args:
        results: Mapping from concept identifier to its result.

    Returns:
        One row per concept, ordered by descending lift over the random
        control, so the concepts whose written direction the probe recognises
        most clearly come first.
    """
    rows: list[dict[str, float | int | str]] = [
        {
            "concept_id": result.concept,
            "inject_layer": result.inject_layer,
            "read_layer": result.read_layer,
            "strength": result.strength,
            "strength_mode": result.strength_mode,
            "baseline_rate": round(result.baseline_rate, 6),
            "steered_rate": round(result.steered_rate, 6),
            "mean_random_rate": round(result.mean_random_rate, 6),
            "lift_over_random": round(result.lift_over_random, 6),
            "n_random": len(result.random_rates),
            "probe_accuracy": round(result.probe_accuracy, 6),
            "n_prompts": result.n_prompts,
        }
        for result in results.values()
    ]
    return sorted(rows, key=lambda row: float(row["lift_over_random"]), reverse=True)


@dataclass(frozen=True)
class SuppressionResult:
    """Whether negative steering removes a concept the probe can already see.

    Amplification is measured on neutral prompts, where the probe starts low
    and has room to rise. Suppression has to be measured where the probe starts
    *high*, which means the concept's own exemplars - and a probe that was
    trained on them would recognise them no matter what was injected. Every
    rate here is therefore pooled over held-out folds: each exemplar is scored
    by a probe that never saw it.

    Attributes:
        concept: Concept identifier.
        inject_layer: Layer the direction was subtracted from.
        read_layer: Layer the probes read from, strictly deeper.
        strength: Injection coefficient. Negative, or the check is measuring
            amplification under another name.
        strength_mode: ``"relative"`` or ``"absolute"``.
        baseline_rate: Share of held-out exemplars the probes recognised with
            no steering. This is the probes' recall, and it caps how far
            suppression can possibly push: a concept its probe barely
            recognises to begin with cannot be shown to be removable.
        steered_rate: The same share with the concept direction subtracted.
        random_rates: The same share for each matched-norm random direction,
            subtracted at the same layer and strength.
        n_exemplars: Held-out exemplars pooled across folds.
        n_folds: Folds the pooling ran over.
        probe_balanced_accuracy: Held-out balanced accuracy of the same
            probes, pooled the same way. Without it a baseline of 1.00 is
            ambiguous: a probe that answered "positive" to everything would
            reach it too, and any perturbation that unsettled it would look
            like suppression. A value near 0.5 means the drop below is not
            evidence about the concept.
    """

    concept: str
    inject_layer: int
    read_layer: int
    strength: float
    strength_mode: str
    baseline_rate: float
    steered_rate: float
    random_rates: list[float] = field(default_factory=list)
    n_exemplars: int = 0
    n_folds: int = 0
    probe_balanced_accuracy: float = 0.0

    @property
    def mean_random_rate(self) -> float:
        """Average held-out recognition rate under the random controls."""
        if not self.random_rates:
            return self.baseline_rate
        return float(np.mean(self.random_rates))

    @property
    def drop_from_baseline(self) -> float:
        """How far steering pushed recognition down. Positive means down."""
        return self.baseline_rate - self.steered_rate

    @property
    def drop_beyond_random(self) -> float:
        """How much of that drop a random direction did not also produce.

        The number worth reading. Subtracting a large enough vector damages
        the representation whatever its direction, and a probe stops
        recognising damaged activations; only the part a random direction of
        the same norm fails to reproduce is evidence about the concept.
        """
        return self.mean_random_rate - self.steered_rate


def suppression(
    engine: CulturalRepE,
    concept: str,
    inject_layer: int,
    read_layer: int,
    strength: float = -DEFAULT_RELATIVE_STRENGTH,
    strength_mode: str = "relative",
    n_random: int = DEFAULT_N_RANDOM,
    seed: int = DEFAULT_SEED,
    n_splits: int = DEFAULT_N_SPLITS,
) -> SuppressionResult:
    """Subtract a concept from its own exemplars and see whether it disappears.

    This is the claim representation engineering is most often reached for and
    least often checked: that a direction can be removed, not merely that
    adding it changes something. Amplification and suppression are not
    symmetric - a model can have a direction that is easy to add along and hard
    to remove, because removing it takes the activation somewhere the model
    never puts it.

    Args:
        engine: Engine with a loaded model.
        concept: Concept identifier.
        inject_layer: Layer to subtract the direction at.
        read_layer: Layer to probe, strictly deeper than ``inject_layer``.
        strength: Injection coefficient. Must be negative.
        strength_mode: ``"relative"`` or ``"absolute"``.
        n_random: Matched-norm random control directions.
        seed: Seed for the probes, the folds and the control directions.
        n_splits: Requested cross-validation folds, capped by the smaller class.

    Returns:
        The rates, with the control alongside.

    Raises:
        ValueError: If the layers are not ordered, ``strength`` is not
            negative, or the concept cannot be resolved.
        RuntimeError: If the model has not been loaded.
    """
    if read_layer <= inject_layer:
        raise ValueError(
            f"read_layer must be deeper than inject_layer, got "
            f"read={read_layer} inject={inject_layer}."
        )
    if strength >= 0:
        raise ValueError(
            f"strength must be negative to suppress, got {strength}. "
            "A positive coefficient measures amplification; use readback() for that."
        )

    engine._require_model()
    positives, curated = engine._resolve_examples(concept, None)
    negatives = build_contrast_examples(positives, curated)

    features = torch.cat(
        [
            engine.collect_activations(positives, read_layer),
            engine.collect_activations(negatives, read_layer),
        ],
        dim=0,
    )
    labels = np.array([1] * len(positives) + [0] * len(negatives))
    prompts = [*positives, *negatives]

    concept_direction = engine.contrast_direction(
        positives, negatives, layer=inject_layer, label=f"{concept}@L{inject_layer}"
    )
    d_model = int(engine.model.cfg.d_model)  # type: ignore[union-attr]
    controls = random_directions(d_model, n_random, seed) if n_random > 0 else []

    effective_splits = min(n_splits, int(min(np.bincount(labels)[np.bincount(labels) > 0])))
    if effective_splits < 2:
        raise ValueError(
            f"Concept {concept!r} has too few exemplars per class to hold any out "
            f"({effective_splits} in the smallest class)."
        )
    splitter = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=seed)

    baseline_hits = 0
    steered_hits = 0
    control_hits = [0] * len(controls)
    held_out_total = 0
    # Held-out negatives are tracked only to show the probes are discriminative.
    # A probe answering "positive" to everything would reach a baseline of 1.00
    # on the exemplars, and anything that unsettled it would read as suppression.
    negative_total = 0
    negative_hits = 0

    for train_index, test_index in splitter.split(features.numpy(), labels):
        held_out_positive = [
            prompts[i] for i in test_index if labels[i] == 1
        ]  # only exemplars can lose their label
        held_out_negative = [prompts[i] for i in test_index if labels[i] == 0]
        if not held_out_positive:
            continue

        probe = LinearProbe(seed=seed)
        probe.fit(features[train_index], labels[train_index])

        if held_out_negative:
            negative_total += len(held_out_negative)
            negative_hits += round(
                _positive_rate(probe, engine.collect_activations(held_out_negative, read_layer))
                * len(held_out_negative)
            )

        held_out_total += len(held_out_positive)
        baseline_hits += round(
            _positive_rate(probe, engine.collect_activations(held_out_positive, read_layer))
            * len(held_out_positive)
        )
        steered_hits += round(
            _rate_under(
                engine,
                concept_direction,
                probe,
                held_out_positive,
                inject_layer=inject_layer,
                read_layer=read_layer,
                strength=strength,
                strength_mode=strength_mode,
            )
            * len(held_out_positive)
        )
        for index, direction in enumerate(controls):
            control_hits[index] += round(
                _rate_under(
                    engine,
                    direction,
                    probe,
                    held_out_positive,
                    inject_layer=inject_layer,
                    read_layer=read_layer,
                    strength=strength,
                    strength_mode=strength_mode,
                )
                * len(held_out_positive)
            )

    if held_out_total == 0:  # pragma: no cover - guarded by effective_splits
        raise ValueError(f"No held-out exemplars for {concept!r}")

    # Balanced accuracy is the mean of the two held-out recalls, so it is 0.5
    # for a probe that answers with one class regardless of the input.
    true_positive_rate = baseline_hits / held_out_total
    true_negative_rate = 1.0 - (negative_hits / negative_total) if negative_total else 0.5

    result = SuppressionResult(
        concept=concept,
        inject_layer=inject_layer,
        read_layer=read_layer,
        strength=strength,
        strength_mode=strength_mode,
        baseline_rate=baseline_hits / held_out_total,
        steered_rate=steered_hits / held_out_total,
        random_rates=[hits / held_out_total for hits in control_hits],
        n_exemplars=held_out_total,
        n_folds=effective_splits,
        probe_balanced_accuracy=(true_positive_rate + true_negative_rate) / 2.0,
    )
    logger.info(
        "%s: subtract at L%d -> read L%d | baseline %.2f steered %.2f random %.2f "
        "(drop beyond random %+.2f, probe %.3f)",
        concept,
        inject_layer,
        read_layer,
        result.baseline_rate,
        result.steered_rate,
        result.mean_random_rate,
        result.drop_beyond_random,
        result.probe_balanced_accuracy,
    )
    return result


def summarize_suppression(
    results: dict[str, SuppressionResult],
) -> list[dict[str, float | int | str]]:
    """Flatten suppression results into rows for a table or CSV.

    Args:
        results: Mapping from concept identifier to its result.

    Returns:
        One row per concept, ordered by descending drop beyond the random
        control, so the concepts that can most clearly be removed come first.
    """
    rows: list[dict[str, float | int | str]] = [
        {
            "concept_id": result.concept,
            "inject_layer": result.inject_layer,
            "read_layer": result.read_layer,
            "strength": result.strength,
            "strength_mode": result.strength_mode,
            "baseline_rate": round(result.baseline_rate, 6),
            "steered_rate": round(result.steered_rate, 6),
            "mean_random_rate": round(result.mean_random_rate, 6),
            "drop_beyond_random": round(result.drop_beyond_random, 6),
            "n_random": len(result.random_rates),
            "probe_balanced_accuracy": round(result.probe_balanced_accuracy, 6),
            "n_exemplars": result.n_exemplars,
            "n_folds": result.n_folds,
        }
        for result in results.values()
    ]
    return sorted(rows, key=lambda row: float(row["drop_beyond_random"]), reverse=True)
