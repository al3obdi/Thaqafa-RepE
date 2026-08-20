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

from src.data.contrastive import build_contrast_examples, neutral_prompt_bank
from src.utils.probes import DEFAULT_SEED, LinearProbe

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
