"""Measuring what steering does to a model's output.

Steering has two effects that pull against each other. Turning the strength up
makes the concept more present in the output, but past some point it also
degrades the text: the model starts repeating, drifts off-topic, or produces
fragments. A steering result is only interesting if it reports both.

:func:`evaluate_steering` sweeps the injection strength over a grid and, at each
point, records what the model generates and how surprised it is by the prompts
themselves. Cross-entropy on the prompt is a cheap, automatic proxy for that
second effect: it is measured on text the steering did not produce, so a sharp
rise means the injection is damaging the model's language modelling rather than
merely changing its topic.

Nothing here judges cultural alignment - that needs human raters, and is Phase 4
work. What this module gives is the axis along which those raters should be
asked to look, plus an automatic tripwire for the strengths where the output has
stopped being usable.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from src.models.rep_engine import CulturalRepE

logger = logging.getLogger(__name__)

DEFAULT_STRENGTHS: tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0)
"""Strength grid spanning suppression, no steering, and amplification."""

DEFAULT_MAX_NEW_TOKENS = 24


@dataclass
class SteeringResult:
    """What one injection strength did to the model.

    Attributes:
        strength: The injection coefficient that was applied.
        generations: Mapping from prompt to the text generated under steering.
            Empty when generation was disabled.
        prompt_losses: Mean cross-entropy per prompt, in nats per token.
        mean_loss: Mean of :attr:`prompt_losses` across prompts.
        perplexity: ``exp(mean_loss)``. Inf when the loss overflows, which is
            itself a signal that the strength is far too high.
    """

    strength: float
    generations: dict[str, str] = field(default_factory=dict)
    prompt_losses: dict[str, float] = field(default_factory=dict)
    mean_loss: float = float("nan")
    perplexity: float = float("nan")


def compute_prompt_loss(engine: CulturalRepE, prompt: str) -> float:
    """Return the model's mean cross-entropy on ``prompt``, in nats per token.

    The prompt is run alone rather than in a batch, so no padding enters the
    loss and the number is comparable across prompts of different lengths.

    Args:
        engine: Engine holding the loaded model.
        prompt: Text to score.

    Returns:
        Mean next-token cross-entropy over the prompt's tokens.

    Raises:
        RuntimeError: If the model has not been loaded.
    """
    model = engine.model
    if model is None:
        raise RuntimeError("Model is not loaded. Call load_model() first.")

    with torch.no_grad():
        loss = model(prompt, return_type="loss")
    return float(loss.item())


def _safe_perplexity(mean_loss: float) -> float:
    """Exponentiate a mean loss without raising on overflow.

    Args:
        mean_loss: Mean cross-entropy in nats per token.

    Returns:
        ``exp(mean_loss)``, or infinity when that overflows or the loss is not
        finite.
    """
    if not math.isfinite(mean_loss):
        return float("inf")
    try:
        return math.exp(mean_loss)
    except OverflowError:
        return float("inf")


def evaluate_steering(
    engine: CulturalRepE,
    concept: str,
    prompts: list[str],
    strengths: list[float] | tuple[float, ...] = DEFAULT_STRENGTHS,
    layers: list[int] | None = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    generate: bool = True,
) -> dict[float, SteeringResult]:
    """Sweep the injection strength and record output and fluency at each point.

    Every strength is evaluated inside :meth:`~src.models.rep_engine.CulturalRepE.steering`,
    so the hooks are removed before the next point is measured and a failure
    part-way through the sweep cannot leave the model steered.

    Include ``0.0`` in ``strengths`` to get the unsteered baseline measured
    through exactly the same code path as the steered points - injecting a zero
    offset is a mathematical no-op, which makes it a fair reference rather than a
    separate branch that might differ in some other way.

    Args:
        engine: Engine with a loaded model and the concept vector already
            extracted. The model is taken from the engine rather than passed
            separately, so the model being steered and the model being measured
            cannot drift apart.
        concept: Concept to inject.
        prompts: Texts to score, and to continue when ``generate`` is true.
        strengths: Injection coefficients to sweep over.
        layers: Layers to inject into. Defaults to the extraction layer.
        max_new_tokens: Tokens to generate per prompt.
        generate: Whether to generate continuations. Disable it for a
            fluency-only sweep, which is several times faster.

    Returns:
        A mapping from strength to its :class:`SteeringResult`, in the order the
        strengths were given.

    Raises:
        ValueError: If ``prompts`` or ``strengths`` is empty.
        KeyError: If ``concept`` has no extracted vector.
        RuntimeError: If the model has not been loaded.
    """
    if not prompts:
        raise ValueError("prompts must contain at least one text")
    if not strengths:
        raise ValueError("strengths must contain at least one value")
    if concept not in engine.concept_vectors:
        raise KeyError(
            f"No vector stored for concept {concept!r}. Call extract_vector() first."
        )
    if engine.model is None:
        raise RuntimeError("Model is not loaded. Call load_model() first.")

    results: dict[float, SteeringResult] = {}

    for strength in strengths:
        result = SteeringResult(strength=float(strength))

        with engine.steering(concept, strength=strength, layers=layers):
            for prompt in prompts:
                result.prompt_losses[prompt] = compute_prompt_loss(engine, prompt)
                if generate:
                    result.generations[prompt] = generate_steered(
                        engine, prompt, max_new_tokens=max_new_tokens
                    )

        losses = list(result.prompt_losses.values())
        result.mean_loss = sum(losses) / len(losses)
        result.perplexity = _safe_perplexity(result.mean_loss)
        results[float(strength)] = result

        logger.info(
            "strength=%+.2f mean_loss=%.4f perplexity=%.2f",
            strength,
            result.mean_loss,
            result.perplexity,
        )

    return results


def generate_steered(
    engine: CulturalRepE,
    prompt: str,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> str:
    """Generate a continuation with whatever hooks are currently attached.

    This does not attach hooks of its own; it is meant to be called inside a
    :meth:`~src.models.rep_engine.CulturalRepE.steering` block. Sampling is
    disabled so that two strengths can be compared without generation noise
    confounding the difference.

    Args:
        engine: Engine holding the loaded model.
        prompt: Text to continue.
        max_new_tokens: How many tokens to generate.

    Returns:
        The generated text, including the prompt.

    Raises:
        RuntimeError: If the model has not been loaded.
    """
    model = engine.model
    if model is None:
        raise RuntimeError("Model is not loaded. Call load_model() first.")

    with torch.no_grad():
        output = model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            verbose=False,
        )
    return str(output)


def summarize_sweep(results: dict[float, SteeringResult]) -> dict[str, list[float]]:
    """Reshape a sweep into parallel lists for plotting.

    Args:
        results: Output of :func:`evaluate_steering`.

    Returns:
        A mapping with ``"strengths"``, ``"mean_losses"`` and
        ``"perplexities"``, each sorted by ascending strength, ready to hand to
        :func:`~src.utils.visualization.plot_steering_sweep`.
    """
    ordered = sorted(results)
    return {
        "strengths": list(ordered),
        "mean_losses": [results[strength].mean_loss for strength in ordered],
        "perplexities": [results[strength].perplexity for strength in ordered],
    }
