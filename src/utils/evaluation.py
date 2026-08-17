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
from typing import TYPE_CHECKING, Any

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
        layers: The layers that were injected into, when known.
        effect_kl: Mean KL divergence between the steered and unsteered
            next-token distributions, averaged over prompts. This is the
            *effect* of the injection; ``mean_loss`` is its *cost*. ``nan``
            when the measurement was not requested.
    """

    strength: float
    generations: dict[str, str] = field(default_factory=dict)
    prompt_losses: dict[str, float] = field(default_factory=dict)
    mean_loss: float = float("nan")
    perplexity: float = float("nan")
    layers: tuple[int, ...] = ()
    effect_kl: float = float("nan")


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


def next_token_log_probs(engine: CulturalRepE, prompt: str) -> torch.Tensor:
    """Return the log-probabilities the model assigns to the next token.

    Args:
        engine: Engine holding the loaded model.
        prompt: Text to condition on.

    Returns:
        Log-probabilities over the vocabulary, shape ``(d_vocab,)``, in float32
        on the CPU.

    Raises:
        RuntimeError: If the model has not been loaded.
    """
    model = engine.model
    if model is None:
        raise RuntimeError("Model is not loaded. Call load_model() first.")

    with torch.no_grad():
        logits = model(model.to_tokens([prompt]))
    return torch.log_softmax(logits[0, -1, :].to(torch.float32), dim=-1).cpu()


def _kl_divergence(steered_log_probs: torch.Tensor, baseline_log_probs: torch.Tensor) -> float:
    """Return ``KL(steered || baseline)`` in nats.

    Args:
        steered_log_probs: Log-probabilities under steering.
        baseline_log_probs: Log-probabilities without steering.

    Returns:
        The divergence, clamped at zero to absorb floating point noise.
    """
    steered_probs = steered_log_probs.exp()
    divergence = float((steered_probs * (steered_log_probs - baseline_log_probs)).sum().item())
    return max(divergence, 0.0)


def measure_steering_effect(
    engine: CulturalRepE,
    concept: str,
    prompts: list[str],
    strength: float,
    layers: list[int] | None = None,
) -> float:
    """Return how far steering moves the model's next-token distribution.

    Cross-entropy tells you what an injection *costs*; it says nothing about
    whether the injection did anything at all. A steering setting that leaves
    fluency untouched because it had no effect should not be mistaken for a
    cheap win. This measures the effect directly, as the KL divergence between
    the steered and unsteered next-token distributions, averaged over prompts.

    Args:
        engine: Engine with a loaded model and the concept extracted.
        concept: Concept to inject.
        prompts: Texts to condition on.
        strength: Injection coefficient.
        layers: Layers to inject into. Defaults to the extraction layer.

    Returns:
        Mean KL divergence in nats. Zero means the injection changed nothing.

    Raises:
        ValueError: If ``prompts`` is empty.
        KeyError: If ``concept`` has no extracted vector.
    """
    if not prompts:
        raise ValueError("prompts must contain at least one text")

    baselines = [next_token_log_probs(engine, prompt) for prompt in prompts]

    with engine.steering(concept, strength=strength, layers=layers):
        steered = [next_token_log_probs(engine, prompt) for prompt in prompts]

    divergences = [
        _kl_divergence(after, before) for after, before in zip(steered, baselines, strict=True)
    ]
    return sum(divergences) / len(divergences)


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
    measure_effect: bool = False,
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
        measure_effect: Whether to also record the KL divergence between the
            steered and unsteered next-token distributions. Off by default
            because it costs an extra forward pass per prompt per strength.

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
        raise KeyError(f"No vector stored for concept {concept!r}. Call extract_vector() first.")
    if engine.model is None:
        raise RuntimeError("Model is not loaded. Call load_model() first.")

    results: dict[float, SteeringResult] = {}

    for strength in strengths:
        result = SteeringResult(strength=float(strength))

        with engine.steering(concept, strength=strength, layers=layers) as handles:
            result.layers = tuple(handle.layer for handle in handles)
            for prompt in prompts:
                result.prompt_losses[prompt] = compute_prompt_loss(engine, prompt)
                if generate:
                    result.generations[prompt] = generate_steered(
                        engine, prompt, max_new_tokens=max_new_tokens
                    )

        losses = list(result.prompt_losses.values())
        result.mean_loss = sum(losses) / len(losses)
        result.perplexity = _safe_perplexity(result.mean_loss)

        if measure_effect:
            result.effect_kl = measure_steering_effect(
                engine, concept, prompts, strength=strength, layers=layers
            )

        results[float(strength)] = result

        logger.info(
            "strength=%+.2f layers=%s mean_loss=%.4f perplexity=%.2f effect_kl=%.4f",
            strength,
            result.layers,
            result.mean_loss,
            result.perplexity,
            result.effect_kl,
        )

    return results


def evaluate_layer_sets(
    engine: CulturalRepE,
    concept: str,
    prompts: list[str],
    layer_sets: list[list[int]],
    strengths: list[float] | tuple[float, ...] = DEFAULT_STRENGTHS,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    generate: bool = False,
    measure_effect: bool = True,
) -> dict[tuple[int, ...], dict[float, SteeringResult]]:
    """Sweep both which layers are injected into and how hard.

    Injecting the same direction into several mid-stack layers usually produces
    a larger effect per unit of strength than a single layer, at a larger
    fluency cost. Which trade is better is an empirical question per model, and
    this is the grid that answers it: effect (``effect_kl``) against cost
    (``mean_loss``), for every layer configuration.

    ``layer_sets`` is a separate argument rather than an overload of
    ``evaluate_steering``'s ``layers`` because a bare ``[12, 14]`` is ambiguous -
    one configuration spanning two layers, or two single-layer configurations?
    Keeping the nesting explicit removes the guesswork.

    Args:
        engine: Engine with a loaded model and the concept extracted.
        concept: Concept to inject.
        prompts: Texts to score, and to continue when ``generate`` is true.
        layer_sets: Layer configurations to compare, for example
            ``[[12], [12, 14], [10, 12, 14]]``.
        strengths: Injection coefficients to sweep within each configuration.
        max_new_tokens: Tokens to generate per prompt.
        generate: Whether to generate continuations. Off by default: a full grid
            multiplies generation cost by the number of configurations.
        measure_effect: Whether to record the KL effect size. On by default
            here, since comparing configurations without it is meaningless.

    Returns:
        A mapping from layer configuration (as a sorted tuple) to that
        configuration's strength sweep.

    Raises:
        ValueError: If ``layer_sets`` is empty or contains an empty set.
        KeyError: If ``concept`` has no extracted vector.
        RuntimeError: If the model has not been loaded.
    """
    if not layer_sets:
        raise ValueError("layer_sets must contain at least one configuration")
    if any(not layers for layers in layer_sets):
        raise ValueError("each entry of layer_sets must contain at least one layer")

    grid: dict[tuple[int, ...], dict[float, SteeringResult]] = {}

    for layers in layer_sets:
        logger.info("Evaluating layer configuration %s", layers)
        sweep = evaluate_steering(
            engine,
            concept,
            prompts,
            strengths=strengths,
            layers=layers,
            max_new_tokens=max_new_tokens,
            generate=generate,
            measure_effect=measure_effect,
        )
        key = tuple(sorted({engine._resolve_layer(layer) for layer in layers}))
        grid[key] = sweep

    return grid


def summarize_layer_sets(
    grid: dict[tuple[int, ...], dict[float, SteeringResult]],
) -> list[dict[str, Any]]:
    """Flatten a layer-set grid into rows ready for a table or DataFrame.

    Args:
        grid: Output of :func:`evaluate_layer_sets`.

    Returns:
        One row per (configuration, strength) pair, each holding the layers,
        their count, the strength, the effect and the cost. Rows are ordered by
        configuration then strength.
    """
    rows: list[dict[str, Any]] = []
    for layers in sorted(grid):
        for strength in sorted(grid[layers]):
            result = grid[layers][strength]
            rows.append(
                {
                    "layers": layers,
                    "n_layers": len(layers),
                    "strength": strength,
                    "effect_kl": result.effect_kl,
                    "mean_loss": result.mean_loss,
                    "perplexity": result.perplexity,
                }
            )
    return rows


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
