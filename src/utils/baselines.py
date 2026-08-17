"""Prompt-engineering baselines to compare representation steering against.

Steering is only worth its complexity if it beats the obvious alternative:
telling the model in plain language what you want. "Answer with Arab hospitality
in mind" costs nothing to implement, needs no white-box access, and works
through an API. Any claim that representation engineering helps has to be made
against that baseline, not against an unprompted model.

The two interventions are not symmetric, and the asymmetry matters when
reporting numbers:

* **Steering modifies the model.** Its fluency cost can be measured on any text,
  because the perturbed model scores text differently.
* **Prompting modifies the input.** The model is untouched, so there is no
  "cost" to measure the same way - but the instruction consumes context, and the
  model may echo, refuse, or over-comply with it.

Cross-entropy on the *prompt* is therefore not comparable across the two: the
prompted condition is scoring a different string. What is comparable is the
fluency of the **output**, scored by the same unmodified model in both cases.
:func:`compare_steering_vs_prompting` reports that, and deliberately reports no
single "winner" - which intervention is more culturally grounded is a question
for human raters, not for perplexity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.utils.evaluation import (
    DEFAULT_MAX_NEW_TOKENS,
    compute_prompt_loss,
    generate_steered,
    measure_steering_effect,
)

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from src.models.rep_engine import CulturalRepE

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InstructionTemplate:
    """A natural-language framing of a cultural concept.

    Attributes:
        name: Short identifier used to select the template.
        language: ``"en"`` or ``"ar"``.
        template: Format string taking ``{concept}`` and ``{prompt}``.
        description: What framing this template applies, and why it differs
            from the others.
    """

    name: str
    language: str
    template: str
    description: str

    def format(self, concept_name: str, prompt: str) -> str:
        """Render the instruction around a prompt.

        Args:
            concept_name: Human-readable concept name, for example
                ``"Arab hospitality"``.
            prompt: The user's actual question.

        Returns:
            The full text to send to the model.
        """
        return self.template.format(concept=concept_name, prompt=prompt)


DIRECT_EN = InstructionTemplate(
    name="direct_en",
    language="en",
    template="Answer the following with deep {concept} in mind:\n{prompt}",
    description="The plainest instruction: name the concept and ask for it.",
)

REFLECT_EN = InstructionTemplate(
    name="reflect_en",
    language="en",
    template="Respond reflecting the cultural concept of {concept}.\n\n{prompt}",
    description="Frames the concept as cultural rather than as a style request.",
)

PERSONA_EN = InstructionTemplate(
    name="persona_en",
    language="en",
    template=(
        "You are answering as someone for whom {concept} is a lived, everyday "
        "value rather than an abstract idea.\n\n{prompt}"
    ),
    description=(
        "A persona framing. Often stronger than a direct instruction, and the "
        "harder baseline for steering to beat."
    ),
)

DIRECT_AR = InstructionTemplate(
    name="direct_ar",
    language="ar",
    template="أجب عما يلي مستحضرًا مفهوم {concept}:\n{prompt}",
    description="The direct instruction in Arabic, for Arabic-language prompts.",
)

NEUTRAL = InstructionTemplate(
    name="neutral",
    language="en",
    template="{prompt}",
    description=(
        "No instruction at all. The control condition - include it so the "
        "comparison has a floor."
    ),
)

INSTRUCTION_TEMPLATES: dict[str, InstructionTemplate] = {
    template.name: template
    for template in (DIRECT_EN, REFLECT_EN, PERSONA_EN, DIRECT_AR, NEUTRAL)
}

DEFAULT_TEMPLATE = DIRECT_EN


def get_template(name: str) -> InstructionTemplate:
    """Look up a template by name.

    Args:
        name: Template name, for example ``"persona_en"``.

    Returns:
        The matching template.

    Raises:
        KeyError: If no template has that name.
    """
    if name not in INSTRUCTION_TEMPLATES:
        known = ", ".join(sorted(INSTRUCTION_TEMPLATES))
        raise KeyError(f"Unknown instruction template {name!r}. Known templates: {known}")
    return INSTRUCTION_TEMPLATES[name]


@dataclass
class ConditionResult:
    """What one intervention produced for one set of prompts.

    Attributes:
        condition: Which intervention this was, for example ``"steering"`` or
            ``"prompt:persona_en"``.
        generations: Mapping from the original prompt to the continuation the
            model produced. Keyed by the *original* prompt even when an
            instruction was prepended, so conditions line up side by side.
        full_inputs: Mapping from the original prompt to the text actually sent
            to the model. Identical to the key for steering; longer for
            prompting.
        continuation_losses: Cross-entropy of each continuation under the
            *unmodified* model, in nats per token. Comparable across
            conditions, because the scorer is the same in all of them.
        mean_continuation_loss: Mean of :attr:`continuation_losses`.
        extra_input_tokens: Roughly how many extra tokens the instruction cost,
            measured in whitespace-separated words. Zero for steering.
    """

    condition: str
    generations: dict[str, str] = field(default_factory=dict)
    full_inputs: dict[str, str] = field(default_factory=dict)
    continuation_losses: dict[str, float] = field(default_factory=dict)
    mean_continuation_loss: float = float("nan")
    extra_input_tokens: int = 0


def _mean(values: list[float]) -> float:
    """Return the mean of ``values``, or ``nan`` when empty.

    Args:
        values: Numbers to average.

    Returns:
        The arithmetic mean.
    """
    return sum(values) / len(values) if values else float("nan")


def _continuation_of(full_output: str, full_input: str) -> str:
    """Strip the input back off a generation.

    Args:
        full_output: What the model returned, including the input.
        full_input: The text that was sent in.

    Returns:
        Just the newly generated text, stripped of surrounding whitespace. The
        full output is returned unchanged if it does not start with the input,
        which can happen when the tokenizer round-trip is not exact.
    """
    if full_output.startswith(full_input):
        return full_output[len(full_input) :].strip()
    return full_output.strip()


def generate_prompt_baseline(
    engine: CulturalRepE,
    concept_name: str,
    prompts: list[str],
    instruction_template: InstructionTemplate | str = DEFAULT_TEMPLATE,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> ConditionResult:
    """Generate answers with a natural-language instruction instead of steering.

    No hooks are attached: the model is untouched and only the input changes.
    This is the baseline representation engineering has to beat.

    Args:
        engine: Engine with a loaded model. The model is taken from the engine
            for consistency with the rest of the evaluation code, so the model
            being prompted and the model being scored cannot drift apart.
        concept_name: Human-readable concept name to interpolate into the
            instruction, for example ``"Arab hospitality"``.
        prompts: The questions to answer, without any instruction attached.
        instruction_template: A template or the name of a registered one.
        max_new_tokens: Tokens to generate per prompt.

    Returns:
        The condition's generations, the exact inputs used, and the fluency of
        each continuation under the unmodified model.

    Raises:
        ValueError: If ``prompts`` is empty or ``concept_name`` is blank.
        KeyError: If a template name is not registered.
        RuntimeError: If the model has not been loaded.
    """
    if not prompts:
        raise ValueError("prompts must contain at least one text")
    if not concept_name or not concept_name.strip():
        raise ValueError("concept_name must be a non-empty string")
    if engine.model is None:
        raise RuntimeError("Model is not loaded. Call load_model() first.")

    template = (
        get_template(instruction_template)
        if isinstance(instruction_template, str)
        else instruction_template
    )

    result = ConditionResult(condition=f"prompt:{template.name}")
    extra_tokens: list[int] = []

    for prompt in prompts:
        full_input = template.format(concept_name, prompt)
        result.full_inputs[prompt] = full_input
        extra_tokens.append(len(full_input.split()) - len(prompt.split()))

        output = generate_steered(engine, full_input, max_new_tokens=max_new_tokens)
        continuation = _continuation_of(output, full_input)
        result.generations[prompt] = continuation

        if continuation:
            result.continuation_losses[prompt] = compute_prompt_loss(engine, continuation)

    result.mean_continuation_loss = _mean(list(result.continuation_losses.values()))
    result.extra_input_tokens = max(extra_tokens) if extra_tokens else 0

    logger.info(
        "%s: mean continuation loss %.4f over %d prompt(s), +%d instruction tokens",
        result.condition,
        result.mean_continuation_loss,
        len(prompts),
        result.extra_input_tokens,
    )
    return result


def generate_steering_condition(
    engine: CulturalRepE,
    concept: str,
    prompts: list[str],
    strength: float = 1.0,
    layers: list[int] | None = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> ConditionResult:
    """Generate answers with vector steering instead of an instruction.

    The continuations are scored *after* the steering scope exits, so the
    fluency number comes from the unmodified model and is directly comparable
    with :func:`generate_prompt_baseline`.

    Args:
        engine: Engine with a loaded model and the concept extracted.
        concept: Concept identifier to inject.
        prompts: The questions to answer.
        strength: Injection coefficient.
        layers: Layers to inject into. Defaults to the extraction layer.
        max_new_tokens: Tokens to generate per prompt.

    Returns:
        The condition's generations and their fluency under the clean model.

    Raises:
        ValueError: If ``prompts`` is empty.
        KeyError: If ``concept`` has no extracted vector.
        RuntimeError: If the model has not been loaded.
    """
    if not prompts:
        raise ValueError("prompts must contain at least one text")

    result = ConditionResult(condition=f"steering@{strength:+.2f}")

    with engine.steering(concept, strength=strength, layers=layers):
        for prompt in prompts:
            result.full_inputs[prompt] = prompt
            output = generate_steered(engine, prompt, max_new_tokens=max_new_tokens)
            result.generations[prompt] = _continuation_of(output, prompt)

    # Scored outside the steering scope: the clean model is the common yardstick.
    for prompt, continuation in result.generations.items():
        if continuation:
            result.continuation_losses[prompt] = compute_prompt_loss(engine, continuation)

    result.mean_continuation_loss = _mean(list(result.continuation_losses.values()))

    logger.info(
        "%s: mean continuation loss %.4f over %d prompt(s)",
        result.condition,
        result.mean_continuation_loss,
        len(prompts),
    )
    return result


@dataclass
class ComparisonResult:
    """Steering and prompting measured side by side.

    Attributes:
        prompts: The questions all conditions answered.
        conditions: Every condition that was run, keyed by its name. Always
            includes the unprompted control.
        steering_effect_kl: How far steering moved the next-token distribution.
            Prompting has no counterpart: it does not perturb the model, so
            there is no distribution shift to attribute to it.
    """

    prompts: list[str] = field(default_factory=list)
    conditions: dict[str, ConditionResult] = field(default_factory=dict)
    steering_effect_kl: float = float("nan")

    def rows(self) -> list[dict[str, object]]:
        """Flatten the comparison into rows for a table.

        Returns:
            One row per condition, holding its name, mean continuation loss and
            instruction overhead.
        """
        return [
            {
                "condition": name,
                "mean_continuation_loss": result.mean_continuation_loss,
                "extra_input_tokens": result.extra_input_tokens,
                "n_generations": len(result.generations),
            }
            for name, result in self.conditions.items()
        ]


def compare_steering_vs_prompting(
    engine: CulturalRepE,
    concept: str,
    concept_name: str,
    prompts: list[str],
    strength: float = 1.0,
    layers: list[int] | None = None,
    template_names: list[str] | None = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    measure_effect: bool = False,
) -> ComparisonResult:
    """Run steering and prompting on the same prompts and collect both outputs.

    This produces the comparison table the paper needs. It deliberately does not
    declare a winner: the only automatic number here is continuation fluency
    under the unmodified model, which detects damage but cannot tell you which
    answer is more culturally grounded. That judgement needs human raters, and
    the generations are returned so they can be handed to them blind.

    Args:
        engine: Engine with a loaded model and the concept extracted.
        concept: Concept identifier, for the steering condition.
        concept_name: Human-readable name, for the prompting conditions.
        prompts: Questions every condition answers.
        strength: Injection coefficient for the steering condition.
        layers: Layers to inject into. Defaults to the extraction layer.
        template_names: Instruction templates to run. Defaults to the direct
            and persona English framings. The unprompted control is always
            included.
        max_new_tokens: Tokens to generate per prompt.
        measure_effect: Whether to also record the steering KL effect size.

    Returns:
        Every condition's generations and fluency, ready to tabulate.

    Raises:
        ValueError: If ``prompts`` is empty.
        KeyError: If ``concept`` has no extracted vector, or a template name is
            not registered.
        RuntimeError: If the model has not been loaded.
    """
    if not prompts:
        raise ValueError("prompts must contain at least one text")
    if concept not in engine.concept_vectors:
        raise KeyError(f"No vector stored for concept {concept!r}. Call extract_vector() first.")

    names = ["direct_en", "persona_en"] if template_names is None else list(template_names)
    if NEUTRAL.name not in names:
        names.insert(0, NEUTRAL.name)

    comparison = ComparisonResult(prompts=list(prompts))

    for name in names:
        result = generate_prompt_baseline(
            engine,
            concept_name,
            prompts,
            instruction_template=name,
            max_new_tokens=max_new_tokens,
        )
        comparison.conditions[result.condition] = result

    steering_result = generate_steering_condition(
        engine,
        concept,
        prompts,
        strength=strength,
        layers=layers,
        max_new_tokens=max_new_tokens,
    )
    comparison.conditions[steering_result.condition] = steering_result

    if measure_effect:
        comparison.steering_effect_kl = measure_steering_effect(
            engine, concept, prompts, strength=strength, layers=layers
        )

    return comparison
