"""Do the Arabic and English exemplars of a concept find the same direction?

If a model represents *diyafa* as one thing rather than as two unrelated
language-specific habits, then a direction extracted from the Arabic exemplars
alone should point roughly where the one extracted from the English exemplars
alone points. That is a claim about the model's representation, and it is
checkable without any human raters.

The check is only interpretable with a control. Two directions extracted from
the same model at the same layer can be similar simply because the residual
stream at that layer has a dominant axis that every mean-difference partly
lies along, and a cosine of 0.4 means nothing on its own. Every aligned pair is
therefore reported next to *mismatched* pairs - the Arabic direction of one
concept against the English direction of the others - measured the same way.
The quantity that carries information is the gap between them.

A model with little Arabic capability will fail this check for reasons that
have nothing to do with culture, so a low separation is evidence about the
model before it is evidence about the concept.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from src.data.contrastive import neutral_prompt_bank
from src.data.dataset_builder import CulturalConcept, load_concepts
from src.utils.probes import DEFAULT_SEED

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from src.models.rep_engine import CulturalRepE

logger = logging.getLogger(__name__)

ARABIC = "ar"
ENGLISH = "en"
LANGUAGES: tuple[str, str] = (ARABIC, ENGLISH)


@dataclass(frozen=True)
class AlignmentResult:
    """How well one concept's two language directions agree.

    Attributes:
        concept: Concept identifier.
        layer: Layer both directions were extracted at.
        aligned_cosine: Cosine between the Arabic and English directions of
            *this* concept.
        mismatched_cosines: Cosine between this concept's Arabic direction and
            every *other* concept's English direction, keyed by that concept.
        n_arabic: Arabic exemplars behind the Arabic direction.
        n_english: English exemplars behind the English direction.
    """

    concept: str
    layer: int
    aligned_cosine: float
    mismatched_cosines: dict[str, float] = field(default_factory=dict)
    n_arabic: int = 0
    n_english: int = 0

    @property
    def mean_mismatched(self) -> float:
        """Average cosine against the other concepts' English directions.

        Returns:
            The mean, or 0.0 when there was no other concept to compare with -
            in which case :attr:`separation` is uninformative and says so by
            being equal to :attr:`aligned_cosine`.
        """
        if not self.mismatched_cosines:
            return 0.0
        return sum(self.mismatched_cosines.values()) / len(self.mismatched_cosines)

    @property
    def separation(self) -> float:
        """How much closer the aligned pair is than the mismatched ones.

        This, not :attr:`aligned_cosine`, is the number worth reading: a high
        aligned cosine that every mismatched pair also reaches says the layer
        has a dominant axis, not that the concept crosses languages.
        """
        return self.aligned_cosine - self.mean_mismatched


def cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    """Return the cosine similarity between two 1-D tensors.

    Args:
        first: A vector.
        second: A vector of the same length.

    Returns:
        The cosine in ``[-1, 1]``, or 0.0 if either vector is the zero vector,
        for which no angle is defined.

    Raises:
        ValueError: If the shapes disagree or either input is not 1-D.
    """
    if first.ndim != 1 or second.ndim != 1:
        raise ValueError(f"both vectors must be 1-D, got {first.shape} and {second.shape}")
    if first.shape != second.shape:
        raise ValueError(f"shape mismatch: {first.shape} against {second.shape}")

    left = first.to(torch.float32)
    right = second.to(torch.float32)
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if denominator <= torch.finfo(torch.float32).eps:
        return 0.0
    return float(torch.dot(left, right) / denominator)


def _entry(engine: CulturalRepE, concept: str) -> CulturalConcept:
    """Look one concept up in the engine's dataset.

    Args:
        engine: Engine holding the dataset path.
        concept: Concept identifier.

    Returns:
        The matching entry.

    Raises:
        ValueError: If the concept is not in the dataset.
    """
    concepts = load_concepts(engine.dataset_path)
    for entry in concepts:
        if entry.concept_id == concept:
            return entry
    known = ", ".join(item.concept_id for item in concepts)
    raise ValueError(f"Concept {concept!r} was not found in {engine.dataset_path}. Known: {known}")


def language_direction(
    engine: CulturalRepE,
    concept: str,
    language: str,
    layer: int | None = None,
) -> torch.Tensor:
    """Extract a concept direction from one language's exemplars only.

    Both the positive and the negative side come from the same language, so the
    result cannot be a direction that merely separates Arabic text from English
    text.

    Args:
        engine: Engine with a loaded model.
        concept: Concept identifier.
        language: ``"ar"`` or ``"en"``.
        layer: Block to read from. ``None`` selects the engine's middle layer.

    Returns:
        A unit-norm direction of shape ``(d_model,)``.

    Raises:
        ValueError: If the language is unknown, the concept is not in the
            dataset, or that language has no exemplars for it.
        RuntimeError: If the model has not been loaded.
    """
    if language not in LANGUAGES:
        raise ValueError(f"language must be one of {LANGUAGES}, got {language!r}")

    entry = _entry(engine, concept)
    positives = entry.examples_ar if language == ARABIC else entry.examples_en
    contrasts = entry.contrast_ar if language == ARABIC else entry.contrast_en
    if not positives:
        raise ValueError(f"Concept {concept!r} has no {language} exemplars")

    return engine.contrast_direction(
        list(positives),
        list(contrasts) or None,
        layer=layer,
        label=f"{concept}[{language}]",
    )


def alignment(
    engine: CulturalRepE,
    concepts: list[str],
    layer: int | None = None,
) -> dict[str, AlignmentResult]:
    """Measure Arabic-English agreement for every concept, against a control.

    Args:
        engine: Engine with a loaded model.
        concepts: Concept identifiers to measure. At least two are needed for
            the mismatched control to mean anything; with one, every result's
            ``separation`` equals its ``aligned_cosine`` and should be ignored.
        layer: Block to extract at. ``None`` selects the engine's middle layer.
            One layer for all concepts, so the cosines are comparable.

    Returns:
        A mapping from concept identifier to its result, in the order given.

    Raises:
        ValueError: If ``concepts`` is empty, or a concept lacks exemplars in
            either language.
        RuntimeError: If the model has not been loaded.
    """
    if not concepts:
        raise ValueError("concepts must contain at least one identifier")
    if len(concepts) < 2:
        logger.warning(
            "Only one concept given: the mismatched control cannot be computed, "
            "so separation will not be interpretable."
        )

    resolved_layer = engine._resolve_layer(layer)
    arabic = {c: language_direction(engine, c, ARABIC, resolved_layer) for c in concepts}
    english = {c: language_direction(engine, c, ENGLISH, resolved_layer) for c in concepts}

    results: dict[str, AlignmentResult] = {}
    for concept in concepts:
        entry = _entry(engine, concept)
        mismatched = {
            other: cosine(arabic[concept], english[other]) for other in concepts if other != concept
        }
        result = AlignmentResult(
            concept=concept,
            layer=resolved_layer,
            aligned_cosine=cosine(arabic[concept], english[concept]),
            mismatched_cosines=mismatched,
            n_arabic=len(entry.examples_ar),
            n_english=len(entry.examples_en),
        )
        results[concept] = result
        logger.info(
            "%s at layer %d: aligned %.3f, mismatched %.3f, separation %+.3f",
            concept,
            resolved_layer,
            result.aligned_cosine,
            result.mean_mismatched,
            result.separation,
        )
    return results


def summarize_alignment(results: dict[str, AlignmentResult]) -> list[dict[str, float | int | str]]:
    """Flatten alignment results into rows for a table or CSV.

    Args:
        results: Output of :func:`alignment`.

    Returns:
        One row per concept, ordered by descending separation so the concepts
        that cross languages most clearly come first.
    """
    rows: list[dict[str, float | int | str]] = [
        {
            "concept_id": result.concept,
            "layer": result.layer,
            "aligned_cosine": round(result.aligned_cosine, 6),
            "mean_mismatched_cosine": round(result.mean_mismatched, 6),
            "separation": round(result.separation, 6),
            "n_arabic": result.n_arabic,
            "n_english": result.n_english,
        }
        for result in results.values()
    ]
    return sorted(rows, key=lambda row: float(row["separation"]), reverse=True)


@dataclass(frozen=True)
class TransferResult:
    """Whether one language's direction steers the other language's reader.

    :func:`alignment` asks a geometric question - do the two directions point
    the same way - and answers it with a cosine. This asks the behavioural one:
    inject the Arabic direction and see whether a probe trained on English
    exemplars notices. The two can disagree. A cosine of 0.2 between vectors in
    a 768-dimensional space still leaves a large shared component, and a probe
    reads a projection, not an angle.

    Every rate needs two references to be read. The **random** arm is the floor:
    any large perturbation moves a probe. The **same-language** arm is the
    ceiling: it is the best this reader could do with a direction extracted from
    its own exemplars, so it says how much of the effect was ever available to
    transfer.

    Attributes:
        concept: Concept identifier.
        reader_language: Language whose exemplars trained the reading probe.
        inject_layer: Layer the direction was added to.
        read_layer: Layer the probe read from, strictly deeper.
        strength: Injection coefficient, as a fraction of the residual norm.
        probe_accuracy: Cross-validated balanced accuracy of the reading probe.
            A transfer measured through a probe near 0.5 says nothing.
        baseline_rate: Positive rate with no steering.
        same_language_rate: Positive rate with the reader's own language
            direction injected.
        other_language_rate: Positive rate with the *other* language's
            direction injected. This is the transfer.
        random_rates: Positive rate under each matched-norm random direction.
        n_prompts: Neutral prompts behind every rate, in the reader's language.
    """

    concept: str
    reader_language: str
    inject_layer: int
    read_layer: int
    strength: float
    probe_accuracy: float
    baseline_rate: float
    same_language_rate: float
    other_language_rate: float
    random_rates: list[float] = field(default_factory=list)
    n_prompts: int = 0

    @property
    def mean_random_rate(self) -> float:
        """Average positive rate across the random control directions."""
        if not self.random_rates:
            return self.baseline_rate
        return float(sum(self.random_rates) / len(self.random_rates))

    @property
    def same_language_lift(self) -> float:
        """The ceiling: what the reader's own language direction achieved."""
        return self.same_language_rate - self.mean_random_rate

    @property
    def transfer_lift(self) -> float:
        """What the other language's direction achieved, over the same floor."""
        return self.other_language_rate - self.mean_random_rate

    @property
    def transfer_ratio(self) -> float:
        """Transfer as a fraction of what was available to transfer.

        1.0 means the other language's direction moved this reader exactly as
        far as its own did; 0.0 means it did no better than noise. Returns 0.0
        when the ceiling is not positive, because a concept whose own direction
        does not move its own reader has nothing to transfer and a ratio there
        would divide by a number that means nothing.
        """
        if self.same_language_lift <= 0:
            return 0.0
        return self.transfer_lift / self.same_language_lift


def transfer(
    engine: CulturalRepE,
    concept: str,
    inject_layer: int,
    read_layer: int,
    reader_language: str = ENGLISH,
    strengths: Sequence[float] = (0.02, 0.05, 0.10, 0.20),
    strength_mode: str = "relative",
    n_random: int = 3,
    seed: int = DEFAULT_SEED,
) -> list[TransferResult]:
    """Steer with one language's direction and read with the other's probe.

    Both the probe and the prompts come from ``reader_language``, so nothing in
    the measurement is bilingual except the injected direction. A rise in the
    probe's positive rate therefore cannot be the reader recognising Arabic
    text - it never sees any.

    Args:
        engine: Engine with a loaded model.
        concept: Concept identifier.
        inject_layer: Layer to add the direction to.
        read_layer: Layer to probe, strictly deeper than ``inject_layer``.
        reader_language: ``"en"`` or ``"ar"``. The other language supplies the
            transferred direction.
        strengths: Injection coefficients to sweep. All are reported: the
            effect saturates, and at a saturated point both arms sit at 1.00
            so the ratio reads 1.00 for free. A grid shows whether the
            transfer holds where there is still room to move.
        strength_mode: ``"relative"`` or ``"absolute"``.
        n_random: Matched-norm random control directions.
        seed: Seed for the probe, its folds and the controls.

    Returns:
        One result per strength, in the order given. The probe and both
        directions are built once and shared, so the points differ only in
        the coefficient.

    Raises:
        ValueError: If the layers are not ordered, the language is unknown, or
            the concept lacks exemplars in either language.
        RuntimeError: If the model has not been loaded.
    """
    from src.data.contrastive import build_contrast_examples
    from src.utils.causal import _rate_under, random_directions
    from src.utils.probes import LinearProbe

    if reader_language not in LANGUAGES:
        raise ValueError(f"reader_language must be one of {LANGUAGES}, got {reader_language!r}")
    if read_layer <= inject_layer:
        raise ValueError(
            f"read_layer must be deeper than inject_layer, got "
            f"read={read_layer} inject={inject_layer}."
        )

    engine._require_model()
    entry = _entry(engine, concept)
    other_language = ENGLISH if reader_language == ARABIC else ARABIC

    reader_positives = entry.examples_ar if reader_language == ARABIC else entry.examples_en
    reader_contrasts = entry.contrast_ar if reader_language == ARABIC else entry.contrast_en
    if not reader_positives:
        raise ValueError(f"Concept {concept!r} has no {reader_language} exemplars")

    # The reader is trained and tested entirely within its own language.
    positives = list(reader_positives)
    negatives = build_contrast_examples(positives, list(reader_contrasts) or None)
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

    prompts = neutral_prompt_bank(reader_language)  # type: ignore[arg-type]
    same = language_direction(engine, concept, reader_language, inject_layer)
    other = language_direction(engine, concept, other_language, inject_layer)
    d_model = int(engine.model.cfg.d_model)  # type: ignore[union-attr]
    controls = random_directions(d_model, n_random, seed) if n_random > 0 else []
    baseline = _positive_rate_via(engine, probe, prompts, read_layer)

    results: list[TransferResult] = []
    for strength in strengths:

        def rate(direction: torch.Tensor, coefficient: float = strength) -> float:
            """Positive rate under one injected direction at this strength."""
            return _rate_under(
                engine,
                direction,
                probe,
                prompts,
                inject_layer=inject_layer,
                read_layer=read_layer,
                strength=coefficient,
                strength_mode=strength_mode,
            )

        result = TransferResult(
            concept=concept,
            reader_language=reader_language,
            inject_layer=inject_layer,
            read_layer=read_layer,
            strength=strength,
            probe_accuracy=accuracy,
            baseline_rate=baseline,
            same_language_rate=rate(same),
            other_language_rate=rate(other),
            random_rates=[rate(control) for control in controls],
            n_prompts=len(prompts),
        )
        results.append(result)
        logger.info(
            "%s read by %s at L%d s=%.2f: baseline %.2f own %.2f other %.2f "
            "random %.2f (ratio %.2f, probe %.3f)",
            concept,
            reader_language,
            read_layer,
            strength,
            result.baseline_rate,
            result.same_language_rate,
            result.other_language_rate,
            result.mean_random_rate,
            result.transfer_ratio,
            result.probe_accuracy,
        )
    return results


def _positive_rate_via(
    engine: CulturalRepE,
    probe: Any,
    prompts: list[str],
    read_layer: int,
) -> float:
    """Unsteered positive rate, for the baseline arm.

    Args:
        engine: Engine with a loaded model.
        probe: A fitted probe.
        prompts: Texts to score.
        read_layer: Layer to read activations from.

    Returns:
        The share of prompts the probe calls positive.
    """
    from src.utils.causal import _positive_rate

    return _positive_rate(probe, engine.collect_activations(prompts, read_layer))


def summarize_transfer(results: dict[str, TransferResult]) -> list[dict[str, float | int | str]]:
    """Flatten transfer results into rows for a table or CSV.

    Args:
        results: Mapping from a key to its result.

    Returns:
        One row per result, ordered by descending transfer ratio, so the
        concepts whose direction carries furthest across languages come first.
    """
    rows: list[dict[str, float | int | str]] = [
        {
            "concept_id": result.concept,
            "reader_language": result.reader_language,
            "inject_layer": result.inject_layer,
            "read_layer": result.read_layer,
            "strength": result.strength,
            "probe_accuracy": round(result.probe_accuracy, 6),
            "baseline_rate": round(result.baseline_rate, 6),
            "same_language_rate": round(result.same_language_rate, 6),
            "other_language_rate": round(result.other_language_rate, 6),
            "mean_random_rate": round(result.mean_random_rate, 6),
            "same_language_lift": round(result.same_language_lift, 6),
            "transfer_lift": round(result.transfer_lift, 6),
            "transfer_ratio": round(result.transfer_ratio, 6),
            "n_random": len(result.random_rates),
            "n_prompts": result.n_prompts,
        }
        for result in results.values()
    ]
    return sorted(rows, key=lambda row: float(row["transfer_ratio"]), reverse=True)
