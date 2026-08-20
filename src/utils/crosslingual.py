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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from src.data.dataset_builder import CulturalConcept, load_concepts

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
