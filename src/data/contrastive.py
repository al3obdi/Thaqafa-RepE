"""Baseline (neutral) prompts used as the negative side of the contrast.

The contrastive mean-difference recipe needs two sets of prompts per concept:

* **Positive** prompts, in which the cultural concept is present. These come
  from the ``examples_ar`` / ``examples_en`` fields of the concept dataset.
* **Negative** prompts, in which the concept is absent. Ideally these are
  hand-written minimal pairs, but the seed dataset does not carry them yet.

Until curated negatives exist, this module supplies a fixed bank of culturally
neutral, everyday sentences in Arabic and English. Subtracting their mean
activation removes the components of the residual stream that merely encode
"this is an ordinary sentence in language X", leaving the concept-specific
direction behind.

The selection is deterministic on purpose: two runs with the same arguments
produce the same baseline, so extracted vectors stay reproducible.
"""

from __future__ import annotations

import logging
from itertools import zip_longest

logger = logging.getLogger(__name__)

Language = str
"""Language selector: ``"ar"``, ``"en"`` or ``"both"``."""

NEUTRAL_PROMPTS_EN: tuple[str, ...] = (
    "The weather is nice today.",
    "I went to the store to buy some bread.",
    "The train leaves the station at eight in the morning.",
    "She opened the window because the room was warm.",
    "There are four chairs around the wooden table.",
    "He read the instructions before starting the machine.",
    "The book was returned to the library last week.",
    "Water boils at one hundred degrees Celsius.",
    "They walked along the river for about an hour.",
    "The meeting was moved to a different room.",
    "I need to charge my phone before we leave.",
    "The garden needs watering twice a week.",
)

NEUTRAL_PROMPTS_AR: tuple[str, ...] = (
    "الطقس جميل اليوم.",
    "ذهبت إلى المتجر لشراء بعض الخبز.",
    "يغادر القطار المحطة في الثامنة صباحًا.",
    "فتحت النافذة لأن الغرفة كانت دافئة.",
    "هناك أربعة كراسي حول الطاولة الخشبية.",
    "قرأ التعليمات قبل تشغيل الآلة.",
    "أُعيد الكتاب إلى المكتبة الأسبوع الماضي.",
    "يغلي الماء عند مئة درجة مئوية.",
    "مشوا بمحاذاة النهر لمدة ساعة تقريبًا.",
    "نُقل الاجتماع إلى غرفة أخرى.",
    "أحتاج إلى شحن هاتفي قبل أن نغادر.",
    "تحتاج الحديقة إلى الري مرتين في الأسبوع.",
)

VALID_LANGUAGES: frozenset[str] = frozenset({"ar", "en", "both"})


def _interleave(first: tuple[str, ...], second: tuple[str, ...]) -> list[str]:
    """Interleave two sequences, keeping every element.

    Args:
        first: Sequence contributing the even positions.
        second: Sequence contributing the odd positions.

    Returns:
        The interleaved list. When the inputs differ in length the surplus
        elements are appended in order.
    """
    merged: list[str] = []
    for left, right in zip_longest(first, second):
        if left is not None:
            merged.append(left)
        if right is not None:
            merged.append(right)
    return merged


def neutral_prompt_bank(language: Language = "both") -> list[str]:
    """Return the full bank of neutral prompts for ``language``.

    Args:
        language: ``"ar"`` for Arabic only, ``"en"`` for English only, or
            ``"both"`` to interleave the two banks so that a truncated
            selection stays balanced across languages.

    Returns:
        The neutral prompts, in a deterministic order.

    Raises:
        ValueError: If ``language`` is not one of ``"ar"``, ``"en"`` or
            ``"both"``.
    """
    if language not in VALID_LANGUAGES:
        raise ValueError(f"language must be one of {sorted(VALID_LANGUAGES)}, got {language!r}")

    if language == "en":
        return list(NEUTRAL_PROMPTS_EN)
    if language == "ar":
        return list(NEUTRAL_PROMPTS_AR)
    return _interleave(NEUTRAL_PROMPTS_AR, NEUTRAL_PROMPTS_EN)


def build_neutral_examples(
    n_examples: int | None = None,
    language: Language = "both",
) -> list[str]:
    """Build a deterministic set of neutral baseline prompts.

    Args:
        n_examples: How many prompts to return. ``None`` returns the whole
            bank. If more prompts are requested than the bank holds, the bank
            is repeated until the count is met.
        language: Which language bank to draw from. See
            :func:`neutral_prompt_bank`.

    Returns:
        Exactly ``n_examples`` prompts (or the whole bank when ``n_examples``
        is ``None``).

    Raises:
        ValueError: If ``n_examples`` is not positive, or ``language`` is
            invalid.

    Example:
        >>> build_neutral_examples(2, language="en")
        ['The weather is nice today.', 'I went to the store to buy some bread.']
    """
    bank = neutral_prompt_bank(language)

    if n_examples is None:
        return bank
    if n_examples <= 0:
        raise ValueError(f"n_examples must be positive, got {n_examples}")

    if n_examples > len(bank):
        logger.debug(
            "Requested %d neutral prompts but the %s bank holds %d; cycling the bank.",
            n_examples,
            language,
            len(bank),
        )
        repeats = -(-n_examples // len(bank))  # ceiling division
        bank = (bank * repeats)[:n_examples]
        return bank

    return bank[:n_examples]


def build_contrast_examples(
    positive_examples: list[str],
    contrast_examples: list[str] | None = None,
    language: Language = "both",
) -> list[str]:
    """Resolve the negative side of a contrastive pair.

    Args:
        positive_examples: The prompts that express the concept. Only their
            count is used, so that the two sides of the contrast are balanced.
        contrast_examples: Curated negatives. Returned unchanged when
            provided and non-empty.
        language: Language bank used when negatives must be generated.

    Returns:
        The negative prompts to subtract from the positive mean.

    Raises:
        ValueError: If ``positive_examples`` is empty.
    """
    if not positive_examples:
        raise ValueError("positive_examples must contain at least one prompt")

    if contrast_examples:
        return list(contrast_examples)

    logger.debug(
        "No contrast examples supplied; falling back to %d generated neutral prompts.",
        len(positive_examples),
    )
    return build_neutral_examples(n_examples=len(positive_examples), language=language)
