"""Tests for the neutral baseline prompt bank."""

from __future__ import annotations

import pytest

from src.data.contrastive import (
    NEUTRAL_PROMPTS_AR,
    NEUTRAL_PROMPTS_EN,
    build_contrast_examples,
    build_neutral_examples,
    neutral_prompt_bank,
)


class TestNeutralPromptBank:
    """Language selection and ordering of the bank."""

    def test_single_language_banks(self) -> None:
        assert neutral_prompt_bank("en") == list(NEUTRAL_PROMPTS_EN)
        assert neutral_prompt_bank("ar") == list(NEUTRAL_PROMPTS_AR)

    def test_both_interleaves_the_two_banks(self) -> None:
        bank = neutral_prompt_bank("both")

        assert len(bank) == len(NEUTRAL_PROMPTS_AR) + len(NEUTRAL_PROMPTS_EN)
        assert bank[0] == NEUTRAL_PROMPTS_AR[0]
        assert bank[1] == NEUTRAL_PROMPTS_EN[0]

    def test_invalid_language_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="language must be one of"):
            neutral_prompt_bank("fr")


class TestBuildNeutralExamples:
    """Deterministic selection of baseline prompts."""

    def test_returns_the_requested_count(self) -> None:
        assert len(build_neutral_examples(5)) == 5

    def test_selection_is_deterministic(self) -> None:
        assert build_neutral_examples(6) == build_neutral_examples(6)

    def test_truncated_selection_stays_language_balanced(self) -> None:
        selected = build_neutral_examples(4, language="both")

        assert selected[0] in NEUTRAL_PROMPTS_AR
        assert selected[1] in NEUTRAL_PROMPTS_EN
        assert sum(prompt in NEUTRAL_PROMPTS_AR for prompt in selected) == 2

    def test_none_returns_the_whole_bank(self) -> None:
        assert build_neutral_examples(None, language="en") == list(NEUTRAL_PROMPTS_EN)

    def test_requesting_more_than_the_bank_cycles_it(self) -> None:
        bank_size = len(NEUTRAL_PROMPTS_EN)
        selected = build_neutral_examples(bank_size + 3, language="en")

        assert len(selected) == bank_size + 3
        assert selected[bank_size] == NEUTRAL_PROMPTS_EN[0]

    @pytest.mark.parametrize("count", [0, -1])
    def test_non_positive_count_is_rejected(self, count: int) -> None:
        with pytest.raises(ValueError, match="n_examples must be positive"):
            build_neutral_examples(count)


class TestBuildContrastExamples:
    """Resolution of the negative side of a contrastive pair."""

    def test_curated_negatives_pass_through(self) -> None:
        negatives = ["a neutral sentence", "another one"]

        assert build_contrast_examples(["positive"], negatives) == negatives

    def test_generated_negatives_match_the_positive_count(self) -> None:
        positives = ["one", "two", "three"]

        generated = build_contrast_examples(positives)

        assert len(generated) == len(positives)

    def test_empty_negatives_fall_back_to_the_bank(self) -> None:
        generated = build_contrast_examples(["one"], [])

        assert generated == build_neutral_examples(1)

    def test_empty_positives_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one prompt"):
            build_contrast_examples([])
