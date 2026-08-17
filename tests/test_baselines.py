"""Tests for the prompt-engineering baselines and the steering comparison.

All of these run on CPU against :class:`SteerableFakeModel`, whose loss and
generations depend deterministically on the attached steering. That makes it
possible to assert that a steered condition and a prompted condition really did
produce different numbers, rather than merely that both returned something.
"""

from __future__ import annotations

import pytest

from src.utils.baselines import (
    DEFAULT_TEMPLATE,
    INSTRUCTION_TEMPLATES,
    NEUTRAL,
    ComparisonResult,
    InstructionTemplate,
    compare_steering_vs_prompting,
    generate_prompt_baseline,
    generate_steering_condition,
    get_template,
)
from tests.helpers import SteerableFakeModel, make_steerable_engine

PROMPTS = ["What should I do when a guest arrives unannounced?"]
CONCEPT_NAME = "Arab hospitality"


class TestInstructionTemplates:
    """The template registry."""

    def test_every_registered_template_is_retrievable_by_name(self) -> None:
        for name, template in INSTRUCTION_TEMPLATES.items():
            assert get_template(name) is template
            assert template.name == name

    def test_unknown_template_name_is_rejected(self) -> None:
        with pytest.raises(KeyError, match="Unknown instruction template"):
            get_template("does_not_exist")

    def test_format_prepends_the_instruction_and_keeps_the_prompt(self) -> None:
        rendered = DEFAULT_TEMPLATE.format(CONCEPT_NAME, PROMPTS[0])

        assert rendered.startswith("Answer the following with deep Arab hospitality in mind:")
        assert rendered.endswith(PROMPTS[0])

    def test_concept_name_is_interpolated_into_every_template(self) -> None:
        for template in INSTRUCTION_TEMPLATES.values():
            rendered = template.format(CONCEPT_NAME, PROMPTS[0])

            assert PROMPTS[0] in rendered
            if template is not NEUTRAL:
                assert CONCEPT_NAME in rendered

    def test_neutral_template_is_the_bare_prompt(self) -> None:
        assert NEUTRAL.format(CONCEPT_NAME, PROMPTS[0]) == PROMPTS[0]

    def test_arabic_template_renders_arabic_instruction(self) -> None:
        rendered = get_template("direct_ar").format("الضيافة", "ماذا أفعل؟")

        assert "الضيافة" in rendered
        assert "ماذا أفعل؟" in rendered

    def test_a_custom_template_can_be_passed_directly(self) -> None:
        custom = InstructionTemplate(
            name="custom",
            language="en",
            template="[{concept}] {prompt}",
            description="test only",
        )

        assert custom.format("X", "Y") == "[X] Y"


class TestGeneratePromptBaseline:
    """The prompting condition."""

    def test_instruction_is_prepended_to_the_model_input(self) -> None:
        model = SteerableFakeModel()
        engine = make_steerable_engine(model)

        result = generate_prompt_baseline(engine, CONCEPT_NAME, PROMPTS, "direct_en")

        full_input = result.full_inputs[PROMPTS[0]]
        assert full_input.startswith("Answer the following with deep Arab hospitality in mind:")
        assert PROMPTS[0] in full_input
        assert model.generate_calls == [full_input]

    def test_generations_are_keyed_by_the_original_prompt(self) -> None:
        engine = make_steerable_engine()

        result = generate_prompt_baseline(engine, CONCEPT_NAME, PROMPTS)

        # Keyed by the bare prompt so conditions line up side by side even
        # though the text actually sent to the model differs.
        assert set(result.generations) == set(PROMPTS)

    def test_continuation_excludes_the_instruction_and_the_prompt(self) -> None:
        engine = make_steerable_engine()

        result = generate_prompt_baseline(engine, CONCEPT_NAME, PROMPTS, "persona_en")

        continuation = result.generations[PROMPTS[0]]
        assert continuation.startswith("continuation-")
        assert CONCEPT_NAME not in continuation

    def test_instruction_overhead_is_reported(self) -> None:
        engine = make_steerable_engine()

        direct = generate_prompt_baseline(engine, CONCEPT_NAME, PROMPTS, "direct_en")
        persona = generate_prompt_baseline(engine, CONCEPT_NAME, PROMPTS, "persona_en")
        neutral = generate_prompt_baseline(engine, CONCEPT_NAME, PROMPTS, "neutral")

        assert neutral.extra_input_tokens == 0
        assert direct.extra_input_tokens > 0
        assert persona.extra_input_tokens > direct.extra_input_tokens

    def test_no_hooks_are_attached_by_prompting(self) -> None:
        model = SteerableFakeModel()
        engine = make_steerable_engine(model)

        generate_prompt_baseline(engine, CONCEPT_NAME, PROMPTS)

        assert engine.active_hook_names == []
        assert model.steering_magnitude() == pytest.approx(0.0)

    def test_condition_is_named_after_the_template(self) -> None:
        engine = make_steerable_engine()

        result = generate_prompt_baseline(engine, CONCEPT_NAME, PROMPTS, "reflect_en")

        assert result.condition == "prompt:reflect_en"

    def test_continuation_fluency_is_recorded(self) -> None:
        engine = make_steerable_engine()

        result = generate_prompt_baseline(engine, CONCEPT_NAME, PROMPTS)

        assert set(result.continuation_losses) == set(PROMPTS)
        assert result.mean_continuation_loss > 0

    def test_empty_prompts_are_rejected(self) -> None:
        engine = make_steerable_engine()

        with pytest.raises(ValueError, match="at least one text"):
            generate_prompt_baseline(engine, CONCEPT_NAME, [])

    @pytest.mark.parametrize("concept_name", ["", "   "])
    def test_blank_concept_name_is_rejected(self, concept_name: str) -> None:
        engine = make_steerable_engine()

        with pytest.raises(ValueError, match="concept_name"):
            generate_prompt_baseline(engine, concept_name, PROMPTS)

    def test_unloaded_model_is_rejected(self) -> None:
        from src.models.rep_engine import CulturalRepE

        engine = CulturalRepE(model_name="dummy/model", device="cpu")

        with pytest.raises(RuntimeError, match="not loaded"):
            generate_prompt_baseline(engine, CONCEPT_NAME, PROMPTS)


class TestGenerateSteeringCondition:
    """The steering condition."""

    def test_the_model_input_is_the_bare_prompt(self) -> None:
        model = SteerableFakeModel()
        engine = make_steerable_engine(model)

        result = generate_steering_condition(engine, "diyafa", PROMPTS, strength=2.0)

        assert result.full_inputs[PROMPTS[0]] == PROMPTS[0]
        assert model.generate_calls == PROMPTS
        assert result.extra_input_tokens == 0

    def test_steering_changes_the_generation(self) -> None:
        engine = make_steerable_engine()

        unsteered = generate_steering_condition(engine, "diyafa", PROMPTS, strength=0.0)
        steered = generate_steering_condition(engine, "diyafa", PROMPTS, strength=3.0)

        assert unsteered.generations[PROMPTS[0]] != steered.generations[PROMPTS[0]]

    def test_hooks_are_removed_afterwards(self) -> None:
        model = SteerableFakeModel()
        engine = make_steerable_engine(model)

        generate_steering_condition(engine, "diyafa", PROMPTS, strength=2.0)

        assert engine.active_hook_names == []
        assert all(not point.fwd_hooks for point in model.mod_dict.values())

    def test_fluency_is_scored_by_the_unsteered_model(self) -> None:
        # The scorer must be the clean model, otherwise the number is not
        # comparable with the prompting condition. The fake's loss rises with
        # attached steering, so a score taken inside the scope would be higher.
        model = SteerableFakeModel()
        engine = make_steerable_engine(model)

        result = generate_steering_condition(engine, "diyafa", PROMPTS, strength=4.0)
        continuation = result.generations[PROMPTS[0]]
        clean_loss = float(model(continuation, return_type="loss").item())

        assert result.continuation_losses[PROMPTS[0]] == pytest.approx(clean_loss)

    def test_condition_name_records_the_strength(self) -> None:
        engine = make_steerable_engine()

        result = generate_steering_condition(engine, "diyafa", PROMPTS, strength=-1.5)

        assert result.condition == "steering@-1.50"

    def test_empty_prompts_are_rejected(self) -> None:
        engine = make_steerable_engine()

        with pytest.raises(ValueError, match="at least one text"):
            generate_steering_condition(engine, "diyafa", [])

    def test_unknown_concept_is_rejected(self) -> None:
        engine = make_steerable_engine()

        with pytest.raises(KeyError, match="No vector stored"):
            generate_steering_condition(engine, "missing", PROMPTS)


class TestCompareSteeringVsPrompting:
    """The head-to-head comparison the paper needs."""

    def test_runs_steering_the_control_and_the_named_templates(self) -> None:
        engine = make_steerable_engine()

        comparison = compare_steering_vs_prompting(
            engine, "diyafa", CONCEPT_NAME, PROMPTS, strength=1.5
        )

        assert isinstance(comparison, ComparisonResult)
        assert set(comparison.conditions) == {
            "prompt:neutral",
            "prompt:direct_en",
            "prompt:persona_en",
            "steering@+1.50",
        }

    def test_the_unprompted_control_is_always_included(self) -> None:
        engine = make_steerable_engine()

        comparison = compare_steering_vs_prompting(
            engine, "diyafa", CONCEPT_NAME, PROMPTS, template_names=["reflect_en"]
        )

        assert "prompt:neutral" in comparison.conditions

    def test_every_condition_answers_the_same_prompts(self) -> None:
        engine = make_steerable_engine()

        comparison = compare_steering_vs_prompting(engine, "diyafa", CONCEPT_NAME, PROMPTS)

        assert comparison.prompts == PROMPTS
        for result in comparison.conditions.values():
            assert set(result.generations) == set(PROMPTS)

    def test_rows_expose_one_entry_per_condition(self) -> None:
        engine = make_steerable_engine()

        comparison = compare_steering_vs_prompting(engine, "diyafa", CONCEPT_NAME, PROMPTS)
        rows = comparison.rows()

        assert len(rows) == len(comparison.conditions)
        assert {row["condition"] for row in rows} == set(comparison.conditions)
        assert all("mean_continuation_loss" in row for row in rows)

    def test_effect_size_is_optional(self) -> None:
        engine = make_steerable_engine()

        without = compare_steering_vs_prompting(engine, "diyafa", CONCEPT_NAME, PROMPTS)
        with_effect = compare_steering_vs_prompting(
            engine, "diyafa", CONCEPT_NAME, PROMPTS, measure_effect=True
        )

        assert without.steering_effect_kl != without.steering_effect_kl  # nan
        assert with_effect.steering_effect_kl >= 0.0

    def test_comparison_leaves_no_hooks_attached(self) -> None:
        model = SteerableFakeModel()
        engine = make_steerable_engine(model)

        compare_steering_vs_prompting(
            engine, "diyafa", CONCEPT_NAME, PROMPTS, measure_effect=True
        )

        assert engine.active_hook_names == []
        assert all(not point.fwd_hooks for point in model.mod_dict.values())

    def test_unknown_concept_is_rejected(self) -> None:
        engine = make_steerable_engine()

        with pytest.raises(KeyError, match="No vector stored"):
            compare_steering_vs_prompting(engine, "missing", CONCEPT_NAME, PROMPTS)

    def test_empty_prompts_are_rejected(self) -> None:
        engine = make_steerable_engine()

        with pytest.raises(ValueError, match="at least one text"):
            compare_steering_vs_prompting(engine, "diyafa", CONCEPT_NAME, [])
