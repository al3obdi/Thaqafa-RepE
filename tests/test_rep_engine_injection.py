"""Tests for vector injection, hook lifecycle and the steering context manager.

The suite runs on CPU in two tiers, mirroring ``test_rep_engine_extraction.py``:

* **Unit tests** use :class:`FakeHookedModel`, which reproduces the part of the
  TransformerLens contract the engine depends on - ``mod_dict``, ``add_hook``
  appending a handle to the hook point's ``fwd_hooks`` list, and a removable
  PyTorch handle. That makes it possible to assert on the hook lifecycle
  precisely, including cases a real model makes awkward to set up, such as an
  unrelated third-party hook that must survive cleanup.
* **Integration tests** attach real hooks to a genuinely tiny
  ``HookedTransformer`` and check the residual stream numerically: the
  activation must shift by exactly ``strength * vector`` and return bit-for-bit
  to its unsteered value afterwards. That is the assertion the fake cannot make
  on its own behalf.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from src.models.rep_engine import RESID_POST_HOOK, CulturalRepE, InjectionHandle
from src.utils.evaluation import (
    SteeringResult,
    evaluate_steering,
    summarize_sweep,
)

TINY_TOKENIZER_NAME = "sshleifer/tiny-gpt2"
DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "datasets" / "cultural_concepts.jsonl"

D_MODEL = 4
N_LAYERS = 6


# ---------------------------------------------------------------------------
# A stand-in that honours the parts of the TransformerLens hook contract we use
# ---------------------------------------------------------------------------


class FakeRemovableHandle:
    """Stands in for ``torch.utils.hooks.RemovableHandle``."""

    def __init__(self) -> None:
        self.remove_count = 0

    def remove(self) -> None:
        """Record a removal. Idempotent, like the real handle."""
        self.remove_count += 1


class FakeLensHandle:
    """Stands in for ``transformer_lens.hook_points.LensHandle``."""

    def __init__(self, user_hook: Any) -> None:
        self.hook = FakeRemovableHandle()
        self.user_hook = user_hook
        self.is_permanent = False


class FakeHookPoint:
    """Stands in for ``transformer_lens.hook_points.HookPoint``."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.fwd_hooks: list[FakeLensHandle] = []


class FakeConfig:
    """Minimal config exposing the two fields injection reads."""

    def __init__(self, n_layers: int, d_model: int) -> None:
        self.n_layers = n_layers
        self.d_model = d_model


class FakeHookedModel:
    """A model that registers hooks the way TransformerLens does.

    ``add_hook`` returns ``None`` and appends the handle it created to the hook
    point, which is exactly the behaviour the engine has to work around.
    ``__call__`` and ``generate`` return values that depend on the attached
    hooks, so the evaluation sweep can be tested without a real forward pass.
    """

    def __init__(self, n_layers: int = N_LAYERS, d_model: int = D_MODEL) -> None:
        self.cfg = FakeConfig(n_layers, d_model)
        self.tokenizer = None
        self.mod_dict: dict[str, FakeHookPoint] = {
            RESID_POST_HOOK.format(layer=layer): FakeHookPoint(RESID_POST_HOOK.format(layer=layer))
            for layer in range(n_layers)
        }
        self.add_hook_calls: list[tuple[str, str]] = []

    def eval(self) -> FakeHookedModel:
        """Match the ``torch.nn.Module`` interface."""
        return self

    def add_hook(self, name: str, hook: Any, dir: str = "fwd") -> None:  # noqa: A002
        """Register a hook and append its handle to the hook point."""
        self.add_hook_calls.append((name, dir))
        self.mod_dict[name].fwd_hooks.append(FakeLensHandle(hook))

    def total_abs_strength(self) -> float:
        """Sum of the offsets every attached hook would apply to a unit input."""
        total = 0.0
        for hook_point in self.mod_dict.values():
            for handle in hook_point.fwd_hooks:
                probe = torch.zeros(1, 1, self.cfg.d_model)
                shifted = handle.user_hook(probe, None)
                total += float(shifted.abs().max().item())
        return total

    def __call__(self, prompt: str, return_type: str = "loss") -> torch.Tensor:
        """Return a deterministic loss that rises with the attached steering."""
        return torch.tensor(2.0 + 0.5 * self.total_abs_strength())

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Return a deterministic continuation that reflects the steering."""
        return f"{prompt} [steered by {self.total_abs_strength():.2f}]"


def make_engine(model: FakeHookedModel | None = None) -> CulturalRepE:
    """Build an engine wired to a fake model with one cached concept vector.

    Args:
        model: Stand-in model to attach. A default one is built when omitted.

    Returns:
        An engine holding a unit vector under the concept ``"diyafa"``.
    """
    engine = CulturalRepE(
        model_name="dummy/model",
        device="cpu",
        dtype="float32",
        dataset_path=DATASET_PATH,
    )
    attached = model if model is not None else FakeHookedModel()
    engine.model = attached  # type: ignore[assignment]
    engine.tokenizer = attached.tokenizer
    engine.concept_vectors["diyafa"] = torch.ones(D_MODEL)
    engine.extraction_layers["diyafa"] = 2
    return engine


# ---------------------------------------------------------------------------
# inject_vector
# ---------------------------------------------------------------------------


class TestInjectVector:
    """Attaching steering hooks."""

    def test_hook_is_added_to_the_extraction_layer_by_default(self) -> None:
        model = FakeHookedModel()
        engine = make_engine(model)

        handles = engine.inject_vector("diyafa", strength=1.5)

        assert len(handles) == 1
        assert engine.active_hook_names == ["blocks.2.hook_resid_post"]
        assert len(model.mod_dict["blocks.2.hook_resid_post"].fwd_hooks) == 1
        assert model.add_hook_calls == [("blocks.2.hook_resid_post", "fwd")]

    def test_default_layer_falls_back_to_the_middle(self) -> None:
        engine = make_engine()
        del engine.extraction_layers["diyafa"]

        engine.inject_vector("diyafa")

        assert engine.active_hook_names == [f"blocks.{N_LAYERS // 2}.hook_resid_post"]

    def test_multiple_layers_each_get_a_hook(self) -> None:
        model = FakeHookedModel()
        engine = make_engine(model)

        handles = engine.inject_vector("diyafa", strength=1.0, layers=[0, 3, 5])

        assert [handle.layer for handle in handles] == [0, 3, 5]
        assert engine.active_hook_names == [
            "blocks.0.hook_resid_post",
            "blocks.3.hook_resid_post",
            "blocks.5.hook_resid_post",
        ]

    def test_duplicate_layers_are_collapsed_and_sorted(self) -> None:
        engine = make_engine()

        handles = engine.inject_vector("diyafa", layers=[4, 1, 4, 1])

        assert [handle.layer for handle in handles] == [1, 4]

    def test_negative_layer_indices_are_resolved(self) -> None:
        engine = make_engine()

        handles = engine.inject_vector("diyafa", layers=[-1])

        assert handles[0].layer == N_LAYERS - 1

    def test_handle_records_the_injection_metadata(self) -> None:
        engine = make_engine()

        handle = engine.inject_vector("diyafa", strength=-0.75, layers=[3])[0]

        assert handle.concept == "diyafa"
        assert handle.layer == 3
        assert handle.hook_name == "blocks.3.hook_resid_post"
        assert handle.strength == pytest.approx(-0.75)
        assert handle.is_active

    def test_handles_are_tracked_on_the_engine(self) -> None:
        engine = make_engine()

        first = engine.inject_vector("diyafa", layers=[0])
        second = engine.inject_vector("diyafa", layers=[1])

        assert engine._active_hooks == [*first, *second]

    def test_unknown_concept_is_rejected(self) -> None:
        engine = make_engine()

        with pytest.raises(KeyError, match="No vector stored"):
            engine.inject_vector("not_extracted")

    def test_empty_layer_list_is_rejected(self) -> None:
        engine = make_engine()

        with pytest.raises(ValueError, match="at least one layer"):
            engine.inject_vector("diyafa", layers=[])

    def test_out_of_range_layer_is_rejected(self) -> None:
        engine = make_engine()

        with pytest.raises(IndexError, match="out of range"):
            engine.inject_vector("diyafa", layers=[N_LAYERS])

    def test_multi_dimensional_vector_is_rejected(self) -> None:
        engine = make_engine()
        engine.concept_vectors["bad"] = torch.ones(2, D_MODEL)

        with pytest.raises(ValueError, match="must be 1-D"):
            engine.inject_vector("bad")

    def test_vector_width_must_match_d_model(self) -> None:
        engine = make_engine()
        engine.concept_vectors["bad"] = torch.ones(D_MODEL + 1)

        with pytest.raises(ValueError, match="d_model"):
            engine.inject_vector("bad")

    def test_nothing_is_attached_when_validation_fails(self) -> None:
        model = FakeHookedModel()
        engine = make_engine(model)
        engine.concept_vectors["bad"] = torch.ones(D_MODEL + 1)

        with pytest.raises(ValueError):
            engine.inject_vector("bad")

        assert engine.active_hook_names == []
        assert model.add_hook_calls == []


# ---------------------------------------------------------------------------
# The hook function itself
# ---------------------------------------------------------------------------


class TestHookBroadcasting:
    """The offset a hook applies to an activation."""

    def _hook_for(self, engine: CulturalRepE, strength: float) -> Any:
        """Attach a hook and return the callable the engine registered."""
        model = engine.model
        assert model is not None
        handle = engine.inject_vector("diyafa", strength=strength, layers=[0])[0]
        return model.mod_dict[handle.hook_name].fwd_hooks[-1].user_hook  # type: ignore[union-attr]

    def test_one_dimensional_vector_broadcasts_over_batch_and_sequence(self) -> None:
        engine = make_engine()
        hook = self._hook_for(engine, strength=2.0)
        activation = torch.zeros(3, 5, D_MODEL)

        result = hook(activation, None)

        assert result.shape == (3, 5, D_MODEL)
        assert torch.allclose(result, torch.full((3, 5, D_MODEL), 2.0))

    def test_offset_is_added_not_replaced(self) -> None:
        engine = make_engine()
        hook = self._hook_for(engine, strength=1.0)
        activation = torch.arange(2 * 3 * D_MODEL, dtype=torch.float32).reshape(2, 3, D_MODEL)

        result = hook(activation, None)

        assert torch.allclose(result, activation + 1.0)

    def test_zero_strength_is_a_no_op(self) -> None:
        engine = make_engine()
        hook = self._hook_for(engine, strength=0.0)
        activation = torch.randn(2, 4, D_MODEL)

        assert torch.equal(hook(activation, None), activation)

    def test_negative_strength_subtracts_the_direction(self) -> None:
        engine = make_engine()
        hook = self._hook_for(engine, strength=-3.0)
        activation = torch.zeros(1, 2, D_MODEL)

        assert torch.allclose(hook(activation, None), torch.full((1, 2, D_MODEL), -3.0))

    def test_offset_adopts_the_activation_dtype(self) -> None:
        # Vectors are cached in float32; a bfloat16 model must not be upcast.
        engine = make_engine()
        hook = self._hook_for(engine, strength=1.0)
        activation = torch.zeros(1, 2, D_MODEL, dtype=torch.bfloat16)

        result = hook(activation, None)

        assert result.dtype is torch.bfloat16

    def test_input_activation_is_not_mutated(self) -> None:
        engine = make_engine()
        hook = self._hook_for(engine, strength=5.0)
        activation = torch.zeros(1, 2, D_MODEL)
        original = activation.clone()

        hook(activation, None)

        assert torch.equal(activation, original)


# ---------------------------------------------------------------------------
# remove_hooks
# ---------------------------------------------------------------------------


class TestRemoveHooks:
    """Detaching steering hooks."""

    def test_removes_every_hook_and_clears_the_list(self) -> None:
        model = FakeHookedModel()
        engine = make_engine(model)
        engine.inject_vector("diyafa", layers=[0, 1, 2])

        removed = engine.remove_hooks()

        assert removed == 3
        assert engine.active_hook_names == []
        assert engine._active_hooks == []
        assert all(not point.fwd_hooks for point in model.mod_dict.values())

    def test_removing_with_nothing_attached_is_safe(self) -> None:
        engine = make_engine()

        assert engine.remove_hooks() == 0

    def test_calling_twice_is_safe(self) -> None:
        engine = make_engine()
        engine.inject_vector("diyafa", layers=[0])

        assert engine.remove_hooks() == 1
        assert engine.remove_hooks() == 0

    def test_a_subset_can_be_removed(self) -> None:
        model = FakeHookedModel()
        engine = make_engine(model)
        first = engine.inject_vector("diyafa", layers=[0])
        engine.inject_vector("diyafa", layers=[4])

        removed = engine.remove_hooks(first)

        assert removed == 1
        assert engine.active_hook_names == ["blocks.4.hook_resid_post"]
        assert not model.mod_dict["blocks.0.hook_resid_post"].fwd_hooks
        assert len(model.mod_dict["blocks.4.hook_resid_post"].fwd_hooks) == 1

    def test_handle_reports_itself_inactive_once_removed(self) -> None:
        engine = make_engine()
        handle = engine.inject_vector("diyafa", layers=[0])[0]

        engine.remove_hooks()

        assert not handle.is_active

    def test_handle_remove_is_idempotent(self) -> None:
        model = FakeHookedModel()
        engine = make_engine(model)
        handle = engine.inject_vector("diyafa", layers=[0])[0]
        lens_handle = model.mod_dict["blocks.0.hook_resid_post"].fwd_hooks[0]

        handle.remove()
        handle.remove()

        assert lens_handle.hook.remove_count == 1

    def test_unrelated_hooks_are_left_alone(self) -> None:
        # A caller's own caching or ablation hook must survive our cleanup.
        model = FakeHookedModel()
        engine = make_engine(model)
        model.add_hook("blocks.2.hook_resid_post", lambda act, hook: act)
        engine.inject_vector("diyafa", layers=[2])

        engine.remove_hooks()

        assert len(model.mod_dict["blocks.2.hook_resid_post"].fwd_hooks) == 1


# ---------------------------------------------------------------------------
# steering context manager
# ---------------------------------------------------------------------------


class TestSteeringContextManager:
    """Scoped steering."""

    def test_hook_is_present_inside_and_absent_after(self) -> None:
        model = FakeHookedModel()
        engine = make_engine(model)

        with engine.steering("diyafa", strength=2.0):
            assert engine.active_hook_names == ["blocks.2.hook_resid_post"]
            assert len(model.mod_dict["blocks.2.hook_resid_post"].fwd_hooks) == 1

        assert engine.active_hook_names == []
        assert not model.mod_dict["blocks.2.hook_resid_post"].fwd_hooks

    def test_handles_are_yielded(self) -> None:
        engine = make_engine()

        with engine.steering("diyafa", layers=[1, 3]) as handles:
            assert [handle.layer for handle in handles] == [1, 3]
            assert all(isinstance(handle, InjectionHandle) for handle in handles)

    def test_hooks_are_removed_when_the_body_raises(self) -> None:
        model = FakeHookedModel()
        engine = make_engine(model)

        with pytest.raises(RuntimeError, match="boom"), engine.steering("diyafa"):
            raise RuntimeError("boom")

        assert engine.active_hook_names == []
        assert not model.mod_dict["blocks.2.hook_resid_post"].fwd_hooks

    def test_nested_scopes_unwind_independently(self) -> None:
        engine = make_engine()

        with engine.steering("diyafa", strength=1.0, layers=[0]):
            with engine.steering("diyafa", strength=-1.0, layers=[5]):
                assert engine.active_hook_names == [
                    "blocks.0.hook_resid_post",
                    "blocks.5.hook_resid_post",
                ]
            assert engine.active_hook_names == ["blocks.0.hook_resid_post"]

        assert engine.active_hook_names == []

    def test_hooks_attached_outside_the_scope_survive_it(self) -> None:
        engine = make_engine()
        outer = engine.inject_vector("diyafa", layers=[0])

        with engine.steering("diyafa", layers=[4]):
            assert len(engine.active_hook_names) == 2

        assert engine.active_hook_names == ["blocks.0.hook_resid_post"]
        assert outer[0].is_active

    def test_unknown_concept_raises_before_entering(self) -> None:
        engine = make_engine()

        with pytest.raises(KeyError, match="No vector stored"), engine.steering("nope"):
            pytest.fail("the context body must not run")

        assert engine.active_hook_names == []


# ---------------------------------------------------------------------------
# evaluate_steering
# ---------------------------------------------------------------------------


class TestEvaluateSteering:
    """The strength sweep."""

    def test_returns_one_result_per_strength(self) -> None:
        engine = make_engine()

        results = evaluate_steering(
            engine, "diyafa", ["a prompt"], strengths=[-1.0, 0.0, 1.0], max_new_tokens=2
        )

        assert list(results) == [-1.0, 0.0, 1.0]
        assert all(isinstance(result, SteeringResult) for result in results.values())

    def test_sweep_leaves_no_hooks_attached(self) -> None:
        model = FakeHookedModel()
        engine = make_engine(model)

        evaluate_steering(engine, "diyafa", ["a prompt"], strengths=[1.0, 2.0])

        assert engine.active_hook_names == []
        assert all(not point.fwd_hooks for point in model.mod_dict.values())

    def test_records_loss_and_perplexity_per_prompt(self) -> None:
        engine = make_engine()
        prompts = ["first prompt", "second prompt"]

        results = evaluate_steering(engine, "diyafa", prompts, strengths=[1.0], generate=False)
        result = results[1.0]

        assert set(result.prompt_losses) == set(prompts)
        assert result.mean_loss == pytest.approx(2.5)  # 2.0 + 0.5 * strength 1.0
        assert result.perplexity == pytest.approx(pow(2.718281828459045, 2.5), rel=1e-6)

    def test_zero_strength_reproduces_the_unsteered_loss(self) -> None:
        engine = make_engine()

        results = evaluate_steering(engine, "diyafa", ["a prompt"], strengths=[0.0], generate=False)

        assert results[0.0].mean_loss == pytest.approx(2.0)

    def test_generation_can_be_disabled(self) -> None:
        engine = make_engine()

        results = evaluate_steering(engine, "diyafa", ["a prompt"], strengths=[1.0], generate=False)

        assert results[1.0].generations == {}

    def test_generations_are_recorded_per_prompt(self) -> None:
        engine = make_engine()

        results = evaluate_steering(engine, "diyafa", ["a prompt"], strengths=[2.0])

        assert "a prompt" in results[2.0].generations
        assert results[2.0].generations["a prompt"].startswith("a prompt")

    def test_empty_prompts_are_rejected(self) -> None:
        engine = make_engine()

        with pytest.raises(ValueError, match="at least one text"):
            evaluate_steering(engine, "diyafa", [])

    def test_empty_strengths_are_rejected(self) -> None:
        engine = make_engine()

        with pytest.raises(ValueError, match="at least one value"):
            evaluate_steering(engine, "diyafa", ["a prompt"], strengths=[])

    def test_unknown_concept_is_rejected(self) -> None:
        engine = make_engine()

        with pytest.raises(KeyError, match="No vector stored"):
            evaluate_steering(engine, "missing", ["a prompt"])

    def test_unloaded_model_is_rejected(self) -> None:
        engine = CulturalRepE(model_name="dummy/model", device="cpu")
        engine.concept_vectors["diyafa"] = torch.ones(D_MODEL)

        with pytest.raises(RuntimeError, match="not loaded"):
            evaluate_steering(engine, "diyafa", ["a prompt"])

    def test_summarize_sweep_sorts_by_strength(self) -> None:
        engine = make_engine()
        results = evaluate_steering(
            engine, "diyafa", ["a prompt"], strengths=[2.0, -2.0, 0.0], generate=False
        )

        summary = summarize_sweep(results)

        assert summary["strengths"] == [-2.0, 0.0, 2.0]
        assert len(summary["mean_losses"]) == 3
        assert len(summary["perplexities"]) == 3


# ---------------------------------------------------------------------------
# Integration tier: real hooks on a real (tiny) HookedTransformer
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_engine() -> CulturalRepE:
    """Build an engine backed by a randomly initialised 4-layer transformer.

    Only the ``sshleifer/tiny-gpt2`` tokenizer is fetched; the weights are
    random, which is all that is needed to check that real TransformerLens hooks
    shift the residual stream by the right amount.

    Returns:
        An engine with the tiny model attached and one vector extracted.
    """
    pytest.importorskip("transformer_lens")
    from transformer_lens import HookedTransformer, HookedTransformerConfig

    try:
        config = HookedTransformerConfig(
            n_layers=4,
            d_model=16,
            n_ctx=64,
            d_head=4,
            n_heads=4,
            d_mlp=32,
            act_fn="gelu",
            d_vocab=50257,
            tokenizer_name=TINY_TOKENIZER_NAME,
            device="cpu",
        )
        model = HookedTransformer(config)
    except Exception as exc:  # pragma: no cover - depends on network access
        pytest.skip(f"Could not build the tiny model, is the runner offline? {exc}")

    engine = CulturalRepE(
        model_name=TINY_TOKENIZER_NAME,
        device="cpu",
        dtype="float32",
        dataset_path=DATASET_PATH,
        batch_size=4,
    )
    engine.model = model
    engine.tokenizer = model.tokenizer
    engine.extract_vector("diyafa_001", layer=2)
    return engine


def _resid_post(engine: CulturalRepE, prompts: list[str], layer: int) -> torch.Tensor:
    """Read the residual stream after ``layer`` for ``prompts``.

    Args:
        engine: Engine with a loaded model.
        prompts: Texts to run.
        layer: Block to read after.

    Returns:
        Activations of shape ``(batch, seq, d_model)``.
    """
    model = engine.model
    assert model is not None
    hook_name = RESID_POST_HOOK.format(layer=layer)
    with torch.no_grad():
        _, cache = model.run_with_cache(
            model.to_tokens(prompts), names_filter=hook_name, return_type=None
        )
    return torch.as_tensor(cache[hook_name]).clone()


class TestTinyModelInjection:
    """Numerical behaviour of real hooks."""

    def test_activation_shifts_by_exactly_strength_times_vector(
        self, tiny_engine: CulturalRepE
    ) -> None:
        prompts = ["A guest arrives at the door."]
        baseline = _resid_post(tiny_engine, prompts, layer=2)
        vector = tiny_engine.concept_vectors["diyafa_001"]

        with tiny_engine.steering("diyafa_001", strength=3.0):
            steered = _resid_post(tiny_engine, prompts, layer=2)

        expected = (3.0 * vector).expand_as(baseline)
        assert torch.allclose(steered - baseline, expected, atol=1e-5)

    def test_offset_is_identical_at_every_position_and_batch_row(
        self, tiny_engine: CulturalRepE
    ) -> None:
        # Different lengths, so the batch is padded: the offset must still land
        # on every position of every row.
        prompts = ["Short one.", "A considerably longer sentence than the first one."]
        baseline = _resid_post(tiny_engine, prompts, layer=1)

        with tiny_engine.steering("diyafa_001", strength=1.0, layers=[1]):
            steered = _resid_post(tiny_engine, prompts, layer=1)

        delta = steered - baseline
        assert delta.shape == baseline.shape
        assert torch.allclose(delta, delta[0, 0].expand_as(delta), atol=1e-5)

    def test_model_returns_to_its_unsteered_state(self, tiny_engine: CulturalRepE) -> None:
        prompts = ["Hospitality is a duty."]
        before = _resid_post(tiny_engine, prompts, layer=2)

        with tiny_engine.steering("diyafa_001", strength=5.0):
            pass

        after = _resid_post(tiny_engine, prompts, layer=2)

        assert torch.equal(before, after)
        assert tiny_engine.active_hook_names == []

    def test_negative_strength_mirrors_positive(self, tiny_engine: CulturalRepE) -> None:
        prompts = ["A guest arrives."]
        baseline = _resid_post(tiny_engine, prompts, layer=2)

        with tiny_engine.steering("diyafa_001", strength=2.0):
            amplified = _resid_post(tiny_engine, prompts, layer=2)
        with tiny_engine.steering("diyafa_001", strength=-2.0):
            suppressed = _resid_post(tiny_engine, prompts, layer=2)

        assert torch.allclose(amplified - baseline, baseline - suppressed, atol=1e-5)

    def test_zero_strength_leaves_the_activation_untouched(
        self, tiny_engine: CulturalRepE
    ) -> None:
        prompts = ["A guest arrives."]
        baseline = _resid_post(tiny_engine, prompts, layer=2)

        with tiny_engine.steering("diyafa_001", strength=0.0):
            steered = _resid_post(tiny_engine, prompts, layer=2)

        assert torch.allclose(steered, baseline, atol=1e-6)

    def test_injecting_into_several_layers_shifts_each_of_them(
        self, tiny_engine: CulturalRepE
    ) -> None:
        prompts = ["A guest arrives."]
        vector = tiny_engine.concept_vectors["diyafa_001"]
        baseline_0 = _resid_post(tiny_engine, prompts, layer=0)

        with tiny_engine.steering("diyafa_001", strength=1.0, layers=[0, 1]):
            assert len(tiny_engine.active_hook_names) == 2
            steered_0 = _resid_post(tiny_engine, prompts, layer=0)

        # Layer 0 is upstream of layer 1, so its shift is exactly the injection.
        assert torch.allclose(steered_0 - baseline_0, vector.expand_as(baseline_0), atol=1e-5)

    def test_forward_pass_still_produces_finite_logits(self, tiny_engine: CulturalRepE) -> None:
        model = tiny_engine.model
        assert model is not None

        with tiny_engine.steering("diyafa_001", strength=2.0), torch.no_grad():
            logits = model(model.to_tokens(["A guest arrives."]))

        assert torch.isfinite(logits).all()

    def test_evaluate_steering_runs_against_a_real_model(self, tiny_engine: CulturalRepE) -> None:
        results = evaluate_steering(
            tiny_engine,
            "diyafa_001",
            ["A guest arrives at the door."],
            strengths=[-1.0, 0.0, 1.0],
            max_new_tokens=3,
        )

        assert list(results) == [-1.0, 0.0, 1.0]
        for result in results.values():
            assert result.generations
            assert result.mean_loss > 0
        assert tiny_engine.active_hook_names == []
