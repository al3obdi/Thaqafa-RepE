"""Deterministic stand-in models shared by the Phase 4 test modules.

Both fakes below reproduce only the slice of the TransformerLens interface the
code under test touches, and both are fully deterministic, so assertions can be
about exact numbers rather than "did it run".
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.models.rep_engine import RESID_POST_HOOK, CulturalRepE

DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "datasets" / "cultural_concepts.jsonl"

POSITIVE_MARKER = "[POS]"
NEGATIVE_MARKER = "[NEG]"

MARKER_D_MODEL = 4
MARKER_N_LAYERS = 6
INFORMATIVE_LAYERS: frozenset[int] = frozenset({2, 3})
"""Layers at which MarkerTransformer makes the two classes separable."""


class _Config:
    """Minimal config exposing the fields the engine reads."""

    def __init__(self, n_layers: int, d_model: int) -> None:
        self.n_layers = n_layers
        self.d_model = d_model


class MarkerTransformer:
    """A model whose activations are separable only at certain layers.

    Each prompt is tokenized into two ids: a class marker (10 for a prompt
    starting with ``[POS]``, 20 otherwise) and a length code shared by both
    classes. The residual stream carries the marker **only at**
    :data:`INFORMATIVE_LAYERS`; everywhere else the marker is zeroed and only the
    class-independent length code survives.

    A layer probe should therefore score perfectly at the informative layers and
    at chance everywhere else - which is exactly the shape of result a real layer
    sweep is looking for, made exact.
    """

    def __init__(
        self,
        n_layers: int = MARKER_N_LAYERS,
        d_model: int = MARKER_D_MODEL,
    ) -> None:
        self.cfg = _Config(n_layers, d_model)
        self.tokenizer = None
        self.probed_layers: list[int] = []

    def eval(self) -> MarkerTransformer:
        """Match the ``torch.nn.Module`` interface."""
        return self

    def to_tokens(self, prompts: list[str]) -> torch.Tensor:
        """Encode each prompt as ``[class_marker, length_code]``."""
        rows = [
            [
                10 if prompt.startswith(POSITIVE_MARKER) else 20,
                len(prompt) % 7 + 1,
            ]
            for prompt in prompts
        ]
        return torch.tensor(rows, dtype=torch.long)

    def run_with_cache(
        self,
        tokens: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[None, dict[str, torch.Tensor]]:
        """Return activations that carry the marker only at informative layers."""
        hook_name = str(kwargs.get("names_filter") or RESID_POST_HOOK.format(layer=0))
        layer = int(hook_name.split(".")[1])
        self.probed_layers.append(layer)

        marker = tokens[:, 0].to(torch.float32)
        code = tokens[:, 1].to(torch.float32)
        signal = marker if layer in INFORMATIVE_LAYERS else torch.zeros_like(marker)

        per_token = torch.stack([signal, code], dim=1)  # (batch, seq=2)
        activations = per_token.unsqueeze(-1).expand(-1, -1, self.cfg.d_model)
        return None, {hook_name: activations.clone()}


class _FakeRemovableHandle:
    """Stands in for ``torch.utils.hooks.RemovableHandle``."""

    def __init__(self) -> None:
        self.remove_count = 0

    def remove(self) -> None:
        """Record a removal."""
        self.remove_count += 1


class _FakeLensHandle:
    """Stands in for ``transformer_lens.hook_points.LensHandle``."""

    def __init__(self, user_hook: Any) -> None:
        self.hook = _FakeRemovableHandle()
        self.user_hook = user_hook
        self.is_permanent = False


class _FakeHookPoint:
    """Stands in for ``transformer_lens.hook_points.HookPoint``."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.fwd_hooks: list[_FakeLensHandle] = []


class SteerableFakeModel:
    """A hookable model that can also score text and generate continuations.

    ``add_hook`` behaves like TransformerLens's: it returns ``None`` and appends
    a handle to the hook point. Loss and generation both depend deterministically
    on the attached steering, so a comparison between a steered and a prompted
    condition produces different, predictable numbers without any real weights.
    """

    def __init__(self, n_layers: int = 4, d_model: int = 4, d_vocab: int = 8) -> None:
        self.cfg = _Config(n_layers, d_model)
        self.cfg.d_vocab = d_vocab  # type: ignore[attr-defined]
        self.tokenizer = None
        self.mod_dict: dict[str, _FakeHookPoint] = {
            RESID_POST_HOOK.format(layer=layer): _FakeHookPoint(RESID_POST_HOOK.format(layer=layer))
            for layer in range(n_layers)
        }
        self.generate_calls: list[str] = []

    def eval(self) -> SteerableFakeModel:
        """Match the ``torch.nn.Module`` interface."""
        return self

    def add_hook(self, name: str, hook: Any, dir: str = "fwd") -> None:  # noqa: A002
        """Register a hook and append its handle to the hook point."""
        self.mod_dict[name].fwd_hooks.append(_FakeLensHandle(hook))

    def steering_magnitude(self) -> float:
        """Total offset the attached hooks would apply to a zero activation."""
        total = 0.0
        for hook_point in self.mod_dict.values():
            for handle in hook_point.fwd_hooks:
                probe = torch.zeros(1, 1, self.cfg.d_model)
                total += float(handle.user_hook(probe, None).abs().max().item())
        return total

    def to_tokens(self, prompts: list[str]) -> torch.Tensor:
        """Encode each prompt as one token per word, capped to the vocabulary."""
        rows = [[len(word) % self.cfg.d_vocab for word in prompt.split()] or [0] for prompt in prompts]  # type: ignore[attr-defined]
        width = max(len(row) for row in rows)
        return torch.tensor([row + [0] * (width - len(row)) for row in rows], dtype=torch.long)

    def run_with_cache(
        self,
        tokens: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[None, dict[str, torch.Tensor]]:
        """Return activations whose residual norm is a known function of depth.

        Every token vector at layer L has L2 norm ``10 * (L + 1)``, so
        ``calibrate_layer_norms`` has an exact answer to find and relative
        steering can be asserted numerically. This mirrors the real shape of
        the problem: norms that grow with depth.
        """
        hook_name = str(kwargs.get("names_filter") or RESID_POST_HOOK.format(layer=0))
        layer = int(hook_name.split(".")[1])
        target_norm = 10.0 * (layer + 1)

        d_model = self.cfg.d_model
        # A vector of d_model identical components c has norm sqrt(d_model)*c.
        component = target_norm / (d_model**0.5)
        activations = torch.full((*tokens.shape, d_model), component, dtype=torch.float32)
        return None, {hook_name: activations}

    def __call__(self, text: Any, return_type: str = "logits") -> torch.Tensor:
        """Return a deterministic loss for a string, or logits for tokens."""
        if isinstance(text, str):
            word_count = len(text.split())
            return torch.tensor(1.0 + 0.01 * word_count + 0.5 * self.steering_magnitude())

        batch, seq = text.shape
        base = torch.arange(self.cfg.d_vocab, dtype=torch.float32)  # type: ignore[attr-defined]
        # Scale rather than offset: adding a constant to every logit leaves the
        # softmax unchanged, so an offset could not express a distribution shift.
        logits = base.expand(batch, seq, -1) * (1.0 + self.steering_magnitude())
        return logits.clone()

    def generate(self, prompt: str, max_new_tokens: int = 8, **kwargs: Any) -> str:
        """Return the prompt plus a continuation that reflects the steering."""
        self.generate_calls.append(prompt)
        return f"{prompt} continuation-{self.steering_magnitude():.2f}"


def make_marker_engine(model: MarkerTransformer | None = None) -> CulturalRepE:
    """Build an engine backed by :class:`MarkerTransformer`.

    Args:
        model: Stand-in model to attach. A default one is built when omitted.

    Returns:
        An engine that will not attempt to download anything.
    """
    engine = CulturalRepE(
        model_name="dummy/marker",
        device="cpu",
        dtype="float32",
        dataset_path=DATASET_PATH,
    )
    attached = model if model is not None else MarkerTransformer()
    engine.model = attached  # type: ignore[assignment]
    engine.tokenizer = attached.tokenizer
    return engine


def make_steerable_engine(model: SteerableFakeModel | None = None) -> CulturalRepE:
    """Build an engine backed by :class:`SteerableFakeModel` with a cached vector.

    Args:
        model: Stand-in model to attach. A default one is built when omitted.

    Returns:
        An engine holding a unit vector under the concept ``"diyafa"``.
    """
    engine = CulturalRepE(
        model_name="dummy/steerable",
        device="cpu",
        dtype="float32",
        dataset_path=DATASET_PATH,
    )
    attached = model if model is not None else SteerableFakeModel()
    engine.model = attached  # type: ignore[assignment]
    engine.tokenizer = attached.tokenizer
    engine.concept_vectors["diyafa"] = torch.ones(attached.cfg.d_model)
    engine.extraction_layers["diyafa"] = 1
    return engine


def marked_prompts(count: int = 8) -> tuple[list[str], list[str]]:
    """Build matched positive and negative prompt sets.

    The two sets differ only in their marker: lengths, and therefore the
    class-independent length code, are identical pairwise. A probe at a
    non-informative layer consequently sees the very same feature values for
    both classes and cannot do better than chance.

    Args:
        count: How many prompts per class.

    Returns:
        A ``(positive, negative)`` pair of prompt lists.

    Raises:
        ValueError: If ``count`` is not positive.
    """
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")

    positives = [f"{POSITIVE_MARKER} {'x' * index}" for index in range(count)]
    negatives = [f"{NEGATIVE_MARKER} {'x' * index}" for index in range(count)]
    return positives, negatives
