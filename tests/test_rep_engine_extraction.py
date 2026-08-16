"""Tests for the contrastive vector extraction engine.

The suite runs on CPU and is split into two tiers:

* **Unit tests** drive :class:`~src.models.rep_engine.CulturalRepE` with a
  deterministic in-process stand-in (:class:`DummyTransformer`) whose
  activations are an exact function of the token ids. That makes masking and
  averaging assertions numerically exact, needs no weights, and keeps the whole
  tier at a few milliseconds.
* **Integration tests** build a genuinely tiny ``HookedTransformer`` (4 layers,
  ``d_model=16``) from a config, borrowing only the ``sshleifer/tiny-gpt2``
  tokenizer. No pretrained weights are downloaded, so the tier stays fast and
  never needs a GPU. It skips itself when ``transformer_lens`` is missing or
  the tokenizer cannot be fetched, so an offline runner degrades to the unit
  tier instead of failing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from src.models.rep_engine import (
    RESID_POST_HOOK,
    CulturalRepE,
    resolve_dtype,
)

TINY_TOKENIZER_NAME = "sshleifer/tiny-gpt2"
DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "datasets" / "cultural_concepts.jsonl"

PAD_ID = 0
D_MODEL = 4


# ---------------------------------------------------------------------------
# Deterministic stand-in for a HookedTransformer
# ---------------------------------------------------------------------------


class DummyTokenizer:
    """Minimal tokenizer exposing only what the engine reads."""

    def __init__(self, padding_side: str = "right", bos_token_id: int = PAD_ID) -> None:
        self.pad_token_id = PAD_ID
        self.bos_token_id = bos_token_id
        self.padding_side = padding_side


class DummyConfig:
    """Minimal config exposing only ``n_layers``."""

    def __init__(self, n_layers: int) -> None:
        self.n_layers = n_layers


class DummyTransformer:
    """A fake model whose activations are a known function of the token ids.

    Every prompt is tokenized as one token per word, where the token id is the
    word length. Position ``(batch, seq)`` of the residual stream is filled
    with the token id in every channel, so the masked mean over a prompt is
    exactly the mean word length. That makes it possible to assert on numbers
    rather than merely on shapes.
    """

    def __init__(self, n_layers: int = 4, padding_side: str = "right") -> None:
        self.cfg = DummyConfig(n_layers)
        self.tokenizer = DummyTokenizer(padding_side=padding_side)
        self.run_calls: list[dict[str, Any]] = []

    def eval(self) -> DummyTransformer:
        """Match the ``torch.nn.Module`` interface used by ``load_model``."""
        return self

    def to_tokens(self, prompts: list[str]) -> torch.Tensor:
        """Tokenize prompts into word-length ids, right-padded with zeros."""
        rows = [[len(word) for word in prompt.split()] for prompt in prompts]
        width = max(len(row) for row in rows)
        padded = [row + [PAD_ID] * (width - len(row)) for row in rows]
        return torch.tensor(padded, dtype=torch.long)

    def run_with_cache(
        self,
        tokens: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[None, dict[str, torch.Tensor]]:
        """Return activations equal to the token id, broadcast over channels."""
        self.run_calls.append({"tokens": tokens, **kwargs})
        activations = tokens.to(torch.float32).unsqueeze(-1).expand(-1, -1, D_MODEL)
        hook_name = kwargs.get("names_filter") or RESID_POST_HOOK.format(layer=0)
        return None, {hook_name: activations.clone()}


def make_engine(model: DummyTransformer | None = None, **kwargs: Any) -> CulturalRepE:
    """Build an engine wired to a dummy model.

    Args:
        model: Stand-in model to attach. A default one is built when omitted.
        **kwargs: Overrides forwarded to the :class:`CulturalRepE` constructor.

    Returns:
        An engine that will not attempt to download anything.
    """
    engine = CulturalRepE(
        model_name="dummy/model",
        device="cpu",
        dtype="float32",
        dataset_path=DATASET_PATH,
        **kwargs,
    )
    attached = model if model is not None else DummyTransformer()
    engine.model = attached  # type: ignore[assignment]
    engine.tokenizer = attached.tokenizer
    return engine


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


class TestDtypeResolution:
    """Translation of dtype names into torch dtypes."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("bfloat16", torch.bfloat16),
            ("BF16", torch.bfloat16),
            ("float16", torch.float16),
            ("float32", torch.float32),
            (" fp32 ", torch.float32),
        ],
    )
    def test_known_names(self, name: str, expected: torch.dtype) -> None:
        assert resolve_dtype(name) is expected

    def test_torch_dtype_passes_through(self) -> None:
        assert resolve_dtype(torch.float64) is torch.float64

    def test_unknown_name_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unsupported dtype"):
            resolve_dtype("float8")

    def test_constructor_rejects_bad_dtype(self) -> None:
        with pytest.raises(ValueError, match="Unsupported dtype"):
            CulturalRepE(model_name="dummy/model", dtype="float8")

    def test_constructor_rejects_bad_batch_size(self) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            CulturalRepE(model_name="dummy/model", batch_size=0)


# ---------------------------------------------------------------------------
# load_model
# ---------------------------------------------------------------------------


class TestLoadModel:
    """Loading and caching of the base model."""

    def test_load_model_caches_the_instance(self, monkeypatch: pytest.MonkeyPatch) -> None:
        transformer_lens = pytest.importorskip("transformer_lens")
        stub = DummyTransformer()
        calls: list[dict[str, Any]] = []

        def fake_from_pretrained(model_name: str, **kwargs: Any) -> DummyTransformer:
            calls.append({"model_name": model_name, **kwargs})
            return stub

        monkeypatch.setattr(
            transformer_lens.HookedTransformer,
            "from_pretrained",
            staticmethod(fake_from_pretrained),
        )

        engine = CulturalRepE(model_name="dummy/model", device="cpu", dtype="float32")
        first = engine.load_model()
        second = engine.load_model()

        assert first is stub
        assert second is stub
        assert len(calls) == 1, "the model must not be reloaded on the second call"
        assert engine.model is stub
        assert engine.tokenizer is stub.tokenizer

    def test_load_model_forwards_device_and_dtype(self, monkeypatch: pytest.MonkeyPatch) -> None:
        transformer_lens = pytest.importorskip("transformer_lens")
        calls: list[dict[str, Any]] = []

        def fake_from_pretrained(model_name: str, **kwargs: Any) -> DummyTransformer:
            calls.append({"model_name": model_name, **kwargs})
            return DummyTransformer()

        monkeypatch.setattr(
            transformer_lens.HookedTransformer,
            "from_pretrained",
            staticmethod(fake_from_pretrained),
        )

        CulturalRepE(model_name="dummy/model", device="cpu", dtype="bfloat16").load_model()

        assert calls[0]["model_name"] == "dummy/model"
        assert calls[0]["device"] == "cpu"
        assert calls[0]["dtype"] is torch.bfloat16

    def test_load_failure_is_wrapped_in_runtime_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        transformer_lens = pytest.importorskip("transformer_lens")

        def boom(model_name: str, **kwargs: Any) -> DummyTransformer:
            raise OSError("gated repository")

        monkeypatch.setattr(
            transformer_lens.HookedTransformer,
            "from_pretrained",
            staticmethod(boom),
        )

        engine = CulturalRepE(model_name="dummy/model", device="cpu", dtype="float32")
        with pytest.raises(RuntimeError, match="Failed to load model"):
            engine.load_model()

    def test_no_token_means_no_auth_kwargs(self) -> None:
        engine = CulturalRepE(model_name="dummy/model", device="cpu")

        def loader(model_name: str, token: str | None = None) -> None:  # pragma: no cover
            return None

        assert engine._auth_kwargs(loader) == {}

    def test_token_is_passed_under_the_supported_name(self) -> None:
        engine = CulturalRepE(model_name="dummy/model", device="cpu", hf_token="secret-value")

        def modern_loader(model_name: str, token: str | None = None) -> None:  # pragma: no cover
            return None

        def legacy_loader(
            model_name: str, hf_api_key: str | None = None
        ) -> None:  # pragma: no cover
            return None

        assert engine._auth_kwargs(modern_loader) == {"token": "secret-value"}
        assert engine._auth_kwargs(legacy_loader) == {"hf_api_key": "secret-value"}


# ---------------------------------------------------------------------------
# Layer resolution
# ---------------------------------------------------------------------------


class TestLayerResolution:
    """Selection of the extraction layer."""

    def test_default_layer_is_the_middle_of_the_stack(self) -> None:
        engine = make_engine(DummyTransformer(n_layers=12))

        assert engine.n_layers == 12
        assert engine.middle_layer == 6
        assert engine._resolve_layer(None) == 6

    def test_negative_indices_count_from_the_end(self) -> None:
        engine = make_engine(DummyTransformer(n_layers=8))

        assert engine._resolve_layer(-1) == 7
        assert engine._resolve_layer(-8) == 0

    @pytest.mark.parametrize("layer", [8, 99, -9])
    def test_out_of_range_layer_is_rejected(self, layer: int) -> None:
        engine = make_engine(DummyTransformer(n_layers=8))

        with pytest.raises(IndexError, match="out of range"):
            engine._resolve_layer(layer)

    def test_n_layers_requires_a_loaded_model(self) -> None:
        engine = CulturalRepE(model_name="dummy/model", device="cpu")

        with pytest.raises(RuntimeError, match="not loaded"):
            _ = engine.n_layers


# ---------------------------------------------------------------------------
# Attention masking
# ---------------------------------------------------------------------------


class TestAttentionMask:
    """Padding masks used to exclude pad tokens from the mean."""

    def test_trailing_pads_are_masked_with_right_padding(self) -> None:
        engine = make_engine(DummyTransformer(padding_side="right"))
        tokens = torch.tensor([[5, 3, PAD_ID, PAD_ID], [2, 2, 2, 4]])

        mask = engine._build_attention_mask(tokens)

        assert mask.tolist() == [[1, 1, 0, 0], [1, 1, 1, 1]]

    def test_interior_pad_ids_are_kept(self) -> None:
        # A pad id in the middle of a sequence is real content (GPT-2 style
        # tokenizers reuse the same id for pad, bos and eos).
        engine = make_engine(DummyTransformer(padding_side="right"))
        tokens = torch.tensor([[5, PAD_ID, 3, PAD_ID]])

        mask = engine._build_attention_mask(tokens)

        assert mask.tolist() == [[1, 1, 1, 0]]

    def test_leading_pads_are_masked_with_left_padding(self) -> None:
        model = DummyTransformer(padding_side="left")
        model.tokenizer.bos_token_id = 99  # distinct from pad, so no BOS rescue
        engine = make_engine(model)
        tokens = torch.tensor([[PAD_ID, PAD_ID, 5, 3], [1, 2, 3, 4]])

        mask = engine._build_attention_mask(tokens)

        assert mask.tolist() == [[0, 0, 1, 1], [1, 1, 1, 1]]

    def test_left_padding_keeps_bos_when_it_shares_the_pad_id(self) -> None:
        engine = make_engine(DummyTransformer(padding_side="left"))
        tokens = torch.tensor([[PAD_ID, PAD_ID, 5, 3]])

        mask = engine._build_attention_mask(tokens)

        # The last leading pad is the prepended BOS and must stay visible.
        assert mask.tolist() == [[0, 1, 1, 1]]


# ---------------------------------------------------------------------------
# _compute_mean_activation
# ---------------------------------------------------------------------------


class TestComputeMeanActivation:
    """Collection and averaging of residual stream activations."""

    def test_returns_a_one_dimensional_hidden_state(self) -> None:
        engine = make_engine()

        mean = engine._compute_mean_activation(["one two", "three four five"], layer=1)

        assert mean.ndim == 1
        assert mean.shape == (D_MODEL,)
        assert mean.dtype is torch.float32

    def test_mean_excludes_padding_positions(self) -> None:
        # "aa" -> [2]; "bbb cccc" -> [3, 4]. Batched together the first row is
        # padded to length two. Per-prompt means are 2.0 and 3.5, so the mean
        # over prompts is 2.75. Counting the pad token would give 2.25.
        engine = make_engine()

        mean = engine._compute_mean_activation(["aa", "bbb cccc"], layer=0)

        assert torch.allclose(mean, torch.full((D_MODEL,), 2.75))

    def test_batching_does_not_change_the_result(self) -> None:
        prompts = ["a", "bb ccc", "dddd", "ee fff gggg", "h"]
        single = make_engine(batch_size=1)._compute_mean_activation(prompts, layer=0)
        batched = make_engine(batch_size=8)._compute_mean_activation(prompts, layer=0)

        assert torch.allclose(single, batched)

    def test_reads_the_requested_layer_hook(self) -> None:
        model = DummyTransformer(n_layers=6)
        engine = make_engine(model)

        engine._compute_mean_activation(["hello there"], layer=3)

        assert model.run_calls[0]["names_filter"] == "blocks.3.hook_resid_post"
        assert model.run_calls[0]["stop_at_layer"] == 4

    def test_negative_layer_is_resolved(self) -> None:
        model = DummyTransformer(n_layers=6)
        engine = make_engine(model)

        engine._compute_mean_activation(["hello there"], layer=-1)

        assert model.run_calls[0]["names_filter"] == "blocks.5.hook_resid_post"

    def test_empty_prompts_are_rejected(self) -> None:
        engine = make_engine()

        with pytest.raises(ValueError, match="at least one text"):
            engine._compute_mean_activation([], layer=0)


# ---------------------------------------------------------------------------
# extract_vector
# ---------------------------------------------------------------------------


class TestExtractVector:
    """The contrastive mean-difference recipe."""

    def test_returns_unit_norm_vector(self) -> None:
        engine = make_engine()

        vector = engine.extract_vector(
            "diyafa",
            examples=["generous host welcomed guests"],
            layer=1,
        )

        assert vector.shape == (D_MODEL,)
        assert torch.linalg.vector_norm(vector).item() == pytest.approx(1.0, abs=1e-5)

    def test_vector_is_the_normalized_difference_of_the_means(self) -> None:
        engine = make_engine()
        positives = ["aaa bbb", "cc"]
        negatives = ["d", "ee"]

        expected_direction = engine._compute_mean_activation(
            positives, layer=1
        ) - engine._compute_mean_activation(negatives, layer=1)
        expected = expected_direction / torch.linalg.vector_norm(expected_direction)

        vector = engine.extract_vector(
            "concept", examples=positives, contrast_examples=negatives, layer=1
        )

        assert torch.allclose(vector, expected, atol=1e-6)

    def test_normalization_can_be_disabled(self) -> None:
        engine = make_engine()

        raw = engine.extract_vector(
            "concept",
            examples=["aaaa bbbb cccc"],
            contrast_examples=["d"],
            layer=1,
            normalize=False,
        )

        # Positive mean is 4.0 per channel, negative mean 1.0, difference 3.0.
        assert torch.allclose(raw, torch.full((D_MODEL,), 3.0))

    def test_vector_is_cached_with_its_layer(self) -> None:
        engine = make_engine(DummyTransformer(n_layers=6))

        engine.extract_vector("wasta", examples=["personal connections helped"], layer=2)

        assert "wasta" in engine.concept_vectors
        assert engine.extraction_layers["wasta"] == 2
        assert torch.equal(engine.concept_vectors["wasta"], engine.concept_vectors["wasta"])

    def test_default_layer_is_the_middle_layer(self) -> None:
        model = DummyTransformer(n_layers=10)
        engine = make_engine(model)

        engine.extract_vector("wasta", examples=["personal connections helped"])

        assert engine.extraction_layers["wasta"] == 5
        assert model.run_calls[0]["names_filter"] == "blocks.5.hook_resid_post"

    def test_examples_are_loaded_from_the_dataset_by_concept_id(self) -> None:
        engine = make_engine()

        vector = engine.extract_vector("diyafa_001", layer=1)

        assert vector.shape == (D_MODEL,)
        assert "diyafa_001" in engine.concept_vectors

    def test_unknown_concept_id_is_reported(self) -> None:
        engine = make_engine()

        with pytest.raises(ValueError, match="was not found"):
            engine.extract_vector("not_a_concept_999", layer=1)

    def test_empty_concept_is_rejected(self) -> None:
        engine = make_engine()

        with pytest.raises(ValueError, match="concept"):
            engine.extract_vector("   ", examples=["something"])

    def test_empty_example_list_is_rejected(self) -> None:
        engine = make_engine()

        with pytest.raises(ValueError, match="at least one example"):
            engine.extract_vector("diyafa", examples=[])

    def test_identical_prompt_sets_cannot_be_normalized(self) -> None:
        engine = make_engine()
        prompts = ["same words here"]

        with pytest.raises(ValueError, match="identical"):
            engine.extract_vector("concept", examples=prompts, contrast_examples=prompts, layer=1)

    def test_neutral_baseline_is_used_when_no_contrast_is_given(self) -> None:
        model = DummyTransformer()
        engine = make_engine(model)

        engine.extract_vector("diyafa", examples=["one two three"], layer=1)

        # Two forward passes: one for the positives, one for the generated
        # neutral baseline.
        assert len(model.run_calls) == 2

    def test_extract_all_vectors_covers_the_dataset(self) -> None:
        engine = make_engine()

        vectors = engine.extract_all_vectors(layer=1)

        assert set(vectors) == {"wasta_001", "muruah_001", "diyafa_001"}
        for vector in vectors.values():
            assert torch.linalg.vector_norm(vector).item() == pytest.approx(1.0, abs=1e-5)


class TestSaveVectors:
    """Persistence of extracted vectors."""

    def test_save_and_reload_round_trip(self, tmp_path: Path) -> None:
        engine = make_engine()
        engine.extract_vector("diyafa", examples=["generous host"], layer=1)

        destination = engine.save_vectors(tmp_path / "nested" / "vectors.pt")
        payload = torch.load(destination, weights_only=False)

        assert destination.exists()
        assert payload["model_name"] == "dummy/model"
        assert torch.equal(payload["concept_vectors"]["diyafa"], engine.concept_vectors["diyafa"])
        assert payload["extraction_layers"] == {"diyafa": 1}

    def test_saving_without_vectors_is_rejected(self, tmp_path: Path) -> None:
        engine = make_engine()

        with pytest.raises(ValueError, match="No concept vectors"):
            engine.save_vectors(tmp_path / "vectors.pt")


# ---------------------------------------------------------------------------
# Integration tier: a real, tiny HookedTransformer on CPU
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_engine() -> CulturalRepE:
    """Build an engine backed by a randomly initialised 4-layer transformer.

    Only the ``sshleifer/tiny-gpt2`` tokenizer is fetched; the weights are
    random, which is enough to exercise the real ``run_with_cache`` code path
    on CPU in well under a second.

    Returns:
        An engine with the tiny model already attached.
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
    return engine


class TestTinyModelIntegration:
    """End-to-end checks against a real HookedTransformer."""

    def test_mean_activation_has_model_hidden_size(self, tiny_engine: CulturalRepE) -> None:
        mean = tiny_engine._compute_mean_activation(
            ["He hosted him generously", "أكرم ضيافته لمدة ثلاثة أيام"],
            layer=2,
        )

        assert mean.shape == (16,)
        assert torch.isfinite(mean).all()

    def test_extract_vector_is_normalized(self, tiny_engine: CulturalRepE) -> None:
        vector = tiny_engine.extract_vector("diyafa_001", layer=2)

        assert vector.shape == (16,)
        assert torch.linalg.vector_norm(vector).item() == pytest.approx(1.0, abs=1e-5)
        assert torch.isfinite(vector).all()

    def test_extraction_defaults_to_the_middle_layer(self, tiny_engine: CulturalRepE) -> None:
        tiny_engine.extract_vector("wasta_001")

        assert tiny_engine.extraction_layers["wasta_001"] == 2

    def test_masking_makes_the_result_independent_of_batching(
        self, tiny_engine: CulturalRepE
    ) -> None:
        # Prompts of very different lengths: batched together the short one is
        # heavily padded, run one at a time it is not padded at all. If pad
        # positions were included in the mean the two results would diverge.
        prompts = [
            "Hospitality is a duty.",
            "This considerably longer sentence forces the batch to be padded out by several tokens.",
            "الضيافة واجب.",
        ]

        unbatched = CulturalRepE(
            model_name=TINY_TOKENIZER_NAME,
            device="cpu",
            dtype="float32",
            dataset_path=DATASET_PATH,
            batch_size=1,
        )
        unbatched.model = tiny_engine.model
        unbatched.tokenizer = tiny_engine.tokenizer

        batched_mean = tiny_engine._compute_mean_activation(prompts, layer=1)
        unbatched_mean = unbatched._compute_mean_activation(prompts, layer=1)

        assert torch.allclose(batched_mean, unbatched_mean, atol=1e-5)
