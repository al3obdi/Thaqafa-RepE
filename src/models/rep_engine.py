"""Representation engineering engine for Arab cultural concepts.

This module exposes :class:`CulturalRepE`, the central abstraction of the
project. It is responsible for three things:

1. Loading a causal language model through ``transformer_lens`` so that every
   internal activation is addressable by hook name.
2. Extracting a *concept vector* - a direction in the residual stream that
   corresponds to a cultural concept such as ``الضيافة`` (hospitality).
3. Injecting a previously extracted concept vector back into the forward pass
   in order to steer generation towards (or away from) that concept.

Extraction follows the **contrastive mean-difference** recipe from the
representation engineering literature:

.. code-block:: text

    v_concept = mean(resid_post[layer] | positive prompts)
              - mean(resid_post[layer] | neutral  prompts)
    v_concept = v_concept / ||v_concept||_2

Averaging over a set of neutral prompts removes the components of the residual
stream that merely encode "this is an ordinary sentence", which would otherwise
dominate the raw mean and make every concept vector point in roughly the same
direction. The result is L2-normalised so that the injection strength is the
only magnitude knob.

Injection is the mirror image of extraction. A forward hook on the same
residual stream point adds ``strength * v_concept`` to every position of the
activation as it flows past:

.. code-block:: text

    resid_post[layer] <- resid_post[layer] + strength * v_concept

Because the vector is a unit direction, ``strength`` is measured in residual
stream norms: positive values amplify the concept, negative values suppress it,
and zero reproduces the unsteered model exactly. Hooks mutate the model until
they are removed, so the recommended entry point is the :meth:`
CulturalRepE.steering` context manager, which guarantees cleanup even when the
body raises.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from inspect import signature
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from src.data.contrastive import build_contrast_examples
from src.data.dataset_builder import DEFAULT_DATASET_PATH, load_concepts

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from transformer_lens import HookedTransformer

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
DEFAULT_BATCH_SIZE = 8

RESID_POST_HOOK = "blocks.{layer}.hook_resid_post"
"""Hook name template for the residual stream after block ``layer``."""

_DTYPE_ALIASES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "half": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def resolve_dtype(dtype: str | torch.dtype) -> torch.dtype:
    """Translate a dtype name into a :class:`torch.dtype`.

    Args:
        dtype: Either a ``torch.dtype`` (returned unchanged) or one of the
            supported names, for example ``"bfloat16"``, ``"fp16"`` or
            ``"float32"``.

    Returns:
        The corresponding torch dtype.

    Raises:
        ValueError: If the name is not recognised.
    """
    if isinstance(dtype, torch.dtype):
        return dtype

    key = dtype.strip().lower()
    if key not in _DTYPE_ALIASES:
        raise ValueError(f"Unsupported dtype {dtype!r}. Supported: {sorted(_DTYPE_ALIASES)}")
    return _DTYPE_ALIASES[key]


@dataclass
class InjectionHandle:
    """A single live steering hook, and the means to take it back off.

    TransformerLens's ``add_hook`` returns nothing; it appends a ``LensHandle``
    to the hook point's ``fwd_hooks`` list. This wrapper captures that handle so
    an individual injection can be undone without resetting every hook on the
    model - which matters because callers may have their own caching or
    ablation hooks attached that must survive.

    Attributes:
        concept: Concept whose vector this hook injects.
        layer: Block the hook is attached to.
        hook_name: Full TransformerLens hook name.
        strength: Coefficient applied to the concept vector.
    """

    concept: str
    layer: int
    hook_name: str
    strength: float
    _hook_point: Any = field(repr=False)
    _lens_handle: Any = field(repr=False)
    _removed: bool = field(default=False, repr=False)

    @property
    def is_active(self) -> bool:
        """Whether the hook is still attached to the model."""
        return not self._removed

    def remove(self) -> None:
        """Detach the hook and drop it from the hook point's bookkeeping.

        Removing the PyTorch handle alone would leave a stale ``LensHandle``
        behind, which makes ``model.fwd_hooks`` and TransformerLens's own hook
        accounting disagree with reality. Both are cleaned up here. Calling this
        twice is safe.
        """
        if self._removed:
            return

        self._lens_handle.hook.remove()
        self._hook_point.fwd_hooks = [
            handle for handle in self._hook_point.fwd_hooks if handle is not self._lens_handle
        ]
        self._removed = True


class CulturalRepE:
    """Extract and inject cultural concept vectors for a causal language model.

    Args:
        model_name: Hugging Face identifier of the base model, for example
            ``"meta-llama/Meta-Llama-3-8B-Instruct"``.
        device: Device the model should be placed on (``"cuda"``, ``"cpu"``
            or ``"mps"``).
        dtype: Name of the torch dtype used for the model weights, for example
            ``"bfloat16"`` or ``"float16"``.
        hf_token: Hugging Face access token for gated repositories. Read it
            from the environment (``HF_TOKEN``); never hard-code it.
        dataset_path: JSONL file used to resolve concept identifiers into
            example prompts.
        batch_size: Number of prompts pushed through the model at once during
            activation collection.

    Attributes:
        concept_vectors: Mapping from concept name to the extracted direction.
        extraction_layers: Layer each cached vector was extracted from.
        model: The loaded model, or ``None`` until :meth:`load_model` is called.
        tokenizer: The loaded tokenizer, or ``None`` until :meth:`load_model`
            is called.
        _active_hooks: Steering hooks currently attached by this engine. Use
            :meth:`remove_hooks` rather than mutating it directly.

    Example:
        >>> engine = CulturalRepE("meta-llama/Meta-Llama-3-8B-Instruct")
        >>> engine.concept_vectors
        {}
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "cuda",
        dtype: str = "bfloat16",
        hf_token: str | None = None,
        dataset_path: Path | str = DEFAULT_DATASET_PATH,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        if not model_name or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")

        # Validate the dtype eagerly so that a typo fails at construction time
        # rather than after a multi-minute model download.
        resolve_dtype(dtype)

        self.model_name: str = model_name
        self.device: str = device
        self.dtype: str = dtype
        self.hf_token: str | None = hf_token
        self.dataset_path: Path = Path(dataset_path)
        self.batch_size: int = batch_size

        self.model: HookedTransformer | None = None
        self.tokenizer: Any | None = None
        self.concept_vectors: dict[str, torch.Tensor] = {}
        self.extraction_layers: dict[str, int] = {}
        self._active_hooks: list[InjectionHandle] = []

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self) -> HookedTransformer:
        """Load the base model onto :attr:`device` and cache it.

        The model is wrapped in a ``transformer_lens.HookedTransformer`` so
        that residual stream activations can be read by hook name. Calling the
        method twice is cheap: the cached instance is returned without
        reloading the weights.

        Returns:
            The loaded :class:`~transformer_lens.HookedTransformer`.

        Raises:
            ImportError: If ``transformer_lens`` is not installed.
            RuntimeError: If the weights cannot be loaded, for example because
                the repository is gated and no valid token was supplied.
        """
        if self.model is not None:
            logger.debug("Reusing cached model %s", self.model_name)
            return self.model

        try:
            from transformer_lens import HookedTransformer
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "transformer_lens is required to load models. Install it with "
                "`poetry install` or `pip install transformer-lens`."
            ) from exc

        logger.info(
            "Loading %s onto %s with dtype %s",
            self.model_name,
            self.device,
            self.dtype,
        )

        try:
            model = HookedTransformer.from_pretrained(
                self.model_name,
                device=self.device,
                dtype=resolve_dtype(self.dtype),
                **self._auth_kwargs(HookedTransformer.from_pretrained),
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load model {self.model_name!r} on device {self.device!r}. "
                "Check the model name, that the device is available, and that a "
                "valid HF_TOKEN is set for gated repositories."
            ) from exc

        model.eval()
        self.model = model
        self.tokenizer = model.tokenizer
        logger.info("Loaded %s with %d layers", self.model_name, model.cfg.n_layers)
        return model

    def _auth_kwargs(self, loader: Any) -> dict[str, Any]:
        """Build the authentication keyword arguments for the model loader.

        TransformerLens has moved the token argument around across releases:
        older versions accept ``hf_api_key`` directly, newer ones forward
        ``**from_pretrained_kwargs`` to the underlying ``transformers`` loader,
        which expects ``token``. The correct spelling is chosen by inspecting
        the loader signature so that the engine works on both.

        Args:
            loader: The ``from_pretrained`` callable being invoked.

        Returns:
            Keyword arguments to forward. Empty when no token is configured.
        """
        if not self.hf_token:
            return {}

        try:
            parameters = signature(loader).parameters
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return {"token": self.hf_token}

        if "hf_api_key" in parameters:
            return {"hf_api_key": self.hf_token}
        return {"token": self.hf_token}

    def _require_model(self) -> HookedTransformer:
        """Return the loaded model, loading it on first use.

        Returns:
            The cached :class:`~transformer_lens.HookedTransformer`.
        """
        if self.model is None:
            return self.load_model()
        return self.model

    # ------------------------------------------------------------------
    # Activation collection
    # ------------------------------------------------------------------

    @property
    def n_layers(self) -> int:
        """Number of transformer blocks in the loaded model.

        Returns:
            The block count.

        Raises:
            RuntimeError: If the model has not been loaded yet.
        """
        model = self.model
        if model is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")
        return int(model.cfg.n_layers)

    @property
    def middle_layer(self) -> int:
        """Index of the middle block, the default extraction site.

        Mid-stack layers tend to carry the most linearly separable semantic
        features, which is why extraction defaults here rather than to the last
        layer where representations are already specialised for next-token
        prediction.

        Returns:
            ``n_layers // 2``.
        """
        return self.n_layers // 2

    def _resolve_layer(self, layer: int | None) -> int:
        """Normalise a possibly negative or omitted layer index.

        Args:
            layer: Requested layer. ``None`` selects :attr:`middle_layer`;
                negative values count back from the end of the stack.

        Returns:
            A non-negative layer index.

        Raises:
            IndexError: If the index falls outside the model's block range.
        """
        n_layers = self.n_layers
        if layer is None:
            return self.middle_layer

        resolved = layer + n_layers if layer < 0 else layer
        if not 0 <= resolved < n_layers:
            raise IndexError(f"Layer {layer} is out of range for a model with {n_layers} layers")
        return resolved

    def _build_attention_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        """Build a padding mask for a batch of token ids.

        Only leading pads (left padding) or trailing pads (right padding) are
        treated as padding. An interior token that happens to share the pad id -
        common for GPT-2 style tokenizers where ``pad == eos == bos`` - is real
        content and stays unmasked.

        Args:
            tokens: Token ids of shape ``(batch, seq)``.

        Returns:
            A mask of shape ``(batch, seq)`` holding 1 for real tokens and 0
            for padding.
        """
        mask = torch.ones_like(tokens, dtype=torch.long)

        tokenizer = self.tokenizer
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if tokenizer is None or pad_token_id is None:
            return mask

        is_content = tokens.ne(pad_token_id).long()
        if getattr(tokenizer, "padding_side", "right") == "right":
            # A position is trailing padding when no content follows it.
            trailing = is_content.flip(-1).cumsum(-1).flip(-1) == 0
            mask[trailing] = 0
        else:
            leading = is_content.cumsum(-1) == 0
            mask[leading] = 0
            # With left padding and pad == bos, the prepended BOS looks like
            # padding; restore the last leading pad position for each row.
            if getattr(tokenizer, "bos_token_id", None) == pad_token_id:
                bos_positions = leading.sum(-1) - 1
                rows = torch.arange(mask.shape[0], device=mask.device)
                keep = bos_positions >= 0
                mask[rows[keep], bos_positions[keep]] = 1

        return mask

    def collect_activations(
        self,
        prompts: list[str],
        layer: int,
    ) -> torch.Tensor:
        """Return one residual stream vector per prompt, read at ``layer``.

        Each prompt is run through the model and the ``hook_resid_post``
        activations of ``layer`` are collected, then averaged over that prompt's
        real tokens. Padding positions are excluded via an attention mask, so a
        short prompt batched with long ones is not diluted by pad tokens.

        Activations are accumulated in float32 even when the model runs in
        bfloat16, because summing hundreds of low-precision values loses more
        signal than the cast costs.

        This is the per-prompt view that linear probes need.
        :meth:`_compute_mean_activation` reduces it to a single direction.

        Args:
            prompts: Texts to run through the model.
            layer: Index of the transformer block to read from. Negative
                indices count from the end of the stack.

        Returns:
            A tensor of shape ``(len(prompts), d_model)`` on the CPU, in
            float32, in the order the prompts were given.

        Raises:
            ValueError: If ``prompts`` is empty.
            IndexError: If ``layer`` is out of range.
            RuntimeError: If the model has not been loaded.
        """
        if not prompts:
            raise ValueError("prompts must contain at least one text")

        model = self._require_model()
        resolved_layer = self._resolve_layer(layer)
        hook_name = RESID_POST_HOOK.format(layer=resolved_layer)

        per_prompt_means: list[torch.Tensor] = []

        with torch.no_grad():
            for start in range(0, len(prompts), self.batch_size):
                batch = prompts[start : start + self.batch_size]
                tokens = model.to_tokens(batch)
                mask = self._build_attention_mask(tokens).to(tokens.device)

                _, cache = model.run_with_cache(
                    tokens,
                    attention_mask=mask,
                    names_filter=hook_name,
                    stop_at_layer=resolved_layer + 1,
                    return_type=None,
                )

                # (batch, seq, d_model)
                activations = cache[hook_name].to(torch.float32)
                weights = mask.to(torch.float32).unsqueeze(-1)

                summed = (activations * weights).sum(dim=1)
                counts = weights.sum(dim=1).clamp(min=1.0)
                per_prompt_means.append((summed / counts).cpu())

                del cache

        return torch.cat(per_prompt_means, dim=0)

    def _compute_mean_activation(
        self,
        prompts: list[str],
        layer: int,
    ) -> torch.Tensor:
        """Return the mean residual stream activation of ``prompts`` at ``layer``.

        Every prompt contributes equally regardless of its length: each is
        averaged over its own real tokens first (see
        :meth:`collect_activations`), and those per-prompt vectors are then
        averaged together.

        Args:
            prompts: Texts to run through the model.
            layer: Index of the transformer block to read from.

        Returns:
            A 1-D tensor of shape ``(d_model,)`` on the CPU, in float32.

        Raises:
            ValueError: If ``prompts`` is empty.
            IndexError: If ``layer`` is out of range.
            RuntimeError: If the model has not been loaded.
        """
        return self.collect_activations(prompts, layer).mean(dim=0)

    # ------------------------------------------------------------------
    # Concept vectors
    # ------------------------------------------------------------------

    def _resolve_examples(self, concept: str, examples: list[str] | None) -> list[str]:
        """Resolve positive examples for ``concept``.

        Args:
            concept: Concept identifier, matched against ``concept_id`` in the
                dataset when ``examples`` is omitted.
            examples: Explicit prompts. ``None`` triggers a dataset lookup; an
                empty list is an error rather than a lookup, so that a caller
                passing a filtered-to-empty list is told about it.

        Returns:
            The positive prompts.

        Raises:
            ValueError: If ``examples`` is empty, the concept is unknown, or
                the matching dataset entry carries no examples.
        """
        if examples is not None:
            if not examples:
                raise ValueError("at least one example is required to extract a vector")
            return list(examples)

        concepts = load_concepts(self.dataset_path)
        matches = [entry for entry in concepts if entry.concept_id == concept]
        if not matches:
            known = ", ".join(entry.concept_id for entry in concepts)
            raise ValueError(
                f"Concept {concept!r} was not found in {self.dataset_path}. Known ids: {known}"
            )

        resolved = matches[0].all_examples
        if not resolved:
            raise ValueError(
                f"Concept {concept!r} has no examples in {self.dataset_path}; "
                "add examples_ar/examples_en or pass examples explicitly."
            )
        logger.debug("Resolved %d examples for concept %s", len(resolved), concept)
        return resolved

    def extract_vector(
        self,
        concept: str,
        examples: list[str] | None = None,
        layer: int | None = None,
        contrast_examples: list[str] | None = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Extract the representation vector of ``concept`` at ``layer``.

        Implements the contrastive mean-difference recipe:

        1. Mean residual stream activation over the positive prompts.
        2. Mean residual stream activation over the negative prompts. When no
           negatives are supplied, a deterministic bank of culturally neutral
           Arabic and English sentences is used as the baseline.
        3. The concept direction is the difference of the two means.
        4. The difference is scaled to unit L2 norm.

        The result is cached in :attr:`concept_vectors` and the layer it came
        from in :attr:`extraction_layers`.

        Args:
            concept: Concept identifier, used as the cache key. When
                ``examples`` is omitted this is also looked up as a
                ``concept_id`` in :attr:`dataset_path`.
            examples: Prompts that express the concept. Omit to load them from
                the dataset.
            layer: Block to read from. ``None`` selects :attr:`middle_layer`;
                negative values count back from the end of the stack.
            contrast_examples: Prompts that deliberately lack the concept.
                Omit to fall back on the generated neutral baseline.
            normalize: Whether to scale the result to unit L2 norm. Disable
                only when the raw effect magnitude is the object of study.

        Returns:
            A 1-D tensor of shape ``(d_model,)`` holding the concept direction.

        Raises:
            ValueError: If ``concept`` is empty, ``examples`` is an empty list,
                the concept cannot be resolved from the dataset, or the two
                means are identical so that no direction exists.
            IndexError: If ``layer`` is out of range.
        """
        if not concept or not concept.strip():
            raise ValueError("concept must be a non-empty string")

        positive_examples = self._resolve_examples(concept, examples)

        self._require_model()
        resolved_layer = self._resolve_layer(layer)
        negative_examples = build_contrast_examples(positive_examples, contrast_examples)

        logger.info(
            "Extracting %s at layer %d from %d positive and %d negative prompts",
            concept,
            resolved_layer,
            len(positive_examples),
            len(negative_examples),
        )

        positive_mean = self._compute_mean_activation(positive_examples, resolved_layer)
        negative_mean = self._compute_mean_activation(negative_examples, resolved_layer)
        direction = positive_mean - negative_mean

        if normalize:
            norm = torch.linalg.vector_norm(direction)
            if norm <= torch.finfo(direction.dtype).eps:
                raise ValueError(
                    f"The positive and negative means for {concept!r} are identical, so no "
                    "direction can be normalised. Check that the two prompt sets differ."
                )
            direction = direction / norm

        self.concept_vectors[concept] = direction
        self.extraction_layers[concept] = resolved_layer
        return direction

    def extract_all_vectors(
        self,
        layer: int | None = None,
        contrast_examples: list[str] | None = None,
        normalize: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Extract a vector for every concept in :attr:`dataset_path`.

        Args:
            layer: Block to read from, forwarded to :meth:`extract_vector`.
            contrast_examples: Shared negatives, forwarded to
                :meth:`extract_vector`.
            normalize: Whether to L2-normalise each vector.

        Returns:
            A mapping from ``concept_id`` to the extracted direction. Concepts
            without examples are skipped with a warning rather than aborting
            the whole run.
        """
        vectors: dict[str, torch.Tensor] = {}
        for entry in load_concepts(self.dataset_path):
            try:
                vectors[entry.concept_id] = self.extract_vector(
                    concept=entry.concept_id,
                    examples=entry.all_examples or None,
                    layer=layer,
                    contrast_examples=contrast_examples,
                    normalize=normalize,
                )
            except ValueError as exc:
                logger.warning("Skipping concept %s: %s", entry.concept_id, exc)
        return vectors

    def save_vectors_to_hf(
        self,
        dataset_name: str = "al3obdi/thaqafa-repe-vectors",
        token: str | None = None,
    ) -> str:
        """Save all extracted concept vectors to a Hugging Face Dataset.

        Delegates to :func:`src.utils.hf_integration.save_vectors_to_hf`, passing
        the engine's cached vectors, extraction layers, and model name
        as metadata. The token is resolved from the environment when not
        provided.

        Args:
            dataset_name: Target HF dataset repository.
            token: Hugging Face access token. Defaults to ``HF_TOKEN``.

        Returns:
            The URL of the updated dataset.

        Raises:
            ValueError: If no vectors have been extracted yet.
            MissingTokenError: If no token is available.
            HFIntegrationError: If the upload fails.
        """
        if not self.concept_vectors:
            raise ValueError("No concept vectors to save. Call extract_vector() first.")

        from src.utils.hf_integration import save_vectors_to_hf as _save

        metadata = {
            "model_name": self.model_name,
            "extraction_layers": dict(self.extraction_layers),
        }
        return _save(
            self.concept_vectors,
            dataset_name=dataset_name,
            metadata=metadata,
            token=token,
        )

    def load_vectors_from_hf(
        self,
        dataset_name: str = "al3obdi/thaqafa-repe-vectors",
        concept_ids: list[str] | None = None,
        token: str | None = None,
    ) -> dict[str, torch.Tensor]:
        """Load vectors from a Hugging Face Dataset into :attr:`concept_vectors`.

        Delegates to :func:`src.utils.hf_integration.load_vectors_from_hf`. Loaded
        vectors are merged into the engine's cache — existing entries with
        the same key are overwritten.

        Args:
            dataset_name: Source HF dataset repository.
            concept_ids: Optional list of concept IDs to load. ``None``
                loads everything.
            token: Hugging Face access token. Defaults to ``HF_TOKEN``.

        Returns:
            The loaded mapping (also stored in :attr:`concept_vectors`).

        Raises:
            MissingTokenError: If no token is available.
            HFIntegrationError: If the download fails.
        """
        from src.utils.hf_integration import load_vectors_from_hf as _load

        loaded = _load(dataset_name=dataset_name, concept_ids=concept_ids, token=token)
        self.concept_vectors.update(loaded)
        logger.info("Loaded %d vectors from HF into engine cache", len(loaded))
        return loaded

    def sync_with_space(
        self,
        space_name: str = "al3obdi/thaqafa-repe-extraction",
        token: str | None = None,
    ) -> dict[str, Any]:
        """Check the ZeroGPU extraction Space status and sync results.

        Delegates to :func:`src.utils.hf_integration.sync_with_space`.

        Args:
            space_name: HF Space repository.
            token: Hugging Face access token. Defaults to ``HF_TOKEN``.

        Returns:
            A dictionary with Space status information.

        Raises:
            MissingTokenError: If no token is available.
            HFIntegrationError: If the API request fails.
        """
        from src.utils.hf_integration import sync_with_space as _sync

        return _sync(space_name=space_name, token=token)

    def extract_via_space(
        self,
        concept_ids: list[str],
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        space_name: str = "al3obdi/thaqafa-repe-extraction",
        dataset_name: str = "al3obdi/thaqafa-repe-vectors",
        token: str | None = None,
    ) -> dict[str, torch.Tensor]:
        """Trigger extraction on the ZeroGPU Space and load results locally.

        This is the high-level automation entry point. It:

        1. Connects to the HF ZeroGPU Space via ``gradio_client``.
        2. Submits an extraction job for the given concepts and model.
        3. Polls until the job completes.
        4. Loads the resulting vectors from the HF Dataset into
           :attr:`concept_vectors`.

        The local machine never loads the model - all GPU work happens on
        the Space. This method is safe to call from a CPU-only environment.

        Args:
            concept_ids: Concept identifiers to extract, e.g.
                ``["wasta_001", "diyafa_001"]``.
            model_name: Hugging Face model identifier to extract from.
            space_name: HF Space repository to connect to.
            dataset_name: HF dataset to load results from after extraction.
            token: Hugging Face access token. Defaults to ``HF_TOKEN``.

        Returns:
            The loaded vectors, also merged into :attr:`concept_vectors`.

        Raises:
            ImportError: If ``gradio_client`` is not installed.
            RuntimeError: If the Space is unreachable or the job fails.
            MissingTokenError: If no token is available.
        """
        import time

        from src.utils.hf_integration import _resolve_token

        resolved_token = _resolve_token(token)

        try:
            from gradio_client import Client
        except ImportError as exc:
            raise ImportError(
                "gradio_client is required for Space automation. "
                "Install it with: pip install gradio_client"
            ) from exc

        space_url = f"https://{space_name.replace(chr(47), chr(45))}.hf.space"
        logger.info("Connecting to Space: %s", space_url)

        try:
            client = Client(space_url, hf_token=resolved_token)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to connect to Space {space_name!r}: {exc}. "
                "The Space may be sleeping or in an error state."
            ) from exc

        concept_str = ", ".join(concept_ids)
        logger.info("Submitting extraction job: concepts=%s, model=%s", concept_str, model_name)

        try:
            job = client.submit(
                fn_index=0,
                inputs=[concept_str, model_name, dataset_name],
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to submit extraction job: {exc}") from exc

        # Poll for completion
        max_wait = 600
        poll_interval = 5
        elapsed = 0
        while elapsed < max_wait:
            status = job.status()
            code = getattr(status, "code", None)
            if code == "SUCCESS":
                logger.info("Space extraction completed: %s", job.result())
                break
            if code == "ERROR":
                raise RuntimeError(f"Space extraction job failed: {status}")
            logger.debug("Waiting for Space job... (%ds, status=%s)", elapsed, code)
            time.sleep(poll_interval)
            elapsed += poll_interval
        else:
            raise RuntimeError(
                f"Space extraction timed out after {max_wait}s. "
                "The Space may be cold-starting or the model is too large."
            )

        # Load results from HF Dataset
        loaded = self.load_vectors_from_hf(
            dataset_name=dataset_name,
            concept_ids=concept_ids,
            token=resolved_token,
        )
        logger.info("Loaded %d vectors from Space extraction", len(loaded))
        return loaded

    def save_vectors(self, path: Path | str) -> Path:
        """Write the cached concept vectors to disk.

        Args:
            path: Destination ``.pt`` file. Parent directories are created.

        Returns:
            The path that was written.

        Raises:
            ValueError: If no vectors have been extracted yet.
        """
        if not self.concept_vectors:
            raise ValueError("No concept vectors to save. Call extract_vector() first.")

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_name": self.model_name,
                "concept_vectors": self.concept_vectors,
                "extraction_layers": self.extraction_layers,
            },
            destination,
        )
        logger.info("Saved %d concept vectors to %s", len(self.concept_vectors), destination)
        return destination

    # ------------------------------------------------------------------
    # Injection and steering
    # ------------------------------------------------------------------

    @property
    def active_hook_names(self) -> list[str]:
        """Hook names currently carrying an injection, in attachment order."""
        return [handle.hook_name for handle in self._active_hooks if handle.is_active]

    def _resolve_injection_layers(self, concept: str, layers: list[int] | None) -> list[int]:
        """Decide which layers an injection should target.

        Args:
            concept: Concept being injected, used to look up the layer its
                vector was extracted from.
            layers: Explicit layers. ``None`` falls back to the extraction
                layer, or to :attr:`middle_layer` if the vector was cached
                without one (for example assigned directly in a test).

        Returns:
            Non-negative, de-duplicated layer indices in ascending order.

        Raises:
            ValueError: If ``layers`` is an empty list.
            IndexError: If any layer is out of range.
        """
        if layers is None:
            default = self.extraction_layers.get(concept)
            return [self.middle_layer if default is None else default]

        if not layers:
            raise ValueError("layers must contain at least one layer, or be None")

        return sorted({self._resolve_layer(layer) for layer in layers})

    def _make_injection_hook(
        self,
        vector: torch.Tensor,
        strength: float,
    ) -> Any:
        """Build the forward hook that adds a scaled concept vector.

        The hook adds ``strength * vector`` at every sequence position. The
        vector has shape ``(d_model,)`` and the activation ``(batch, seq,
        d_model)``, so ordinary broadcasting over the trailing dimension applies
        the same offset to every token of every sequence in the batch. The
        vector is cast to the activation's device and dtype on each call, which
        keeps the hook correct when the model is sharded or running in bfloat16
        while vectors are cached in float32.

        Args:
            vector: Concept direction of shape ``(d_model,)``.
            strength: Coefficient applied to the vector.

        Returns:
            A callable with the TransformerLens hook signature
            ``(activation, hook) -> activation``.
        """

        def injection_hook(activation: torch.Tensor, hook: Any) -> torch.Tensor:
            offset = vector.to(device=activation.device, dtype=activation.dtype)
            return activation + strength * offset

        return injection_hook

    def inject_vector(
        self,
        concept: str,
        strength: float = 1.0,
        layers: list[int] | None = None,
    ) -> list[InjectionHandle]:
        """Steer generation by adding a cached concept vector to the residual stream.

        A forward hook is registered on ``blocks.{layer}.hook_resid_post`` for
        each requested layer. Each hook adds ``strength * concept_vector`` to
        every position of the activation passing through it. Positive strengths
        amplify the concept, negative strengths suppress it, and ``0.0``
        reproduces the unsteered model.

        The hooks stay attached until removed. Prefer :meth:`steering`, which
        cleans up automatically; call this directly only when the steering has
        to outlive a single block of code.

        Args:
            concept: Name of a concept previously passed to
                :meth:`extract_vector`.
            strength: Scaling coefficient applied to the concept vector.
            layers: Layers to hook. Defaults to the layer recorded in
                :attr:`extraction_layers`, or the middle layer if unknown.

        Returns:
            The handles that were attached, also appended to
            ``self._active_hooks``.

        Raises:
            KeyError: If ``concept`` has no cached vector.
            ValueError: If ``layers`` is empty, or the cached vector is not a
                1-D tensor matching the model's ``d_model``.
            IndexError: If a requested layer is out of range.
            RuntimeError: If the model has not been loaded.
        """
        if concept not in self.concept_vectors:
            raise KeyError(
                f"No vector stored for concept {concept!r}. "
                "Call extract_vector() before inject_vector()."
            )

        model = self._require_model()
        vector = self.concept_vectors[concept]
        self._validate_vector(concept, vector)
        target_layers = self._resolve_injection_layers(concept, layers)

        handles: list[InjectionHandle] = []
        for layer in target_layers:
            hook_name = RESID_POST_HOOK.format(layer=layer)
            # nn.Module.__getattr__ is typed as returning Tensor | Module, so a
            # HookPoint reached through mod_dict has no usable static type. The
            # explicit Any keeps that unavoidable looseness in one place.
            hook_point: Any = model.mod_dict[hook_name]

            model.add_hook(hook_name, self._make_injection_hook(vector, strength), dir="fwd")

            # add_hook returns None; TransformerLens appends the handle it
            # created to the hook point, so the newest entry is ours.
            handles.append(
                InjectionHandle(
                    concept=concept,
                    layer=layer,
                    hook_name=hook_name,
                    strength=strength,
                    _hook_point=hook_point,
                    _lens_handle=hook_point.fwd_hooks[-1],
                )
            )

        self._active_hooks.extend(handles)
        logger.info(
            "Injected %s at strength %.3f into layer(s) %s",
            concept,
            strength,
            ", ".join(str(layer) for layer in target_layers),
        )
        return handles

    def _validate_vector(self, concept: str, vector: torch.Tensor) -> None:
        """Check that a cached vector can be broadcast onto the residual stream.

        Args:
            concept: Concept the vector belongs to, used in the error message.
            vector: The cached direction.

        Raises:
            ValueError: If the vector is not 1-D or its width does not match
                the model's ``d_model``.
        """
        if vector.ndim != 1:
            raise ValueError(
                f"Concept vector for {concept!r} must be 1-D of shape (d_model,), "
                f"got shape {tuple(vector.shape)}"
            )

        model = self.model
        d_model = int(model.cfg.d_model) if model is not None else None
        if d_model is not None and vector.shape[0] != d_model:
            raise ValueError(
                f"Concept vector for {concept!r} has width {vector.shape[0]}, but the model's "
                f"d_model is {d_model}. Re-extract the vector with this model."
            )

    def remove_hooks(self, handles: list[InjectionHandle] | None = None) -> int:
        """Detach steering hooks and forget them.

        Args:
            handles: Specific handles to remove. ``None`` removes every hook
                this engine has attached, which is the usual cleanup path.
                Passing a subset lets nested steering scopes unwind
                independently.

        Returns:
            How many hooks were actually detached. Zero is a valid, silent
            outcome: calling this with nothing attached is not an error.
        """
        targets = self._active_hooks if handles is None else handles

        removed = 0
        for handle in list(targets):
            if handle.is_active:
                handle.remove()
                removed += 1

        if handles is None:
            self._active_hooks.clear()
        else:
            requested = {id(handle) for handle in handles}
            self._active_hooks = [
                handle for handle in self._active_hooks if id(handle) not in requested
            ]

        if removed:
            logger.info("Removed %d steering hook(s)", removed)
        return removed

    @contextmanager
    def steering(
        self,
        concept: str,
        strength: float = 1.0,
        layers: list[int] | None = None,
    ) -> Iterator[list[InjectionHandle]]:
        """Steer the model for the duration of a ``with`` block.

        This is the recommended way to inject: the hooks are removed on exit
        whether the body returns normally or raises, so a failed generation
        cannot leave a model silently steered for the rest of the session. Only
        the hooks this scope added are removed, so nested scopes and any
        unrelated hooks the caller attached are left intact.

        Args:
            concept: Name of a concept previously passed to
                :meth:`extract_vector`.
            strength: Scaling coefficient applied to the concept vector.
            layers: Layers to hook. Defaults to the extraction layer.

        Yields:
            The handles attached for this scope.

        Raises:
            KeyError: If ``concept`` has no cached vector.

        Example:
            >>> with engine.steering("diyafa_001", strength=2.0):  # doctest: +SKIP
            ...     steered = engine.model.generate("A guest arrives", max_new_tokens=20)
        """
        handles = self.inject_vector(concept=concept, strength=strength, layers=layers)
        try:
            yield handles
        finally:
            self.remove_hooks(handles)
