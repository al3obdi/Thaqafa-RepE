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
direction. The result is L2-normalised so that the injection strength used in
Phase 3 is the only magnitude knob.

Injection is still a stub; it is the subject of Phase 3.
"""

from __future__ import annotations

import logging
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
        self._active_hooks: list[Any] = []

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

    def _compute_mean_activation(
        self,
        prompts: list[str],
        layer: int,
    ) -> torch.Tensor:
        """Return the mean residual stream activation of ``prompts`` at ``layer``.

        Each prompt is run through the model and the ``hook_resid_post``
        activations of ``layer`` are collected. Padding positions are excluded
        via an attention mask, so a short prompt in a batch of long ones is not
        diluted by pad tokens. The per-prompt mean over real tokens is computed
        first, then averaged across prompts, so that every prompt contributes
        equally regardless of its length.

        Activations are accumulated in float32 even when the model runs in
        bfloat16, because summing hundreds of low-precision values loses more
        signal than the cast costs.

        Args:
            prompts: Texts to run through the model.
            layer: Non-negative index of the transformer block to read from.

        Returns:
            A 1-D tensor of shape ``(d_model,)`` on the CPU, in float32.

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

        stacked = torch.cat(per_prompt_means, dim=0)
        return stacked.mean(dim=0)

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
    # Injection (Phase 3)
    # ------------------------------------------------------------------

    def inject_vector(
        self,
        concept: str,
        strength: float = 1.0,
        layers: list[int] | None = None,
    ) -> None:
        """Steer generation by adding a cached concept vector to the residual stream.

        The final implementation registers forward hooks on the requested
        layers; each hook adds ``strength * self.concept_vectors[concept]`` to
        the hidden states flowing through that layer. Negative ``strength``
        values suppress the concept instead of amplifying it. Handles are kept
        in ``self._active_hooks`` so that they can be removed later.

        Args:
            concept: Name of a concept previously passed to
                :meth:`extract_vector`.
            strength: Scaling coefficient applied to the concept vector.
            layers: Layers to hook. Defaults to the layer recorded in
                :attr:`extraction_layers` when omitted.

        Raises:
            KeyError: If ``concept`` has no cached vector.
            NotImplementedError: Always, until the injection logic is written.
        """
        if concept not in self.concept_vectors:
            raise KeyError(
                f"No vector stored for concept {concept!r}. "
                "Call extract_vector() before inject_vector()."
            )

        raise NotImplementedError("Vector injection is not implemented yet.")
