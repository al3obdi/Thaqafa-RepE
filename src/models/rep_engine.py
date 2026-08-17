"""Representation engineering engine for Arab cultural concepts.

This module exposes :class:`CulturalRepE`, the central abstraction of the
project. It is responsible for three things:

1. Loading a causal language model and its tokenizer.
2. Extracting a *concept vector* - a direction in the residual stream that
   corresponds to a cultural concept such as ``الضيافة`` (hospitality).
3. Injecting a previously extracted concept vector back into the forward pass
   in order to steer generation towards (or away from) that concept.

The implementations here are intentionally left as stubs: the extraction and
injection algorithms are the subject of the research phase of the project.
Every stub documents the contract that the final implementation must honour so
that downstream scripts, notebooks and tests can be written against it today.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    import torch

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"


class CulturalRepE:
    """Extract and inject cultural concept vectors for a causal language model.

    Args:
        model_name: Hugging Face identifier of the base model, for example
            ``"meta-llama/Meta-Llama-3-8B-Instruct"``.
        device: Device the model should be placed on (``"cuda"``, ``"cpu"``
            or ``"mps"``).
        dtype: String name of the torch dtype used for the model weights, for
            example ``"bfloat16"`` or ``"float16"``.

    Attributes:
        concept_vectors: Mapping from concept name to the extracted direction.
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
    ) -> None:
        if not model_name or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")

        self.model_name: str = model_name
        self.device: str = device
        self.dtype: str = dtype

        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self.concept_vectors: dict[str, torch.Tensor] = {}
        self._active_hooks: list[Any] = []

    def load_model(self) -> None:
        """Load the base model and tokenizer onto :attr:`device`.

        The final implementation is expected to load the model through
        ``transformers.AutoModelForCausalLM`` (or ``transformer_lens`` when
        fine-grained activation access is required), move it to
        :attr:`device`, cast it to :attr:`dtype` and put it in evaluation
        mode. It must be idempotent: calling it twice should not reload the
        weights.

        Raises:
            NotImplementedError: Always, until the loading logic is written.
        """
        raise NotImplementedError("Model loading is not implemented yet.")

    def extract_vector(
        self,
        concept: str,
        examples: list[str],
        layer: int = -1,
        contrast_examples: list[str] | None = None,
    ) -> torch.Tensor:
        """Extract the representation vector of ``concept`` at ``layer``.

        The planned algorithm is the standard reading-vector recipe from the
        representation engineering literature: run every prompt through the
        model, collect the hidden state of ``layer`` at the final token, and
        take the difference between the mean positive activation and the mean
        contrastive (or neutral) activation. The resulting direction is
        normalised and cached in :attr:`concept_vectors`.

        Args:
            concept: Human readable concept name used as the cache key, for
                example ``"diyafa"``.
            examples: Prompts that express the concept.
            layer: Index of the transformer block to read from. Negative
                indices count from the end of the stack.
            contrast_examples: Optional prompts that deliberately lack the
                concept. When omitted the mean activation over ``examples`` is
                used directly.

        Returns:
            A 1-D tensor of shape ``(hidden_size,)`` holding the concept
            direction.

        Raises:
            ValueError: If ``concept`` is empty or ``examples`` is empty.
            NotImplementedError: Always, until the extraction logic is written.
        """
        if not concept or not concept.strip():
            raise ValueError("concept must be a non-empty string")
        if not examples:
            raise ValueError("at least one example is required to extract a vector")

        raise NotImplementedError("Vector extraction is not implemented yet.")

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
            layers: Layers to hook. Defaults to the layer the vector was
                extracted from when omitted.

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

    def _compute_mean_activation(
        self,
        prompts: list[str],
        layer: int,
    ) -> torch.Tensor:
        """Return the mean last-token hidden state of ``prompts`` at ``layer``.

        Args:
            prompts: Texts to run through the model.
            layer: Index of the transformer block to read from.

        Returns:
            A 1-D tensor of shape ``(hidden_size,)``.

        Raises:
            NotImplementedError: Always, until the forward pass is written.
        """
        raise NotImplementedError("Mean activation computation is not implemented yet.")
