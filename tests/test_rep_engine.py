"""Unit tests for :mod:`src.models.rep_engine`.

The extraction and injection algorithms are still stubs, so these tests pin
down the parts of the contract that already exist: construction, the concept
vector cache, and input validation.
"""

from __future__ import annotations

import pytest
import torch

from src.models.rep_engine import DEFAULT_MODEL_NAME, CulturalRepE


class TestClassInitialization:
    """Construction of :class:`CulturalRepE`."""

    def test_class_initialization(self) -> None:
        engine = CulturalRepE(model_name="test/model", device="cpu", dtype="float32")

        assert engine.model_name == "test/model"
        assert engine.device == "cpu"
        assert engine.dtype == "float32"
        assert engine.model is None
        assert engine.tokenizer is None
        assert engine.concept_vectors == {}

    def test_default_arguments(self) -> None:
        engine = CulturalRepE()

        assert engine.model_name == DEFAULT_MODEL_NAME
        assert engine.device == "cuda"
        assert engine.dtype == "bfloat16"

    @pytest.mark.parametrize("model_name", ["", "   "])
    def test_empty_model_name_is_rejected(self, model_name: str) -> None:
        with pytest.raises(ValueError, match="model_name"):
            CulturalRepE(model_name=model_name)


class TestConceptVectorStorage:
    """The concept vector cache."""

    def test_concept_vector_storage(self) -> None:
        engine = CulturalRepE(model_name="test/model", device="cpu")
        vector = torch.ones(8)

        engine.concept_vectors["diyafa"] = vector

        assert "diyafa" in engine.concept_vectors
        assert torch.equal(engine.concept_vectors["diyafa"], vector)
        assert engine.concept_vectors["diyafa"].shape == (8,)

    def test_cache_is_not_shared_between_instances(self) -> None:
        first = CulturalRepE(model_name="test/model", device="cpu")
        second = CulturalRepE(model_name="test/model", device="cpu")

        first.concept_vectors["muruah"] = torch.zeros(4)

        assert "muruah" not in second.concept_vectors


class TestInvalidConceptHandling:
    """Validation of concept arguments."""

    def test_invalid_concept_handling(self) -> None:
        engine = CulturalRepE(model_name="test/model", device="cpu")

        with pytest.raises(KeyError, match="wasta"):
            engine.inject_vector("wasta")

    @pytest.mark.parametrize("concept", ["", "   "])
    def test_empty_concept_is_rejected(self, concept: str) -> None:
        engine = CulturalRepE(model_name="test/model", device="cpu")

        with pytest.raises(ValueError, match="concept"):
            engine.extract_vector(concept, examples=["some example"])

    def test_extraction_requires_examples(self) -> None:
        engine = CulturalRepE(model_name="test/model", device="cpu")

        with pytest.raises(ValueError, match="example"):
            engine.extract_vector("diyafa", examples=[])

    def test_unimplemented_methods_raise(self) -> None:
        engine = CulturalRepE(model_name="test/model", device="cpu")

        with pytest.raises(NotImplementedError):
            engine.load_model()

        with pytest.raises(NotImplementedError):
            engine.extract_vector("diyafa", examples=["أكرم ضيافته"])
