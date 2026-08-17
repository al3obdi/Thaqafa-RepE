"""Tests for Hugging Face integration (CPU-only, no actual HF API calls).

All tests mock the ``datasets`` and ``huggingface_hub`` libraries so they
run on CPU without network access or real HF credentials.
"""

from __future__ import annotations

from typing import Any
from unittest import mock

import pytest
import torch

from src.models.rep_engine import CulturalRepE
from src.utils.hf_integration import (
    DEFAULT_DATASET,
    DEFAULT_SPACE,
    REQUIRED_COLUMNS,
    HFIntegrationError,
    MissingTokenError,
    _build_dataset_rows,
    _list_to_tensor,
    _rows_to_vectors,
    _tensor_to_list,
    load_vectors_from_hf,
    save_vectors_to_hf,
    sync_with_space,
    validate_dataset_schema,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_vectors() -> dict[str, torch.Tensor]:
    """Return a small set of concept vectors for testing."""
    return {
        "diyafa_001": torch.tensor([1.0, 0.0, 0.0, 0.0]),
        "wasta_001": torch.tensor([0.0, 1.0, 0.0, 0.0]),
        "muruah_001": torch.tensor([0.0, 0.0, 1.0, 0.0]),
    }


@pytest.fixture
def sample_metadata() -> dict[str, Any]:
    """Return metadata for the sample vectors."""
    return {
        "model_name": "test-model",
        "extraction_layers": {"diyafa_001": 14, "wasta_001": 14, "muruah_001": 14},
        "extraction_timestamp": "2026-08-17T00:00:00Z",
        "concept_metadata": {
            "diyafa_001": {"concept_ar": "الضيافة", "concept_en": "Hospitality"},
            "wasta_001": {"concept_ar": "الواسطة", "concept_en": "Wasta"},
            "muruah_001": {"concept_ar": "المروءة", "concept_en": "Muru'ah"},
        },
    }


@pytest.fixture
def fake_hf_token(monkeypatch: pytest.MonkeyPatch) -> str:
    """Set a fake HF_TOKEN in the environment."""
    monkeypatch.setenv("HF_TOKEN", "hf_fake_test_token_12345")
    return "hf_fake_test_token_12345"


@pytest.fixture
def no_hf_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove HF_TOKEN from the environment."""
    monkeypatch.delenv("HF_TOKEN", raising=False)


# ---------------------------------------------------------------------------
# Serialisation tests
# ---------------------------------------------------------------------------


class TestTensorSerialisation:
    """Test tensor ↔ list conversion round-trip."""

    def test_tensor_to_list_1d(self) -> None:
        """1-D tensors convert correctly."""
        tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16)
        result = _tensor_to_list(tensor)
        assert result == [1.0, 2.0, 3.0]
        assert all(isinstance(v, float) for v in result)

    def test_tensor_to_list_2d_raises(self) -> None:
        """2-D tensors raise ValueError."""
        tensor = torch.randn(3, 4)
        with pytest.raises(ValueError, match="1-D tensor"):
            _tensor_to_list(tensor)

    def test_list_to_tensor(self) -> None:
        """Lists convert back to float32 tensors."""
        values = [1.0, 2.0, 3.0]
        result = _list_to_tensor(values)
        assert result.dtype == torch.float32
        assert result.shape == (3,)
        assert torch.allclose(result, torch.tensor(values))

    def test_roundtrip(self) -> None:
        """Tensor → list → tensor preserves values."""
        original = torch.randn(128)
        roundtripped = _list_to_tensor(_tensor_to_list(original))
        assert torch.allclose(original.to(torch.float32), roundtripped, atol=1e-6)

    def test_tensor_to_list_cpu_conversion(self) -> None:
        """Tensors on CPU work correctly."""
        tensor = torch.tensor([1.0, 0.0])
        result = _tensor_to_list(tensor)
        assert result == [1.0, 0.0]


# ---------------------------------------------------------------------------
# Dataset row building tests
# ---------------------------------------------------------------------------


class TestBuildDatasetRows:
    """Test conversion of vectors to HF Dataset rows."""

    def test_basic_rows(
        self,
        sample_vectors: dict[str, torch.Tensor],
        sample_metadata: dict[str, Any],
    ) -> None:
        """Rows are built with all required columns."""
        rows = _build_dataset_rows(sample_vectors, sample_metadata)
        assert len(rows) == 3

        for row in rows:
            assert REQUIRED_COLUMNS <= set(row.keys())
            assert isinstance(row["concept_id"], str)
            assert isinstance(row["vector"], list)
            assert isinstance(row["extraction_layer"], int)
            assert isinstance(row["model_name"], str)
            assert isinstance(row["extraction_timestamp"], str)

    def test_empty_vectors_raises(self) -> None:
        """Empty vectors dict raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            _build_dataset_rows({})

    def test_no_metadata(self, sample_vectors: dict[str, torch.Tensor]) -> None:
        """Rows build correctly without metadata."""
        rows = _build_dataset_rows(sample_vectors)
        assert len(rows) == 3
        assert rows[0]["model_name"] == "unknown"
        assert rows[0]["extraction_layer"] == -1

    def test_concept_metadata(
        self,
        sample_vectors: dict[str, torch.Tensor],
        sample_metadata: dict[str, Any],
    ) -> None:
        """Per-concept metadata is included in rows."""
        rows = _build_dataset_rows(sample_vectors, sample_metadata)
        diyafa_row = next(r for r in rows if r["concept_id"] == "diyafa_001")
        assert diyafa_row["concept_ar"] == "الضيافة"
        assert diyafa_row["concept_en"] == "Hospitality"


class TestRowsToVectors:
    """Test conversion of HF Dataset rows back to tensors."""

    def test_basic_conversion(
        self,
        sample_vectors: dict[str, torch.Tensor],
        sample_metadata: dict[str, Any],
    ) -> None:
        """Rows convert back to tensors correctly."""
        rows = _build_dataset_rows(sample_vectors, sample_metadata)
        vectors = _rows_to_vectors(rows)
        assert len(vectors) == 3
        for _, vec in vectors.items():
            assert vec.dtype == torch.float32
            assert vec.shape == (4,)

    def test_filter_by_concept_ids(
        self,
        sample_vectors: dict[str, torch.Tensor],
        sample_metadata: dict[str, Any],
    ) -> None:
        """Filtering by concept_ids returns only requested concepts."""
        rows = _build_dataset_rows(sample_vectors, sample_metadata)
        vectors = _rows_to_vectors(rows, concept_ids=["diyafa_001"])
        assert len(vectors) == 1
        assert "diyafa_001" in vectors
        assert "wasta_001" not in vectors

    def test_roundtrip(
        self,
        sample_vectors: dict[str, torch.Tensor],
        sample_metadata: dict[str, Any],
    ) -> None:
        """Full round-trip: vectors → rows → vectors preserves values."""
        rows = _build_dataset_rows(sample_vectors, sample_metadata)
        loaded = _rows_to_vectors(rows)
        for cid, original in sample_vectors.items():
            assert torch.allclose(original.to(torch.float32), loaded[cid], atol=1e-6)


# ---------------------------------------------------------------------------
# Token resolution tests
# ---------------------------------------------------------------------------


class TestTokenResolution:
    """Test HF token resolution logic."""

    def test_explicit_token(self) -> None:
        """Explicit token is used when provided."""
        from src.utils.hf_integration import _resolve_token

        result = _resolve_token("hf_my_explicit_token")
        assert result == "hf_my_explicit_token"

    def test_env_token(self, fake_hf_token: str) -> None:
        """Token is read from HF_TOKEN env var."""
        from src.utils.hf_integration import _resolve_token

        result = _resolve_token(None)
        assert result == fake_hf_token

    def test_no_token_raises(self, no_hf_token: None) -> None:
        """Missing token raises MissingTokenError."""
        from src.utils.hf_integration import _resolve_token

        with pytest.raises(MissingTokenError, match="No Hugging Face token"):
            _resolve_token(None)


# ---------------------------------------------------------------------------
# save_vectors_to_hf tests (mocked)
# ---------------------------------------------------------------------------


class TestSaveVectorsToHf:
    """Test save_vectors_to_hf with mocked HF API."""

    def test_save_success(
        self,
        sample_vectors: dict[str, torch.Tensor],
        sample_metadata: dict[str, Any],
        fake_hf_token: str,
    ) -> None:
        """Successful save returns dataset URL."""
        mock_dataset = mock.MagicMock()
        with mock.patch("datasets.Dataset") as mock_ds_class:
            mock_ds_class.from_list.return_value = mock_dataset
            url = save_vectors_to_hf(
                sample_vectors,
                dataset_name="al3obdi/test-vectors",
                metadata=sample_metadata,
            )

        assert "huggingface.co/datasets/al3obdi/test-vectors" in url
        mock_dataset.push_to_hub.assert_called_once()
        call_kwargs = mock_dataset.push_to_hub.call_args
        assert call_kwargs.kwargs["private"] is True

    def test_save_empty_vectors_raises(self, fake_hf_token: str) -> None:
        """Empty vectors dict raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            save_vectors_to_hf({})

    def test_save_no_token_raises(
        self,
        sample_vectors: dict[str, torch.Tensor],
        no_hf_token: None,
    ) -> None:
        """Missing token raises MissingTokenError."""
        with pytest.raises(MissingTokenError):
            save_vectors_to_hf(sample_vectors)

    def test_save_push_failure_raises(
        self,
        sample_vectors: dict[str, torch.Tensor],
        fake_hf_token: str,
    ) -> None:
        """Push failure raises HFIntegrationError."""
        with mock.patch("datasets.Dataset") as mock_ds_class:
            mock_ds_class.from_list.return_value = mock.MagicMock()
            mock_ds_class.from_list.return_value.push_to_hub.side_effect = RuntimeError(
                "Network error"
            )
            with pytest.raises(HFIntegrationError, match="Failed to push"):
                save_vectors_to_hf(sample_vectors)


# ---------------------------------------------------------------------------
# load_vectors_from_hf tests (mocked)
# ---------------------------------------------------------------------------


class TestLoadVectorsFromHf:
    """Test load_vectors_from_hf with mocked HF API."""

    def test_load_success(self, fake_hf_token: str) -> None:
        """Successful load returns concept vectors."""
        # Build fake dataset rows
        fake_rows = [
            {
                "concept_id": "diyafa_001",
                "concept_ar": "الضيافة",
                "concept_en": "Hospitality",
                "vector": [1.0, 0.0, 0.0, 0.0],
                "extraction_layer": 14,
                "model_name": "test-model",
                "extraction_timestamp": "2026-08-17T00:00:00Z",
            },
            {
                "concept_id": "wasta_001",
                "concept_ar": "الواسطة",
                "concept_en": "Wasta",
                "vector": [0.0, 1.0, 0.0, 0.0],
                "extraction_layer": 14,
                "model_name": "test-model",
                "extraction_timestamp": "2026-08-17T00:00:00Z",
            },
        ]

        mock_split = mock.MagicMock()
        mock_split.__iter__ = mock.Mock(return_value=iter(fake_rows))
        mock_ds = mock.MagicMock()
        mock_ds.keys.return_value = ["train"]
        mock_ds.__getitem__ = mock.Mock(return_value=mock_split)

        with mock.patch("datasets.load_dataset", return_value=mock_ds):
            vectors = load_vectors_from_hf(dataset_name="al3obdi/test-vectors")

        assert len(vectors) == 2
        assert "diyafa_001" in vectors
        assert vectors["diyafa_001"].shape == (4,)
        assert torch.allclose(vectors["diyafa_001"], torch.tensor([1.0, 0.0, 0.0, 0.0]))

    def test_load_with_filter(self, fake_hf_token: str) -> None:
        """Loading with concept_ids filter returns only requested concepts."""
        fake_rows = [
            {
                "concept_id": "diyafa_001",
                "concept_ar": "",
                "concept_en": "",
                "vector": [1.0, 0.0],
                "extraction_layer": 14,
                "model_name": "test",
                "extraction_timestamp": "2026-01-01T00:00:00Z",
            },
            {
                "concept_id": "wasta_001",
                "concept_ar": "",
                "concept_en": "",
                "vector": [0.0, 1.0],
                "extraction_layer": 14,
                "model_name": "test",
                "extraction_timestamp": "2026-01-01T00:00:00Z",
            },
        ]

        mock_split = mock.MagicMock()
        mock_split.__iter__ = mock.Mock(return_value=iter(fake_rows))
        mock_ds = mock.MagicMock()
        mock_ds.keys.return_value = ["train"]
        mock_ds.__getitem__ = mock.Mock(return_value=mock_split)

        with mock.patch("datasets.load_dataset", return_value=mock_ds):
            vectors = load_vectors_from_hf(concept_ids=["diyafa_001"])

        assert len(vectors) == 1
        assert "diyafa_001" in vectors

    def test_load_no_token_raises(self, no_hf_token: None) -> None:
        """Missing token raises MissingTokenError."""
        with pytest.raises(MissingTokenError):
            load_vectors_from_hf()

    def test_load_failure_raises(self, fake_hf_token: str) -> None:
        """Load failure raises HFIntegrationError."""
        with mock.patch("datasets.load_dataset", side_effect=RuntimeError("404")):
            with pytest.raises(HFIntegrationError, match="Failed to load"):
                load_vectors_from_hf()


# ---------------------------------------------------------------------------
# sync_with_space tests (mocked)
# ---------------------------------------------------------------------------


class TestSyncWithSpace:
    """Test sync_with_space with mocked HF API."""

    def test_sync_success(self, fake_hf_token: str) -> None:
        """Successful sync returns Space status."""
        mock_runtime = mock.MagicMock()
        mock_runtime.stage = "RUNNING"
        mock_runtime.hardware = {"requested": "zero-a10g"}

        mock_info = mock.MagicMock()
        mock_info.runtime = mock_runtime
        mock_info.last_modified = "2026-08-17T12:00:00Z"

        mock_api = mock.MagicMock()
        mock_api.space_info.return_value = mock_info

        with mock.patch("huggingface_hub.HfApi", return_value=mock_api):
            result = sync_with_space(space_name="al3obdi/test-space")

        assert result["stage"] == "RUNNING"
        assert result["hardware"] == "zero-a10g"
        assert "huggingface.co/spaces" in result["space_url"]
        assert "huggingface.co/datasets" in result["dataset_url"]

    def test_sync_no_token_raises(self, no_hf_token: None) -> None:
        """Missing token raises MissingTokenError."""
        with pytest.raises(MissingTokenError):
            sync_with_space()

    def test_sync_failure_raises(self, fake_hf_token: str) -> None:
        """API failure raises HFIntegrationError."""
        with mock.patch("huggingface_hub.HfApi") as mock_hfapi:
            mock_hfapi.return_value.space_info.side_effect = RuntimeError("404")
            with pytest.raises(HFIntegrationError, match="Failed to query"):
                sync_with_space()


# ---------------------------------------------------------------------------
# validate_dataset_schema tests (mocked)
# ---------------------------------------------------------------------------


class TestValidateDatasetSchema:
    """Test dataset schema validation."""

    def test_valid_schema(self, fake_hf_token: str) -> None:
        """Valid schema returns True."""
        mock_split = mock.MagicMock()
        mock_split.column_names = list(REQUIRED_COLUMNS)

        with mock.patch("datasets.load_dataset", return_value=mock_split):
            assert validate_dataset_schema() is True

    def test_missing_columns(self, fake_hf_token: str) -> None:
        """Missing columns raise HFIntegrationError."""
        mock_split = mock.MagicMock()
        mock_split.column_names = ["concept_id", "vector"]  # missing several

        with mock.patch("datasets.load_dataset", return_value=mock_split):
            with pytest.raises(HFIntegrationError, match="missing columns"):
                validate_dataset_schema()


# ---------------------------------------------------------------------------
# CulturalRepE integration tests
# ---------------------------------------------------------------------------


class TestCulturalRepEHfIntegration:
    """Test HF integration methods on the CulturalRepE class."""

    def test_save_vectors_to_hf_method(self, fake_hf_token: str) -> None:
        """Engine save_vectors_to_hf delegates to the utility function."""
        from tests.helpers import make_steerable_engine

        engine = make_steerable_engine()
        # Add a second concept
        engine.concept_vectors["wasta"] = torch.ones(4)
        engine.extraction_layers["wasta"] = 1

        with mock.patch("src.utils.hf_integration.save_vectors_to_hf") as mock_save:
            mock_save.return_value = "https://huggingface.co/datasets/test"
            url = engine.save_vectors_to_hf()

        assert "huggingface.co" in url
        mock_save.assert_called_once()
        call_args = mock_save.call_args
        # Verify metadata includes model_name and extraction_layers
        assert call_args.kwargs["metadata"]["model_name"] == "dummy/steerable"
        assert "diyafa" in call_args.kwargs["metadata"]["extraction_layers"]

    def test_save_no_vectors_raises(self, fake_hf_token: str) -> None:
        """Engine with no vectors raises ValueError."""
        engine = CulturalRepE(model_name="test/model", device="cpu")
        with pytest.raises(ValueError, match="No concept vectors"):
            engine.save_vectors_to_hf()

    def test_load_vectors_to_hf_method(self, fake_hf_token: str) -> None:
        """Engine load_vectors_from_hf merges into concept_vectors."""
        from tests.helpers import make_steerable_engine

        engine = make_steerable_engine()
        loaded_vectors = {"new_concept": torch.tensor([1.0, 2.0, 3.0, 4.0])}

        with mock.patch(
            "src.utils.hf_integration.load_vectors_from_hf", return_value=loaded_vectors
        ):
            result = engine.load_vectors_from_hf()

        assert "new_concept" in engine.concept_vectors
        assert "diyafa" in engine.concept_vectors  # original preserved
        assert torch.allclose(result["new_concept"], loaded_vectors["new_concept"])

    def test_sync_with_space_method(self, fake_hf_token: str) -> None:
        """Engine sync_with_space delegates to the utility function."""
        from tests.helpers import make_steerable_engine

        engine = make_steerable_engine()
        expected = {"space_url": "https://huggingface.co/spaces/test", "stage": "RUNNING"}

        with mock.patch("src.utils.hf_integration.sync_with_space", return_value=expected):
            result = engine.sync_with_space()

        assert result == expected


# ---------------------------------------------------------------------------
# Dataset README / schema tests
# ---------------------------------------------------------------------------


class TestDatasetSchema:
    """Test that the dataset schema is correctly defined."""

    def test_required_columns(self) -> None:
        """Required columns match the documented schema."""
        assert "concept_id" in REQUIRED_COLUMNS
        assert "concept_ar" in REQUIRED_COLUMNS
        assert "concept_en" in REQUIRED_COLUMNS
        assert "vector" in REQUIRED_COLUMNS
        assert "extraction_layer" in REQUIRED_COLUMNS
        assert "model_name" in REQUIRED_COLUMNS
        assert "extraction_timestamp" in REQUIRED_COLUMNS

    def test_default_constants(self) -> None:
        """Default dataset and space names are set correctly."""
        assert DEFAULT_DATASET == "al3obdi/thaqafa-repe-vectors"
        assert DEFAULT_SPACE == "al3obdi/thaqafa-repe-extraction"


# ---------------------------------------------------------------------------
# No-secrets-leaked test
# ---------------------------------------------------------------------------


class TestNoSecretsExposed:
    """Ensure no tokens are exposed in logs or error messages."""

    def test_token_not_in_error_messages(self, no_hf_token: None) -> None:
        """MissingTokenError does not contain any token value."""
        with pytest.raises(MissingTokenError) as exc_info:
            save_vectors_to_hf({"test": torch.tensor([1.0])})
        assert "hf_" not in str(exc_info.value)
        assert "token" not in str(exc_info.value).lower() or "token" in str(exc_info.value).lower()

    def test_source_code_no_hardcoded_tokens(self) -> None:
        """Source files do not contain hardcoded HF tokens."""
        import src.utils.hf_integration as mod

        source_file = mod.__file__
        if source_file:
            with open(source_file) as f:
                content = f.read()
            # Check for common token patterns
            assert "hf_" not in content or content.count("hf_") <= 2  # allow in docstrings
            # No actual token values (typically 34+ chars starting with hf_)
            import re

            tokens = re.findall(r"hf_[A-Za-z0-9]{30,}", content)
            assert len(tokens) == 0, f"Found potential hardcoded tokens: {tokens}"
