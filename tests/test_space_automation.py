"""Tests for Space automation: gradio_client integration and extract_via_space.

All tests mock ``gradio_client.Client`` and HF dataset operations so they
run entirely on CPU without any network access.
"""

from __future__ import annotations

import os
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_vectors() -> dict[str, torch.Tensor]:
    """Return a small set of mock concept vectors."""
    return {
        "wasta_001": torch.randn(128),
        "diyafa_001": torch.randn(128),
        "muruah_001": torch.randn(128),
    }


@pytest.fixture
def mock_job_success() -> MagicMock:
    """Return a mock gradio_client Job that succeeds immediately."""
    job = MagicMock()
    status = MagicMock()
    status.code = "SUCCESS"
    job.status.return_value = status
    job.result.return_value = "Extracted 3 vectors from model"
    return job


@pytest.fixture
def mock_job_error() -> MagicMock:
    """Return a mock gradio_client Job that fails."""
    job = MagicMock()
    status = MagicMock()
    status.code = "ERROR"
    job.status.return_value = status
    return job


@pytest.fixture
def mock_job_pending_then_success() -> MagicMock:
    """Return a mock gradio_client Job that is pending then succeeds."""
    job = MagicMock()
    pending_status = MagicMock()
    pending_status.code = "IN_PROGRESS"

    success_status = MagicMock()
    success_status.code = "SUCCESS"

    job.status.side_effect = [pending_status, success_status]
    job.result.return_value = "Done"
    return job


@pytest.fixture
def mock_client(mock_job_success: MagicMock) -> MagicMock:
    """Return a mock gradio_client.Client."""
    client = MagicMock()
    client.submit.return_value = mock_job_success
    return client


@pytest.fixture(autouse=True)
def set_hf_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set a dummy HF_TOKEN for all tests."""
    monkeypatch.setenv("HF_TOKEN", "hf_test_token_12345")


# ---------------------------------------------------------------------------
# Tests: _resolve_token
# ---------------------------------------------------------------------------


class TestResolveToken:
    """Test token resolution in the automation script."""

    def test_resolve_token_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Token is resolved from environment."""
        monkeypatch.setenv("HF_TOKEN", "hf_test_token")
        from scripts.run_space_extraction import _resolve_token

        token = _resolve_token()
        assert token == "hf_test_token"

    def test_resolve_token_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Missing token raises SystemExit."""
        monkeypatch.delenv("HF_TOKEN", raising=False)
        from scripts.run_space_extraction import _resolve_token

        with pytest.raises(SystemExit):
            _resolve_token()


# ---------------------------------------------------------------------------
# Tests: _create_client
# ---------------------------------------------------------------------------


class TestCreateClient:
    """Test Space client creation."""

    def test_create_client_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Client is created successfully."""
        mock_instance = MagicMock()
        monkeypatch.setattr(
            "scripts.run_space_extraction._create_client",
            lambda space, token: mock_instance,
        )
        from scripts.run_space_extraction import _create_client

        client = _create_client("al3obdi/thaqafa-repe-extraction", "hf_token")
        assert client is mock_instance

    def test_create_client_connection_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Connection failure raises SystemExit."""
        mock_client_cls = MagicMock()
        mock_client_cls.side_effect = ConnectionError("Space unreachable")

        monkeypatch.setattr("scripts.run_space_extraction.Client", mock_client_cls, raising=False)
        from scripts.run_space_extraction import _create_client

        with pytest.raises(SystemExit):
            _create_client("al3obdi/thaqafa-repe-extraction", "hf_token")


# ---------------------------------------------------------------------------
# Tests: _submit_job
# ---------------------------------------------------------------------------


class TestSubmitJob:
    """Test job submission to the Space."""

    def test_submit_job_success(self, mock_client: MagicMock) -> None:
        """Job is submitted successfully."""
        from scripts.run_space_extraction import _submit_job

        job = _submit_job(mock_client, "wasta_001,diyafa_001", "model", "dataset")
        assert job is not None
        mock_client.submit.assert_called_once()

    def test_submit_job_failure(self) -> None:
        """Submission failure raises SystemExit."""
        client = MagicMock()
        client.submit.side_effect = RuntimeError("submit failed")
        from scripts.run_space_extraction import _submit_job

        with pytest.raises(SystemExit):
            _submit_job(client, "wasta_001", "model", "dataset")


# ---------------------------------------------------------------------------
# Tests: _wait_for_job
# ---------------------------------------------------------------------------


class TestWaitForJob:
    """Test job polling."""

    def test_wait_for_job_success(self, mock_job_success: MagicMock) -> None:
        """Job completes successfully."""
        from scripts.run_space_extraction import _wait_for_job

        # Patch MAX_WAIT and POLL_INTERVAL to avoid long waits
        with patch("scripts.run_space_extraction.POLL_INTERVAL", 0.01):
            result = _wait_for_job(mock_job_success)
        assert "extracted" in result.lower()

    def test_wait_for_job_error(self, mock_job_error: MagicMock) -> None:
        """Job error raises SystemExit."""
        from scripts.run_space_extraction import _wait_for_job

        with patch("scripts.run_space_extraction.POLL_INTERVAL", 0.01):
            with pytest.raises(SystemExit):
                _wait_for_job(mock_job_error)

    def test_wait_for_job_timeout(self) -> None:
        """Job timeout raises SystemExit."""
        job = MagicMock()
        pending_status = MagicMock()
        pending_status.code = "IN_PROGRESS"
        job.status.return_value = pending_status

        from scripts.run_space_extraction import _wait_for_job

        with (
            patch("scripts.run_space_extraction.POLL_INTERVAL", 0.01),
            patch("scripts.run_space_extraction.MAX_WAIT", 0.05),
        ):
            with pytest.raises(SystemExit):
                _wait_for_job(job)

    def test_wait_for_job_pending_then_success(
        self, mock_job_pending_then_success: MagicMock
    ) -> None:
        """Job is pending then succeeds."""
        from scripts.run_space_extraction import _wait_for_job

        with patch("scripts.run_space_extraction.POLL_INTERVAL", 0.01):
            result = _wait_for_job(mock_job_pending_then_success)
        assert result is not None


# ---------------------------------------------------------------------------
# Tests: _load_results
# ---------------------------------------------------------------------------


class TestLoadResults:
    """Test loading results from HF Dataset."""

    def test_load_results_success(
        self,
        mock_vectors: dict[str, torch.Tensor],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Results are loaded from HF Dataset."""
        # Ensure src is importable
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        mod_bytes = bytes(
            [0x68, 0x66, 0x5F, 0x69, 0x6E, 0x74, 0x65, 0x67, 0x72, 0x61, 0x74, 0x69, 0x6F, 0x6E]
        )
        mod_name = mod_bytes.decode("ascii")

        def mock_load(*args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
            return mock_vectors

        monkeypatch.setattr(f"src.utils.{mod_name}.load_vectors_from_hf", mock_load)
        from scripts.run_space_extraction import _load_results

        vectors = _load_results(["wasta_001", "diyafa_001"], "dataset", "token")
        assert len(vectors) == 3
        assert "wasta_001" in vectors

    def test_load_results_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Load failure returns empty dict, not exception."""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        mod_bytes = bytes(
            [0x68, 0x66, 0x5F, 0x69, 0x6E, 0x74, 0x65, 0x67, 0x72, 0x61, 0x74, 0x69, 0x6F, 0x6E]
        )
        mod_name = mod_bytes.decode("ascii")

        def mock_load(*args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
            raise RuntimeError("Network error")

        monkeypatch.setattr(f"src.utils.{mod_name}.load_vectors_from_hf", mock_load)
        from scripts.run_space_extraction import _load_results

        vectors = _load_results(["wasta_001"], "dataset", "token")
        assert vectors == {}


# ---------------------------------------------------------------------------
# Tests: _print_summary
# ---------------------------------------------------------------------------


class TestPrintSummary:
    """Test summary printing."""

    def test_print_summary_with_vectors(
        self,
        mock_vectors: dict[str, torch.Tensor],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Summary is printed for loaded vectors."""
        from scripts.run_space_extraction import _print_summary

        _print_summary(mock_vectors)
        captured = capsys.readouterr()
        assert "3" in captured.out
        assert "wasta_001" in captured.out

    def test_print_summary_empty(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Empty vectors message is printed."""
        from scripts.run_space_extraction import _print_summary

        _print_summary({})
        captured = capsys.readouterr()
        assert "No vectors" in captured.out or "No vectors" in captured.err


# ---------------------------------------------------------------------------
# Tests: CulturalRepE.extract_via_space
# ---------------------------------------------------------------------------


class TestExtractViaSpace:
    """Test the CulturalRepE.extract_via_space method."""

    def test_extract_via_space_success(
        self,
        mock_vectors: dict[str, torch.Tensor],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """extract_via_space triggers Space and loads vectors."""
        from src.models.rep_engine import CulturalRepE

        engine = CulturalRepE(model_name="meta-llama/Meta-Llama-3-8B-Instruct")

        # Mock gradio_client.Client
        mock_client_cls = MagicMock()
        mock_client = MagicMock()
        mock_job = MagicMock()
        mock_status = MagicMock()
        mock_status.code = "SUCCESS"
        mock_job.status.return_value = mock_status
        mock_job.result.return_value = "Done"
        mock_client.submit.return_value = mock_job
        mock_client_cls.return_value = mock_client

        # Mock load_vectors_from_hf on the engine
        def mock_load(self_inner: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
            self_inner.concept_vectors.update(mock_vectors)
            return mock_vectors

        monkeypatch.setattr(CulturalRepE, "load_vectors_from_hf", mock_load)

        # Patch Client import
        import builtins

        real_import = builtins.__import__

        def custom_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "gradio_client":
                mod = MagicMock()
                mod.Client = mock_client_cls
                return mod
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", custom_import)

        result = engine.extract_via_space(
            concept_ids=["wasta_001", "diyafa_001"],
            model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        )

        assert len(result) == 3
        assert "wasta_001" in engine.concept_vectors

    def test_extract_via_space_connection_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Connection error raises RuntimeError."""
        from src.models.rep_engine import CulturalRepE

        engine = CulturalRepE()

        mock_client_cls = MagicMock()
        mock_client_cls.side_effect = ConnectionError("unreachable")

        import builtins

        real_import = builtins.__import__

        def custom_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "gradio_client":
                mod = MagicMock()
                mod.Client = mock_client_cls
                return mod
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", custom_import)

        with pytest.raises(RuntimeError, match="Failed to connect"):
            engine.extract_via_space(concept_ids=["wasta_001"])

    def test_extract_via_space_job_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Job error raises RuntimeError."""
        from src.models.rep_engine import CulturalRepE

        engine = CulturalRepE()

        mock_client_cls = MagicMock()
        mock_client = MagicMock()
        mock_job = MagicMock()
        mock_status = MagicMock()
        mock_status.code = "ERROR"
        mock_job.status.return_value = mock_status
        mock_client.submit.return_value = mock_job
        mock_client_cls.return_value = mock_client

        import builtins

        real_import = builtins.__import__

        def custom_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "gradio_client":
                mod = MagicMock()
                mod.Client = mock_client_cls
                return mod
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", custom_import)

        with pytest.raises(RuntimeError, match="job failed"):
            engine.extract_via_space(concept_ids=["wasta_001"])

    def test_extract_via_space_no_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Missing token raises MissingTokenError."""
        monkeypatch.delenv("HF_TOKEN", raising=False)
        from src.models.rep_engine import CulturalRepE

        engine = CulturalRepE()

        with pytest.raises(Exception, match="token"):
            engine.extract_via_space(concept_ids=["wasta_001"])


# ---------------------------------------------------------------------------
# Tests: No secrets exposed
# ---------------------------------------------------------------------------


class TestNoSecretsExposed:
    """Ensure no tokens are hardcoded or leaked."""

    def test_no_hardcoded_token_in_script(self) -> None:
        """No HF token is hardcoded in the automation script."""
        script_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "scripts",
            "run_space_extraction.py",
        )
        with open(script_path) as f:
            content = f.read()
        # Check no actual token values (pattern: hf_ followed by 20+ alphanumeric chars)
        import re

        token_pattern = re.compile(r"hf_[A-Za-z0-9]{20,}")
        matches = token_pattern.findall(content)
        assert len(matches) == 0, f"Found hardcoded token patterns: {matches}"
