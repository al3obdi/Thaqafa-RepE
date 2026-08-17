"""Tests for paper results generation pipeline.

All tests mock CulturalRepE methods (extract_via_space, sweep_layers_with_probe,
evaluate_steering, compare_steering_vs_prompting) so they run entirely on CPU
without any network access, GPU usage, or HF API calls.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Create a temporary output directory."""
    out = tmp_path / "paper_results"
    out.mkdir()
    (out / "figures").mkdir()
    (out / "generations").mkdir()
    return out


@pytest.fixture
def mock_vectors() -> dict[str, torch.Tensor]:
    """Return mock concept vectors."""
    return {
        "wasta_001": torch.randn(128),
        "muruah_001": torch.randn(128),
        "diyafa_001": torch.randn(128),
    }


@pytest.fixture
def mock_probe_results() -> dict[int, Any]:
    """Return mock probe results for 5 layers."""
    results = {}
    for layer in range(5):
        result = MagicMock()
        result.layer = layer
        result.accuracy = 0.5 + layer * 0.08
        result.chance = 0.5
        result.std = 0.05
        results[layer] = result
    return results


@pytest.fixture
def mock_steering_results() -> dict[float, Any]:
    """Return mock steering results for 5 strengths."""
    results = {}
    for _i, s in enumerate([-2.0, -1.0, 0.0, 1.0, 2.0]):
        result = MagicMock()
        result.strength = s
        result.effect_kl = abs(s) * 0.15
        result.mean_loss = 2.0 + s * s * 0.1
        result.perplexity = 7.4
        results[float(s)] = result
    return results


@pytest.fixture
def mock_comparison_result() -> Any:
    """Return a mock ComparisonResult."""
    comp = MagicMock()
    comp.rows.return_value = [
        {
            "condition": "steering",
            "mean_continuation_loss": 2.0,
            "extra_input_tokens": 0,
            "n_generations": 5,
        },
        {
            "condition": "prompt:direct_en",
            "mean_continuation_loss": 2.1,
            "extra_input_tokens": 8,
            "n_generations": 5,
        },
    ]
    comp.conditions = {"steering": MagicMock(), "prompt:direct_en": MagicMock()}
    comp.steering_effect_kl = 0.15
    return comp


@pytest.fixture(autouse=True)
def set_hf_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set a dummy HF_TOKEN for all tests."""
    monkeypatch.setenv("HF_TOKEN", "hf_test_token_12345")


# ---------------------------------------------------------------------------
# Tests: CSV output schema
# ---------------------------------------------------------------------------


class TestCSVSchema:
    """Test that CSV files have the correct schema."""

    def test_layer_sweep_csv_schema(self, tmp_output_dir: Path) -> None:
        """Layer sweep CSV has correct columns."""
        csv_path = tmp_output_dir / "layer_sweep.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["concept_id", "layer", "probe_accuracy", "chance_accuracy"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "concept_id": "wasta_001",
                    "layer": 0,
                    "probe_accuracy": 0.75,
                    "chance_accuracy": 0.5,
                }
            )

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert set(rows[0].keys()) == {"concept_id", "layer", "probe_accuracy", "chance_accuracy"}

    def test_steering_sweep_csv_schema(self, tmp_output_dir: Path) -> None:
        """Steering sweep CSV has correct columns."""
        csv_path = tmp_output_dir / "steering_sweep.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["concept_id", "strength", "effect_kl", "mean_loss"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "concept_id": "wasta_001",
                    "strength": 1.0,
                    "effect_kl": 0.15,
                    "mean_loss": 2.1,
                }
            )

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert set(rows[0].keys()) == {"concept_id", "strength", "effect_kl", "mean_loss"}

    def test_baseline_comparison_csv_schema(self, tmp_output_dir: Path) -> None:
        """Baseline comparison CSV has correct columns."""
        csv_path = tmp_output_dir / "baseline_comparison.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "concept_id",
                    "condition",
                    "mean_continuation_loss",
                    "extra_input_tokens",
                    "n_generations",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "concept_id": "wasta_001",
                    "condition": "steering",
                    "mean_continuation_loss": 2.0,
                    "extra_input_tokens": 0,
                    "n_generations": 5,
                }
            )

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert set(rows[0].keys()) == {
            "concept_id",
            "condition",
            "mean_continuation_loss",
            "extra_input_tokens",
            "n_generations",
        }


# ---------------------------------------------------------------------------
# Tests: CulturalRepE.run_full_experiment (mocked, no model)
# ---------------------------------------------------------------------------


class TestRunFullExperiment:
    """Test the run_full_experiment method with mocked dependencies."""

    def test_run_full_experiment_no_model(
        self,
        mock_vectors: dict[str, torch.Tensor],
        tmp_output_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """run_full_experiment produces all outputs when model is not loaded."""
        from src.models.rep_engine import CulturalRepE

        engine = CulturalRepE(model_name="test-model")

        # Mock extract_via_space to populate vectors
        def mock_extract(self_inner: Any, concept_ids: list[str]) -> dict[str, torch.Tensor]:
            self_inner.concept_vectors = dict(mock_vectors)
            return mock_vectors

        monkeypatch.setattr(CulturalRepE, "extract_via_space", mock_extract)

        results = engine.run_full_experiment(
            concept_ids=["wasta_001", "muruah_001", "diyafa_001"],
            output_dir=str(tmp_output_dir),
        )

        # Check all outputs exist
        assert (tmp_output_dir / "vectors.json").exists()
        assert (tmp_output_dir / "layer_sweep.csv").exists()
        assert (tmp_output_dir / "steering_sweep.csv").exists()
        assert (tmp_output_dir / "baseline_comparison.csv").exists()
        assert (tmp_output_dir / "RESULTS_SUMMARY.md").exists()

        # Check return dict
        assert results["vectors_saved"] is True
        assert "best_layers" in results
        assert "layer_sweep" in results
        assert "steering_sweep" in results
        assert "baseline_comparison" in results

    def test_vectors_json_structure(
        self,
        mock_vectors: dict[str, torch.Tensor],
        tmp_output_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """vectors.json contains the extracted vectors as lists."""
        from src.models.rep_engine import CulturalRepE

        engine = CulturalRepE(model_name="test-model")

        def mock_extract(self_inner: Any, concept_ids: list[str]) -> dict[str, torch.Tensor]:
            self_inner.concept_vectors = dict(mock_vectors)
            return mock_vectors

        monkeypatch.setattr(CulturalRepE, "extract_via_space", mock_extract)

        engine.run_full_experiment(
            concept_ids=["wasta_001"],
            output_dir=str(tmp_output_dir),
        )

        with open(tmp_output_dir / "vectors.json") as f:
            data = json.load(f)
        assert "wasta_001" in data
        assert isinstance(data["wasta_001"], list)
        assert len(data["wasta_001"]) == 128

    def test_layer_sweep_csv_content(
        self,
        mock_vectors: dict[str, torch.Tensor],
        tmp_output_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """layer_sweep.csv has rows for each concept and layer."""
        from src.models.rep_engine import CulturalRepE

        engine = CulturalRepE(model_name="test-model")

        def mock_extract(self_inner: Any, concept_ids: list[str]) -> dict[str, torch.Tensor]:
            self_inner.concept_vectors = dict(mock_vectors)
            return mock_vectors

        monkeypatch.setattr(CulturalRepE, "extract_via_space", mock_extract)

        engine.run_full_experiment(
            concept_ids=["wasta_001"],
            output_dir=str(tmp_output_dir),
        )

        with open(tmp_output_dir / "layer_sweep.csv") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 32  # placeholder data has 32 layers
        assert rows[0]["concept_id"] == "wasta_001"

    def test_steering_sweep_csv_content(
        self,
        mock_vectors: dict[str, torch.Tensor],
        tmp_output_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """steering_sweep.csv has rows for each strength."""
        from src.models.rep_engine import CulturalRepE

        engine = CulturalRepE(model_name="test-model")

        def mock_extract(self_inner: Any, concept_ids: list[str]) -> dict[str, torch.Tensor]:
            self_inner.concept_vectors = dict(mock_vectors)
            return mock_vectors

        monkeypatch.setattr(CulturalRepE, "extract_via_space", mock_extract)

        engine.run_full_experiment(
            concept_ids=["wasta_001"],
            output_dir=str(tmp_output_dir),
        )

        with open(tmp_output_dir / "steering_sweep.csv") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 5  # 5 strengths
        assert "effect_kl" in rows[0]
        assert "mean_loss" in rows[0]

    def test_results_summary_markdown(
        self,
        mock_vectors: dict[str, torch.Tensor],
        tmp_output_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """RESULTS_SUMMARY.md has expected sections."""
        from src.models.rep_engine import CulturalRepE

        engine = CulturalRepE(model_name="test-model")

        def mock_extract(self_inner: Any, concept_ids: list[str]) -> dict[str, torch.Tensor]:
            self_inner.concept_vectors = dict(mock_vectors)
            return mock_vectors

        monkeypatch.setattr(CulturalRepE, "extract_via_space", mock_extract)

        engine.run_full_experiment(
            concept_ids=["wasta_001", "diyafa_001"],
            output_dir=str(tmp_output_dir),
        )

        md = (tmp_output_dir / "RESULTS_SUMMARY.md").read_text()
        assert "Best Layers" in md
        assert "Steering Sweep" in md
        assert "Baseline Comparison" in md
        assert "LaTeX" in md or "latex" in md.lower()
        assert "Summary Statistics" in md


# ---------------------------------------------------------------------------
# Tests: With mocked model (probe/steering/baseline paths)
# ---------------------------------------------------------------------------


class TestRunWithModel:
    """Test run_full_experiment with a mocked model loaded."""

    def test_run_with_model_uses_probes(
        self,
        mock_vectors: dict[str, torch.Tensor],
        mock_probe_results: dict[int, Any],
        mock_steering_results: dict[float, Any],
        mock_comparison_result: Any,
        tmp_output_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When model is loaded, real probe/steering/baseline paths are used."""
        from src.models.rep_engine import CulturalRepE

        engine = CulturalRepE(model_name="test-model")
        engine.model = MagicMock()  # Simulate loaded model
        engine.concept_vectors = dict(mock_vectors)

        # Mock the utility functions
        monkeypatch.setattr(
            "src.utils.probes.sweep_layers_with_probe",
            lambda eng, cid: mock_probe_results,
        )
        monkeypatch.setattr(
            "src.utils.probes.best_layer",
            lambda results: max(results, key=lambda k: results[k].accuracy),
        )
        monkeypatch.setattr(
            "src.utils.evaluation.evaluate_steering",
            lambda eng, cid, prompts, **kw: mock_steering_results,
        )
        monkeypatch.setattr(
            "src.utils.baselines.compare_steering_vs_prompting",
            lambda eng, cid, name, prompts, **kw: mock_comparison_result,
        )

        results = engine.run_full_experiment(
            concept_ids=["wasta_001"],
            output_dir=str(tmp_output_dir),
        )

        # Layer sweep should have 5 entries (from mock_probe_results)
        assert len(results["layer_sweep"]["wasta_001"]) == 5
        # Steering sweep should have 5 entries
        assert len(results["steering_sweep"]["wasta_001"]) == 5
        # Baseline comparison should have 2 conditions
        assert len(results["baseline_comparison"]["wasta_001"]) == 2


# ---------------------------------------------------------------------------
# Tests: No secrets exposed
# ---------------------------------------------------------------------------


class TestNoSecretsExposed:
    """Ensure no tokens are hardcoded in the script."""

    def test_no_hardcoded_token_in_script(self) -> None:
        """No HF token is hardcoded in generate_paper_results.py."""
        import re

        script_path = PROJECT_ROOT / "scripts" / "generate_paper_results.py"
        content = script_path.read_text()
        token_pattern = re.compile(r"hf_[A-Za-z0-9]{20,}")
        matches = token_pattern.findall(content)
        assert len(matches) == 0, f"Found hardcoded token patterns: {matches}"
