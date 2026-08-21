"""Tests for the reproducible pilot runner.

The runner's job is to turn a model into a directory of results that a reader
can check. These tests are therefore about the artefacts: that every file is
written, that the numbers in the report are the numbers in the CSVs, and that
the manifest says what produced them.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_pilot  # noqa: E402
from src.models.rep_engine import RESID_POST_HOOK, CulturalRepE  # noqa: E402
from tests.helpers import DATASET_PATH, SteerableFakeModel  # noqa: E402


class PilotFakeModel(SteerableFakeModel):
    """A fake that can be probed, steered and generated from at once.

    :class:`SteerableFakeModel` returns the same activation for every prompt,
    which would leave the contrastive difference at exactly zero. Here one
    dimension carries the prompt's token content, so probes see variance and
    extraction produces a non-degenerate vector, while the remaining dimensions
    still carry the depth-dependent norm relative steering calibrates against.
    """

    def run_with_cache(
        self,
        tokens: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[None, dict[str, torch.Tensor]]:
        """Return depth-scaled activations whose first dimension varies by text."""
        _, cache = super().run_with_cache(tokens, **kwargs)
        name = next(iter(cache))
        activations = cache[name].clone()
        activations[..., 0] = activations[..., 0] + tokens.to(torch.float32) * 0.1
        return None, {name: activations}


def make_pilot_engine() -> CulturalRepE:
    """Build an engine backed by :class:`PilotFakeModel`."""
    engine = CulturalRepE(
        model_name="dummy/pilot",
        device="cpu",
        dtype="float32",
        dataset_path=DATASET_PATH,
    )
    model = PilotFakeModel(n_layers=3, d_model=4, d_vocab=8)
    engine.model = model  # type: ignore[assignment]
    engine.tokenizer = model.tokenizer
    return engine


@pytest.fixture
def engine() -> CulturalRepE:
    """An engine wired to the pilot fake."""
    return make_pilot_engine()


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV into a list of row dicts."""
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class TestLoadConceptNames:
    """Display names come from the dataset, not a hand-maintained table."""

    def test_covers_every_concept_in_the_dataset(self) -> None:
        """Every entry gets a name, so no run can label one by its raw id."""
        names = run_pilot.load_concept_names(DATASET_PATH)
        line_count = sum(
            1 for line in DATASET_PATH.read_text(encoding="utf-8").splitlines() if line.strip()
        )
        assert len(names) == line_count

    def test_maps_ids_to_english_names(self) -> None:
        """The value is the readable name an instruction template can use."""
        names = run_pilot.load_concept_names(DATASET_PATH)
        assert names["wasta_001"] == "Wasta/Nepotism"

    def test_empty_dataset_is_an_error(self, tmp_path: Path) -> None:
        """Silently running over zero concepts would produce empty results."""
        empty = tmp_path / "empty.jsonl"
        empty.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="No concepts"):
            run_pilot.load_concept_names(empty)


class TestParseArgs:
    """The defaults are part of what makes a run reproducible."""

    def test_the_strength_grids_span_the_onset_not_only_saturation(self) -> None:
        """A grid saturated at every point shows no dose-response at all.

        The first grid tried here started at a coefficient that already pinned
        every concept, so every row read the same and the table said nothing
        about how much intervention the effect needs.
        """
        assert min(run_pilot.DEFAULT_READBACK_STRENGTHS) <= 0.02
        assert max(run_pilot.DEFAULT_READBACK_STRENGTHS) >= 0.20

    def test_suppression_mirrors_the_readback_magnitudes(self) -> None:
        """So amplification and suppression compare point for point."""
        assert [-s for s in run_pilot.DEFAULT_SUPPRESSION_STRENGTHS] == list(
            run_pilot.DEFAULT_READBACK_STRENGTHS
        )

    def test_suppression_strengths_are_all_negative(self) -> None:
        assert all(s < 0 for s in run_pilot.DEFAULT_SUPPRESSION_STRENGTHS)

    def test_defaults_are_norm_relative(self) -> None:
        """A fixed absolute grid is not comparable across layers."""
        args = run_pilot.parse_args([])
        assert args.model == run_pilot.DEFAULT_MODEL
        assert [float(s) for s in args.strengths.split(",")] == list(run_pilot.DEFAULT_STRENGTHS)

    def test_default_strengths_include_the_unsteered_reference(self) -> None:
        """Zero has to be in the grid, measured through the same code path."""
        assert 0.0 in run_pilot.DEFAULT_STRENGTHS

    def test_overrides_are_honoured(self) -> None:
        """Every knob the manifest records must be settable."""
        args = run_pilot.parse_args(
            ["--model", "gpt2-medium", "--seed", "7", "--strengths", "0.0,0.5", "--no-baselines"]
        )
        assert args.model == "gpt2-medium"
        assert args.seed == 7
        assert args.strengths == "0.0,0.5"
        assert args.no_baselines is True


class TestWriteCsv:
    """Results are read by other tools, so the header contract matters."""

    def test_writes_a_header_and_rows(self, tmp_path: Path) -> None:
        """A reader must be able to open the file with DictReader."""
        path = tmp_path / "nested" / "out.csv"
        run_pilot._write_csv(path, ["a", "b"], [{"a": 1, "b": 2}, {"a": 3, "b": 4}])
        assert _read_csv(path) == [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}]

    def test_creates_missing_parent_directories(self, tmp_path: Path) -> None:
        """A run must not fail on its last step because a directory is absent."""
        path = tmp_path / "deep" / "deeper" / "out.csv"
        run_pilot._write_csv(path, ["a"], [{"a": 1}])
        assert path.exists()


class TestLayerSweep:
    """The sweep decides which layer everything downstream uses."""

    def test_covers_every_layer_of_every_concept(self, engine: CulturalRepE) -> None:
        """A partial sweep would make the "best" layer an artefact of coverage."""
        rows, best = run_pilot.run_layer_sweep(engine, ["wasta_001", "diyafa_001"], 0)
        n_layers = engine.model.cfg.n_layers  # type: ignore[union-attr]
        assert len(rows) == 2 * n_layers
        assert set(best) == {"wasta_001", "diyafa_001"}

    def test_reports_dispersion_alongside_the_score(self, engine: CulturalRepE) -> None:
        """A score without its spread invites over-reading a small sample."""
        rows, _ = run_pilot.run_layer_sweep(engine, ["wasta_001"], 0)
        assert all("probe_std" in row for row in rows)
        assert all("chance" in row for row in rows)

    def test_the_seed_reaches_the_probes(self, engine: CulturalRepE) -> None:
        """A --seed that controls only some phases makes the manifest a half-truth.

        The estimates are seed-sensitive at this sample size, so a sweep left on
        the library default while later phases used the run's seed produced two
        different numbers for the same concept and layer.
        """
        first, _ = run_pilot.run_layer_sweep(engine, ["wasta_001"], 0, seed=1)
        again, _ = run_pilot.run_layer_sweep(engine, ["wasta_001"], 0, seed=1)
        other, _ = run_pilot.run_layer_sweep(engine, ["wasta_001"], 0, seed=99)

        assert first == again
        assert first != other

    def test_records_the_metric_and_the_class_balance(self, engine: CulturalRepE) -> None:
        """A stored score must say which rule produced it and how even the split was."""
        rows, _ = run_pilot.run_layer_sweep(engine, ["wasta_001"], 0)
        assert {row["metric"] for row in rows} == {"balanced_accuracy"}
        assert all(row["chance"] == 0.5 for row in rows)
        assert all(row["majority_class_rate"] > 0.5 for row in rows)

    def test_best_layer_is_one_that_was_probed(self, engine: CulturalRepE) -> None:
        """The chosen layer has to appear in the rows backing it."""
        rows, best = run_pilot.run_layer_sweep(engine, ["wasta_001"], 0)
        probed = {row["layer"] for row in rows}
        assert best["wasta_001"] in probed


class TestSteeringSweep:
    """Strengths are fractions of the residual norm, not raw coefficients."""

    def test_one_row_per_concept_and_strength(self, engine: CulturalRepE) -> None:
        """Missing points would silently shorten the curve."""
        engine.extract_vector("wasta_001", layer=1)
        rows = run_pilot.run_steering_sweep(
            engine, ["wasta_001"], (-0.2, 0.0, 0.2), ["A guest arrives."]
        )
        assert [row["relative_strength"] for row in rows] == [-0.2, 0.0, 0.2]

    def test_zero_strength_has_no_effect(self, engine: CulturalRepE) -> None:
        """Injecting a zero offset is a no-op, so it is the honest reference."""
        engine.extract_vector("wasta_001", layer=1)
        rows = run_pilot.run_steering_sweep(engine, ["wasta_001"], (0.0,), ["A guest arrives."])
        assert rows[0]["effect_kl"] == pytest.approx(0.0, abs=1e-6)

    def test_records_the_layer_it_injected_into(self, engine: CulturalRepE) -> None:
        """A KL value is meaningless without knowing where it was applied."""
        engine.extract_vector("wasta_001", layer=2)
        rows = run_pilot.run_steering_sweep(engine, ["wasta_001"], (0.2,), ["A guest arrives."])
        assert rows[0]["layer"] == 2


class TestBaselines:
    """Steering is only interesting next to what prompting already does."""

    def test_includes_the_unprompted_control(self, engine: CulturalRepE, tmp_path: Path) -> None:
        """Without a control, neither arm has anything to be compared against."""
        engine.extract_vector("wasta_001", layer=1)
        rows = run_pilot.run_baselines(
            engine, ["wasta_001"], {"wasta_001": "Wasta"}, ["A guest arrives."], 0.2, 4, tmp_path
        )
        conditions = {row["condition"] for row in rows}
        assert any("neutral" in str(name) for name in conditions)
        assert len(conditions) > 1

    def test_saves_generations_for_blind_rating(self, engine: CulturalRepE, tmp_path: Path) -> None:
        """Fluency cannot judge cultural grounding; humans need the text."""
        engine.extract_vector("wasta_001", layer=1)
        run_pilot.run_baselines(
            engine, ["wasta_001"], {"wasta_001": "Wasta"}, ["A guest arrives."], 0.2, 4, tmp_path
        )
        payload = json.loads((tmp_path / "generations" / "wasta_001.json").read_text())
        assert payload["relative_strength"] == 0.2
        assert payload["conditions"]


class TestAlignmentPhase:
    """The cross-lingual check, and the shared layer it needs."""

    def test_uses_one_layer_for_every_concept(self, engine: CulturalRepE) -> None:
        """Cosines from different depths are not comparable."""
        rows = run_pilot.run_alignment(
            engine, ["wasta_001", "diyafa_001"], {"wasta_001": 0, "diyafa_001": 2}
        )
        assert len({row["layer"] for row in rows}) == 1

    def test_reports_the_control_next_to_the_headline(self, engine: CulturalRepE) -> None:
        """An aligned cosine with no mismatched baseline cannot be read."""
        rows = run_pilot.run_alignment(
            engine, ["wasta_001", "diyafa_001"], {"wasta_001": 1, "diyafa_001": 1}
        )
        assert rows
        for row in rows:
            assert "mean_mismatched_cosine" in row
            assert "separation" in row

    def test_a_single_concept_produces_no_rows(self, engine: CulturalRepE) -> None:
        """With nothing to compare against, a separation would be meaningless."""
        assert run_pilot.run_alignment(engine, ["wasta_001"], {"wasta_001": 1}) == []


class TestReadbackPhase:
    """The causal check, and the layer ordering it depends on."""

    @pytest.fixture
    def extracted(self, engine: CulturalRepE) -> CulturalRepE:
        """An engine with the concept already extracted, as the runner leaves it."""
        engine.extract_vector("wasta_001", layer=0)
        return engine

    def test_reports_every_strength_rather_than_choosing_one(self, extracted: CulturalRepE) -> None:
        """Picking the best strength would be selection on the same data."""
        rows = run_pilot.run_readback(
            extracted, ["wasta_001"], {"wasta_001": 2}, strengths=(0.1, 0.2), n_random=1
        )
        assert {float(row["strength"]) for row in rows} == {0.1, 0.2}

    def test_carries_the_random_control(self, extracted: CulturalRepE) -> None:
        """A rise in the positive rate means nothing without it."""
        rows = run_pilot.run_readback(
            extracted, ["wasta_001"], {"wasta_001": 2}, strengths=(0.2,), n_random=2
        )
        assert rows
        assert all(row["n_random"] == 2 for row in rows)
        assert all("lift_over_random" in row for row in rows)

    def test_skips_a_concept_whose_best_layer_is_the_first_block(
        self, extracted: CulturalRepE
    ) -> None:
        """There is nothing below layer 0 to inject into."""
        rows = run_pilot.run_readback(
            extracted, ["wasta_001"], {"wasta_001": 0}, strengths=(0.2,), n_random=1
        )
        assert rows == []

    def test_reads_at_the_best_layer_not_the_last(self, extracted: CulturalRepE) -> None:
        """A probe near chance at the last layer would make the lift meaningless."""
        rows = run_pilot.run_readback(
            extracted, ["wasta_001"], {"wasta_001": 2}, strengths=(0.2,), n_random=1
        )
        assert all(int(row["read_layer"]) == 2 for row in rows)
        assert all(int(row["inject_layer"]) == 1 for row in rows)

    def test_injects_below_the_layer_it_reads(self, extracted: CulturalRepE) -> None:
        rows = run_pilot.run_readback(
            extracted, ["wasta_001"], {"wasta_001": 0}, strengths=(0.2,), n_random=1
        )
        assert all(int(row["read_layer"]) > int(row["inject_layer"]) for row in rows)


class TestSuppressionPhase:
    """The mirror check: does subtracting the concept remove it?"""

    @pytest.fixture
    def extracted(self, engine: CulturalRepE) -> CulturalRepE:
        """An engine with the concept extracted, as the runner leaves it."""
        engine.extract_vector("wasta_001", layer=0)
        return engine

    def test_uses_the_same_layer_pair_as_the_readback(self, extracted: CulturalRepE) -> None:
        """Otherwise any asymmetry could be an artefact of where each was measured."""
        amplify = run_pilot.run_readback(
            extracted, ["wasta_001"], {"wasta_001": 2}, strengths=(0.2,), n_random=1
        )
        suppress = run_pilot.run_suppression(
            extracted, ["wasta_001"], {"wasta_001": 2}, strengths=(-0.2,), n_random=1
        )
        assert amplify[0]["inject_layer"] == suppress[0]["inject_layer"]
        assert amplify[0]["read_layer"] == suppress[0]["read_layer"]

    def test_every_strength_is_negative(self, extracted: CulturalRepE) -> None:
        """A positive coefficient here would be amplification under another name."""
        rows = run_pilot.run_suppression(
            extracted, ["wasta_001"], {"wasta_001": 2}, strengths=(-0.2, -0.4), n_random=1
        )
        assert {float(row["strength"]) for row in rows} == {-0.2, -0.4}

    def test_carries_the_control_and_the_probe_quality(self, extracted: CulturalRepE) -> None:
        """A baseline of 1.00 is ambiguous without the probe's own accuracy."""
        rows = run_pilot.run_suppression(
            extracted, ["wasta_001"], {"wasta_001": 2}, strengths=(-0.2,), n_random=2
        )
        assert rows
        assert all(row["n_random"] == 2 for row in rows)
        assert all("probe_balanced_accuracy" in row for row in rows)
        assert all("drop_beyond_random" in row for row in rows)

    def test_skips_a_concept_whose_best_layer_is_the_first_block(
        self, extracted: CulturalRepE
    ) -> None:
        rows = run_pilot.run_suppression(
            extracted, ["wasta_001"], {"wasta_001": 0}, strengths=(-0.2,), n_random=1
        )
        assert rows == []


class TestTransferPhase:
    """The behavioural half of the cross-lingual question."""

    @pytest.fixture
    def extracted(self, engine: CulturalRepE) -> CulturalRepE:
        """An engine with the concept extracted, as the runner leaves it."""
        engine.extract_vector("wasta_001", layer=0)
        return engine

    def test_covers_both_reader_languages(self, extracted: CulturalRepE) -> None:
        """Transfer need not be symmetric, so one direction is not an answer."""
        rows = run_pilot.run_transfer(
            extracted, ["wasta_001"], {"wasta_001": 2}, strengths=(0.2,), n_random=1
        )
        assert {row["reader_language"] for row in rows} == {"en", "ar"}

    def test_uses_the_same_layer_pair_as_the_other_causal_checks(
        self, extracted: CulturalRepE
    ) -> None:
        """So the three sections can be read side by side."""
        amplify = run_pilot.run_readback(
            extracted, ["wasta_001"], {"wasta_001": 2}, strengths=(0.2,), n_random=1
        )
        across = run_pilot.run_transfer(
            extracted, ["wasta_001"], {"wasta_001": 2}, strengths=(0.2,), n_random=1
        )
        assert across[0]["inject_layer"] == amplify[0]["inject_layer"]
        assert across[0]["read_layer"] == amplify[0]["read_layer"]

    def test_carries_the_ceiling_alongside_the_transfer(self, extracted: CulturalRepE) -> None:
        """A transfer lift means nothing without what was available to transfer."""
        rows = run_pilot.run_transfer(
            extracted, ["wasta_001"], {"wasta_001": 2}, strengths=(0.2,), n_random=1
        )
        assert all("same_language_lift" in row for row in rows)
        assert all("transfer_lift" in row for row in rows)
        assert all("probe_accuracy" in row for row in rows)

    def test_skips_a_concept_whose_best_layer_is_the_first_block(
        self, extracted: CulturalRepE
    ) -> None:
        rows = run_pilot.run_transfer(
            extracted, ["wasta_001"], {"wasta_001": 0}, strengths=(0.2,), n_random=1
        )
        assert rows == []


class TestWriteReport:
    """The report is what a reader sees first, so it must not overstate."""

    def _manifest(self) -> dict[str, Any]:
        """A manifest with known values to assert against."""
        return {
            "model": {"name": "gpt2", "device": "cpu", "dtype": "float32"},
            "git": {"commit": "abcdef0123456789", "branch": "main", "dirty": False},
            "timestamp_utc": "2026-01-01T00:00:00+00:00",
            "seed": 42,
            "dataset": {"sha256": "0" * 64},
        }

    def _rows(self) -> tuple[list[dict[str, Any]], dict[str, int], list[dict[str, Any]]]:
        """Minimal layer, best-layer and steering data."""
        layer_rows = [
            {
                "concept_id": "wasta_001",
                "layer": 0,
                "metric": "balanced_accuracy",
                "probe_score": 0.5,
                "probe_std": 0.1,
                "chance": 0.5,
                "lift_over_chance": 0.0,
                "majority_class_rate": 0.6,
                "n_samples": 20,
            },
            {
                "concept_id": "wasta_001",
                "layer": 1,
                "metric": "balanced_accuracy",
                "probe_score": 0.75,
                "probe_std": 0.2,
                "chance": 0.5,
                "lift_over_chance": 0.25,
                "majority_class_rate": 0.6,
                "n_samples": 20,
            },
        ]
        steering_rows = [
            {
                "concept_id": "wasta_001",
                "relative_strength": 0.2,
                "layer": 1,
                "effect_kl": 0.4321,
                "mean_loss": 5.5,
                "perplexity": 244.7,
            }
        ]
        return layer_rows, {"wasta_001": 1}, steering_rows

    def test_states_the_model_commit_and_seed(self, tmp_path: Path) -> None:
        """A table of numbers with no provenance invites being quoted alone."""
        layer_rows, best, steering_rows = self._rows()
        path = run_pilot.write_report(
            tmp_path, self._manifest(), layer_rows, best, steering_rows, []
        )
        text = path.read_text(encoding="utf-8")
        assert "gpt2" in text
        assert "abcdef012345" in text
        assert "42" in text

    def test_reports_the_best_layer_row(self, tmp_path: Path) -> None:
        """The headline accuracy must be the one at the chosen layer."""
        layer_rows, best, steering_rows = self._rows()
        path = run_pilot.write_report(
            tmp_path, self._manifest(), layer_rows, best, steering_rows, []
        )
        text = path.read_text(encoding="utf-8")
        assert "0.750" in text
        assert "+0.250" in text

    def test_carries_the_limitations(self, tmp_path: Path) -> None:
        """Small samples and unreviewed entries have to travel with the numbers."""
        layer_rows, best, steering_rows = self._rows()
        path = run_pilot.write_report(
            tmp_path, self._manifest(), layer_rows, best, steering_rows, []
        )
        text = path.read_text(encoding="utf-8")
        assert "Limitations" in text
        assert "native-speaker" in text

    def test_says_the_layer_was_chosen_on_the_same_data(self, tmp_path: Path) -> None:
        """Best-of-twelve selection makes an uncorrected p-value optimistic."""
        layer_rows, best, steering_rows = self._rows()
        path = run_pilot.write_report(
            tmp_path, self._manifest(), layer_rows, best, steering_rows, []
        )
        text = path.read_text(encoding="utf-8")
        assert "chosen by the same data" in text
        assert "correction" in text

    def test_flags_uncommitted_code(self, tmp_path: Path) -> None:
        """Results from code that differs from the commit must say so."""
        manifest = self._manifest()
        manifest["git"]["dirty"] = True
        layer_rows, best, steering_rows = self._rows()
        path = run_pilot.write_report(tmp_path, manifest, layer_rows, best, steering_rows, [])
        assert "uncommitted code changes" in path.read_text(encoding="utf-8")

    def test_untracked_output_files_do_not_raise_the_alarm(self, tmp_path: Path) -> None:
        """Every run writes output; flagging that would train readers to ignore it."""
        manifest = self._manifest()
        manifest["git"]["untracked"] = 17
        layer_rows, best, steering_rows = self._rows()
        path = run_pilot.write_report(tmp_path, manifest, layer_rows, best, steering_rows, [])
        assert "uncommitted code changes" not in path.read_text(encoding="utf-8")

    def test_omits_the_baseline_section_when_it_did_not_run(self, tmp_path: Path) -> None:
        """An empty table would read as "prompting did nothing"."""
        layer_rows, best, steering_rows = self._rows()
        path = run_pilot.write_report(
            tmp_path, self._manifest(), layer_rows, best, steering_rows, []
        )
        assert "against prompting" not in path.read_text(encoding="utf-8")


class TestMain:
    """End to end: a directory a reader can pick up and check."""

    @pytest.fixture(autouse=True)
    def _patch_engine(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Run the pipeline against the fake instead of downloading a model."""
        built = make_pilot_engine()
        monkeypatch.setattr(built, "load_model", lambda: built.model)
        monkeypatch.setattr(run_pilot, "CulturalRepE", lambda **_kwargs: built)

    def test_writes_every_artefact(self, tmp_path: Path) -> None:
        """Manifest, CSVs and report all have to land together."""
        exit_code = run_pilot.main(
            [
                "--model",
                "dummy/pilot",
                "--concepts",
                "wasta_001,diyafa_001",
                "--strengths",
                "0.0,0.2",
                "--output-dir",
                str(tmp_path / "run"),
                "--no-baselines",
                "--permutations",
                "0",
                "--no-readback",
            ]
        )
        out = tmp_path / "run"
        assert exit_code == 0
        assert (out / "manifest.json").exists()
        assert (out / "layer_sweep.csv").exists()
        assert (out / "steering_sweep.csv").exists()
        assert (out / "README.md").exists()

    def test_manifest_pins_the_run(self, tmp_path: Path) -> None:
        """Seed, dataset hash and the strength grid all belong on disk."""
        run_pilot.main(
            [
                "--model",
                "dummy/pilot",
                "--concepts",
                "wasta_001",
                "--strengths",
                "0.0,0.2",
                "--seed",
                "7",
                "--output-dir",
                str(tmp_path / "run"),
                "--no-baselines",
                "--permutations",
                "0",
                "--no-readback",
            ]
        )
        manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
        assert manifest["seed"] == 7
        assert manifest["settings"]["relative_strengths"] == [0.0, 0.2]
        assert manifest["settings"]["baselines_run"] is False
        assert len(manifest["dataset"]["sha256"]) == 64

    def test_defaults_to_every_concept_in_the_dataset(self, tmp_path: Path) -> None:
        """Hand-listing concepts is how a sweep quietly loses coverage."""
        run_pilot.main(
            [
                "--model",
                "dummy/pilot",
                "--strengths",
                "0.0",
                "--output-dir",
                str(tmp_path / "run"),
                "--no-baselines",
                "--permutations",
                "0",
                "--no-readback",
            ]
        )
        manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
        assert len(manifest["concepts"]) == len(run_pilot.load_concept_names(DATASET_PATH))

    def test_unknown_concept_fails_loudly(self, tmp_path: Path) -> None:
        """A typo must not silently shrink the experiment."""
        with pytest.raises(SystemExit, match="Unknown concept"):
            run_pilot.main(
                [
                    "--model",
                    "dummy/pilot",
                    "--concepts",
                    "not_a_concept_999",
                    "--output-dir",
                    str(tmp_path / "run"),
                    "--no-baselines",
                    "--permutations",
                    "0",
                    "--no-readback",
                ]
            )

    def test_extracts_at_the_layer_the_probe_chose(self, tmp_path: Path) -> None:
        """Extraction must follow the sweep, not a layer chosen by theory."""
        run_pilot.main(
            [
                "--model",
                "dummy/pilot",
                "--concepts",
                "wasta_001",
                "--strengths",
                "0.2",
                "--output-dir",
                str(tmp_path / "run"),
                "--no-baselines",
                "--permutations",
                "0",
                "--no-readback",
            ]
        )
        out = tmp_path / "run"
        layer_rows = _read_csv(out / "layer_sweep.csv")
        best = max(layer_rows, key=lambda r: float(r["probe_score"]))
        steering_rows = _read_csv(out / "steering_sweep.csv")
        assert steering_rows[0]["layer"] == best["layer"]

    def test_baselines_produce_a_comparison_table(self, tmp_path: Path) -> None:
        """The prompting arm is the point of comparison, not an optional extra."""
        run_pilot.main(
            [
                "--model",
                "dummy/pilot",
                "--concepts",
                "wasta_001",
                "--strengths",
                "0.2",
                "--max-new-tokens",
                "4",
                "--output-dir",
                str(tmp_path / "run"),
                "--permutations",
                "0",
                "--no-readback",
            ]
        )
        out = tmp_path / "run"
        rows = _read_csv(out / "baseline_comparison.csv")
        assert len(rows) > 1
        assert "against prompting" in (out / "README.md").read_text(encoding="utf-8")
        assert (out / "generations" / "wasta_001.json").exists()

    def test_writes_the_alignment_check(self, tmp_path: Path) -> None:
        """The cross-lingual result is part of the run, not an optional extra."""
        run_pilot.main(
            [
                "--model",
                "dummy/pilot",
                "--concepts",
                "wasta_001,diyafa_001",
                "--strengths",
                "0.2",
                "--output-dir",
                str(tmp_path / "run"),
                "--no-baselines",
                "--permutations",
                "0",
                "--no-readback",
            ]
        )
        out = tmp_path / "run"
        rows = _read_csv(out / "crosslingual_alignment.csv")
        assert {row["concept_id"] for row in rows} == {"wasta_001", "diyafa_001"}
        assert "same direction" in (out / "README.md").read_text(encoding="utf-8")

    def test_the_p_value_reaches_the_artefacts(self, tmp_path: Path) -> None:
        """A high score on twenty prompts is unreadable without it."""
        run_pilot.main(
            [
                "--model",
                "dummy/pilot",
                "--concepts",
                "wasta_001",
                "--strengths",
                "0.2",
                "--output-dir",
                str(tmp_path / "run"),
                "--no-baselines",
                "--permutations",
                "20",
            ]
        )
        out = tmp_path / "run"
        rows = _read_csv(out / "layer_sweep.csv")

        assert all(row["p_value"] for row in rows)
        assert {row["n_permutations"] for row in rows} == {"20"}
        assert "permutation p-value" in (out / "README.md").read_text(encoding="utf-8")
        manifest = json.loads((out / "manifest.json").read_text())
        assert manifest["settings"]["probe_permutations"] == 20

    def test_skipping_permutations_leaves_the_column_empty_not_zero(self, tmp_path: Path) -> None:
        """An absent p-value must not be readable as a significant one."""
        run_pilot.main(
            [
                "--model",
                "dummy/pilot",
                "--concepts",
                "wasta_001",
                "--strengths",
                "0.2",
                "--output-dir",
                str(tmp_path / "run"),
                "--no-baselines",
                "--permutations",
                "0",
                "--no-readback",
            ]
        )
        rows = _read_csv(tmp_path / "run" / "layer_sweep.csv")

        assert all(row["p_value"] == "" for row in rows)
        assert "n/a" in (tmp_path / "run" / "README.md").read_text(encoding="utf-8")

    def test_the_readback_reaches_the_artefacts(self, tmp_path: Path) -> None:
        """The causal check is part of the run, and its control travels with it."""
        run_pilot.main(
            [
                "--model",
                "dummy/pilot",
                "--concepts",
                "wasta_001",
                "--strengths",
                "0.2",
                "--output-dir",
                str(tmp_path / "run"),
                "--no-baselines",
                "--permutations",
                "0",
            ]
        )
        out = tmp_path / "run"
        readback_csv = out / "causal_readback.csv"
        if not readback_csv.exists():
            # The fake's best layer can land on block 0, where the check is
            # correctly skipped. Nothing else to assert in that case.
            return

        rows = _read_csv(readback_csv)
        assert all(int(row["read_layer"]) > int(row["inject_layer"]) for row in rows)
        assert all(int(row["n_random"]) > 0 for row in rows)
        text = (out / "README.md").read_text(encoding="utf-8")
        assert "Does steering write what the probe reads" in text
        assert "only column that carries information" in text
        assert "discarded, not explained" in text

    def test_the_suppression_check_reaches_the_artefacts(self, tmp_path: Path) -> None:
        """Removal is the claim RepE is reached for; it needs an artefact too."""
        run_pilot.main(
            [
                "--model",
                "dummy/pilot",
                "--concepts",
                "wasta_001",
                "--strengths",
                "0.2",
                "--output-dir",
                str(tmp_path / "run"),
                "--no-baselines",
                "--permutations",
                "0",
            ]
        )
        out = tmp_path / "run"
        path = out / "suppression.csv"
        if not path.exists():
            # The fake's best layer can land on block 0, where both causal
            # checks are correctly skipped.
            return

        rows = _read_csv(path)
        assert all(float(row["strength"]) < 0 for row in rows)
        assert all(float(row["probe_balanced_accuracy"]) >= 0.0 for row in rows)
        text = (out / "README.md").read_text(encoding="utf-8")
        assert "Does subtracting the concept remove it" in text
        assert "column that carries information" in text

    def test_the_transfer_check_reaches_the_artefacts(self, tmp_path: Path) -> None:
        """The behavioural cross-lingual answer needs an artefact too."""
        run_pilot.main(
            [
                "--model",
                "dummy/pilot",
                "--concepts",
                "wasta_001",
                "--strengths",
                "0.2",
                "--output-dir",
                str(tmp_path / "run"),
                "--no-baselines",
                "--permutations",
                "0",
            ]
        )
        out = tmp_path / "run"
        path = out / "crosslingual_transfer.csv"
        if not path.exists():
            # The fake's best layer can land on block 0, where the causal
            # checks are correctly skipped.
            return

        rows = _read_csv(path)
        assert {row["reader_language"] for row in rows} == {"en", "ar"}
        assert all(int(row["read_layer"]) > int(row["inject_layer"]) for row in rows)
        text = (out / "README.md").read_text(encoding="utf-8")
        assert "steer the other" in text

    def test_is_deterministic_for_a_fixed_seed(self, tmp_path: Path) -> None:
        """Two runs of the same command must produce the same numbers."""
        argv = [
            "--model",
            "dummy/pilot",
            "--concepts",
            "wasta_001",
            "--strengths",
            "0.0,0.2",
            "--seed",
            "11",
            "--no-baselines",
            "--permutations",
            "0",
            "--no-readback",
        ]
        run_pilot.main([*argv, "--output-dir", str(tmp_path / "a")])
        run_pilot.main([*argv, "--output-dir", str(tmp_path / "b")])
        assert _read_csv(tmp_path / "a" / "layer_sweep.csv") == _read_csv(
            tmp_path / "b" / "layer_sweep.csv"
        )
        assert _read_csv(tmp_path / "a" / "steering_sweep.csv") == _read_csv(
            tmp_path / "b" / "steering_sweep.csv"
        )


def test_resid_hook_name_is_the_one_the_fake_answers() -> None:
    """Guards the fake against a rename of the hook the engine reads."""
    assert RESID_POST_HOOK.format(layer=0) == "blocks.0.hook_resid_post"


class TestArtefactsSurviveTheCommitHooks:
    """A committed artefact must be what the runner actually wrote.

    The repository's end-of-file hook appends a trailing newline to files that
    lack one. If the runner omitted it, committing an artefact would silently
    change it, and a reproduction check would report a diff that no rerun could
    ever close.
    """

    def test_every_written_file_ends_with_exactly_one_newline(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Covers the manifest, the report, the CSVs and the generations."""
        engine = make_pilot_engine()
        monkeypatch.setattr(engine, "load_model", lambda: engine.model)
        monkeypatch.setattr(run_pilot, "CulturalRepE", lambda **_kwargs: engine)

        run_pilot.main(
            [
                "--model",
                "dummy/pilot",
                "--concepts",
                "wasta_001,diyafa_001",
                "--strengths",
                "0.2",
                "--max-new-tokens",
                "4",
                "--output-dir",
                str(tmp_path / "run"),
                "--permutations",
                "0",
                "--no-readback",
            ]
        )

        written = [path for path in (tmp_path / "run").rglob("*") if path.is_file()]
        assert written
        for path in written:
            content = path.read_bytes()
            assert content.endswith(b"\n"), f"{path.name} has no trailing newline"
            assert not content.endswith(b"\n\n"), f"{path.name} has a blank line at the end"
