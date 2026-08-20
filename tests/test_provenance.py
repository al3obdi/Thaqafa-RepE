"""Tests for run provenance recording."""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
from pathlib import Path

import pytest
import torch

from src.utils import provenance
from src.utils.provenance import (
    MANIFEST_SCHEMA_VERSION,
    UNKNOWN,
    build_manifest,
    file_digest,
    git_revision,
    package_versions,
    set_global_seed,
)

TIMESTAMP = "2026-01-01T00:00:00+00:00"


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    """Write a small file to stand in for the concept dataset."""
    path = tmp_path / "concepts.jsonl"
    path.write_text('{"concept_id": "wasta_001"}\n', encoding="utf-8")
    return path


class TestFileDigest:
    """The dataset hash is what makes an input claim checkable."""

    def test_matches_hashlib(self, dataset: Path) -> None:
        """The digest is the plain SHA-256 of the file's bytes."""
        expected = hashlib.sha256(dataset.read_bytes()).hexdigest()
        assert file_digest(dataset) == expected

    def test_changes_when_the_file_changes(self, dataset: Path) -> None:
        """Editing the dataset must change the recorded hash."""
        before = file_digest(dataset)
        dataset.write_text('{"concept_id": "karam_001"}\n', encoding="utf-8")
        assert file_digest(dataset) != before

    def test_missing_file_is_unknown_not_an_error(self, tmp_path: Path) -> None:
        """A manifest that fails to write is worse than one with a gap."""
        assert file_digest(tmp_path / "absent.jsonl") == UNKNOWN

    def test_reads_files_larger_than_one_chunk(self, tmp_path: Path) -> None:
        """The chunked read must produce the same digest as a single read."""
        path = tmp_path / "big.bin"
        payload = b"x" * (65536 * 3 + 17)
        path.write_bytes(payload)
        assert file_digest(path) == hashlib.sha256(payload).hexdigest()


class TestGitRevision:
    """The commit is the other half of "what produced this"."""

    def test_reports_commit_branch_and_clean_tree(self, tmp_path: Path) -> None:
        """A fresh repository with one commit reads back cleanly."""
        subprocess.run(["git", "init", "-q", "-b", "trunk"], cwd=tmp_path, check=True)
        subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=tmp_path, check=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
        (tmp_path / "a.txt").write_text("one", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
        subprocess.run(["git", "commit", "-qm", "init"], cwd=tmp_path, check=True)

        revision = git_revision(tmp_path)

        assert len(revision.commit) == 40
        assert revision.branch == "trunk"
        assert revision.dirty is False

    def test_detects_a_dirty_tree(self, tmp_path: Path) -> None:
        """Uncommitted changes mean the commit alone does not identify the code."""
        subprocess.run(["git", "init", "-q", "-b", "trunk"], cwd=tmp_path, check=True)
        subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=tmp_path, check=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
        (tmp_path / "a.txt").write_text("one", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
        subprocess.run(["git", "commit", "-qm", "init"], cwd=tmp_path, check=True)
        (tmp_path / "a.txt").write_text("two", encoding="utf-8")

        assert git_revision(tmp_path).dirty is True

    def test_untracked_files_alone_are_not_dirty(self, tmp_path: Path) -> None:
        """A run writes output into the tree; that must not raise the alarm."""
        subprocess.run(["git", "init", "-q", "-b", "trunk"], cwd=tmp_path, check=True)
        subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=tmp_path, check=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
        (tmp_path / "a.txt").write_text("one", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
        subprocess.run(["git", "commit", "-qm", "init"], cwd=tmp_path, check=True)
        (tmp_path / "results.csv").write_text("x", encoding="utf-8")

        revision = git_revision(tmp_path)

        assert revision.dirty is False
        assert revision.untracked == 1

    def test_untracked_files_are_still_counted(self, tmp_path: Path) -> None:
        """An untracked file can be a module the run imported but nobody committed."""
        subprocess.run(["git", "init", "-q", "-b", "trunk"], cwd=tmp_path, check=True)
        (tmp_path / "one.py").write_text("x", encoding="utf-8")
        (tmp_path / "two.py").write_text("y", encoding="utf-8")

        assert git_revision(tmp_path).untracked == 2

    def test_outside_a_checkout_degrades_to_unknown(self, tmp_path: Path) -> None:
        """Running from a plain directory must not raise."""
        revision = git_revision(tmp_path)
        assert revision.commit == UNKNOWN
        assert revision.dirty is False

    def test_missing_git_binary_degrades_to_unknown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A machine without git still gets a manifest."""

        def explode(*_args: object, **_kwargs: object) -> None:
            raise FileNotFoundError("git")

        monkeypatch.setattr(provenance.subprocess, "run", explode)
        assert git_revision(".").commit == UNKNOWN

    def test_as_dict_is_json_serialisable(self, tmp_path: Path) -> None:
        """The revision has to survive a round trip through JSON."""
        payload = git_revision(tmp_path).as_dict()
        assert json.loads(json.dumps(payload)) == payload


class TestPackageVersions:
    """Results can change because a dependency changed, not because code did."""

    def test_reports_installed_versions(self) -> None:
        """A package that is certainly installed reports a real version."""
        assert package_versions(("torch",))["torch"] != UNKNOWN

    def test_absent_package_is_unknown(self) -> None:
        """An uninstalled name is recorded, not omitted."""
        assert package_versions(("definitely-not-installed-xyz",)) == {
            "definitely-not-installed-xyz": UNKNOWN
        }

    def test_default_covers_the_result_bearing_dependencies(self) -> None:
        """torch and transformer-lens can both move a number on their own."""
        versions = package_versions()
        assert "torch" in versions
        assert "transformer-lens" in versions


class TestSetGlobalSeed:
    """Recording a seed is only useful if something applies it."""

    def test_python_random_is_reproducible(self) -> None:
        """The same seed yields the same stdlib random draw."""
        set_global_seed(7)
        first = random.random()
        set_global_seed(7)
        assert random.random() == first

    def test_torch_is_reproducible(self) -> None:
        """The same seed yields the same torch draw."""
        set_global_seed(7)
        first = torch.rand(4)
        set_global_seed(7)
        assert torch.equal(torch.rand(4), first)

    def test_different_seeds_differ(self) -> None:
        """Seeding must not collapse everything to one stream."""
        set_global_seed(1)
        first = torch.rand(4)
        set_global_seed(2)
        assert not torch.equal(torch.rand(4), first)


class TestBuildManifest:
    """The manifest is the contract between a results file and its reader."""

    def _manifest(self, dataset: Path, **overrides: object) -> dict[str, object]:
        """Build a manifest with test defaults."""
        kwargs: dict[str, object] = {
            "experiment": "pilot_gpt2",
            "model_name": "gpt2",
            "device": "cpu",
            "dtype": "float32",
            "seed": 42,
            "dataset_path": dataset,
            "concepts": ["wasta_001", "diyafa_001"],
            "timestamp": TIMESTAMP,
        }
        kwargs.update(overrides)
        return build_manifest(**kwargs)  # type: ignore[arg-type]

    def test_records_every_field_a_rerun_needs(self, dataset: Path) -> None:
        """Model, seed, dataset hash and concepts all have to be present."""
        manifest = self._manifest(dataset)

        assert manifest["schema_version"] == MANIFEST_SCHEMA_VERSION
        assert manifest["experiment"] == "pilot_gpt2"
        assert manifest["timestamp_utc"] == TIMESTAMP
        assert manifest["model"] == {"name": "gpt2", "device": "cpu", "dtype": "float32"}
        assert manifest["seed"] == 42
        assert manifest["dataset"]["sha256"] == file_digest(dataset)  # type: ignore[index]
        assert manifest["concepts"] == ["wasta_001", "diyafa_001"]

    def test_records_the_environment(self, dataset: Path) -> None:
        """A version bump in torch can move a number without any code change."""
        environment = self._manifest(dataset)["environment"]
        assert environment["python"]  # type: ignore[index]
        assert environment["packages"]["torch"] != UNKNOWN  # type: ignore[index]

    def test_settings_are_carried_verbatim(self, dataset: Path) -> None:
        """Run-specific knobs belong in the manifest, not only in the shell."""
        manifest = self._manifest(dataset, extra={"relative_strengths": [-0.2, 0.0, 0.2]})
        assert manifest["settings"] == {"relative_strengths": [-0.2, 0.0, 0.2]}

    def test_settings_default_to_empty(self, dataset: Path) -> None:
        """Omitting extras yields an empty mapping rather than None."""
        assert self._manifest(dataset)["settings"] == {}

    def test_is_json_serialisable(self, dataset: Path) -> None:
        """The manifest is written to disk as JSON, so it must round trip."""
        manifest = self._manifest(dataset, extra={"max_new_tokens": 24})
        assert json.loads(json.dumps(manifest, ensure_ascii=False)) == manifest

    def test_concepts_are_copied_not_aliased(self, dataset: Path) -> None:
        """A later mutation of the caller's list must not rewrite the manifest."""
        concepts = ["wasta_001"]
        manifest = self._manifest(dataset, concepts=concepts)
        concepts.append("karam_001")
        assert manifest["concepts"] == ["wasta_001"]
