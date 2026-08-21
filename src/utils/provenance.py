"""Recording exactly what produced a set of results.

A results file without provenance cannot be reproduced and cannot be trusted:
six months later nobody can tell whether a number came from a different model,
a different dataset revision, or a different version of this code. Every
experiment run in this repository therefore writes a manifest alongside its
CSVs, and :func:`build_manifest` is the single place that decides what a
manifest contains.

The manifest is deliberately cheap to produce - a few subprocess calls and a
file hash - so there is never a reason to skip it.
"""

from __future__ import annotations

import hashlib
import platform
import random
import subprocess
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA_VERSION = "1.0"
"""Bumped whenever the manifest's keys change incompatibly."""

TRACKED_PACKAGES: tuple[str, ...] = (
    "torch",
    "transformer-lens",
    "transformers",
    "scikit-learn",
    "numpy",
)
"""Packages whose versions can change a result without changing this repository."""

UNKNOWN = "unknown"
"""Placeholder used when a value genuinely cannot be determined."""

OUTPUT_PREFIX = "results/"
"""Tree holding committed run artefacts.

Changes here are excluded from :attr:`GitRevision.dirty` and counted
separately. Nothing in the pipeline reads this tree - it is written by runs
and read by people - so a difference in it cannot change what a run computes.
"""


@dataclass(frozen=True)
class GitRevision:
    """The commit a run was produced from.

    Attributes:
        commit: Full commit SHA, or :data:`UNKNOWN` outside a git checkout.
        branch: Branch name, or :data:`UNKNOWN` when detached or unavailable.
        dirty: Whether *tracked* files differed from the commit. This is the
            flag that matters: it means the commit alone does not identify the
            code that ran, so results produced from one are provisional.
        untracked: How many untracked paths were present, excluding
            :data:`OUTPUT_PREFIX`. A non-zero count is worth seeing: an
            untracked file can be a module the run imported but nobody
            committed.
        output_paths_changed: How many paths under :data:`OUTPUT_PREFIX`
            differed. Counted, not ignored, so the exclusion is visible - but
            kept out of :attr:`dirty`, because results are outputs of runs and
            never inputs to them. Nothing in the pipeline reads that tree, so a
            run that rewrote an earlier run's output has not changed the code
            it is about to execute, and flagging it would raise the alarm on
            every sequential run until nobody read the flag at all.
    """

    commit: str
    branch: str
    dirty: bool
    untracked: int = 0
    output_paths_changed: int = 0

    def as_dict(self) -> dict[str, Any]:
        """Return the revision as plain JSON-serialisable data."""
        return {
            "commit": self.commit,
            "branch": self.branch,
            "dirty": self.dirty,
            "untracked": self.untracked,
            "output_paths_changed": self.output_paths_changed,
        }


def _git(args: list[str], repo_root: Path) -> str | None:
    """Run a git command, returning its stripped stdout or None on any failure.

    Args:
        args: Arguments following ``git``.
        repo_root: Directory to run in.

    Returns:
        The command's output, or None if git is missing, errors, or times out.
    """
    try:
        completed = subprocess.run(  # noqa: S603
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def git_revision(repo_root: Path | str = ".") -> GitRevision:
    """Describe the git checkout at *repo_root*.

    Never raises: a manifest that fails to be written is worse than one with an
    ``unknown`` field, so every failure mode degrades to :data:`UNKNOWN`.

    Args:
        repo_root: Directory inside the checkout to inspect.

    Returns:
        The commit, branch and dirty flag.
    """
    root = Path(repo_root)
    commit = _git(["rev-parse", "HEAD"], root) or UNKNOWN
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], root) or UNKNOWN

    everything = _git(["status", "--porcelain"], root)
    tracked_changes = 0
    untracked = 0
    output_changes = 0
    for line in (everything or "").splitlines():
        # Split on the first space rather than slicing a fixed width: the
        # command's output is stripped as a whole, so the first line loses the
        # leading space that porcelain format puts before a single-letter
        # status and every later line keeps it.
        status, _, path = line.strip().partition(" ")
        cleaned = path.strip().strip('"')
        if cleaned.startswith(OUTPUT_PREFIX):
            output_changes += 1
        elif status == "??":
            untracked += 1
        else:
            tracked_changes += 1

    return GitRevision(
        commit=commit,
        branch=branch,
        dirty=bool(tracked_changes),
        untracked=untracked,
        output_paths_changed=output_changes,
    )


def file_digest(path: Path | str) -> str:
    """Return the SHA-256 of a file's bytes.

    Hashing the dataset file, rather than recording its path, is what lets a
    reader confirm that a rerun used the same inputs even after the file has
    been edited in place.

    Args:
        path: File to hash.

    Returns:
        The hex digest, or :data:`UNKNOWN` if the file cannot be read.
    """
    file_path = Path(path)
    digest = hashlib.sha256()
    try:
        with open(file_path, "rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""):
                digest.update(chunk)
    except OSError:
        return UNKNOWN
    return digest.hexdigest()


def package_versions(names: tuple[str, ...] | list[str] = TRACKED_PACKAGES) -> dict[str, str]:
    """Look up installed versions for the given distributions.

    Args:
        names: Distribution names as they appear on PyPI.

    Returns:
        A mapping from name to version, with :data:`UNKNOWN` for anything not
        installed.
    """
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = UNKNOWN
    return versions


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy and torch so a rerun reproduces the same numbers.

    Seeding is separate from the manifest on purpose: the manifest records the
    seed, this applies it, and a caller that forgets one of the two ends up with
    results that are either irreproducible or silently mislabelled.

    Args:
        seed: The seed to apply.
    """
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is a hard dependency
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # pragma: no cover - torch is a hard dependency
        pass


def build_manifest(
    *,
    experiment: str,
    model_name: str,
    device: str,
    dtype: str,
    seed: int,
    dataset_path: Path | str,
    concepts: list[str],
    timestamp: str,
    repo_root: Path | str = ".",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the manifest that accompanies a set of results.

    Args:
        experiment: Short identifier for the run, e.g. ``"pilot_gpt2"``.
        model_name: The model the numbers came from.
        device: Device the model ran on.
        dtype: Numeric precision the model ran at.
        seed: Seed passed to :func:`set_global_seed`.
        dataset_path: Concept dataset the run read.
        concepts: Concept identifiers evaluated.
        timestamp: ISO-8601 UTC time the run started. Passed in rather than
            read here so the caller controls it and tests stay deterministic.
        repo_root: Checkout to describe.
        extra: Run-specific settings to record verbatim, such as the strength
            grid or the number of cross-validation folds.

    Returns:
        A JSON-serialisable manifest.
    """
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment": experiment,
        "timestamp_utc": timestamp,
        "git": git_revision(repo_root).as_dict(),
        "model": {"name": model_name, "device": device, "dtype": dtype},
        "seed": seed,
        "dataset": {
            "path": str(dataset_path),
            "sha256": file_digest(dataset_path),
        },
        "concepts": list(concepts),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": package_versions(),
        },
        "settings": dict(extra or {}),
    }
