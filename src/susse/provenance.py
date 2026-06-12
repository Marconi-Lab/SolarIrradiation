"""Build-time provenance helpers shared across persistence layers.

Both :mod:`susse.datasets.snapshot_io` (for :class:`DatasetManifest`)
and :mod:`susse.training.metadata` (for :class:`TrainingMetadata`)
record the same two things — git SHA and the running ``susse`` package
version — when they materialise an artifact. Keeping the
implementations in one neutral module prevents the two layers from
drifting (e.g. one returning ``None`` from a missing git context and
the other raising).
"""

from __future__ import annotations

import subprocess
from importlib.metadata import PackageNotFoundError, version


def git_sha() -> str | None:
    """Best-effort ``git rev-parse HEAD``.

    Returns the 40-char hex SHA of ``HEAD`` when invoked inside a git
    repository with a working ``git`` binary; ``None`` if either is
    missing, the command times out, or the working tree has no commits.
    Never raises — provenance is best-effort and should not fail an
    artifact build.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def susse_version() -> str:
    """Return the installed ``susse`` package version, or ``"unknown"``.

    The package is registered as ``SuSSE`` in ``pyproject.toml``; the
    casing is preserved here. Returns ``"unknown"`` when the package
    metadata can't be resolved (e.g. when running directly from a
    source checkout without an editable install).
    """
    try:
        return version("SuSSE")
    except PackageNotFoundError:
        return "unknown"
