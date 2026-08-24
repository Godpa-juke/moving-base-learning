"""Record which code produced a run.

Experiments here are configuration variants rather than branches, so several of them
share one working tree and differ only in a reward weight or an observation term. That
makes the commit a run was launched from, plus any uncommitted diff, the only way to
reconstruct what a checkpoint actually optimised. Without it a number in a results
table cannot be reproduced.

``rsl_rl`` already stores a diff for the Isaac Lab tree it was launched against; this
covers the project's own tree, which is the half that changes between experiments.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def _git(project_dir: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ("git", "-C", str(project_dir), *args),
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    return result.stdout


def collect(project_dir: str | Path) -> dict:
    """Commit, dirty flag and untracked files for ``project_dir``.

    Returns ``{"available": False}`` when the directory is not a git repository or git
    is unusable, so callers never have to guard the call.
    """
    project_dir = Path(project_dir)
    commit = _git(project_dir, "rev-parse", "HEAD")
    if commit is None:
        return {"available": False}
    diff = _git(project_dir, "diff", "HEAD") or ""
    untracked = _git(project_dir, "ls-files", "--others", "--exclude-standard") or ""
    untracked_files = [line for line in untracked.splitlines() if line]
    return {
        "available": True,
        "commit": commit.strip(),
        "branch": (_git(project_dir, "rev-parse", "--abbrev-ref", "HEAD") or "").strip() or None,
        "dirty": bool(diff.strip()) or bool(untracked_files),
        "untracked_files": untracked_files,
        "_diff": diff,
    }


def record(project_dir: str | Path, out_dir: str | Path) -> dict:
    """Write any uncommitted diff next to a run's outputs and return summary fields.

    The diff goes to a file rather than into ``summary.json`` so the summary stays
    readable; the returned dict is what belongs in the summary.
    """
    info = collect(project_dir)
    diff = info.pop("_diff", "")
    if info.get("available") and diff.strip():
        diff_path = Path(out_dir) / "project.diff"
        diff_path.write_text(diff)
        info["diff_file"] = diff_path.name
    return info
