"""
gocia/database/status.py

Sentinel file helpers for tracking calculator pipeline state on disk.

Why sentinel files?
-------------------
When a GOCIA run is interrupted (e.g. the HPC login node session times out),
the SQLite database may be out of sync with the actual state of jobs on the
cluster.  Rather than querying the scheduler (which may itself be unavailable),
the main loop reads a single sentinel file in each structure's directory to
determine its current state.

Exactly one sentinel file is present in a structure directory at any time.
The filename maps directly to the Individual.status string:

    None              → "pending"
    PENDING           → "pending"
    SUBMITTED         → "submitted"
    RUNNING_1         → "running_stage_1"
    CONVERGED_1       → "converged_stage_1"
    RUNNING_2         → "running_stage_2"
    CONVERGED_2       → "converged_stage_2"
    CONVERGED         → "converged"
    DESORBED          → "desorbed"
    FAILED            → "failed"
    DUPLICATE         → "duplicate"
    ISOMER            → "isomer"

Usage
-----
    from pathlib import Path
    from gocia.database.status import write_sentinel, read_sentinel, clear_sentinels

    struct_dir = Path("gen_000/struct_0001")

    # Write a sentinel (clears any existing one first)
    write_sentinel(struct_dir, "pending")

    # Read current status from disk
    status = read_sentinel(struct_dir)   # returns "pending" or None

    # Transition to submitted
    write_sentinel(struct_dir, "submitted")
"""

from __future__ import annotations

from pathlib import Path

from gocia.population.individual import STATUS


# ---------------------------------------------------------------------------
# Mapping: status string  ↔  sentinel filename
# ---------------------------------------------------------------------------

def _status_to_filename(status: str) -> str:
    """
    Convert a status string to its sentinel filename.

    Examples
    --------
    "pending"           → "PENDING"
    "running_stage_1"   → "RUNNING_1"
    "converged_stage_2" → "CONVERGED_2"
    "converged"         → "CONVERGED"
    "desorbed"          → "DESORBED"
    """
    if STATUS.is_stage_running(status):
        n = STATUS.stage_number(status)
        return f"RUNNING_{n}"
    if STATUS.is_stage_converged(status):
        n = STATUS.stage_number(status)
        return f"CONVERGED_{n}"
    # Simple 1:1 mapping for all other statuses
    return status.upper()


def _filename_to_status(filename: str) -> str:
    """
    Convert a sentinel filename back to a status string.

    Examples
    --------
    "PENDING"      → "pending"
    "RUNNING_1"    → "running_stage_1"
    "CONVERGED_2"  → "converged_stage_2"
    "CONVERGED"    → "converged"
    """
    if filename.startswith("RUNNING_"):
        n = filename[len("RUNNING_"):]
        return f"running_stage_{n}"
    if filename.startswith("CONVERGED_") and filename != "CONVERGED":
        n = filename[len("CONVERGED_"):]
        return f"converged_stage_{n}"
    return filename.lower()


# All known non-stage sentinel filenames (for cleanup / scanning)
_STATIC_SENTINELS = {
    "PENDING", "SUBMITTED", "CONVERGED", "DESORBED", "FAILED",
    "DUPLICATE", "ISOMER",
}


def _all_sentinel_files(struct_dir: Path) -> list[Path]:
    """
    Return every sentinel file currently present in struct_dir.

    This includes both static sentinels (PENDING, CONVERGED, …) and
    dynamic stage sentinels (RUNNING_1, CONVERGED_2, …).
    """
    found = []
    for f in struct_dir.iterdir():
        name = f.name
        if name in _STATIC_SENTINELS:
            found.append(f)
            continue
        # Dynamic: RUNNING_N or CONVERGED_N where N is an integer
        for prefix in ("RUNNING_", "CONVERGED_"):
            if name.startswith(prefix):
                suffix = name[len(prefix):]
                if suffix.isdigit():
                    found.append(f)
                    break
    return found


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def write_sentinel(struct_dir: str | Path, status: str) -> Path:
    """
    Write a sentinel file for the given status, removing any existing ones.

    Parameters
    ----------
    struct_dir:
        The structure's working directory (e.g. gen_000/struct_0001).
    status:
        A valid Individual status string (see STATUS constants).

    Returns
    -------
    Path
        The path of the newly written sentinel file.

    Raises
    ------
    FileNotFoundError
        If struct_dir does not exist.
    """
    struct_dir = Path(struct_dir)
    if not struct_dir.is_dir():
        raise FileNotFoundError(f"Structure directory not found: {struct_dir}")

    # Remove any existing sentinels first
    clear_sentinels(struct_dir)

    filename = _status_to_filename(status)
    sentinel_path = struct_dir / filename
    sentinel_path.touch()
    return sentinel_path


def read_sentinel(struct_dir: str | Path) -> str | None:
    """
    Read the current status from the sentinel file in struct_dir.

    Parameters
    ----------
    struct_dir:
        The structure's working directory.

    Returns
    -------
    str or None
        The status string corresponding to the sentinel file found, or None
        if no sentinel file is present (should not happen in normal operation).
    """
    struct_dir = Path(struct_dir)
    if not struct_dir.is_dir():
        return None

    sentinels = _all_sentinel_files(struct_dir)

    if not sentinels:
        return None

    if len(sentinels) > 1:
        # More than one sentinel is a bug — log and return the most recent
        names = [s.name for s in sentinels]
        import warnings
        warnings.warn(
            f"Multiple sentinel files found in {struct_dir}: {names}. "
            "This indicates a corrupted state.  Using most recently modified.",
            stacklevel=2,
        )
        sentinels.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    return _filename_to_status(sentinels[0].name)


def clear_sentinels(struct_dir: str | Path) -> int:
    """
    Remove all sentinel files from struct_dir.

    Parameters
    ----------
    struct_dir:
        The structure's working directory.

    Returns
    -------
    int
        Number of sentinel files removed.
    """
    struct_dir = Path(struct_dir)
    sentinels = _all_sentinel_files(struct_dir)
    for s in sentinels:
        s.unlink(missing_ok=True)
    return len(sentinels)


def sentinel_exists(struct_dir: str | Path, status: str) -> bool:
    """
    Return True if the sentinel for the given status exists in struct_dir.

    Useful for quick checks without reading the full status:

        if sentinel_exists(struct_dir, STATUS.CONVERGED):
            ...
    """
    struct_dir = Path(struct_dir)
    filename = _status_to_filename(status)
    return (struct_dir / filename).exists()


def scan_generation(gen_dir: str | Path) -> dict[str, str]:
    """
    Scan an entire generation directory and return a mapping of
    structure_id → status for every structure found.

    Assumes the generation directory layout:

        gen_dir/
            struct_0001/   (directory name is used as the structure key)
            struct_0002/
            ...

    Parameters
    ----------
    gen_dir:
        Path to a generation directory (e.g. gen_000/).

    Returns
    -------
    dict[str, str]
        Mapping from subdirectory name to status string.
        Directories with no sentinel file are mapped to None.
    """
    gen_dir = Path(gen_dir)
    result = {}
    for entry in sorted(gen_dir.iterdir()):
        if entry.is_dir():
            result[entry.name] = read_sentinel(entry)
    return result
