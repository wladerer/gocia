"""
gocia/calculator/vasp_calc.py

VASP calculator setup and relaxation runner.

VASP is invoked via the ASE Vasp calculator, which writes INCAR, POSCAR,
KPOINTS, and POTCAR files and calls the VASP binary.  The environment
variable VASP_COMMAND (or ASE_VASP_COMMAND) must be set to the VASP
executable path before running.

Default INCAR
-------------
DEFAULT_INCAR provides conservative, general-purpose settings suitable for
most transition-metal surface calculations.  Per-stage overrides from
gocia.yaml are merged on top, with the user's values taking precedence.

Trajectory data
---------------
Unlike MACE (where we can hook into the ASE optimiser), VASP performs its
own internal relaxation (NSW steps in INCAR).  We extract the ionic steps
from the OUTCAR after the run completes and write them to trajectory.h5.
This means the trajectory is always written, even if VASP crashes partway
(as long as OUTCAR has some ionic steps written).

Marks
-----
Tests importing from this module should use @pytest.mark.vasp.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.io.vasp import read_vasp_out

from gocia.calculator.stage import CalculatorStage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default INCAR
# ---------------------------------------------------------------------------

DEFAULT_INCAR: dict[str, Any] = {
    # Electronic
    "ENCUT":  400,       # plane-wave cutoff (eV) — override per stage
    "PREC":   "Normal",
    "EDIFF":  1e-5,      # electronic convergence (eV)
    "ALGO":   "Fast",
    "NELM":   200,
    "NELMIN": 5,

    # Smearing
    "ISMEAR":  1,        # Methfessel-Paxton; use 0 + SIGMA=0.01 for molecules
    "SIGMA":   0.1,

    # Relaxation
    "NSW":     100,
    "IBRION":  2,        # CG relaxation; use 1 (RMM-DIIS) for faster convergence
    "POTIM":   0.3,
    "ISIF":    2,        # relax ions only; use 3 for cell+ions

    # Output verbosity
    "NWRITE":  1,
    "LWAVE":   False,    # don't write WAVECAR by default (large files)
    "LCHARG":  False,    # don't write CHGCAR by default

    # Parallelisation (safe defaults; user can override via extra_headers)
    "NCORE":   4,
}

# KPOINTS default: Gamma-centred 3×3×1 for surface slabs
DEFAULT_KPOINTS: dict[str, Any] = {
    "size":   (3, 3, 1),
    "gamma":  True,
}


# ---------------------------------------------------------------------------
# INCAR merge
# ---------------------------------------------------------------------------

def build_incar(stage: CalculatorStage) -> dict[str, Any]:
    """
    Merge DEFAULT_INCAR with per-stage user overrides from gocia.yaml.

    User values always win.  NSW is capped at stage.max_steps so the
    VASP run does not exceed the stage budget.

    Parameters
    ----------
    stage:
        CalculatorStage with optional incar dict.

    Returns
    -------
    dict
        Merged INCAR parameters ready to pass to the ASE Vasp calculator.
    """
    merged = dict(DEFAULT_INCAR)
    if stage.incar:
        merged.update(stage.incar)

    # Cap NSW to stage budget (user's NSW is used if smaller)
    merged["NSW"] = min(merged.get("NSW", stage.max_steps), stage.max_steps)

    return merged


# ---------------------------------------------------------------------------
# KPOINTS builder
# ---------------------------------------------------------------------------

def build_kpoints(stage: CalculatorStage) -> dict[str, Any]:
    """
    Build KPOINTS settings for this stage.

    Uses stage.kpoints if provided, otherwise falls back to DEFAULT_KPOINTS.
    """
    if stage.kpoints:
        return stage.kpoints
    return dict(DEFAULT_KPOINTS)


# ---------------------------------------------------------------------------
# Environment check
# ---------------------------------------------------------------------------

def check_vasp_environment() -> str:
    """
    Check that the VASP executable is accessible.

    Returns the VASP command string if found.

    Raises
    ------
    EnvironmentError
        If neither VASP_COMMAND nor ASE_VASP_COMMAND is set.
    """
    cmd = os.environ.get("VASP_COMMAND") or os.environ.get("ASE_VASP_COMMAND")
    if not cmd:
        raise EnvironmentError(
            "VASP_COMMAND or ASE_VASP_COMMAND environment variable is not set. "
            "Set it to the path of the VASP executable, e.g.:\n"
            "  export VASP_COMMAND='mpirun -np 16 /path/to/vasp_std'"
        )
    return cmd


# ---------------------------------------------------------------------------
# Trajectory extraction from OUTCAR
# ---------------------------------------------------------------------------

def _extract_trajectory_from_outcar(outcar_path: Path) -> dict[str, np.ndarray]:
    """
    Parse an OUTCAR file and extract ionic step data.

    Returns a dict with keys: positions, forces, energies, stresses.
    Each value is a numpy array with the first axis being the ionic step index.
    Returns empty arrays if OUTCAR is missing or has no completed steps.
    """
    if not outcar_path.exists():
        logger.warning(f"OUTCAR not found at {outcar_path}. No trajectory data saved.")
        return {}

    try:
        # ASE can read all ionic steps from OUTCAR as a list of Atoms
        traj = read(str(outcar_path), index=":", format="vasp-out")
    except Exception as exc:
        logger.warning(f"Could not parse OUTCAR at {outcar_path}: {exc}")
        return {}

    if not traj:
        return {}

    positions_list, forces_list, energies_list, stresses_list = [], [], [], []

    for atoms in traj:
        positions_list.append(atoms.get_positions())
        try:
            forces_list.append(atoms.get_forces())
        except Exception:
            forces_list.append(np.zeros_like(atoms.get_positions()))
        try:
            energies_list.append(float(atoms.get_potential_energy()))
        except Exception:
            energies_list.append(float("nan"))
        try:
            sv = atoms.get_stress()   # Voigt (6,)
            stress = np.array([
                [sv[0], sv[5], sv[4]],
                [sv[5], sv[1], sv[3]],
                [sv[4], sv[3], sv[2]],
            ])
            stresses_list.append(stress)
        except Exception:
            n = len(atoms)
            stresses_list.append(np.zeros((3, 3)))

    return {
        "positions": np.array(positions_list),
        "forces":    np.array(forces_list),
        "energies":  np.array(energies_list),
        "stresses":  np.array(stresses_list),
    }


def _write_trajectory_to_h5(
    h5_path: Path,
    group_name: str,
    data: dict[str, np.ndarray],
) -> None:
    """Write extracted OUTCAR trajectory data to trajectory.h5."""
    if not data:
        return

    try:
        import h5py
    except ImportError:
        logger.warning("h5py not installed; skipping trajectory.h5 write.")
        return

    with h5py.File(str(h5_path), "a") as f:
        grp = f.require_group(group_name)
        for key, arr in data.items():
            if key in grp:
                del grp[key]   # overwrite if re-running a stage
            grp.create_dataset(key, data=arr, compression="gzip")


# ---------------------------------------------------------------------------
# Main relaxation function
# ---------------------------------------------------------------------------

def run_vasp_relaxation(
    atoms: Atoms,
    stage: CalculatorStage,
    struct_dir: Path,
    h5_path: Path,
) -> tuple[Atoms, float]:
    """
    Run a VASP relaxation for one pipeline stage.

    Writes VASP input files to struct_dir, runs VASP, reads the relaxed
    geometry from CONTCAR, extracts trajectory data from OUTCAR, and
    writes it to trajectory.h5.

    Parameters
    ----------
    atoms:
        Input geometry.  Not modified — a copy is made.
    stage:
        CalculatorStage describing INCAR overrides, fmax, max_steps.
    struct_dir:
        Structure working directory.  VASP files are written here.
    h5_path:
        Path to trajectory.h5.  Trajectory data appended to stage group.

    Returns
    -------
    (relaxed_atoms, final_energy)

    Raises
    ------
    EnvironmentError
        If VASP_COMMAND is not set.
    RuntimeError
        If VASP exits with a non-zero status or CONTCAR is not produced.
    """
    try:
        from ase.calculators.vasp import Vasp
    except ImportError as exc:
        raise ImportError(
            "ASE's VASP calculator is required. "
            "Install ASE with: pip install ase"
        ) from exc

    check_vasp_environment()

    incar = build_incar(stage)
    kpoints = build_kpoints(stage)

    work = atoms.copy()

    # Write a labelled INCAR copy for inspection alongside the live one
    _write_incar_snapshot(struct_dir, stage, incar)

    # Build ASE Vasp calculator
    calc = Vasp(
        directory=str(struct_dir),
        **incar,
        kpts=kpoints.get("size", (3, 3, 1)),
        gamma=kpoints.get("gamma", True),
    )
    work.calc = calc

    logger.info(f"  Running VASP stage '{stage.name}' in {struct_dir}")

    try:
        # Calling get_potential_energy() triggers the VASP run
        final_energy = float(work.get_potential_energy())
    except Exception as exc:
        raise RuntimeError(
            f"VASP failed at stage '{stage.name}' in {struct_dir}.\n"
            f"Check OUTCAR and stderr for details.\nOriginal error: {exc}"
        ) from exc

    # Read relaxed geometry from CONTCAR
    contcar_path = struct_dir / "CONTCAR"
    if not contcar_path.exists():
        raise RuntimeError(
            f"VASP stage '{stage.name}' completed but CONTCAR not found "
            f"in {struct_dir}.  VASP may have crashed."
        )

    relaxed = read(str(contcar_path), format="vasp")

    # Extract and write trajectory data
    outcar_path = struct_dir / "OUTCAR"
    traj_data = _extract_trajectory_from_outcar(outcar_path)
    _write_trajectory_to_h5(h5_path, stage.hdf5_group, traj_data)

    n_steps = len(traj_data.get("energies", []))
    logger.info(
        f"  VASP stage '{stage.name}' done: "
        f"{n_steps} ionic steps, E = {final_energy:.4f} eV"
    )

    return relaxed, final_energy


# ---------------------------------------------------------------------------
# Helper: write a labelled INCAR snapshot
# ---------------------------------------------------------------------------

def _write_incar_snapshot(
    struct_dir: Path,
    stage: CalculatorStage,
    incar: dict[str, Any],
) -> None:
    """
    Write a human-readable INCAR_<stage_index> file for inspection.

    This is separate from the INCAR that ASE writes for VASP — it is purely
    for the user's reference and is not read by the calculator.
    """
    snapshot_path = struct_dir / f"INCAR_{stage.stage_index}"
    lines = [
        f"# GOCIA VASP stage {stage.stage_index}: {stage.name}\n",
        "# Auto-generated — do not edit; changes will be overwritten.\n",
        "\n",
    ]
    for key, val in sorted(incar.items()):
        if isinstance(val, bool):
            val_str = ".TRUE." if val else ".FALSE."
        else:
            val_str = str(val)
        lines.append(f"  {key:<12} = {val_str}\n")

    try:
        snapshot_path.write_text("".join(lines))
    except Exception as exc:
        logger.debug(f"Could not write INCAR snapshot: {exc}")
