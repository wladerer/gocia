"""
gocia/calculator/pipeline.py

Staged optimisation pipeline orchestrator.

The pipeline takes an Atoms object and runs it through an ordered list of
CalculatorStages.  Each stage:

  1. Reads the geometry produced by the previous stage (or the initial
     geometry for stage 1).
  2. Writes the RUNNING_N sentinel file.
  3. Runs the appropriate calculator (MACE or VASP).
  4. Writes ionic step data to trajectory.h5.
  5. Writes the CONVERGED_N sentinel file.
  6. Passes the relaxed geometry to the next stage.

After all stages complete, the final geometry is written as CONTCAR and
the CONVERGED sentinel is written.

Restart behaviour
-----------------
On restart (after HPC timeout), the pipeline reads the current sentinel
file and resumes from the appropriate stage:

  CONVERGED_N  → skip stages 1..N, resume from stage N+1
  RUNNING_N    → re-run stage N from scratch (treat as if PENDING)
  CONVERGED    → all done; return early
  FAILED       → raise RuntimeError (user must inspect and reset manually)

This means the pipeline is safe to call multiple times on the same
structure directory — it will not re-run completed stages.

Public API
----------
    run_pipeline(atoms, stages, struct_dir, h5_path) → (Atoms, float)
    resume_stage(struct_dir, stages) → int   (index of next stage to run, 0-based)
"""

from __future__ import annotations

import logging
from pathlib import Path

from ase import Atoms
from ase.io import write, read

from gocia.calculator.stage import CalculatorStage
from gocia.database.status import (
    write_sentinel,
    read_sentinel,
    sentinel_exists,
)
from gocia.population.individual import STATUS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Restart helpers
# ---------------------------------------------------------------------------

def resume_stage(struct_dir: Path, stages: list[CalculatorStage]) -> int:
    """
    Determine which stage to resume from based on sentinel files.

    Parameters
    ----------
    struct_dir:
        Structure working directory.
    stages:
        Ordered list of CalculatorStage objects (1-indexed internally).

    Returns
    -------
    int
        0-based index of the next stage to run.
        0 means start from the beginning.
        len(stages) means all stages are already complete.

    Raises
    ------
    RuntimeError
        If the structure is in FAILED status — user must intervene.
    """
    current_status = read_sentinel(struct_dir)

    if current_status in (None, STATUS.PENDING, STATUS.SUBMITTED):
        # None      → directory just created, no sentinel yet
        # PENDING   → written by loop before submission, pipeline not yet started
        # SUBMITTED → job was submitted but died before pipeline wrote running_stage_1
        return 0

    if current_status == STATUS.FAILED:
        raise RuntimeError(
            f"Structure at {struct_dir} has status FAILED. "
            "Inspect the OUTCAR/mace log, fix the issue, remove the FAILED "
            "sentinel file, and re-run."
        )

    if current_status == STATUS.CONVERGED:
        return len(stages)   # all done

    if STATUS.is_stage_converged(current_status):
        completed_n = STATUS.stage_number(current_status)
        return completed_n   # 1-based completed → 0-based next = same number

    if STATUS.is_stage_running(current_status):
        # Running means the job was interrupted — re-run this stage
        running_n = STATUS.stage_number(current_status)
        return running_n - 1   # 1-based running → 0-based restart = N-1

    # Unknown status (e.g. DESORBED, DUPLICATE set externally) — skip
    logger.warning(
        f"Unexpected status '{current_status}' at {struct_dir}. "
        "Pipeline will not run."
    )
    return len(stages)


def _read_previous_geometry(struct_dir: Path, stage: CalculatorStage, stages: list[CalculatorStage]) -> Atoms | None:
    """
    Read the geometry produced by the previous stage, if it exists.

    Looks for CONTCAR_{prev_stage_name}, falling back to CONTCAR, then
    POSCAR.  Returns None if nothing is found.
    """
    if stage.stage_index > 1:
        # mace_calc.py writes CONTCAR_{stage.name} (e.g. CONTCAR_mace_preopt)
        # vasp_calc.py writes the same convention.
        # Also accept the generic CONTCAR_stage{N} as a fallback.
        prev_stage = stages[stage.stage_index - 2]  # 1-based → 0-based
        prev_name_candidates = [
            struct_dir / f"CONTCAR_{prev_stage.name}",
            struct_dir / f"CONTCAR_stage{stage.stage_index - 1}",
        ]
        for path in prev_name_candidates:
            if path.exists():
                return read(str(path), format="vasp")

    # Fall back to CONTCAR then POSCAR
    for fname in ("CONTCAR", "POSCAR"):
        p = struct_dir / fname
        if p.exists():
            try:
                return read(str(p), format="vasp")
            except Exception:
                continue

    return None


# ---------------------------------------------------------------------------
# Main pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(
    atoms: Atoms,
    stages: list[CalculatorStage],
    struct_dir: Path,
    h5_path: Path | None = None,
    mace_model: str = "medium",
    mace_device: str = "auto",
    slab_energy: float | None = None,
    n_slab: int | None = None,
) -> tuple[Atoms, float]:
    """
    Run an Atoms object through an ordered list of CalculatorStages.

    Manages sentinel files, trajectory writing, and restart behaviour.

    Parameters
    ----------
    atoms:
        Initial geometry.  Must already be placed in struct_dir (POSCAR
        should exist there before calling this function).
    stages:
        Ordered list of CalculatorStage objects from build_pipeline().
    struct_dir:
        Structure working directory.  Created if it does not exist.
    h5_path:
        Path to trajectory.h5.  Defaults to struct_dir/trajectory.h5.
    mace_model:
        MACE-MP model size, passed through to run_mace_relaxation().
    mace_device:
        Compute device for MACE.
    slab_energy:
        Energy of the bare slab in eV, used for the post-relaxation
        energy-per-atom sanity check.  If None the check is skipped.
    n_slab:
        Number of slab atoms.  Required when slab_energy is provided.

    Returns
    -------
    (relaxed_atoms, final_energy)
        relaxed_atoms: Atoms at the final converged geometry.
        final_energy: potential energy in eV from the last stage.

    Raises
    ------
    RuntimeError
        If any stage fails (calculator error, sanity check, missing files).
    ValueError
        If stages list is empty.
    """
    if not stages:
        raise ValueError("stages list is empty — nothing to run.")

    struct_dir = Path(struct_dir)
    struct_dir.mkdir(parents=True, exist_ok=True)

    if h5_path is None:
        h5_path = struct_dir / "trajectory.h5"

    # Write initial geometry if POSCAR doesn't exist yet
    poscar_path = struct_dir / "POSCAR"
    if not poscar_path.exists():
        write(str(poscar_path), atoms, format="vasp")

    # Determine restart point
    start_idx = resume_stage(struct_dir, stages)

    if start_idx >= len(stages):
        # All stages already complete — read final geometry and return
        logger.info(f"  All stages already complete for {struct_dir.name}. Skipping.")
        final_atoms, final_energy = _load_final_result(struct_dir, stages[-1])
        return final_atoms, final_energy

    current_atoms = atoms.copy()

    # If restarting mid-pipeline, load the geometry from the last completed stage
    if start_idx > 0:
        prev_stage = stages[start_idx - 1]
        resumed = _read_previous_geometry(struct_dir, prev_stage, stages)
        if resumed is not None:
            current_atoms = resumed
            logger.info(
                f"  Resuming {struct_dir.name} from stage "
                f"{start_idx + 1}/{len(stages)} ({stages[start_idx].name})"
            )

    final_energy = float("nan")

    for idx in range(start_idx, len(stages)):
        stage = stages[idx]

        logger.info(
            f"  [{struct_dir.name}] Stage {stage.stage_index}/{len(stages)}: "
            f"{stage.name} ({stage.calculator_type})"
        )

        # Mark as running
        write_sentinel(struct_dir, STATUS.running_stage(stage.stage_index))

        try:
            if stage.is_mace:
                from gocia.calculator.mace_calc import run_mace_relaxation
                current_atoms, final_energy = run_mace_relaxation(
                    atoms=current_atoms,
                    stage=stage,
                    struct_dir=struct_dir,
                    h5_path=h5_path,
                    model=mace_model,
                    device=mace_device,
                    slab_energy=slab_energy,
                    n_slab=n_slab,
                )

            elif stage.is_vasp:
                from gocia.calculator.vasp_calc import run_vasp_relaxation
                current_atoms, final_energy = run_vasp_relaxation(
                    atoms=current_atoms,
                    stage=stage,
                    struct_dir=struct_dir,
                    h5_path=h5_path,
                )

            else:
                raise ValueError(
                    f"Unknown calculator type '{stage.calculator_type}' "
                    f"in stage '{stage.name}'."
                )

        except Exception as exc:
            # Write FAILED sentinel and re-raise so the runner can handle it
            write_sentinel(struct_dir, STATUS.FAILED)
            logger.error(
                f"  Stage '{stage.name}' FAILED for {struct_dir.name}: {exc}"
            )
            raise RuntimeError(
                f"Pipeline failed at stage '{stage.name}' for {struct_dir}: {exc}"
            ) from exc

        # Stage succeeded — write intermediate sentinel
        write_sentinel(struct_dir, STATUS.converged_stage(stage.stage_index))
        logger.info(
            f"  [{struct_dir.name}] Stage {stage.name} converged: "
            f"E = {final_energy:.4f} eV"
        )

    # All stages complete — write final CONTCAR and CONVERGED sentinel
    _write_final_output(struct_dir, current_atoms, final_energy)
    write_sentinel(struct_dir, STATUS.CONVERGED)

    logger.info(
        f"  [{struct_dir.name}] Pipeline complete. "
        f"Final energy: {final_energy:.4f} eV"
    )

    return current_atoms, final_energy


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _write_final_output(
    struct_dir: Path,
    atoms: Atoms,
    energy: float,
) -> None:
    """Write final CONTCAR and a one-line energy summary."""
    contcar_path = struct_dir / "CONTCAR"
    write(str(contcar_path), atoms, format="vasp")

    energy_path = struct_dir / "FINAL_ENERGY"
    energy_path.write_text(f"{energy:.8f}\n")

    logger.debug(f"  Written {contcar_path} and FINAL_ENERGY")


def _load_final_result(
    struct_dir: Path,
    last_stage: CalculatorStage,
) -> tuple[Atoms, float]:
    """
    Load the final geometry and energy for a already-completed structure.

    Used when the pipeline is called on a structure that has CONVERGED
    already (e.g. restarted run that checks status before submitting).
    """
    contcar_path = struct_dir / "CONTCAR"
    energy_path  = struct_dir / "FINAL_ENERGY"

    if not contcar_path.exists():
        raise RuntimeError(
            f"CONVERGED sentinel found at {struct_dir} but CONTCAR is missing."
        )

    atoms = read(str(contcar_path), format="vasp")

    energy = float("nan")
    if energy_path.exists():
        try:
            energy = float(energy_path.read_text().strip())
        except ValueError:
            pass

    return atoms, energy
