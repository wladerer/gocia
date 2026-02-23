"""
gocia/calculator/mace_calc.py

MACE-MP-0 calculator setup and relaxation wrapper.

MACE-MP-0 is a universal machine-learning interatomic potential trained on
the Materials Project database.  It covers most element combinations and
works reasonably well for surface science without system-specific fine-tuning.

This module provides:
  - get_mace_calculator(): build and cache a MACE-MP-0 ASE calculator
  - run_mace_relaxation(): relax an Atoms object, write trajectory data,
    and manage sentinel files

The calculator is cached as a module-level singleton so it is only loaded
once per process — MACE model loading takes a few seconds on first call.

MACE requires the mace-torch package:
    pip install gocia[mace]

Marks
-----
Tests that import from this module should be decorated:
    @pytest.mark.mace
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.optimize import BFGS, LBFGS, GPMin

from gocia.calculator.stage import CalculatorStage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MACE model cache
# ---------------------------------------------------------------------------

_MACE_CALCULATOR = None   # module-level singleton


def get_mace_calculator(
    model: str = "medium",
    device: str = "auto",
    dispersion: bool = False,
) -> Any:
    """
    Return a cached MACE-MP-0 ASE calculator instance.

    The model is loaded once and reused for all subsequent calls within the
    same process.  This avoids the overhead of reloading the neural network
    weights for every structure.

    Parameters
    ----------
    model:
        MACE-MP model size: "small", "medium" (default), or "large".
        "medium" offers a good speed/accuracy trade-off for most surfaces.
    device:
        Compute device: "cpu", "cuda", "mps", or "auto".
        "auto" selects CUDA if available, then MPS, then CPU.
    dispersion:
        If True, add a D3 dispersion correction on top of MACE.
        Requires the torch-dftd package.

    Returns
    -------
    MACECalculator
        An ASE-compatible calculator with get_potential_energy(),
        get_forces(), and get_stress() methods.

    Raises
    ------
    ImportError
        If mace-torch is not installed.
    """
    global _MACE_CALCULATOR

    if _MACE_CALCULATOR is not None:
        return _MACE_CALCULATOR

    try:
        from mace.calculators import mace_mp
    except ImportError as exc:
        raise ImportError(
            "MACE requires the mace-torch package. "
            "Install it with: pip install gocia[mace]"
        ) from exc

    if device == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        except ImportError:
            device = "cpu"

    logger.info(f"Loading MACE-MP-0 ({model}) on {device}…")
    _MACE_CALCULATOR = mace_mp(model=model, device=device, dispersion=dispersion)
    logger.info("MACE-MP-0 loaded.")
    return _MACE_CALCULATOR


def clear_mace_cache() -> None:
    """Clear the cached MACE calculator (useful in tests)."""
    global _MACE_CALCULATOR
    _MACE_CALCULATOR = None


# ---------------------------------------------------------------------------
# Optimiser factory
# ---------------------------------------------------------------------------

_OPTIMISERS = {
    "BFGS":  BFGS,
    "LBFGS": LBFGS,
    "GPMin": GPMin,
}


def _get_optimiser_class(name: str):
    if name not in _OPTIMISERS:
        raise ValueError(
            f"Unknown optimiser '{name}'. "
            f"Available: {sorted(_OPTIMISERS)}."
        )
    return _OPTIMISERS[name]


def _get_cell_filter(name: str | None):
    """Return an ASE filter class, or None for positions-only optimisation."""
    if name is None:
        return None
    try:
        import ase.filters as ase_filters
        return getattr(ase_filters, name)
    except AttributeError:
        raise ValueError(
            f"Unknown ASE filter '{name}'. "
            "Check the ASE documentation for valid filter names."
        )


# ---------------------------------------------------------------------------
# Trajectory writer
# ---------------------------------------------------------------------------

class _HDF5TrajectoryWriter:
    """
    Writes ionic step data (positions, forces, energies, stresses) to an
    HDF5 file as the optimiser runs.

    Appends to the dataset on each step so memory usage stays bounded.
    """

    def __init__(self, h5_path: Path, group_name: str, atoms: Atoms) -> None:
        try:
            import h5py
        except ImportError as exc:
            raise ImportError(
                "h5py is required for trajectory writing. "
                "Install it with: pip install h5py"
            ) from exc

        self._h5py = h5py
        self.h5_path = h5_path
        self.group_name = group_name
        self.n_atoms = len(atoms)

        # Initialise resizable datasets
        self._file = h5py.File(str(h5_path), "a")
        grp = self._file.require_group(group_name)

        chunks_atoms = (1, self.n_atoms, 3)
        self._pos    = grp.require_dataset("positions", shape=(0, self.n_atoms, 3),
                                            maxshape=(None, self.n_atoms, 3),
                                            dtype="f8", chunks=chunks_atoms)
        self._forces = grp.require_dataset("forces",    shape=(0, self.n_atoms, 3),
                                            maxshape=(None, self.n_atoms, 3),
                                            dtype="f8", chunks=chunks_atoms)
        self._energies = grp.require_dataset("energies", shape=(0,),
                                              maxshape=(None,),
                                              dtype="f8", chunks=(1,))
        self._stresses = grp.require_dataset("stresses",  shape=(0, 3, 3),
                                              maxshape=(None, 3, 3),
                                              dtype="f8", chunks=(1, 3, 3))
        self._step = 0

    def write_step(self, atoms: Atoms) -> None:
        """Append current atomic state to the trajectory datasets."""
        n = self._step + 1

        pos = atoms.get_positions()
        try:
            forces = atoms.get_forces()
        except Exception:
            forces = np.zeros((self.n_atoms, 3))
        try:
            energy = float(atoms.get_potential_energy())
        except Exception:
            energy = float("nan")
        try:
            stress_voigt = atoms.get_stress()
            # Convert Voigt (6,) → full (3,3) symmetric tensor
            s = stress_voigt
            stress = np.array([
                [s[0], s[5], s[4]],
                [s[5], s[1], s[3]],
                [s[4], s[3], s[2]],
            ])
        except Exception:
            stress = np.zeros((3, 3))

        for dataset, data, new_shape in [
            (self._pos,      pos[np.newaxis],       (n, self.n_atoms, 3)),
            (self._forces,   forces[np.newaxis],    (n, self.n_atoms, 3)),
            (self._energies, np.array([energy]),    (n,)),
            (self._stresses, stress[np.newaxis],    (n, 3, 3)),
        ]:
            dataset.resize(new_shape)
            dataset[-1] = data

        self._step += 1

    def close(self) -> None:
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ---------------------------------------------------------------------------
# Sanity filter
# ---------------------------------------------------------------------------

def _check_sanity(
    atoms: Atoms,
    final_energy: float,
    stage: CalculatorStage,
    slab_energy: float | None,
    n_slab: int | None,
) -> None:
    """
    Raise RuntimeError if the relaxed structure looks unphysical.

    Two checks are applied:

    1. Energy-per-atom deviation from the slab reference.
       |E_total/N_total − E_slab/N_slab| > energy_per_atom_tol
       Catches Coulomb explosions, surface reconstructions that eject
       bulk atoms into vacuum, and other catastrophic failures where
       MACE reaches a wildly wrong basin.

    2. Maximum residual force component.
       max(|F|) > max_force_tol
       Catches cases where MACE "converged" by hitting max_steps but
       left atoms with unphysically large forces — a sign the structure
       is in a flat but wrong region of the PES.

    Both thresholds come from the stage config (defaults 10 eV/atom and
    10 eV/Å) and can be tightened in gocia.yaml per stage.
    """
    n_total = len(atoms)

    # --- Check 1: energy per atom vs slab reference ---
    if slab_energy is not None and n_slab is not None and n_slab > 0:
        ref_epa = slab_energy / n_slab          # eV/atom (slab reference)
        final_epa = final_energy / n_total      # eV/atom (this structure)
        delta_epa = abs(final_epa - ref_epa)

        if delta_epa > stage.energy_per_atom_tol:
            raise RuntimeError(
                f"Sanity check FAILED (stage '{stage.name}'): "
                f"energy per atom deviates {delta_epa:.2f} eV/atom from slab "
                f"reference ({ref_epa:.3f} eV/atom) — threshold is "
                f"{stage.energy_per_atom_tol} eV/atom.  "
                f"Final E = {final_energy:.3f} eV over {n_total} atoms.  "
                "Likely a Coulomb explosion or unphysical surface reconstruction."
            )

    # --- Check 2: maximum residual force ---
    try:
        forces = atoms.get_forces()
        max_force = float(np.abs(forces).max())
        if max_force > stage.max_force_tol:
            raise RuntimeError(
                f"Sanity check FAILED (stage '{stage.name}'): "
                f"max residual force {max_force:.2f} eV/Å exceeds threshold "
                f"{stage.max_force_tol} eV/Å after relaxation.  "
                "Structure is likely unphysical or the optimiser stalled."
            )
    except RuntimeError:
        raise
    except Exception as exc:
        # get_forces() can fail on a broken structure — treat as unphysical
        raise RuntimeError(
            f"Sanity check FAILED (stage '{stage.name}'): "
            f"could not evaluate forces on relaxed structure: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Main relaxation function
# ---------------------------------------------------------------------------

def run_mace_relaxation(
    atoms: Atoms,
    stage: CalculatorStage,
    struct_dir: Path,
    h5_path: Path,
    model: str = "medium",
    device: str = "auto",
    slab_energy: float | None = None,
    n_slab: int | None = None,
) -> tuple[Atoms, float]:
    """
    Relax an Atoms object using MACE-MP-0 and write trajectory data.

    Parameters
    ----------
    atoms:
        Input geometry.  Not modified — a copy is made internally.
    stage:
        The CalculatorStage describing fmax, max_steps, optimiser, filter,
        and sanity-filter thresholds (energy_per_atom_tol, max_force_tol).
    struct_dir:
        Structure working directory.  The relaxed geometry is written here
        as "CONTCAR_{stage.name}" (VASP format for easy inspection).
    h5_path:
        Path to the trajectory.h5 file.  Data is appended to a new group
        named stage.hdf5_group.
    model:
        MACE-MP model size ("small", "medium", "large").
    device:
        Compute device ("cpu", "cuda", "mps", "auto").
    slab_energy:
        DFT/MACE energy of the bare slab (eV).  Used as reference for the
        energy-per-atom sanity check.  If None the check is skipped.
    n_slab:
        Number of slab atoms.  Required when slab_energy is provided.

    Returns
    -------
    (relaxed_atoms, final_energy)
        relaxed_atoms: Atoms with optimised positions.
        final_energy: potential energy in eV at the relaxed geometry.

    Raises
    ------
    RuntimeError
        If the optimisation fails, or if the relaxed structure fails the
        post-relaxation sanity checks (unphysical energy or forces).
    """
    from ase.io import write

    calc = get_mace_calculator(model=model, device=device)
    work = atoms.copy()
    work.calc = calc

    # Apply cell filter if requested
    filter_cls = _get_cell_filter(stage.cell_filter)
    target = filter_cls(work) if filter_cls is not None else work

    # Optimiser
    opt_cls = _get_optimiser_class(stage.optimizer)

    logfile_path = str(struct_dir / f"mace_{stage.name}.log")

    with _HDF5TrajectoryWriter(h5_path, stage.hdf5_group, work) as writer:

        def _step_callback():
            writer.write_step(work)

        try:
            opt = opt_cls(target, logfile=logfile_path)
            opt.attach(_step_callback)
            converged = opt.run(fmax=stage.fmax, steps=stage.max_steps)
        except Exception as exc:
            raise RuntimeError(
                f"MACE relaxation failed at stage '{stage.name}': {exc}"
            ) from exc

    if not converged:
        logger.warning(
            f"MACE stage '{stage.name}' did not converge within "
            f"{stage.max_steps} steps (fmax={stage.fmax} eV/Å). "
            "Proceeding with best geometry reached."
        )

    final_energy = float(work.get_potential_energy())

    # ------------------------------------------------------------------
    # Post-relaxation sanity checks
    # ------------------------------------------------------------------
    _check_sanity(work, final_energy, stage, slab_energy, n_slab)

    # Write relaxed geometry for inspection
    contcar_path = struct_dir / f"CONTCAR_{stage.name}"
    write(str(contcar_path), work, format="vasp")
    logger.debug(f"  Written {contcar_path}")

    return work, final_energy
