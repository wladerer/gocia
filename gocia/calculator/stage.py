"""
gocia/calculator/stage.py

CalculatorStage: a single step in the staged optimisation pipeline.

Each stage wraps one calculator (MACE or VASP) with its own convergence
criteria, ASE optimiser, and optional cell/strain filter.  Stages are
built from CalculatorStageConfig (from gocia.yaml) by build_stage().

The pipeline (pipeline.py) runs stages in order, passing the output
geometry of each stage as input to the next.

Design notes
------------
- Stages are plain dataclasses — no inheritance, no magic.
- The ASE optimiser and filter are stored as strings and instantiated
  lazily so that importing this module never requires MACE or VASP.
- VASP-specific fields (incar, kpoints) live directly on the stage so
  vasp_calc.py can read them without reaching back into the config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# CalculatorStage dataclass
# ---------------------------------------------------------------------------

@dataclass
class CalculatorStage:
    """
    A single stage in the optimisation pipeline.

    Attributes
    ----------
    name:
        Human-readable label used as the HDF5 group name in trajectory.h5
        and in sentinel filenames (RUNNING_1, CONVERGED_1, …).
    stage_index:
        1-based position in the pipeline (1 = first stage).  Set by
        build_pipeline() — do not set manually.
    calculator_type:
        "mace" or "vasp".
    fmax:
        Force convergence threshold in eV/Å.
    max_steps:
        Maximum number of ionic steps.
    optimizer:
        ASE optimiser class name.  Default "BFGS" for MACE,
        ignored for VASP (uses NSW in INCAR).
    cell_filter:
        ASE filter class name for cell/strain relaxation, or None for
        positions-only optimisation.  E.g. "FrechetCellFilter".
        Ignored for VASP stages (handled by ISIF in INCAR).
    incar:
        VASP INCAR key-value overrides.  Merged on top of DEFAULT_INCAR
        in vasp_calc.py.  None for MACE stages.
    kpoints:
        VASP KPOINTS specification.  None uses the ASE default.
    extra:
        Forward-compatible overflow dict for any future per-stage options.
    """

    name: str
    stage_index: int = 1
    calculator_type: str = "mace"       # "mace" | "vasp"
    fmax: float = 0.05
    max_steps: int = 500
    optimizer: str = "BFGS"
    cell_filter: str | None = None
    incar: dict[str, Any] | None = None
    kpoints: dict[str, Any] | None = None
    energy_per_atom_tol: float = 10.0   # max |ΔE/atom| vs slab reference (eV/atom)
    max_force_tol: float = 10.0         # max residual force after relaxation (eV/Å)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_mace(self) -> bool:
        return self.calculator_type == "mace"

    @property
    def is_vasp(self) -> bool:
        return self.calculator_type == "vasp"

    @property
    def hdf5_group(self) -> str:
        """Name of the HDF5 group in trajectory.h5 for this stage."""
        return f"stage_{self.name}"

    def __repr__(self) -> str:
        return (
            f"CalculatorStage(name={self.name!r}, type={self.calculator_type}, "
            f"fmax={self.fmax}, max_steps={self.max_steps})"
        )


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_stage(config, stage_index: int) -> CalculatorStage:
    """
    Build a CalculatorStage from a CalculatorStageConfig.

    Parameters
    ----------
    config:
        A CalculatorStageConfig pydantic model instance.
    stage_index:
        1-based position in the pipeline.

    Returns
    -------
    CalculatorStage
    """
    return CalculatorStage(
        name=config.name,
        stage_index=stage_index,
        calculator_type=config.type,
        fmax=config.fmax,
        max_steps=config.max_steps,
        incar=config.incar,
        energy_per_atom_tol=config.energy_per_atom_tol,
        max_force_tol=config.max_force_tol,
    )


def build_pipeline(configs: list) -> list[CalculatorStage]:
    """
    Build an ordered list of CalculatorStages from a list of
    CalculatorStageConfig objects.

    Parameters
    ----------
    configs:
        List of CalculatorStageConfig instances, in execution order.

    Returns
    -------
    list[CalculatorStage]
        Stages with stage_index set (1-based).
    """
    return [build_stage(cfg, i + 1) for i, cfg in enumerate(configs)]
