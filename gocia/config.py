"""
gocia/config.py

Load and validate a gocia.yaml file into typed configuration models.

Usage
-----
    from gocia.config import load_config

    cfg = load_config("gocia.yaml")
    print(cfg.slab.energy)
    print(cfg.ga.population_size)

All models use pydantic v2.  Integer inputs for float fields (e.g. pH: 7,
temperature: 298) are automatically coerced without extra validators.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, field_validator, model_validator


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class AdsorbateConfig(BaseModel):
    """
    One adsorbate species that can be placed on the surface.

    For single atoms (e.g. O, H) leave `geometry` as None.
    For molecules (e.g. OH, CO) provide a path to a geometry file
    OR a list of [x, y, z] coordinates for each atom in the molecule
    via the `coordinates` field.
    """

    symbol: str
    chemical_potential: float           # eV referenced to standard state
    geometry: str | None = None         # path to geometry file (VASP, XYZ, …)
    coordinates: list[list[float]] | None = None  # [[x,y,z], …] alternative to file
    n_orientations: int = 6             # orientations trialled at placement

    @model_validator(mode="after")
    def _geometry_or_coordinates_not_both(self) -> "AdsorbateConfig":
        if self.geometry is not None and self.coordinates is not None:
            raise ValueError(
                f"Adsorbate '{self.symbol}': provide either 'geometry' or "
                "'coordinates', not both."
            )
        return self


class SlabConfig(BaseModel):
    """
    The fixed substrate slab.

    The geometry file should already contain selective dynamics tags so that
    bulk atoms are frozen and surface atoms are free.  The clean slab energy
    is used as a reference when computing the grand canonical fitness.
    """

    geometry: str                       # path to slab geometry file
    energy: float                       # DFT energy of the clean slab (eV)
    sampling_zmin: float                # min z for adsorbate placement (Å)
    sampling_zmax: float                # max z for adsorbate placement (Å)
    # Per-element chemical potentials for surface stoichiometry changes (future use)
    chemical_potentials: dict[str, float] = {}

    @model_validator(mode="after")
    def _zmin_below_zmax(self) -> "SlabConfig":
        if self.sampling_zmin >= self.sampling_zmax:
            raise ValueError(
                f"sampling_zmin ({self.sampling_zmin}) must be less than "
                f"sampling_zmax ({self.sampling_zmax})."
            )
        return self


class CalculatorStageConfig(BaseModel):
    """
    One stage in the staged optimization pipeline.

    Stages are executed in the order they appear in gocia.yaml.  Each stage
    reads the geometry produced by the previous stage.

    type options
    ------------
    "mace"  – MACE-MP-0 universal potential via mace-torch
    "vasp"  – VASP via ASE; requires `incar` dict

    The `incar` dict is merged on top of a default INCAR inside
    gocia/calculator/vasp_calc.py.  Keys in `incar` take precedence.
    """

    name: str                           # human-readable label, used as HDF5 group name
    type: str                           # "mace" | "vasp"
    fmax: float = 0.05                  # force convergence threshold (eV/Å)
    max_steps: int = 500                # max ionic steps
    incar: dict[str, Any] | None = None # VASP INCAR overrides

    # Sanity filters applied after relaxation to catch unphysical outcomes
    # (e.g. Coulomb explosions, surface reconstruction artefacts).
    # Structures that fail either check are marked FAILED and excluded from
    # selection — they are never used as parents or counted in fitness ranking.
    energy_per_atom_tol: float = 10.0   # max |ΔE/atom| from slab reference (eV/atom)
    max_force_tol: float = 10.0         # max residual force component (eV/Å)

    @field_validator("type")
    @classmethod
    def _valid_type(cls, v: str) -> str:
        allowed = {"mace", "vasp"}
        if v not in allowed:
            raise ValueError(f"calculator type must be one of {allowed}, got '{v}'.")
        return v

    @model_validator(mode="after")
    def _vasp_needs_incar(self) -> "CalculatorStageConfig":
        # Warn (not error) if a VASP stage has no INCAR overrides — defaults will be used
        return self


class SchedulerResources(BaseModel):
    """
    Structured, cross-scheduler resource specification.

    All fields are optional. The scheduler backend renders these into the
    appropriate directive syntax (e.g. ``mem`` -> ``#SBATCH --mem=32G`` on
    Slurm, ``#PBS -l mem=32gb`` on PBS).

    For anything not covered here, use SchedulerConfig.extra_directives.

    Fields
    ------
    nodes:
        Number of compute nodes.
    tasks_per_node:
        MPI ranks per node (Slurm: ``--ntasks-per-node``, PBS: ``-l mpiprocs``).
    cpus_per_task:
        Threads per MPI rank (Slurm: ``--cpus-per-task``).
        Set this to match NCORE in VASP INCAR.
    mem:
        Memory per node as a string with units, e.g. ``"32G"``, ``"128G"``.
        Slurm: ``--mem``. PBS: ``-l mem``.
    mem_per_cpu:
        Memory per CPU core. Slurm only: ``--mem-per-cpu``.
        Cannot be set together with ``mem``.
    gpus:
        Number of GPUs per node. Slurm: ``--gres=gpu:N``. PBS: ``-l ngpus=N``.
    account:
        Billing account / project code. Slurm: ``--account``. PBS: ``-A``.
    partition:
        Slurm partition or PBS queue name.
    qos:
        Quality of service string. Slurm: ``--qos``. PBS: ``-q``.
    constraint:
        Node feature constraint. Slurm: ``--constraint``.
    """

    nodes: int | None = None
    tasks_per_node: int | None = None
    cpus_per_task: int | None = None
    mem: str | None = None
    mem_per_cpu: str | None = None
    gpus: int | None = None
    account: str | None = None
    partition: str | None = None
    qos: str | None = None
    constraint: str | None = None

    @model_validator(mode="after")
    def _mem_exclusive(self) -> "SchedulerResources":
        if self.mem is not None and self.mem_per_cpu is not None:
            raise ValueError(
                "Specify either 'mem' (total per node) or 'mem_per_cpu', not both."
            )
        return self


class SchedulerConfig(BaseModel):
    """
    HPC job scheduler settings.

    Common resources (memory, CPUs, GPUs, account, partition) go in the
    structured ``resources`` block, validated and rendered correctly per
    scheduler type.

    Site-specific directives with no structured field go in
    ``extra_directives`` as raw option strings (without leading ``#SBATCH``
    or ``#PBS``). Inserted verbatim after the structured resource lines.

    Example gocia.yaml::

        scheduler:
          type: slurm
          nworkers: 20
          walltime: "04:00:00"
          resources:
            nodes: 1
            tasks_per_node: 16
            mem: "32G"
            account: "myproject"
            partition: "regular"
            constraint: "skylake"
          extra_directives:
            - "--mail-type=FAIL"
            - "--mail-user=user@uni.edu"
    """

    type: str                               # "slurm" | "pbs" | "local"
    nworkers: int                           # max concurrent jobs
    walltime: str                           # e.g. "04:00:00"
    resources: SchedulerResources = SchedulerResources()
    extra_directives: list[str] = []

    @field_validator("type")
    @classmethod
    def _valid_type(cls, v: str) -> str:
        allowed = {"slurm", "pbs", "local"}
        if v not in allowed:
            raise ValueError(f"scheduler type must be one of {allowed}, got '{v}'.")
        return v

    @field_validator("nworkers")
    @classmethod
    def _positive_nworkers(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"nworkers must be >= 1, got {v}.")
        return v

    @field_validator("walltime")
    @classmethod
    def _valid_walltime(cls, v: str) -> str:
        import re
        if not re.match(r"^(\d+-)?\d{1,3}:\d{2}:\d{2}$", v):
            raise ValueError(
                f"walltime must be HH:MM:SS or D-HH:MM:SS format, got '{v}'. "
                "Example: '04:00:00' or '1-12:00:00'."
            )
        return v


class GAConfig(BaseModel):
    """
    Genetic algorithm hyperparameters.

    The run stops when BOTH of the following are true:
      - current generation >= min_generations
      - current generation >= max_generations  OR
        best fitness has not improved for max_stall_generations generations

    isomer_weight is the selection weight assigned to near-duplicate structures
    (isomers).  Unique converged structures have weight 1.0.
    """

    population_size: int
    max_generations: int
    min_generations: int
    max_stall_generations: int          # stop after this many generations with no improvement
    isomer_weight: float = 0.01         # must be in (0, 1]

    @model_validator(mode="after")
    def _generation_bounds(self) -> "GAConfig":
        if self.min_generations > self.max_generations:
            raise ValueError(
                f"min_generations ({self.min_generations}) must be <= "
                f"max_generations ({self.max_generations})."
            )
        return self

    @field_validator("isomer_weight")
    @classmethod
    def _valid_isomer_weight(cls, v: float) -> float:
        if not (0 < v <= 1.0):
            raise ValueError(f"isomer_weight must be in (0, 1], got {v}.")
        return v

    @field_validator("population_size", "max_generations", "min_generations", "max_stall_generations")
    @classmethod
    def _positive_int(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"GA integer parameters must be >= 1, got {v}.")
        return v


class ConditionsConfig(BaseModel):
    """
    Thermodynamic conditions used for grand canonical fitness evaluation.

    These conditions fix the selection pressure during a GA run.  Post-hoc
    re-ranking at different conditions is supported via `gocia inspect`.

    All fields accept int or float (pydantic coerces automatically).
    """

    temperature: float = 298.15        # K
    pressure: float = 1.0              # atm
    potential: float = 0.0             # V vs RHE
    pH: float = 0.0

    @field_validator("temperature")
    @classmethod
    def _positive_temperature(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"temperature must be > 0 K, got {v}.")
        return v

    @field_validator("pressure")
    @classmethod
    def _positive_pressure(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"pressure must be > 0 atm, got {v}.")
        return v


# ---------------------------------------------------------------------------
# Root config model
# ---------------------------------------------------------------------------


class GociaConfig(BaseModel):
    """
    Root configuration object loaded from gocia.yaml.

    Example
    -------
    .. code-block:: yaml

        slab:
          geometry: slab.vasp
          energy: -42.31
          sampling_zmin: 8.0
          sampling_zmax: 12.0

        adsorbates:
          - symbol: O
            chemical_potential: -4.92

        calculator_stages:
          - name: mace_preopt
            type: mace
            fmax: 0.10

        scheduler:
          type: slurm
          nworkers: 20
          walltime: "04:00:00"

        ga:
          population_size: 30
          max_generations: 50
          min_generations: 5
          max_stall_generations: 10

        conditions:
          temperature: 298.15
          potential: 0.0
          pH: 7
    """

    slab: SlabConfig
    adsorbates: list[AdsorbateConfig]
    calculator_stages: list[CalculatorStageConfig]
    scheduler: SchedulerConfig
    ga: GAConfig
    conditions: ConditionsConfig = ConditionsConfig()

    @model_validator(mode="after")
    def _at_least_one_adsorbate(self) -> "GociaConfig":
        if len(self.adsorbates) == 0:
            raise ValueError("At least one adsorbate must be defined.")
        return self

    @model_validator(mode="after")
    def _at_least_one_stage(self) -> "GociaConfig":
        if len(self.calculator_stages) == 0:
            raise ValueError("At least one calculator_stage must be defined.")
        return self

    @model_validator(mode="after")
    def _unique_stage_names(self) -> "GociaConfig":
        names = [s.name for s in self.calculator_stages]
        if len(names) != len(set(names)):
            raise ValueError(
                f"calculator_stage names must be unique, got: {names}"
            )
        return self

    @model_validator(mode="after")
    def _unique_adsorbate_symbols(self) -> "GociaConfig":
        symbols = [a.symbol for a in self.adsorbates]
        if len(symbols) != len(set(symbols)):
            raise ValueError(
                f"Adsorbate symbols must be unique, got: {symbols}"
            )
        return self


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> GociaConfig:
    """
    Load and validate a gocia.yaml file.

    Parameters
    ----------
    path:
        Path to the YAML configuration file.

    Returns
    -------
    GociaConfig
        Fully validated configuration object.

    Raises
    ------
    FileNotFoundError
        If the YAML file does not exist.
    pydantic.ValidationError
        If the YAML content fails validation.  The error message lists every
        field that failed and why, which is useful for debugging configs.
    yaml.YAMLError
        If the file is not valid YAML.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    if path.stat().st_size == 0:
        raise ValueError(
            f"Configuration file is empty: {path}\n"
            "Generate a template with: gocia init > gocia.yaml"
        )

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if raw is None:
        raise ValueError(
            f"{path} contains only comments or whitespace — no YAML keys found.\n"
            "Make sure the file has content like:\n"
            "  slab:\n"
            "    geometry: slab.vasp\n"
            "    energy: -125.0\n"
            "    ...\n"
            "Generate a template with: gocia init > gocia.yaml"
        )
    if not isinstance(raw, dict):
        raise ValueError(
            f"Expected a YAML mapping at the top level, got {type(raw).__name__}.  "
            "Make sure gocia.yaml starts with a key like 'slab:' at column 0."
        )

    return GociaConfig.model_validate(raw)


def generate_example_config(path: str | Path = "gocia.yaml.example") -> None:
    """
    Write a fully commented example gocia.yaml to disk.

    Called by `gocia init` when no config file is found.
    """
    example = """\
# gocia.yaml — GOCIA configuration file
# All paths are relative to this file's location unless absolute.

# ---------------------------------------------------------------------------
# Slab definition
# ---------------------------------------------------------------------------
slab:
  geometry: slab.vasp          # VASP POSCAR/CONTCAR with selective dynamics
  energy: -42.31               # DFT energy of clean slab (eV)
  sampling_zmin: 8.0           # Min z-height for adsorbate placement (Å)
  sampling_zmax: 12.0          # Max z-height for adsorbate placement (Å)
  chemical_potentials:         # Per-element μ for surface stoich changes (future)
    Pt: -6.05

# ---------------------------------------------------------------------------
# Adsorbate species
# ---------------------------------------------------------------------------
adsorbates:
  - symbol: O
    chemical_potential: -4.92  # eV (referenced to standard state)
    # No geometry needed for single atoms
    n_orientations: 1

  - symbol: OH
    chemical_potential: -3.75
    geometry: OH.vasp          # Molecule geometry file (or use 'coordinates')
    n_orientations: 6          # Number of random orientations to trial

# ---------------------------------------------------------------------------
# Staged optimization pipeline
# ---------------------------------------------------------------------------
# Stages run in order. Each reads the output geometry of the previous stage.
calculator_stages:
  - name: mace_preopt
    type: mace
    fmax: 0.10                 # Force convergence threshold (eV/Å)
    max_steps: 300

  - name: vasp_coarse
    type: vasp
    fmax: 0.05
    max_steps: 100
    incar:                     # Merged on top of default INCAR in gocia
      ENCUT: 400
      EDIFF: 1.0e-4
      NSW: 100
      IBRION: 2

  - name: vasp_fine
    type: vasp
    fmax: 0.02
    max_steps: 200
    incar:
      ENCUT: 520
      EDIFF: 1.0e-6
      NSW: 200
      IBRION: 2

# ---------------------------------------------------------------------------
# HPC scheduler
# ---------------------------------------------------------------------------
scheduler:
  type: slurm                  # slurm | pbs | local
  nworkers: 20                 # Max concurrent jobs
  walltime: "04:00:00"
  partition: gpu
  extra_headers:               # Inserted verbatim into job script header
    - "#SBATCH --mem=32G"
    - "#SBATCH --gres=gpu:1"

# ---------------------------------------------------------------------------
# Genetic algorithm
# ---------------------------------------------------------------------------
ga:
  population_size: 30
  max_generations: 50
  min_generations: 5           # Run at least this many generations
  max_stall_generations: 10    # Stop if best fitness unchanged for N generations
  isomer_weight: 0.01          # Selection weight for near-duplicate structures

# ---------------------------------------------------------------------------
# Thermodynamic conditions (CHE framework)
# ---------------------------------------------------------------------------
# These fix the selection pressure during the run.
# Use `gocia inspect --potential X --pH Y` to re-rank post-hoc.
conditions:
  temperature: 298.15          # K
  pressure: 1.0                # atm
  potential: 0.0               # V vs RHE
  pH: 0                        # int or float both accepted
"""
    path = Path(path)
    path.write_text(example)
    print(f"Example configuration written to {path}")
