"""
gocia/population/individual.py

The Individual model represents a single slab+adsorbate structure within the
genetic algorithm population.  It carries all metadata needed to:

  - reconstruct the structure's history (genealogy)
  - track progress through the calculator pipeline (status / sentinel files)
  - compute and store fitness under CHE conditions
  - flag desorption and duplicate / isomer relationships
  - assign selection weights

Status lifecycle
----------------
Every Individual moves through a well-defined sequence of statuses that map
1-to-1 onto sentinel files written in the structure's directory on disk.  This
allows the main loop to reconstruct run state from the filesystem after an HPC
timeout without querying the scheduler.

    pending
        → submitted
        → running_stage_{n}       (n = 1, 2, … per calculator stage)
        → converged_stage_{n}     (triggers submission of stage n+1)
        → converged               (all stages done; ready for fitness eval)
        → desorbed                (desorption detected post-opt; logged as reference)
        → failed                  (calculator error)

    converged
        → duplicate               (exact fingerprint match found)
        → isomer                  (near-match; kept with low weight)

Sentinel filenames
------------------
    PENDING, SUBMITTED, RUNNING_1, CONVERGED_1, RUNNING_2, CONVERGED_2,
    CONVERGED, DESORBED, FAILED, DUPLICATE, ISOMER

These are managed by gocia/database/status.py, not by this module.

Usage
-----
    from gocia.population.individual import Individual, STATUS

    ind = Individual(id="abc123", generation=0)
    ind = ind.with_status(STATUS.CONVERGED)
    ind = ind.with_energy(raw=-42.5, grand_canonical=-40.1)
"""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Status constants
# ---------------------------------------------------------------------------

class STATUS:
    """
    Namespace of valid status strings.

    Using a plain class rather than an Enum keeps it simple for scientists
    reading the code.  String comparison works everywhere (DB queries, YAML,
    print statements) without importing Enum machinery.
    """
    PENDING             = "pending"
    SUBMITTED           = "submitted"
    CONVERGED           = "converged"
    DESORBED            = "desorbed"
    FAILED              = "failed"
    DUPLICATE           = "duplicate"
    ISOMER              = "isomer"

    # Stage-specific statuses are built dynamically: running_stage_1, converged_stage_1, …
    @staticmethod
    def running_stage(n: int) -> str:
        return f"running_stage_{n}"

    @staticmethod
    def converged_stage(n: int) -> str:
        return f"converged_stage_{n}"

    @classmethod
    def all_terminal(cls) -> set[str]:
        """Statuses after which no further calculator work is expected."""
        return {cls.CONVERGED, cls.DESORBED, cls.FAILED, cls.DUPLICATE, cls.ISOMER}

    @classmethod
    def is_terminal(cls, status: str) -> bool:
        return status in cls.all_terminal()

    @classmethod
    def is_stage_running(cls, status: str) -> bool:
        return status.startswith("running_stage_")

    @classmethod
    def is_stage_converged(cls, status: str) -> bool:
        return status.startswith("converged_stage_")

    @classmethod
    def stage_number(cls, status: str) -> int | None:
        """Extract the stage number from a running_stage_N or converged_stage_N status."""
        for prefix in ("running_stage_", "converged_stage_"):
            if status.startswith(prefix):
                try:
                    return int(status[len(prefix):])
                except ValueError:
                    return None
        return None


# ---------------------------------------------------------------------------
# Operator constants
# ---------------------------------------------------------------------------

class OPERATOR:
    """
    Namespace of valid operator strings stored in Individual.operator.
    These record how a structure was produced, forming the genealogy record.
    """
    INIT             = "init"           # random initial placement
    SPLICE           = "splice"         # graph-based splice of two parents
    MERGE            = "merge"          # graph-based merge of two parents
    MUTATE_ADD       = "mutate_add"     # added one adsorbate
    MUTATE_REMOVE    = "mutate_remove"  # removed one adsorbate
    MUTATE_DISPLACE  = "mutate_displace"# displaced one adsorbate

    @classmethod
    def all(cls) -> set[str]:
        return {
            cls.INIT, cls.SPLICE, cls.MERGE,
            cls.MUTATE_ADD, cls.MUTATE_REMOVE, cls.MUTATE_DISPLACE,
        }


# ---------------------------------------------------------------------------
# Individual model
# ---------------------------------------------------------------------------

class Individual(BaseModel):
    """
    A single structure in the GA population.

    Fields
    ------
    id : str
        UUID that uniquely identifies this structure across the entire run.
        Auto-generated if not provided.

    generation : int
        The GA generation in which this structure was created.

    parent_ids : list[str]
        IDs of parent structures used to create this one.
        Empty for initial population (operator == "init").
        One entry for mutations, two for splice/merge.

    operator : str
        The GA operator that created this structure.  See OPERATOR constants.

    status : str
        Current position in the calculator pipeline.  See STATUS constants.

    raw_energy : float | None
        Total DFT (or MACE) energy of the final relaxed structure in eV.
        None until the final calculator stage completes.

    grand_canonical_energy : float | None
        CHE-corrected grand canonical energy in eV, evaluated at the run
        conditions.  None until fitness is computed after convergence.
        Stored separately so raw_energy is always available for re-ranking
        at different conditions.

    weight : float
        Selection weight used by the population selector.
        1.0 for unique converged structures.
        isomer_weight (default 0.01) for isomers.
        0.0 for duplicates, desorbed, and failed structures.

    fingerprint : list[float] | None
        Fast distance-histogram fingerprint of the adsorbate layer.
        Computed pre-submission and stored for duplicate detection.

    geometry_path : str | None
        Absolute path to the final converged geometry file (CONTCAR).
        None until the final stage completes.

    desorption_flag : bool
        True if any adsorbate was detected as desorbed post-optimisation.

    is_isomer : bool
        True if this structure is a near-duplicate of another.

    isomer_of : str | None
        ID of the representative (lowest-energy) structure this is an
        isomer of.  None if not an isomer.

    extra_data : dict
        Catch-all for any additional metadata (e.g. per-stage energies,
        custom flags).  Stored as JSON in the database.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    generation: int
    parent_ids: list[str] = Field(default_factory=list)
    operator: str = OPERATOR.INIT
    status: str = STATUS.PENDING

    # Energetics (populated after relaxation)
    raw_energy: float | None = None
    grand_canonical_energy: float | None = None

    # Selection
    weight: float = 1.0

    # Fingerprint (populated before submission)
    fingerprint: list[float] | None = None

    # Filesystem location
    geometry_path: str | None = None

    # Desorption and isomer bookkeeping
    desorption_flag: bool = False
    is_isomer: bool = False
    isomer_of: str | None = None

    # Forward-compatible overflow bucket
    extra_data: dict[str, Any] = Field(default_factory=dict)

    class Config:
        # Allow fields not in the model (e.g. when loading old DB records with new fields)
        extra = "allow"

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("generation")
    @classmethod
    def _non_negative_generation(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"generation must be >= 0, got {v}.")
        return v

    @field_validator("weight")
    @classmethod
    def _valid_weight(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"weight must be >= 0, got {v}.")
        return v

    @field_validator("operator")
    @classmethod
    def _valid_operator(cls, v: str) -> str:
        if v not in OPERATOR.all():
            raise ValueError(
                f"operator must be one of {sorted(OPERATOR.all())}, got '{v}'."
            )
        return v

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_init(cls, generation: int, **kwargs: Any) -> "Individual":
        """Create an Individual for the initial random population."""
        return cls(generation=generation, operator=OPERATOR.INIT, **kwargs)

    @classmethod
    def from_parents(
        cls,
        generation: int,
        parents: list["Individual"],
        operator: str,
        **kwargs: Any,
    ) -> "Individual":
        """
        Create an offspring Individual with genealogy automatically populated.

        Parameters
        ----------
        generation:
            The generation number of the offspring.
        parents:
            The parent Individual objects (1 for mutation, 2 for splice/merge).
        operator:
            The OPERATOR constant describing how this offspring was made.
        """
        if operator not in OPERATOR.all():
            raise ValueError(f"Unknown operator: '{operator}'.")
        return cls(
            generation=generation,
            parent_ids=[p.id for p in parents],
            operator=operator,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Immutable update helpers
    # ------------------------------------------------------------------
    # These return a new Individual rather than mutating in place.
    # This keeps the model predictable and makes testing easier.

    def with_status(self, status: str) -> "Individual":
        """Return a copy with an updated status."""
        return self.model_copy(update={"status": status})

    def with_energy(
        self,
        raw: float,
        grand_canonical: float | None = None,
    ) -> "Individual":
        """Return a copy with updated energy fields."""
        return self.model_copy(update={
            "raw_energy": raw,
            "grand_canonical_energy": grand_canonical,
        })

    def with_fingerprint(self, fingerprint: list[float]) -> "Individual":
        """Return a copy with an updated fingerprint."""
        return self.model_copy(update={"fingerprint": fingerprint})

    def with_geometry_path(self, path: str) -> "Individual":
        """Return a copy with the geometry path set."""
        return self.model_copy(update={"geometry_path": path})

    def mark_desorbed(self) -> "Individual":
        """Return a copy flagged as desorbed with weight zeroed."""
        return self.model_copy(update={
            "status": STATUS.DESORBED,
            "desorption_flag": True,
            "weight": 0.0,
        })

    def mark_duplicate(self) -> "Individual":
        """Return a copy flagged as a duplicate with weight zeroed."""
        return self.model_copy(update={
            "status": STATUS.DUPLICATE,
            "weight": 0.0,
        })

    def mark_isomer(self, of_id: str, isomer_weight: float) -> "Individual":
        """Return a copy flagged as an isomer with reduced weight."""
        return self.model_copy(update={
            "status": STATUS.ISOMER,
            "is_isomer": True,
            "isomer_of": of_id,
            "weight": isomer_weight,
        })

    def mark_failed(self) -> "Individual":
        """Return a copy flagged as failed with weight zeroed."""
        return self.model_copy(update={
            "status": STATUS.FAILED,
            "weight": 0.0,
        })

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def is_converged(self) -> bool:
        return self.status == STATUS.CONVERGED

    @property
    def is_selectable(self) -> bool:
        """True if this Individual can participate in selection."""
        return self.weight > 0 and self.status in {STATUS.CONVERGED, STATUS.ISOMER}

    @property
    def has_fitness(self) -> bool:
        return self.grand_canonical_energy is not None

    @property
    def structure_dir(self) -> str | None:
        """
        The directory containing this structure's files, inferred from
        geometry_path.  Returns None if geometry_path is not set.
        """
        if self.geometry_path is None:
            return None
        from pathlib import Path
        return str(Path(self.geometry_path).parent)

    def __repr__(self) -> str:
        gce = (
            f"{self.grand_canonical_energy:.4f} eV"
            if self.grand_canonical_energy is not None
            else "N/A"
        )
        return (
            f"Individual(id={self.id[:8]}…, gen={self.generation}, "
            f"op={self.operator}, status={self.status}, G={gce})"
        )
