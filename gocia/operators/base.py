"""
gocia/operators/base.py

Abstract base class for all genetic operators.

Every operator takes one or two parent Atoms objects and returns one or two
child Atoms objects.  The operator never modifies inputs in place.

The base class is intentionally thin — it exists to document the interface
contract and make it easy to register new operators without touching the GA
loop.

Operator registry
-----------------
Operators are registered by name in OPERATOR_REGISTRY so the runner can
instantiate them from the YAML config string:

    from gocia.operators.base import OPERATOR_REGISTRY
    op = OPERATOR_REGISTRY["splice"]

Adding a new operator
---------------------
1. Create a new file in gocia/operators/
2. Subclass GeneticOperator and implement apply()
3. Add an instance to OPERATOR_REGISTRY at the bottom of this file
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from ase import Atoms


class GeneticOperator(ABC):
    """
    Abstract base class for GOCIA genetic operators.

    Subclasses must implement apply() and declare n_parents.
    """

    #: Number of parent structures this operator consumes (1 or 2)
    n_parents: int = 2

    #: OPERATOR constant string this maps to (must match OPERATOR namespace)
    operator_name: str = ""

    @abstractmethod
    def apply(
        self,
        parents: list[Atoms],
        n_slab: int,
        rng: np.random.Generator | None = None,
        **kwargs,
    ) -> list[Atoms]:
        """
        Apply the operator to the given parent structures.

        Parameters
        ----------
        parents:
            List of parent Atoms objects.  Length must equal self.n_parents.
            Inputs are never modified.
        n_slab:
            Number of atoms belonging to the bare slab substrate.  Adsorbate
            atoms are indexed from n_slab onward.
        rng:
            NumPy random generator for reproducible runs.  A new one is
            created internally if None.
        **kwargs:
            Operator-specific keyword arguments (e.g. symbol for mutations,
            position for add).

        Returns
        -------
        list[Atoms]
            One or two child Atoms objects.  Slab atoms are identical to the
            input slab.  Adsorbate atoms may differ.
        """
        ...

    def __call__(
        self,
        parents: list[Atoms],
        n_slab: int,
        rng: np.random.Generator | None = None,
        **kwargs,
    ) -> list[Atoms]:
        """Convenience: allows calling the operator directly."""
        if len(parents) != self.n_parents:
            raise ValueError(
                f"{self.__class__.__name__} expects {self.n_parents} parent(s), "
                f"got {len(parents)}."
            )
        return self.apply(parents, n_slab, rng=rng, **kwargs)


# ---------------------------------------------------------------------------
# Shared geometry utilities used by multiple operators
# ---------------------------------------------------------------------------


def remove_clashes(
    slab: Atoms,
    adsorbate_atoms: Atoms,
    n_slab: int,
    min_distance: float = 1.2,
) -> Atoms:
    """
    Remove adsorbate atoms that clash with any other atom.

    Iterates over adsorbate atoms and removes those that are closer than
    min_distance to any other atom (slab or other adsorbate).  Removal
    is done greedily in order of worst clash first to preserve as many
    atoms as possible.

    This is the shared clash-removal routine used by splice, merge, and
    mutate_add.  Stoichiometry correction after this call is the
    responsibility of the calling operator.

    Parameters
    ----------
    slab:
        The bare slab (first n_slab atoms), used as an immutable reference.
    adsorbate_atoms:
        The adsorbate atoms to check and filter.
    n_slab:
        Number of slab atoms.
    min_distance:
        Minimum allowed interatomic distance (Å).

    Returns
    -------
    Atoms
        New adsorbate Atoms object with clashing atoms removed.
        May be empty if all atoms clashed.
    """
    if len(adsorbate_atoms) == 0:
        return adsorbate_atoms.copy()

    # Build combined structure: slab + adsorbates (for distance checks)
    combined = slab.copy()
    combined += adsorbate_atoms

    # Compute minimum distance from each adsorbate atom to all other atoms
    pos = combined.positions
    adsorbate_global_indices = list(range(n_slab, len(combined)))

    # Score each adsorbate atom by its minimum distance to any other atom
    scores = {}
    for i in adsorbate_global_indices:
        min_d = float("inf")
        for j in range(len(combined)):
            if i == j:
                continue
            d = float(np.linalg.norm(pos[i] - pos[j]))
            if d < min_d:
                min_d = d
        scores[i] = min_d

    # Greedily remove worst clashers until no clashes remain
    keep = set(adsorbate_global_indices)
    changed = True
    while changed:
        changed = False
        for i in sorted(keep, key=lambda x: scores[x]):
            if scores[i] < min_distance:
                keep.discard(i)
                # Recompute scores for neighbours of the removed atom
                for j in keep:
                    d = float(np.linalg.norm(pos[i] - pos[j]))
                    if d < scores[j]:
                        # The removed atom was the closest neighbour for j;
                        # recompute j's min distance excluding removed atoms
                        min_d = float("inf")
                        for k in list(keep) + list(range(n_slab)):
                            if k == j:
                                continue
                            dk = float(np.linalg.norm(pos[j] - pos[k]))
                            if dk < min_d:
                                min_d = dk
                        scores[j] = min_d
                changed = True
                break  # restart the loop after each removal

    # Build filtered adsorbate Atoms
    kept_local_indices = [i - n_slab for i in sorted(keep)]
    if not kept_local_indices:
        result = Atoms(cell=adsorbate_atoms.cell, pbc=adsorbate_atoms.pbc)
        return result

    return adsorbate_atoms[kept_local_indices]


def cut_plane_partition(
    positions: np.ndarray,
    cut_value: float,
    axis: int = 0,
) -> tuple[list[int], list[int]]:
    """
    Partition atom indices into two groups by a plane perpendicular to `axis`.

    Atoms with coordinate < cut_value go to group A; the rest to group B.
    Atoms exactly on the plane go to group B.

    Parameters
    ----------
    positions:
        (N, 3) array of positions.
    cut_value:
        Plane position along the given axis.
    axis:
        0 = x, 1 = y, 2 = z.

    Returns
    -------
    (group_a_indices, group_b_indices)
    """
    group_a = [i for i, p in enumerate(positions) if p[axis] < cut_value]
    group_b = [i for i, p in enumerate(positions) if p[axis] >= cut_value]
    return group_a, group_b


# ---------------------------------------------------------------------------
# Operator registry (populated by importing each operator module)
# ---------------------------------------------------------------------------

OPERATOR_REGISTRY: dict[str, GeneticOperator] = {}
