"""
gocia/operators/mutation.py

Single-structure mutation operators for the GOCIA genetic algorithm.

Three mutations
---------------
mutate_add(atoms, n_slab, symbol, position)
    Add one adsorbate atom at the given position.  Clash removal is applied
    if the position conflicts with an existing atom.

mutate_remove(atoms, n_slab, symbol)
    Remove one randomly selected adsorbate atom of the given species.
    Raises ValueError if no atoms of that species exist.

mutate_displace(atoms, n_slab, symbol, new_position)
    Move one randomly selected adsorbate atom of the given species to
    new_position.  Stoichiometry is unchanged.

All three:
  - Never modify the input Atoms object
  - Never modify slab atoms (indices < n_slab)
  - Apply clash removal after addition/displacement
  - Return a new Atoms object

Public API
----------
    mutate_add(atoms, n_slab, symbol, position, min_distance) → Atoms
    mutate_remove(atoms, n_slab, symbol, rng)                 → Atoms
    mutate_displace(atoms, n_slab, symbol, new_position, min_distance) → Atoms
"""

from __future__ import annotations

import numpy as np
from ase import Atoms

from gocia.operators.base import GeneticOperator, remove_clashes, OPERATOR_REGISTRY
from gocia.population.individual import OPERATOR as OP_NAMES


# ---------------------------------------------------------------------------
# Public mutation functions
# ---------------------------------------------------------------------------

def mutate_add(
    atoms: Atoms,
    n_slab: int,
    symbol: str,
    position: tuple[float, float, float],
) -> Atoms:
    """
    Add one adsorbate atom of the given species at the specified position.

    Always increases the adsorbate count by exactly one.  No clash removal
    is performed: it is the caller's responsibility to supply a position
    that does not clash with existing atoms (e.g. from place_adsorbate).

    Slab atoms are never modified.

    Parameters
    ----------
    atoms:
        Full slab + adsorbate Atoms object.  Not modified.
    n_slab:
        Number of slab atoms.
    symbol:
        Chemical symbol of the atom to add (e.g. "O", "H").
    position:
        (x, y, z) Cartesian coordinates in Å.

    Returns
    -------
    Atoms
        New structure with exactly one additional adsorbate atom.
    """
    result = atoms.copy()
    new_atom = Atoms(symbol, positions=[list(position)])
    result += new_atom
    return result


def mutate_remove(
    atoms: Atoms,
    n_slab: int,
    symbol: str,
    rng: np.random.Generator | None = None,
) -> Atoms:
    """
    Remove one randomly selected adsorbate atom of the given species.

    Parameters
    ----------
    atoms:
        Full slab + adsorbate Atoms object.  Not modified.
    n_slab:
        Number of slab atoms.
    symbol:
        Chemical symbol of the atom to remove.
    rng:
        NumPy random generator.

    Returns
    -------
    Atoms
        New structure with one fewer adsorbate atom of the given species.

    Raises
    ------
    ValueError
        If no adsorbate atoms of the given species exist in the structure.
        Message starts with "No adsorbate" (tested by test suite).
    """
    if rng is None:
        rng = np.random.default_rng()

    result = atoms.copy()
    symbols = result.get_chemical_symbols()

    # Find global indices of adsorbate atoms matching the symbol
    candidate_indices = [
        i for i in range(n_slab, len(result))
        if symbols[i] == symbol
    ]

    if not candidate_indices:
        raise ValueError(
            f"No adsorbate atoms of species '{symbol}' found in the structure. "
            f"Present adsorbate species: "
            f"{list(set(symbols[n_slab:]))}."
        )

    # Remove a randomly chosen one
    remove_idx = int(rng.choice(candidate_indices))
    del result[remove_idx]
    return result


def mutate_displace(
    atoms: Atoms,
    n_slab: int,
    symbol: str,
    new_position: tuple[float, float, float],
    rng: np.random.Generator | None = None,
) -> Atoms:
    """
    Displace one randomly selected adsorbate atom of the given species
    to a new position.

    Stoichiometry is preserved: one atom is moved, not added or removed.
    Clash removal is applied at the new position if needed.

    Parameters
    ----------
    atoms:
        Full slab + adsorbate Atoms object.  Not modified.
    n_slab:
        Number of slab atoms.
    symbol:
        Chemical symbol of the atom to displace.
    new_position:
        (x, y, z) target coordinates in Å.
    min_distance:
        Minimum allowed distance after displacement (Å).
    rng:
        NumPy random generator.

    Returns
    -------
    Atoms
        New structure with one adsorbate atom moved to new_position.

    Raises
    ------
    ValueError
        If no adsorbate atoms of the given species exist.
    """
    if rng is None:
        rng = np.random.default_rng()

    result = atoms.copy()
    symbols = result.get_chemical_symbols()

    candidate_indices = [
        i for i in range(n_slab, len(result))
        if symbols[i] == symbol
    ]

    if not candidate_indices:
        raise ValueError(
            f"No adsorbate atoms of species '{symbol}' found. "
            f"Present adsorbate species: {list(set(symbols[n_slab:]))}."
        )

    move_idx = int(rng.choice(candidate_indices))
    result.positions[move_idx] = list(new_position)
    return result


# ---------------------------------------------------------------------------
# GeneticOperator subclasses (for registry)
# ---------------------------------------------------------------------------

class MutateAddOperator(GeneticOperator):
    n_parents = 1
    operator_name = OP_NAMES.MUTATE_ADD

    def apply(self, parents, n_slab, rng=None, **kwargs):
        symbol = kwargs["symbol"]
        position = kwargs["position"]
        return [mutate_add(parents[0], n_slab, symbol, position)]


class MutateRemoveOperator(GeneticOperator):
    n_parents = 1
    operator_name = OP_NAMES.MUTATE_REMOVE

    def apply(self, parents, n_slab, rng=None, **kwargs):
        symbol = kwargs["symbol"]
        return [mutate_remove(parents[0], n_slab, symbol, rng=rng)]


class MutateDisplaceOperator(GeneticOperator):
    n_parents = 1
    operator_name = OP_NAMES.MUTATE_DISPLACE

    def apply(self, parents, n_slab, rng=None, **kwargs):
        symbol = kwargs["symbol"]
        new_position = kwargs["new_position"]
        return [mutate_displace(parents[0], n_slab, symbol, new_position, rng=rng)]


OPERATOR_REGISTRY[OP_NAMES.MUTATE_ADD]      = MutateAddOperator()
OPERATOR_REGISTRY[OP_NAMES.MUTATE_REMOVE]   = MutateRemoveOperator()
OPERATOR_REGISTRY[OP_NAMES.MUTATE_DISPLACE] = MutateDisplaceOperator()
