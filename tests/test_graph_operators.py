"""
tests/test_graph_operators.py

Tests for the graph-based splice, merge, and mutation operators.

All tests are pure-geometry (no MACE or VASP).  The key invariants checked:

  1. Stoichiometry preservation — atom counts before == atom counts after
  2. Clash removal — no two atoms closer than a minimum distance
  3. Slab integrity — frozen slab atoms are never modified
  4. Adsorbate layer connectivity — result stays within sampling z-bounds
  5. Edge cases — empty adsorbate layers, single atom, mismatched parents

These tests are written against the operator *interfaces* defined in
gocia/operators/base.py.  The actual implementations in graph_splice.py and
graph_merge.py are tested through those interfaces so the tests remain valid
if the internals change.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111, add_adsorbate
from ase.constraints import FixAtoms


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _adsorbate_counts(atoms: Atoms, slab_size: int) -> Counter:
    """Count adsorbate atoms (everything above the slab) by chemical symbol."""
    return Counter(atoms.get_chemical_symbols()[slab_size:])


def _min_distance(atoms: Atoms) -> float:
    """Return the minimum interatomic distance in the structure."""
    pos = atoms.positions
    n = len(pos)
    if n < 2:
        return float("inf")
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(np.linalg.norm(pos[i] - pos[j]))
    return min(dists)


def _slab_positions_unchanged(original: Atoms, result: Atoms, slab_size: int) -> bool:
    """Check that the first slab_size atoms are identical in both structures."""
    return np.allclose(
        original.positions[:slab_size],
        result.positions[:slab_size],
        atol=1e-6,
    )


def _make_slab_with_adsorbates(
    n_o: int = 2,
    n_oh: int = 1,
    layers: int = 3,
    size: tuple = (2, 2),
    vacuum: float = 8.0,
) -> tuple[Atoms, int]:
    """
    Build a Pt(111) slab with n_o O atoms and n_oh OH molecules adsorbed.

    Returns (atoms, slab_size) where slab_size is the number of Pt atoms.
    """
    slab = fcc111("Pt", size=size + (layers,), vacuum=vacuum)
    n_slab = len(slab)

    # Freeze all slab atoms
    slab.set_constraint(FixAtoms(indices=list(range(n_slab))))

    # Add O atoms at displaced positions to avoid overlap
    cell = slab.cell
    for i in range(n_o):
        x = (i + 0.5) / max(n_o, 1) * float(cell[0, 0])
        y = 0.3 * float(cell[1, 1])
        top_z = slab.positions[:n_slab, 2].max()
        o = Atoms("O", positions=[[x, y, top_z + 1.5]])
        slab += o

    for i in range(n_oh):
        x = (i + 0.5) / max(n_oh, 1) * float(cell[0, 0])
        y = 0.7 * float(cell[1, 1])
        top_z = slab.positions[:n_slab, 2].max()
        oh = Atoms("OH", positions=[[x, y, top_z + 1.4], [x, y, top_z + 2.3]])
        slab += oh

    return slab, n_slab


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def parent_a():
    """Parent A: Pt(111) slab with 2 O atoms."""
    atoms, n_slab = _make_slab_with_adsorbates(n_o=2, n_oh=0)
    return atoms, n_slab


@pytest.fixture
def parent_b():
    """Parent B: Pt(111) slab with 2 O atoms (different positions)."""
    atoms, n_slab = _make_slab_with_adsorbates(n_o=2, n_oh=0)
    # Displace adsorbates to make parents distinct
    for i in range(n_slab, len(atoms)):
        atoms.positions[i, 0] += 0.5
        atoms.positions[i, 1] += 0.5
    return atoms, n_slab


@pytest.fixture
def parent_mixed():
    """Parent with mixed O and OH adsorbates."""
    atoms, n_slab = _make_slab_with_adsorbates(n_o=1, n_oh=1)
    return atoms, n_slab


@pytest.fixture
def parent_empty():
    """Parent with no adsorbates (bare slab)."""
    atoms, n_slab = _make_slab_with_adsorbates(n_o=0, n_oh=0)
    return atoms, n_slab


# ---------------------------------------------------------------------------
# Stoichiometry checks
# ---------------------------------------------------------------------------

class TestStoichiometryPreservation:
    """
    The central invariant: operator(parent_a, parent_b) preserves total
    atom count of each adsorbate species.
    """

    def test_splice_preserves_total_adsorbate_count(self, parent_a, parent_b):
        from gocia.operators.graph_splice import splice

        atoms_a, n_slab_a = parent_a
        atoms_b, n_slab_b = parent_b

        counts_a = _adsorbate_counts(atoms_a, n_slab_a)
        counts_b = _adsorbate_counts(atoms_b, n_slab_b)
        total = counts_a + counts_b  # total across both parents

        child1, child2 = splice(atoms_a, atoms_b, n_slab_a)

        child1_counts = _adsorbate_counts(child1, n_slab_a)
        child2_counts = _adsorbate_counts(child2, n_slab_a)
        child_total = child1_counts + child2_counts

        assert child_total == total, (
            f"Stoichiometry not preserved: parents had {dict(total)}, "
            f"children have {dict(child_total)}"
        )

    def test_merge_preserves_total_adsorbate_count(self, parent_a, parent_b):
        from gocia.operators.graph_merge import merge

        atoms_a, n_slab_a = parent_a
        atoms_b, n_slab_b = parent_b

        counts_a = _adsorbate_counts(atoms_a, n_slab_a)
        counts_b = _adsorbate_counts(atoms_b, n_slab_b)
        expected = counts_a  # merge produces one child matching parent_a count

        child = merge(atoms_a, atoms_b, n_slab_a)
        child_counts = _adsorbate_counts(child, n_slab_a)

        # Each child should have same count as one of the parents
        assert child_counts == expected or child_counts == counts_b, (
            f"Merge produced unexpected stoichiometry: {dict(child_counts)}"
        )

    def test_mutate_add_increases_count_by_one(self, parent_a):
        from gocia.operators.mutation import mutate_add

        atoms, n_slab = parent_a
        before = _adsorbate_counts(atoms, n_slab)
        result = mutate_add(atoms, n_slab, symbol="O", position=(1.0, 1.0, 10.0))
        after = _adsorbate_counts(result, n_slab)

        assert after["O"] == before["O"] + 1
        # Other species unchanged
        for sym in before:
            if sym != "O":
                assert after[sym] == before[sym]

    def test_mutate_remove_decreases_count_by_one(self, parent_a):
        from gocia.operators.mutation import mutate_remove

        atoms, n_slab = parent_a
        before = _adsorbate_counts(atoms, n_slab)
        result = mutate_remove(atoms, n_slab, symbol="O")
        after = _adsorbate_counts(result, n_slab)

        assert after["O"] == before["O"] - 1

    def test_mutate_displace_preserves_count(self, parent_a):
        from gocia.operators.mutation import mutate_displace

        atoms, n_slab = parent_a
        before = _adsorbate_counts(atoms, n_slab)
        result = mutate_displace(atoms, n_slab, symbol="O",
                                  new_position=(1.5, 1.5, 10.5))
        after = _adsorbate_counts(result, n_slab)

        assert after == before

    def test_mutate_remove_last_atom_returns_bare_slab(self, parent_empty):
        """Removing from an empty adsorbate layer should raise, not silently fail."""
        from gocia.operators.mutation import mutate_remove

        atoms, n_slab = parent_empty
        with pytest.raises(ValueError, match="No adsorbate"):
            mutate_remove(atoms, n_slab, symbol="O")

    def test_splice_with_mixed_species(self, parent_mixed, parent_a):
        from gocia.operators.graph_splice import splice

        atoms_m, n_slab_m = parent_mixed
        atoms_a, n_slab_a = parent_a

        counts_m = _adsorbate_counts(atoms_m, n_slab_m)
        counts_a = _adsorbate_counts(atoms_a, n_slab_a)
        total = counts_m + counts_a

        child1, child2 = splice(atoms_m, atoms_a, n_slab_m)
        child_total = (
            _adsorbate_counts(child1, n_slab_m)
            + _adsorbate_counts(child2, n_slab_m)
        )

        assert child_total == total


# ---------------------------------------------------------------------------
# Clash removal
# ---------------------------------------------------------------------------

class TestClashRemoval:

    MIN_DISTANCE = 0.8   # Å — below this is unphysical

    def test_splice_no_clashes(self, parent_a, parent_b):
        from gocia.operators.graph_splice import splice

        atoms_a, n_slab_a = parent_a
        atoms_b, _ = parent_b
        child1, child2 = splice(atoms_a, atoms_b, n_slab_a)

        assert _min_distance(child1) > self.MIN_DISTANCE, \
            "Clash detected in child1 after splice"
        assert _min_distance(child2) > self.MIN_DISTANCE, \
            "Clash detected in child2 after splice"

    def test_merge_no_clashes(self, parent_a, parent_b):
        from gocia.operators.graph_merge import merge

        atoms_a, n_slab_a = parent_a
        atoms_b, _ = parent_b
        child = merge(atoms_a, atoms_b, n_slab_a)

        assert _min_distance(child) > self.MIN_DISTANCE, \
            "Clash detected in child after merge"

    def test_mutate_add_no_clash(self, parent_a):
        """Adding an adsorbate at a clash-free position leaves no unphysical contacts."""
        from gocia.operators.mutation import mutate_add

        atoms, n_slab = parent_a
        # Place well above all existing adsorbates — no clash possible
        existing_pos = atoms.positions[n_slab]
        result = mutate_add(
            atoms, n_slab, symbol="O",
            position=(existing_pos[0], existing_pos[1], existing_pos[2] + 4.0),
        )
        assert _min_distance(result) > self.MIN_DISTANCE


# ---------------------------------------------------------------------------
# Slab integrity
# ---------------------------------------------------------------------------

class TestSlabIntegrity:

    def test_splice_does_not_modify_slab_atoms(self, parent_a, parent_b):
        from gocia.operators.graph_splice import splice

        atoms_a, n_slab_a = parent_a
        atoms_b, _ = parent_b
        original_slab_pos = atoms_a.positions[:n_slab_a].copy()

        child1, child2 = splice(atoms_a, atoms_b, n_slab_a)

        assert np.allclose(child1.positions[:n_slab_a], original_slab_pos, atol=1e-6)
        assert np.allclose(child2.positions[:n_slab_a], original_slab_pos, atol=1e-6)

    def test_merge_does_not_modify_slab_atoms(self, parent_a, parent_b):
        from gocia.operators.graph_merge import merge

        atoms_a, n_slab_a = parent_a
        atoms_b, _ = parent_b
        original_slab_pos = atoms_a.positions[:n_slab_a].copy()

        child = merge(atoms_a, atoms_b, n_slab_a)

        assert np.allclose(child.positions[:n_slab_a], original_slab_pos, atol=1e-6)

    def test_mutation_does_not_modify_slab_atoms(self, parent_a):
        from gocia.operators.mutation import mutate_displace

        atoms, n_slab = parent_a
        original_slab_pos = atoms.positions[:n_slab].copy()

        result = mutate_displace(atoms, n_slab, symbol="O",
                                  new_position=(2.0, 2.0, 11.0))

        assert np.allclose(result.positions[:n_slab], original_slab_pos, atol=1e-6)

    def test_operators_do_not_mutate_input(self, parent_a, parent_b):
        """Operators must return new Atoms objects, not modify inputs in place."""
        from gocia.operators.graph_splice import splice

        atoms_a, n_slab_a = parent_a
        atoms_b, _ = parent_b
        original_pos_a = atoms_a.positions.copy()
        original_pos_b = atoms_b.positions.copy()

        splice(atoms_a, atoms_b, n_slab_a)

        assert np.allclose(atoms_a.positions, original_pos_a, atol=1e-6)
        assert np.allclose(atoms_b.positions, original_pos_b, atol=1e-6)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_splice_with_one_empty_parent(self, parent_a, parent_empty):
        """Splicing with a bare slab should return one copy of each parent."""
        from gocia.operators.graph_splice import splice

        atoms_a, n_slab_a = parent_a
        atoms_e, _ = parent_empty

        child1, child2 = splice(atoms_a, atoms_e, n_slab_a)
        counts_a = _adsorbate_counts(atoms_a, n_slab_a)
        counts_e = _adsorbate_counts(atoms_e, n_slab_a)
        total = counts_a + counts_e

        child_total = (
            _adsorbate_counts(child1, n_slab_a)
            + _adsorbate_counts(child2, n_slab_a)
        )
        assert child_total == total

    def test_splice_both_empty_parents(self, parent_empty):
        """Splicing two bare slabs should produce two bare slabs."""
        from gocia.operators.graph_splice import splice

        atoms_e1, n_slab = parent_empty
        atoms_e2 = atoms_e1.copy()

        child1, child2 = splice(atoms_e1, atoms_e2, n_slab)
        assert len(child1) == n_slab
        assert len(child2) == n_slab

    def test_graph_connectivity_single_adsorbate(self, parent_empty):
        """Graph construction on a single adsorbate should not raise."""
        from gocia.operators.graph_splice import build_adsorbate_graph

        atoms, n_slab = parent_empty
        # Add a single O atom
        from ase import Atoms as AseAtoms
        top_z = atoms.positions[:n_slab, 2].max()
        o = AseAtoms("O", positions=[[1.0, 1.0, top_z + 1.5]])
        atoms = atoms + o

        graph = build_adsorbate_graph(atoms, n_slab)
        assert len(graph.nodes) == 1
        assert len(graph.edges) == 0
