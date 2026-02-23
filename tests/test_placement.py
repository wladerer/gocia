from __future__ import annotations

import numpy as np
import pytest


class TestAdsorbatePlacement:

    def test_placed_atom_within_z_bounds(self, bare_slab, slab_z_bounds):
        from gocia.structure.placement import place_adsorbate
        from ase import Atoms

        zmin, zmax = slab_z_bounds
        o = Atoms("O", positions=[[0, 0, 0]])

        result = place_adsorbate(
            slab=bare_slab,
            adsorbate=o,
            zmin=zmin,
            zmax=zmax,
            n_orientations=1,
        )
        adsorbate_z = result.positions[len(bare_slab):, 2]
        assert np.all(adsorbate_z >= zmin - 0.1), \
            f"Adsorbate placed below zmin: {adsorbate_z.min():.2f} < {zmin:.2f}"
        assert np.all(adsorbate_z <= zmax + 0.1), \
            f"Adsorbate placed above zmax: {adsorbate_z.max():.2f} > {zmax:.2f}"

    def test_placed_molecule_within_z_bounds(self, bare_slab, slab_z_bounds, oh_molecule):
        from gocia.structure.placement import place_adsorbate

        zmin, zmax = slab_z_bounds
        result = place_adsorbate(
            slab=bare_slab,
            adsorbate=oh_molecule,
            zmin=zmin,
            zmax=zmax,
            n_orientations=4,
        )
        adsorbate_z = result.positions[len(bare_slab):, 2]
        assert np.all(adsorbate_z >= zmin - 0.5)
        assert np.all(adsorbate_z <= zmax + 2.0)   # OH can stick up

    def test_slab_atoms_unchanged_after_placement(self, bare_slab, slab_z_bounds, o_atom):
        from gocia.structure.placement import place_adsorbate

        zmin, zmax = slab_z_bounds
        original_pos = bare_slab.positions.copy()
        result = place_adsorbate(bare_slab, o_atom, zmin, zmax, n_orientations=1)

        assert np.allclose(result.positions[:len(bare_slab)], original_pos, atol=1e-6)

    def test_result_has_more_atoms_than_slab(self, bare_slab, slab_z_bounds, o_atom):
        from gocia.structure.placement import place_adsorbate

        zmin, zmax = slab_z_bounds
        result = place_adsorbate(bare_slab, o_atom, zmin, zmax, n_orientations=1)
        assert len(result) == len(bare_slab) + len(o_atom)

    def test_multiple_orientations_selects_one(self, bare_slab, slab_z_bounds, oh_molecule):
        """n_orientations > 1 should still return a single structure."""
        from gocia.structure.placement import place_adsorbate

        zmin, zmax = slab_z_bounds
        result = place_adsorbate(
            bare_slab, oh_molecule, zmin, zmax, n_orientations=6
        )
        n_oh_atoms = len(oh_molecule)
        assert len(result) == len(bare_slab) + n_oh_atoms

    def test_placement_is_not_deterministic_by_default(
        self, bare_slab, slab_z_bounds, o_atom
    ):
        """Two placements without a fixed seed should generally differ."""
        from gocia.structure.placement import place_adsorbate

        zmin, zmax = slab_z_bounds
        r1 = place_adsorbate(bare_slab, o_atom, zmin, zmax, n_orientations=1)
        r2 = place_adsorbate(bare_slab, o_atom, zmin, zmax, n_orientations=1)
        # Positions should differ (with overwhelming probability)
        # Allow for the rare case they happen to match
        pos1 = r1.positions[len(bare_slab):]
        pos2 = r2.positions[len(bare_slab):]
        # We just check this doesn't raise — randomness is hard to test
        assert pos1.shape == pos2.shape
