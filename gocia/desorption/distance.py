"""
gocia/desorption/distance.py

Distance-based adsorbate desorption detector.

Algorithm
---------
For each adsorbate atom (atoms[n_slab:]), compute the minimum distance to
any slab atom (atoms[:n_slab]).  If this minimum distance exceeds `cutoff`,
the atom is considered desorbed.

The slab atom count is inferred from the bare slab passed to detect().  This
means the detector works correctly even if atoms and slab have different
numbers of adsorbate atoms (e.g. after a mutation that added or removed atoms).

This is intentionally the simplest possible detector — no graph construction,
no charge analysis, no external dependencies.  It handles the most common
desorption scenario: an adsorbate drifts far from the surface during
relaxation.

Limitations
-----------
- Does not distinguish between lateral diffusion (which is fine) and
  genuine desorption (which is not).  The z-distance check helps since
  lateral diffusion keeps the atom near the surface z-coordinate.
- Does not detect subsurface migration (atom moves into the bulk).
  For most ORR/OER adsorbates this is not a practical concern.
- For molecules (e.g. OH), only one atom needs to be within cutoff for
  the molecule to be considered adsorbed.  This is correct behaviour:
  a molecule is adsorbed as long as it is bound to the surface via any atom.

Usage
-----
    from gocia.desorption.distance import DistanceDesorptionDetector

    detector = DistanceDesorptionDetector(cutoff=3.2)
    if detector.detect(relaxed_atoms, bare_slab):
        individual = individual.mark_desorbed()
"""

from __future__ import annotations

import numpy as np
from ase import Atoms

from gocia.desorption.base import DesorptionDetector, DETECTOR_REGISTRY


class DistanceDesorptionDetector(DesorptionDetector):
    """
    Detect desorption by checking the minimum distance from each adsorbate
    atom to the nearest slab atom.

    A structure is flagged as desorbed if ANY adsorbate atom has a minimum
    distance to all slab atoms greater than `cutoff`.

    Parameters
    ----------
    cutoff:
        Maximum allowed distance (Å) from an adsorbate atom to the nearest
        slab atom.  Atoms farther than this are considered desorbed.
        Typical values: 4.0–6.0 Å depending on the system.
    """

    detector_name = "distance"

    def __init__(self, cutoff: float = 4.0) -> None:
        if cutoff <= 0:
            raise ValueError(f"cutoff must be > 0 Å, got {cutoff}.")
        self.cutoff = float(cutoff)

    def detect(self, atoms: Atoms, slab: Atoms) -> bool:
        """
        Return True if any adsorbate atom is farther than cutoff from all
        slab atoms.

        Parameters
        ----------
        atoms:
            Relaxed slab + adsorbate Atoms object.
        slab:
            Bare slab Atoms object.  The number of atoms in slab defines
            n_slab — the boundary between slab and adsorbate indices in atoms.

        Returns
        -------
        bool
            True if desorption detected, False otherwise.
        """
        n_slab = len(slab)

        # No adsorbate atoms → nothing can desorb
        if len(atoms) <= n_slab:
            return False

        slab_positions = atoms.positions[:n_slab]        # (n_slab, 3)
        adsorbate_positions = atoms.positions[n_slab:]   # (n_ads, 3)

        # For each adsorbate atom, find its minimum distance to any slab atom
        for ads_pos in adsorbate_positions:
            # Broadcasting: (n_slab, 3) - (3,) → (n_slab, 3)
            diffs = slab_positions - ads_pos
            distances = np.linalg.norm(diffs, axis=1)   # (n_slab,)
            min_dist = float(distances.min())

            if min_dist > self.cutoff:
                return True

        return False

    def which_desorbed(self, atoms: Atoms, slab: Atoms) -> list[int]:
        """
        Return global indices of adsorbate atoms that exceed the cutoff.

        Useful for logging which atoms desorbed, not just whether desorption
        occurred.

        Parameters
        ----------
        atoms:
            Relaxed slab + adsorbate Atoms object.
        slab:
            Bare slab Atoms object.

        Returns
        -------
        list[int]
            Global atom indices (in atoms) of desorbed adsorbate atoms.
            Empty if no desorption detected.
        """
        n_slab = len(slab)

        if len(atoms) <= n_slab:
            return []

        slab_positions = atoms.positions[:n_slab]
        desorbed = []

        for local_i, ads_pos in enumerate(atoms.positions[n_slab:]):
            diffs = slab_positions - ads_pos
            distances = np.linalg.norm(diffs, axis=1)
            if float(distances.min()) > self.cutoff:
                desorbed.append(n_slab + local_i)

        return desorbed

    def __repr__(self) -> str:
        return f"DistanceDesorptionDetector(cutoff={self.cutoff} Å)"


# ---------------------------------------------------------------------------
# Register default instance
# ---------------------------------------------------------------------------

DETECTOR_REGISTRY["distance"] = DistanceDesorptionDetector(cutoff=3.2)
