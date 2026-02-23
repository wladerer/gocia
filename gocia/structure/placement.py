"""
gocia/structure/placement.py

Adsorbate placement on slab surfaces with orientation sampling.

Workflow for each placement
---------------------------
1. Sample a random (x, y) position within the surface unit cell and a
   z-height within [zmin, zmax].
2. For molecules: trial n_orientations random rotations of the adsorbate.
3. For each orientation, compute a fast energy estimate (MACE single-point
   if available, otherwise a Lennard-Jones-style repulsion heuristic).
4. Keep the orientation with the lowest energy estimate.
5. Return the slab + best-placed adsorbate as a new Atoms object.

The slab is never modified in place.  All returned structures are copies.

Public API
----------
    place_adsorbate(slab, adsorbate, zmin, zmax, n_orientations, ...)
        → Atoms (slab + adsorbate)

    place_multiple(slab, adsorbate, n, zmin, zmax, n_orientations, ...)
        → list[Atoms]   (for building the initial population)
"""

from __future__ import annotations

import warnings

import numpy as np
from ase import Atoms
from ase.build import add_adsorbate

from gocia.structure.adsorbate import random_rotation_matrix, rotate_atoms


# ---------------------------------------------------------------------------
# Minimum distance clash threshold (Å)
# Used for fast pre-placement clash rejection without a full energy eval.
#
# 0.9 Å is intentionally looser than the post-operator threshold (1.5 Å).
# Pre-relaxation placements just above the surface (zmin = top_z + 0.5) are
# physically valid starting points for the MACE/VASP pre-optimiser.
# 0.9 Å is chosen to:
#   - Stay below the shortest real molecular bond (OH = 0.97 Å), so that
#     intra-adsorbate pairs never trigger a false clash.
#   - Prevent true atomic core overlaps (anything < 0.9 Å is unphysical).
# The calculator (MACE/VASP) will relax all surface-adsorbate distances to
# their equilibrium values.
# ---------------------------------------------------------------------------
_MIN_DISTANCE = 0.9


def place_adsorbate(
    slab: Atoms,
    adsorbate: Atoms,
    zmin: float,
    zmax: float,
    n_orientations: int = 6,
    rng: np.random.Generator | None = None,
    energy_fn: "callable | None" = None,
    max_attempts: int = 50,
) -> Atoms:
    """
    Place an adsorbate on a slab surface, sampling multiple orientations.

    For each of n_orientations random rotations, the adsorbate is placed at
    the same (x, y, z) position and a fast energy estimate is computed.  The
    orientation with the lowest estimated energy is returned.

    If energy_fn is None, a simple steric clash score is used (sum of
    1/r^12 repulsion terms between adsorbate and slab atoms).  This is cheap
    and avoids needing MACE just for placement orientation selection.

    Parameters
    ----------
    slab:
        The bare slab Atoms object.  Not modified.
    adsorbate:
        The adsorbate Atoms object, ideally centred at the origin (as returned
        by gocia.structure.adsorbate.load_adsorbate).
    zmin:
        Minimum z-coordinate for adsorbate placement (Å).
    zmax:
        Maximum z-coordinate for adsorbate placement (Å).
    n_orientations:
        Number of random orientations to trial.  1 for single atoms (no
        rotation needed).  6 is a good default for diatomics.
    rng:
        NumPy random generator.  A new one is created if None.
    energy_fn:
        Optional callable: f(slab_with_adsorbate: Atoms) → float.
        If provided, this is called for each orientation and the result is
        used for ranking.  Pass a MACE calculator's get_potential_energy for
        more physically meaningful orientation selection.
    max_attempts:
        Maximum number of (x, y, z) positions to try before giving up.
        Each attempt generates a fresh random position if all orientations
        at a given position clash with the slab.

    Returns
    -------
    Atoms
        A copy of slab with the adsorbate appended.

    Raises
    ------
    RuntimeError
        If no clash-free placement is found within max_attempts attempts.
    """
    if rng is None:
        rng = np.random.default_rng()

    cell = slab.cell.array

    for attempt in range(max_attempts):
        # Random fractional coordinates → Cartesian (x, y)
        s, t = rng.random(), rng.random()
        xy = s * cell[0] + t * cell[1]
        x, y = float(xy[0]), float(xy[1])

        # Random z within sampling region
        z = float(rng.uniform(zmin, zmax))

        # Try n_orientations rotations
        best_atoms = None
        best_score = float("inf")

        orientations = _sample_orientations(adsorbate, n_orientations, rng)

        for oriented in orientations:
            candidate = _place_at(slab, oriented, x, y, z)

            # Fast clash check before expensive energy evaluation
            if _has_clash(candidate, n_slab=len(slab)):
                continue

            score = (
                energy_fn(candidate)
                if energy_fn is not None
                else _steric_score(candidate, n_slab=len(slab))
            )

            if score < best_score:
                best_score = score
                best_atoms = candidate

        if best_atoms is not None:
            return best_atoms

    raise RuntimeError(
        f"Could not find a clash-free adsorbate placement after "
        f"{max_attempts} attempts.  The surface may be too crowded, or "
        f"zmin/zmax may be set too close to the slab."
    )


def place_multiple(
    slab: Atoms,
    adsorbate: Atoms,
    n: int,
    zmin: float,
    zmax: float,
    n_orientations: int = 6,
    rng: np.random.Generator | None = None,
    energy_fn: "callable | None" = None,
) -> list[Atoms]:
    """
    Place an adsorbate n times independently, returning n separate structures.

    Used when building the initial population: each call produces one member
    of the population with an independently sampled adsorbate position.

    Parameters
    ----------
    slab:
        Bare slab Atoms object.
    adsorbate:
        Adsorbate Atoms object centred at origin.
    n:
        Number of structures to generate.
    zmin, zmax:
        Sampling z-bounds (Å).
    n_orientations:
        Orientations per placement trial.
    rng:
        NumPy random generator (shared across all placements).
    energy_fn:
        Optional energy callable for orientation ranking.

    Returns
    -------
    list[Atoms]
        n independent slab+adsorbate structures.
    """
    if rng is None:
        rng = np.random.default_rng()

    return [
        place_adsorbate(
            slab=slab,
            adsorbate=adsorbate,
            zmin=zmin,
            zmax=zmax,
            n_orientations=n_orientations,
            rng=rng,
            energy_fn=energy_fn,
        )
        for _ in range(n)
    ]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _sample_orientations(
    adsorbate: Atoms,
    n: int,
    rng: np.random.Generator,
) -> list[Atoms]:
    """
    Return n randomly rotated copies of adsorbate.

    For single atoms (n_atoms == 1) rotation has no effect; we return n
    identical copies so the rest of the pipeline stays uniform.
    """
    if len(adsorbate) == 1 or n <= 1:
        return [adsorbate.copy() for _ in range(max(n, 1))]

    orientations = []
    for _ in range(n):
        R = random_rotation_matrix(rng)
        orientations.append(rotate_atoms(adsorbate, R))
    return orientations


def _place_at(slab: Atoms, adsorbate: Atoms, x: float, y: float, z: float) -> Atoms:
    """
    Return a copy of slab with adsorbate positioned at (x, y, z).

    z is the z-coordinate of the lowest adsorbate atom.  The adsorbate is
    translated so its lowest atom sits at z, preserving internal geometry.
    """
    ads = adsorbate.copy()

    # Translate so the lowest atom is at z
    z_min_ads = ads.positions[:, 2].min()
    ads.positions += np.array([x, y, z - z_min_ads])

    combined = slab.copy()
    combined += ads
    return combined


def _has_clash(atoms: Atoms, n_slab: int, min_distance: float = _MIN_DISTANCE) -> bool:
    """
    Return True if any adsorbate atom is closer than min_distance to any
    other atom (including other adsorbate atoms and slab atoms).
    """
    pos = atoms.positions
    n = len(pos)
    adsorbate_indices = list(range(n_slab, n))

    for i in adsorbate_indices:
        for j in range(n):
            if i == j:
                continue
            dist = np.linalg.norm(pos[i] - pos[j])
            if dist < min_distance:
                return True
    return False


def _steric_score(atoms: Atoms, n_slab: int) -> float:
    """
    Fast steric clash score based on inverse-12 repulsion.

    Sum of (1/r)^12 between each adsorbate atom and all other atoms.
    Lower is better (less overlap).  Used when no energy_fn is provided.
    """
    pos = atoms.positions
    n = len(pos)
    adsorbate_indices = list(range(n_slab, n))
    score = 0.0

    for i in adsorbate_indices:
        for j in range(n):
            if i == j:
                continue
            r = np.linalg.norm(pos[i] - pos[j])
            if r < 0.1:
                r = 0.1   # guard against division by zero
            score += (1.0 / r) ** 12

    return score
