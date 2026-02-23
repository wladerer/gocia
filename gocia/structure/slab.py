"""
gocia/structure/slab.py

Slab loading and sampling region utilities.

The slab is the fixed substrate on which adsorbates are placed.  This module
handles:
  - Loading the slab geometry from a VASP POSCAR/CONTCAR file (or any ASE-
    readable format) and validating that selective dynamics are defined.
  - Identifying which atoms are frozen (slab) and which are mobile (surface).
  - Defining the adsorbate sampling region (a z-range above the surface).
  - Computing derived properties needed by the GA operators: the slab atom
    count, the surface cell, and the sampling bounding box.

Usage
-----
    from gocia.structure.slab import load_slab, SlabInfo

    info = load_slab("slab.vasp", sampling_zmin=8.0, sampling_zmax=12.0)
    print(info.n_slab_atoms)
    print(info.surface_normal)   # always [0, 0, 1] for (111) surfaces
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import read


# ---------------------------------------------------------------------------
# SlabInfo dataclass
# ---------------------------------------------------------------------------

@dataclass
class SlabInfo:
    """
    Derived information about a loaded slab, pre-computed once at startup.

    Attributes
    ----------
    atoms:
        The full ASE Atoms object with constraints.
    n_slab_atoms:
        Number of atoms belonging to the slab substrate.  Adsorbate atoms
        are indexed from n_slab_atoms onward.
    frozen_indices:
        Indices of atoms held fixed by FixAtoms constraints.
    free_indices:
        Indices of surface atoms that are allowed to relax (typically the
        top 1-2 layers).
    sampling_zmin:
        Minimum z-coordinate (Å) of the adsorbate placement region.
    sampling_zmax:
        Maximum z-coordinate (Å) of the adsorbate placement region.
    top_surface_z:
        z-coordinate of the topmost slab atom (Å).  Useful for setting
        adsorbate heights.
    cell:
        3×3 unit cell matrix (Å).
    """

    atoms: Atoms
    n_slab_atoms: int
    frozen_indices: list[int]
    free_indices: list[int]
    sampling_zmin: float
    sampling_zmax: float
    top_surface_z: float
    cell: np.ndarray

    @property
    def surface_area(self) -> float:
        """In-plane surface area of the unit cell (Å²)."""
        a = self.cell[0]
        b = self.cell[1]
        return float(np.linalg.norm(np.cross(a, b)))

    @property
    def sampling_height(self) -> float:
        """Thickness of the adsorbate sampling region (Å)."""
        return self.sampling_zmax - self.sampling_zmin

    def random_xy(self, rng: np.random.Generator | None = None) -> tuple[float, float]:
        """
        Sample a random (x, y) position uniformly within the surface unit cell.

        Uses fractional coordinates internally to handle non-orthogonal cells
        correctly, then converts to Cartesian.

        Parameters
        ----------
        rng:
            NumPy random generator.  Uses np.random.default_rng() if None.

        Returns
        -------
        (x, y) in Cartesian coordinates (Å).
        """
        if rng is None:
            rng = np.random.default_rng()

        # Random fractional coordinates in [0, 1)
        s = rng.random()
        t = rng.random()

        # Convert to Cartesian using the in-plane cell vectors
        a = self.cell[0]
        b = self.cell[1]
        pos = s * a + t * b
        return (float(pos[0]), float(pos[1]))

    def random_z(self, rng: np.random.Generator | None = None) -> float:
        """
        Sample a random z-coordinate within the sampling region.

        Parameters
        ----------
        rng:
            NumPy random generator.

        Returns
        -------
        float
            z-coordinate in Å.
        """
        if rng is None:
            rng = np.random.default_rng()
        return float(rng.uniform(self.sampling_zmin, self.sampling_zmax))


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_slab(
    path: str | Path,
    sampling_zmin: float,
    sampling_zmax: float,
) -> SlabInfo:
    """
    Load a slab geometry file and compute derived slab information.

    The file should already contain selective dynamics (FixAtoms constraints)
    marking which atoms are frozen.  If no constraints are found, a warning
    is issued and all atoms are treated as part of the slab (none are free
    surface atoms, which is unusual but valid for rigid-slab calculations).

    Parameters
    ----------
    path:
        Path to the slab geometry file.  Any ASE-readable format is accepted
        (VASP POSCAR/CONTCAR, XYZ, CIF, …).  VASP format is recommended
        because it natively supports selective dynamics.
    sampling_zmin:
        Minimum z-coordinate (Å) of the adsorbate sampling region.
    sampling_zmax:
        Maximum z-coordinate (Å) of the adsorbate sampling region.

    Returns
    -------
    SlabInfo

    Raises
    ------
    FileNotFoundError:
        If the geometry file does not exist.
    ValueError:
        If sampling_zmin >= sampling_zmax, or if the sampling region is
        below the top of the slab.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Slab geometry file not found: {path}")

    atoms = read(str(path))

    if sampling_zmin >= sampling_zmax:
        raise ValueError(
            f"sampling_zmin ({sampling_zmin}) must be less than "
            f"sampling_zmax ({sampling_zmax})."
        )

    # --- Extract frozen and free indices from FixAtoms constraints ----------
    frozen_indices: list[int] = []
    for constraint in atoms.constraints:
        if isinstance(constraint, FixAtoms):
            frozen_indices.extend(constraint.get_indices().tolist())

    frozen_set = set(frozen_indices)
    all_indices = list(range(len(atoms)))
    free_indices = [i for i in all_indices if i not in frozen_set]

    if not frozen_indices:
        import warnings
        warnings.warn(
            f"No FixAtoms constraint found in {path}.  All atoms will be "
            "treated as slab atoms.  For typical GOCIA runs the slab should "
            "have at least the bottom layers frozen.",
            stacklevel=2,
        )

    top_surface_z = float(atoms.positions[:, 2].max())

    if sampling_zmin < top_surface_z:
        raise ValueError(
            f"sampling_zmin ({sampling_zmin:.2f} Å) is below the topmost slab "
            f"atom at z = {top_surface_z:.2f} Å.  Adsorbates would be placed "
            "inside the slab."
        )

    return SlabInfo(
        atoms=atoms,
        n_slab_atoms=len(atoms),
        frozen_indices=frozen_indices,
        free_indices=free_indices,
        sampling_zmin=sampling_zmin,
        sampling_zmax=sampling_zmax,
        top_surface_z=top_surface_z,
        cell=atoms.cell.array.copy(),
    )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_adsorbate_atoms(
    atoms: Atoms,
    n_slab_atoms: int,
) -> Atoms:
    """
    Return a view of the adsorbate atoms only (indices n_slab_atoms onward).

    Parameters
    ----------
    atoms:
        Full slab + adsorbate Atoms object.
    n_slab_atoms:
        Number of atoms in the bare slab (from SlabInfo.n_slab_atoms).

    Returns
    -------
    Atoms
        New Atoms object containing only the adsorbate atoms, with the same
        cell and PBC as the original.
    """
    adsorbate = atoms[n_slab_atoms:]
    adsorbate.set_cell(atoms.cell)
    adsorbate.set_pbc(atoms.pbc)
    return adsorbate


def validate_sampling_region(
    slab_info: SlabInfo,
    atoms: Atoms,
) -> list[int]:
    """
    Return indices of adsorbate atoms that are outside the sampling z-region.

    Useful for post-placement sanity checks.  A non-empty list indicates that
    some adsorbate atoms drifted outside the expected region during relaxation
    (which may indicate desorption — see gocia.desorption).

    Parameters
    ----------
    slab_info:
        The SlabInfo for this slab.
    atoms:
        Full slab + adsorbate Atoms object (after placement or relaxation).

    Returns
    -------
    list[int]
        Global atom indices of adsorbate atoms outside [zmin - 1.0, zmax + 5.0].
        The loose bounds (−1 below, +5 above) account for normal bonding geometry.
    """
    n = slab_info.n_slab_atoms
    adsorbate_positions = atoms.positions[n:]
    zmin = slab_info.sampling_zmin - 1.0
    zmax = slab_info.sampling_zmax + 5.0

    out_of_bounds = []
    for i, pos in enumerate(adsorbate_positions):
        if pos[2] < zmin or pos[2] > zmax:
            out_of_bounds.append(n + i)
    return out_of_bounds
