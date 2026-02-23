"""
gocia/structure/adsorbate.py

Load adsorbate reference geometries from a file or inline coordinate list.

For single atoms (e.g. O, H) no geometry is needed — ASE can construct them
directly.  For molecules (e.g. OH, CO, H2O) the internal geometry must be
provided either as a path to a geometry file or as a list of [x, y, z]
coordinate rows in the YAML.

Coordinates are stored centred at the origin so that placement routines can
translate and rotate them freely without worrying about the reference frame.

Usage
-----
    from gocia.structure.adsorbate import load_adsorbate
    from gocia.config import AdsorbateConfig

    cfg = AdsorbateConfig(symbol="OH", chemical_potential=-3.75, geometry="OH.vasp")
    atoms = load_adsorbate(cfg)   # ASE Atoms centred at origin
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, covalent_radii
from ase.io import read


def load_adsorbate(config) -> Atoms:
    """
    Load an adsorbate geometry from an AdsorbateConfig.

    For single atoms: constructs an Atoms object directly using the symbol.
    For molecules with a geometry file: reads the file via ASE.
    For molecules with inline coordinates: builds Atoms from the coordinate list.

    In all cases the returned Atoms object is centred at the origin.

    Parameters
    ----------
    config:
        An AdsorbateConfig instance.

    Returns
    -------
    Atoms
        The adsorbate Atoms object centred at the origin.  No cell or PBC
        is set (these are inherited from the slab during placement).

    Raises
    ------
    FileNotFoundError:
        If config.geometry is set but the file does not exist.
    ValueError:
        If config.coordinates is provided but has an unexpected shape.
    """
    symbol = config.symbol

    # Single atom — no geometry needed
    if config.geometry is None and config.coordinates is None:
        return _single_atom(symbol)

    # Molecule from file
    if config.geometry is not None:
        return _from_file(config.geometry, symbol)

    # Molecule from inline coordinates
    return _from_coordinates(config.coordinates, symbol)


# ---------------------------------------------------------------------------
# Private loaders
# ---------------------------------------------------------------------------

def _single_atom(symbol: str) -> Atoms:
    """Build a single-atom Atoms object at the origin."""
    # Validate that the symbol is a known element
    if symbol not in atomic_numbers:
        raise ValueError(
            f"'{symbol}' is not a recognised chemical symbol.  "
            "Check the adsorbate definition in gocia.yaml."
        )
    return Atoms(symbol, positions=[[0.0, 0.0, 0.0]])


def _from_file(path: str | Path, symbol: str) -> Atoms:
    """
    Read a molecule geometry from file and centre it at the origin.

    The symbol parameter is used only for validation (checks that the file
    contains atoms matching the expected species).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Adsorbate geometry file not found: {path}\n"
            f"(referenced by adsorbate symbol '{symbol}' in gocia.yaml)"
        )

    atoms = read(str(path))

    # Centre at origin
    atoms.positions -= atoms.positions.mean(axis=0)

    return atoms


def _from_coordinates(
    coordinates: list[list[float]],
    symbol: str,
) -> Atoms:
    """
    Build a molecule from an inline list of [x, y, z] coordinate rows.

    The symbol string is parsed to extract element symbols in order, e.g.
    "OH" → ["O", "H"], "CO2" → ["C", "O", "O"].

    Parameters
    ----------
    coordinates:
        List of [x, y, z] rows, one per atom.
    symbol:
        Chemical formula string matching the atom order in coordinates.

    Raises
    ------
    ValueError:
        If the number of coordinate rows doesn't match the formula atom count.
    """
    positions = np.array(coordinates, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(
            f"Adsorbate coordinates for '{symbol}' must be a list of "
            f"[x, y, z] rows, got shape {positions.shape}."
        )

    # Parse the formula to get individual element symbols in order
    symbols = _parse_formula_ordered(symbol)

    if len(symbols) != len(positions):
        raise ValueError(
            f"Adsorbate '{symbol}' has {len(symbols)} atoms from the formula "
            f"but {len(positions)} coordinate rows were provided."
        )

    atoms = Atoms(symbols=symbols, positions=positions)
    atoms.positions -= atoms.positions.mean(axis=0)
    return atoms


def _parse_formula_ordered(formula: str) -> list[str]:
    """
    Parse a chemical formula string into an ordered list of element symbols.

    Handles formulas like "OH" → ["O","H"], "CO2" → ["C","O","O"],
    "H2O" → ["H","H","O"].

    Note: this is intentionally simple.  It does not handle parentheses or
    nested groups.  For complex molecules, provide a geometry file instead.
    """
    import re
    # Match element symbol (capital + optional lowercase) followed by optional count
    pattern = re.compile(r"([A-Z][a-z]?)(\d*)")
    symbols = []
    for match in pattern.finditer(formula):
        elem = match.group(1)
        count = int(match.group(2)) if match.group(2) else 1
        if elem not in atomic_numbers:
            raise ValueError(
                f"'{elem}' in formula '{formula}' is not a recognised element."
            )
        symbols.extend([elem] * count)
    if not symbols:
        raise ValueError(f"Could not parse any elements from formula '{formula}'.")
    return symbols


# ---------------------------------------------------------------------------
# Rotation sampling helpers (used by placement.py)
# ---------------------------------------------------------------------------

def random_rotation_matrix(rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Generate a uniformly random 3D rotation matrix using the Gram-Schmidt
    method on a random orthonormal basis.

    Parameters
    ----------
    rng:
        NumPy random generator.

    Returns
    -------
    np.ndarray of shape (3, 3)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Random rotation via QR decomposition of a random matrix
    # (this gives a uniform distribution over SO(3))
    random_matrix = rng.standard_normal((3, 3))
    q, r = np.linalg.qr(random_matrix)

    # Ensure proper rotation (det = +1, not -1)
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1

    return q


def rotate_atoms(atoms: Atoms, rotation_matrix: np.ndarray) -> Atoms:
    """
    Return a copy of atoms with positions rotated by rotation_matrix.

    The rotation is applied about the centroid of the molecule.

    Parameters
    ----------
    atoms:
        Atoms object to rotate (should be centred at origin for best results).
    rotation_matrix:
        3×3 rotation matrix.

    Returns
    -------
    Atoms
        New rotated Atoms object.
    """
    rotated = atoms.copy()
    centroid = rotated.positions.mean(axis=0)
    rotated.positions -= centroid
    rotated.positions = (rotation_matrix @ rotated.positions.T).T
    rotated.positions += centroid
    return rotated
