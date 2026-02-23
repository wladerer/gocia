"""
gocia/structure/fingerprint.py

Fast geometry-based fingerprinting for duplicate and isomer detection.

Two-stage detection strategy
-----------------------------
Pre-submission (this module, cheap):
    A sorted interatomic distance histogram computed over all atoms.
    O(N²) in atom count but very fast for typical slab+adsorbate sizes (<200 atoms).
    Used to avoid submitting obvious duplicates before any calculator runs.

Post-relaxation (also this module):
    Same histogram on the relaxed structure, with tighter thresholds.
    SOAP descriptors via DScribe are supported as an optional upgrade
    (requires `pip install gocia[soap]`).

Public API
----------
    distance_histogram(atoms, n_bins, r_max)  → list[float]
    fingerprint_distance(fp1, fp2)            → float
    classify_structure(fp, existing_fps, duplicate_threshold, isomer_threshold)
        → "duplicate" | "isomer" | "unique"
    find_closest(fp, existing_fps)            → (id, distance) | None

    # Optional SOAP interface (raises ImportError if dscribe not installed)
    soap_fingerprint(atoms, species, r_cut, n_max, l_max) → list[float]
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from ase import Atoms


# ---------------------------------------------------------------------------
# Distance histogram fingerprint
# ---------------------------------------------------------------------------

def distance_histogram(
    atoms: Atoms,
    n_bins: int = 50,
    r_max: float = 8.0,
) -> list[float]:
    """
    Compute a sorted interatomic distance histogram fingerprint.

    All pairwise distances up to r_max are binned into n_bins equally-spaced
    bins and normalised by the total number of pairs.  The result is a fixed-
    length vector that is invariant to atom ordering and rigid-body translation.

    Parameters
    ----------
    atoms:
        The ASE Atoms object to fingerprint.
    n_bins:
        Number of histogram bins.  50 is a good default.
    r_max:
        Maximum distance to consider (Å).

    Returns
    -------
    list[float]
        Normalised histogram of length n_bins.  All values are in [0, 1].
    """
    pos = atoms.get_positions()
    n = len(pos)

    if n < 2:
        return [0.0] * n_bins

    # All unique pairwise distances via broadcasting
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]   # (n, n, 3)
    dist_matrix = np.linalg.norm(diff, axis=-1)            # (n, n)

    i_upper, j_upper = np.triu_indices(n, k=1)
    distances = dist_matrix[i_upper, j_upper]
    distances = distances[distances < r_max]

    hist, _ = np.histogram(distances, bins=n_bins, range=(0.0, r_max))

    if hist.max() == 0:
        return [0.0] * n_bins

    # Normalise by the peak bin count so that the fingerprint represents the
    # *shape* of the distance distribution rather than its absolute scale.
    # This makes the fingerprint sensitive to adsorbate differences even on a
    # large slab, where normalising by total pairs would dilute the adsorbate
    # signal in the dominant slab-slab noise.
    normalised = hist / hist.max()
    return normalised.tolist()


def fingerprint_distance(fp1: list[float], fp2: list[float]) -> float:
    """
    Compute the L2 (Euclidean) distance between two fingerprint vectors.

    Parameters
    ----------
    fp1, fp2:
        Fingerprint vectors of equal length.

    Returns
    -------
    float
        Euclidean distance.  0.0 for identical fingerprints.

    Raises
    ------
    ValueError
        If the fingerprints have different lengths.
    """
    a = np.array(fp1, dtype=float)
    b = np.array(fp2, dtype=float)
    if a.shape != b.shape:
        raise ValueError(
            f"Fingerprint length mismatch: {len(fp1)} vs {len(fp2)}. "
            "Ensure both were computed with the same n_bins and r_max."
        )
    return float(np.linalg.norm(a - b))


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

ClassifyResult = Literal["duplicate", "isomer", "unique"]


def classify_structure(
    fp: list[float],
    existing_fps: list[list[float]],
    duplicate_threshold: float = 0.01,
    isomer_threshold: float = 0.10,
) -> ClassifyResult:
    """
    Compare a fingerprint against an existing population and classify it.

    Classification rules
    --------------------
    distance < duplicate_threshold              → "duplicate"
    duplicate_threshold ≤ distance < isomer_threshold  → "isomer"
    distance ≥ isomer_threshold for all existing       → "unique"

    Parameters
    ----------
    fp:
        Fingerprint of the new structure.
    existing_fps:
        Fingerprints of all structures already in the population.
        Pass an empty list for the first structure.
    duplicate_threshold:
        L2 distance below which two structures are considered identical.
    isomer_threshold:
        L2 distance below which two structures are considered near-duplicates.

    Returns
    -------
    "duplicate", "isomer", or "unique"
    """
    if not existing_fps:
        return "unique"

    min_dist = min(fingerprint_distance(fp, existing) for existing in existing_fps)

    if min_dist < duplicate_threshold:
        return "duplicate"
    if min_dist < isomer_threshold:
        return "isomer"
    return "unique"


def find_closest(
    fp: list[float],
    existing_fps: list[tuple[str, list[float]]],
) -> tuple[str, float] | None:
    """
    Find the closest existing structure to fp by fingerprint distance.

    Parameters
    ----------
    fp:
        Fingerprint of the query structure.
    existing_fps:
        List of (individual_id, fingerprint) pairs, e.g. from
        GociaDB.fingerprints().

    Returns
    -------
    (individual_id, distance) of the closest match, or None if existing_fps
    is empty.
    """
    if not existing_fps:
        return None

    best_id, best_dist = None, float("inf")
    for ind_id, existing in existing_fps:
        d = fingerprint_distance(fp, existing)
        if d < best_dist:
            best_dist = d
            best_id = ind_id

    return (best_id, best_dist)


# ---------------------------------------------------------------------------
# Optional SOAP fingerprint (requires dscribe)
# ---------------------------------------------------------------------------

def soap_fingerprint(
    atoms: Atoms,
    species: list[str],
    r_cut: float = 6.0,
    n_max: int = 8,
    l_max: int = 6,
) -> list[float]:
    """
    Compute a SOAP descriptor fingerprint averaged over all atoms.

    Fully rotation-invariant; more expensive than distance_histogram.
    Intended for post-relaxation duplicate detection when high precision
    is needed.

    Requires the optional `dscribe` dependency:
        pip install gocia[soap]

    Parameters
    ----------
    atoms:
        The ASE Atoms object to fingerprint.
    species:
        All element symbols that can appear in the system, e.g. ["Pt","O","H"].
        Must be consistent across all structures being compared.
    r_cut:
        SOAP cutoff radius in Å.
    n_max, l_max:
        SOAP basis set size.

    Returns
    -------
    list[float]
        Averaged SOAP descriptor vector.

    Raises
    ------
    ImportError
        If dscribe is not installed.
    """
    try:
        from dscribe.descriptors import SOAP
    except ImportError as exc:
        raise ImportError(
            "SOAP fingerprinting requires dscribe. "
            "Install it with: pip install gocia[soap]"
        ) from exc

    soap = SOAP(
        species=species,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        average="inner",
        periodic=True,
    )
    descriptor = soap.create(atoms)
    return descriptor.tolist()
