"""
gocia/operators/graph_merge.py

Graph-based merge operator for the GOCIA genetic algorithm.

Algorithm overview
------------------
Merge is a one-child variant of splice.  Given two parents A and B:

1. Build connectivity graphs over both adsorbate layers.
2. Choose a random cut plane (same logic as splice).
3. Build child = left half of A + right half of B.
4. Stoichiometry target is parent A's count (arbitrary but consistent).
5. Remove clashes and correct stoichiometry.

The cut plane axis and position are chosen randomly, so repeated calls on
the same parents produce different children.

Public API
----------
    merge(atoms_a, atoms_b, n_slab, ...) → child
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from ase import Atoms

from gocia.operators.base import GeneticOperator, remove_clashes, OPERATOR_REGISTRY
from gocia.operators.graph_splice import (
    build_adsorbate_graph,
    _connected_component_centroids,
    _correct_stoichiometry,
    _adsorbate_counts,
)
from gocia.population.individual import OPERATOR as OP_NAMES


def merge(
    atoms_a: Atoms,
    atoms_b: Atoms,
    n_slab: int,
    scale_factor: float = 1.25,
    min_distance: float = 1.5,
    rng: np.random.Generator | None = None,
) -> Atoms:
    """
    Graph-based merge of two parent structures into one child.

    The child's stoichiometry matches parent A.  The left half of the
    adsorbate layer comes from A, the right half from B (relative to a
    random cut plane).

    Parameters
    ----------
    atoms_a, atoms_b:
        Parent Atoms objects.  Must share the same slab (first n_slab atoms).
        Not modified.
    n_slab:
        Number of slab atoms.
    scale_factor:
        Covalent radii multiplier for graph edge detection.
    min_distance:
        Minimum allowed interatomic distance after merge (Å).
    rng:
        NumPy random generator.

    Returns
    -------
    Atoms
        One child structure with parent A's stoichiometry.
    """
    if rng is None:
        rng = np.random.default_rng()

    a = atoms_a.copy()
    b = atoms_b.copy()

    target = _adsorbate_counts(a, n_slab)

    # Edge case: parent A is a bare slab
    if len(a) == n_slab:
        return a.copy()

    # Build graphs
    graph_a = build_adsorbate_graph(a, n_slab, scale_factor)
    graph_b = build_adsorbate_graph(b, n_slab, scale_factor)

    pos_ads_a = a.positions[n_slab:]
    pos_ads_b = b.positions[n_slab:]

    # Random cut plane
    axis = int(rng.integers(0, 2))
    cell = a.cell.array
    cut_frac = rng.uniform(0.2, 0.8)
    cut_value = float(cut_frac * np.linalg.norm(cell[axis]))

    # Split A into left / right
    components_a = _connected_component_centroids(graph_a, pos_ads_a)
    left_a, right_a = [], []
    for node_list, centroid in components_a:
        if centroid[axis] < cut_value:
            left_a.extend(node_list)
        else:
            right_a.extend(node_list)

    # Split B into left / right (we take the right half of B)
    components_b = _connected_component_centroids(graph_b, pos_ads_b)
    left_b, right_b = [], []
    for node_list, centroid in components_b:
        if centroid[axis] < cut_value:
            left_b.extend(node_list)
        else:
            right_b.extend(node_list)

    slab = a[:n_slab]
    ads_a = a[n_slab:]
    ads_b = b[n_slab:]

    # Child = left of A + right of B
    part_left  = ads_a[sorted(left_a)]  if left_a  else Atoms()
    part_right = ads_b[sorted(right_b)] if right_b else Atoms()

    combined_ads = part_left + part_right

    # Clash removal for atoms combined from different parents
    combined_ads = remove_clashes(slab, combined_ads, n_slab, min_distance)

    # Stoichiometry correction.  Atoms are added at 2.0–3.0 Å from a reference
    # (always > min_distance), so no second clash-removal is needed.
    combined_ads = _correct_stoichiometry(slab, combined_ads, target, n_slab, rng)

    child = slab.copy()
    child += combined_ads
    return child


# ---------------------------------------------------------------------------
# GeneticOperator subclass (for registry)
# ---------------------------------------------------------------------------

class MergeOperator(GeneticOperator):
    n_parents = 2
    operator_name = OP_NAMES.MERGE

    def __init__(
        self,
        scale_factor: float = 1.25,
        min_distance: float = 1.5,
    ):
        self.scale_factor = scale_factor
        self.min_distance = min_distance

    def apply(self, parents, n_slab, rng=None, **kwargs):
        child = merge(
            parents[0], parents[1], n_slab,
            scale_factor=self.scale_factor,
            min_distance=self.min_distance,
            rng=rng,
        )
        return [child]


OPERATOR_REGISTRY[OP_NAMES.MERGE] = MergeOperator()
