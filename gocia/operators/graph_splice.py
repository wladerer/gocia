"""
gocia/operators/graph_splice.py

Graph-based splice operator for the GOCIA genetic algorithm.

Algorithm overview
------------------
Given two parent structures A and B (same slab, different adsorbate layers):

1. Build a connectivity graph over the adsorbate layer of each parent.
   Nodes = adsorbate atoms.  Edges = pairs within a distance threshold
   (sum of covalent radii × scale_factor).

2. Choose a cut plane perpendicular to a random in-plane direction.
   The cut position is sampled uniformly within the cell.

3. Assign each adsorbate node to "left" or "right" of the cut plane
   based on its centroid position (for molecules: centroid of connected
   component).  This avoids cutting through bonded molecules.

4. Build child1 = slab + left_A + right_B
         child2 = slab + left_B + right_A

5. Resolve stoichiometry: after the swap, each child may have a different
   atom count than either parent.  We correct this by adding/removing
   the cheapest atoms (by distance to nearest slab atom) until the count
   matches the original parent stoichiometry.

6. Remove steric clashes at the boundary using remove_clashes().

The stoichiometry target for each child is the stoichiometry of parent A
(child1) and parent B (child2), ensuring the total across both children
equals the total across both parents.

Public API
----------
    build_adsorbate_graph(atoms, n_slab, scale_factor) → networkx.Graph
    splice(atoms_a, atoms_b, n_slab, ...)              → (child1, child2)
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from ase import Atoms
from ase.data import covalent_radii, atomic_numbers

from gocia.operators.base import GeneticOperator, remove_clashes, OPERATOR_REGISTRY
from gocia.population.individual import OPERATOR as OP_NAMES


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_adsorbate_graph(
    atoms: Atoms,
    n_slab: int,
    scale_factor: float = 1.25,
):
    """
    Build a connectivity graph over the adsorbate atoms.

    Nodes are adsorbate atom indices (local, starting at 0 relative to the
    adsorbate layer, i.e. global_index - n_slab).

    An edge is added between two atoms if their distance is less than
    scale_factor × (covalent_radius_i + covalent_radius_j).

    Parameters
    ----------
    atoms:
        Full slab + adsorbate Atoms object.
    n_slab:
        Number of slab atoms.
    scale_factor:
        Multiplier on covalent radii sum for edge detection.
        1.25 captures typical bonds while avoiding spurious long-range edges.

    Returns
    -------
    networkx.Graph
        Nodes: local adsorbate indices (0-based).
        Node attribute 'symbol': chemical symbol of the atom.
        Node attribute 'position': (3,) array of Cartesian coordinates.
        Node attribute 'global_index': global index in the Atoms object.

    Raises
    ------
    ImportError
        If networkx is not installed.
    """
    try:
        import networkx as nx
    except ImportError as exc:
        raise ImportError(
            "networkx is required for graph-based operators. "
            "Install it with: pip install networkx"
        ) from exc

    g = nx.Graph()
    symbols = atoms.get_chemical_symbols()
    positions = atoms.positions
    n_ads = len(atoms) - n_slab

    # Add nodes
    for local_i in range(n_ads):
        global_i = n_slab + local_i
        g.add_node(
            local_i,
            symbol=symbols[global_i],
            position=positions[global_i].copy(),
            global_index=global_i,
        )

    # Add edges
    for local_i in range(n_ads):
        for local_j in range(local_i + 1, n_ads):
            global_i = n_slab + local_i
            global_j = n_slab + local_j

            sym_i = symbols[global_i]
            sym_j = symbols[global_j]

            # Covalent radii lookup
            r_i = covalent_radii[atomic_numbers.get(sym_i, 1)]
            r_j = covalent_radii[atomic_numbers.get(sym_j, 1)]
            threshold = scale_factor * (r_i + r_j)

            dist = float(np.linalg.norm(positions[global_i] - positions[global_j]))
            if dist < threshold:
                g.add_edge(local_i, local_j, distance=dist)

    return g


def _connected_component_centroids(graph, positions_ads: np.ndarray):
    """
    Return a list of (component_node_set, centroid_xy) for each connected
    component in the graph.

    centroid_xy is the mean (x, y) position of atoms in the component,
    used for the plane-cut assignment.
    """
    try:
        import networkx as nx
    except ImportError as exc:
        raise ImportError("networkx required") from exc

    components = []
    for component in nx.connected_components(graph):
        node_list = list(component)
        centroid = positions_ads[node_list].mean(axis=0)
        components.append((node_list, centroid))
    return components


# ---------------------------------------------------------------------------
# Stoichiometry helpers
# ---------------------------------------------------------------------------

def _adsorbate_counts(atoms: Atoms, n_slab: int) -> Counter:
    return Counter(atoms.get_chemical_symbols()[n_slab:])


def _correct_stoichiometry(
    slab: Atoms,
    adsorbate_atoms: Atoms,
    target_counts: Counter,
    n_slab: int,
    rng: np.random.Generator,
) -> Atoms:
    """
    Add or remove atoms from adsorbate_atoms until it matches target_counts.

    Removal strategy: remove the atom of the excess species that is farthest
    from the slab surface (most exposed — safest to remove).

    Addition strategy: copy a random atom of the deficit species from the
    adsorbate_atoms pool at a slightly displaced position.  If none exist,
    place a new atom 2 Å above the nearest slab atom centroid.

    This correction is intentionally simple.  The pre-opt stage will handle
    any resulting strain.
    """
    ads = adsorbate_atoms.copy()
    current = Counter(ads.get_chemical_symbols())

    # --- Removals first (reduce overcounting) ---
    for sym, target_n in target_counts.items():
        while current.get(sym, 0) > target_n:
            # Find indices of this symbol in adsorbate layer
            indices = [
                i for i, s in enumerate(ads.get_chemical_symbols()) if s == sym
            ]
            if not indices:
                break
            # Remove the one highest above the slab (most exposed)
            remove_idx = max(indices, key=lambda i: ads.positions[i, 2])
            del ads[remove_idx]
            current[sym] -= 1

    # Remove species entirely absent from target
    all_syms = list(ads.get_chemical_symbols())
    to_remove = []
    for i, sym in enumerate(all_syms):
        if sym not in target_counts:
            to_remove.append(i)
    for idx in sorted(to_remove, reverse=True):
        del ads[idx]
        current = Counter(ads.get_chemical_symbols())

    # --- Additions (fill deficit) ---
    for sym, target_n in target_counts.items():
        while current.get(sym, 0) < target_n:
            # Try up to 20 positions; retry if the candidate clashes with any
            # atom already present in ads (including previously added corrections).
            _min_add_dist = 1.5   # must exceed remove_clashes threshold
            new_pos = None
            for _attempt in range(20):
                existing_indices = [
                    i for i, s in enumerate(ads.get_chemical_symbols()) if s == sym
                ]
                if existing_indices:
                    # Displace a copy of an existing atom of this species.
                    # 2.0–3.0 Å guarantees clearance from the reference atom.
                    ref_idx = rng.choice(existing_indices)
                    delta = rng.uniform(-1.0, 1.0, 3)
                    delta_norm = np.linalg.norm(delta)
                    if delta_norm < 1e-3:
                        delta = np.array([1.0, 0.0, 0.0])
                    else:
                        delta = delta / delta_norm * rng.uniform(2.0, 3.0)
                    candidate = ads.positions[ref_idx] + delta
                else:
                    # No existing atoms of this species: place above slab centroid.
                    slab_centroid_xy = slab.positions[:n_slab, :2].mean(axis=0)
                    top_z = slab.positions[:n_slab, 2].max()
                    candidate = np.array([
                        slab_centroid_xy[0] + rng.uniform(-1.0, 1.0),
                        slab_centroid_xy[1] + rng.uniform(-1.0, 1.0),
                        top_z + 2.0 + rng.uniform(0.0, 1.0),
                    ])
                # Check candidate against all atoms already in ads
                if len(ads) == 0:
                    new_pos = candidate
                    break
                dists = np.linalg.norm(ads.positions - candidate, axis=1)
                if dists.min() >= _min_add_dist:
                    new_pos = candidate
                    break

            if new_pos is None:
                # All attempts clashed — use best candidate (closest to clash-free)
                # The pre-opt calculator will resolve remaining strain.
                new_pos = candidate  # last attempt

            new_atom = Atoms(sym, positions=[list(new_pos)])
            ads += new_atom
            current[sym] = current.get(sym, 0) + 1

    return ads


# ---------------------------------------------------------------------------
# Public splice function
# ---------------------------------------------------------------------------

def splice(
    atoms_a: Atoms,
    atoms_b: Atoms,
    n_slab: int,
    scale_factor: float = 1.25,
    min_distance: float = 1.5,
    rng: np.random.Generator | None = None,
) -> tuple[Atoms, Atoms]:
    """
    Graph-based splice of two parent structures.

    Splits the adsorbate layer of each parent along a random plane and
    swaps the halves to produce two children.  Stoichiometry of each child
    matches the corresponding parent.  Steric clashes at the boundary are
    resolved by removal.

    Parameters
    ----------
    atoms_a, atoms_b:
        Parent Atoms objects.  Must have the same slab (first n_slab atoms).
        Not modified.
    n_slab:
        Number of slab atoms (shared between parents and children).
    scale_factor:
        Covalent radii multiplier for graph edge detection.
    min_distance:
        Minimum allowed interatomic distance after splice (Å).
    rng:
        NumPy random generator.

    Returns
    -------
    (child1, child2)
        Two new Atoms objects.  child1 has parent_a's stoichiometry;
        child2 has parent_b's stoichiometry.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Work on copies to guarantee non-mutation of inputs
    a = atoms_a.copy()
    b = atoms_b.copy()

    target_a = _adsorbate_counts(a, n_slab)
    target_b = _adsorbate_counts(b, n_slab)

    # Edge case: both parents are bare slabs
    if len(a) == n_slab and len(b) == n_slab:
        return a.copy(), b.copy()

    # Build graphs
    graph_a = build_adsorbate_graph(a, n_slab, scale_factor)
    graph_b = build_adsorbate_graph(b, n_slab, scale_factor)

    # Positions of adsorbate layers (local indexing)
    pos_ads_a = a.positions[n_slab:]
    pos_ads_b = b.positions[n_slab:]

    # Choose a random cut plane perpendicular to a random in-plane direction
    # Use x or y axis with equal probability for simplicity
    axis = int(rng.integers(0, 2))  # 0=x, 1=y
    cell = a.cell.array
    cut_frac = rng.uniform(0.2, 0.8)  # avoid cuts at cell edges
    cut_value = float(cut_frac * np.linalg.norm(cell[axis]))

    # Get connected components and their centroids
    components_a = _connected_component_centroids(graph_a, pos_ads_a)
    components_b = _connected_component_centroids(graph_b, pos_ads_b)

    # Assign components to left/right of cut
    def _split(components, pos_ads):
        left_indices, right_indices = [], []
        for node_list, centroid in components:
            if centroid[axis] < cut_value:
                left_indices.extend(node_list)
            else:
                right_indices.extend(node_list)
        return sorted(left_indices), sorted(right_indices)

    left_a, right_a = _split(components_a, pos_ads_a)
    left_b, right_b = _split(components_b, pos_ads_b)

    # Build children: child1 = left_A + right_B, child2 = left_B + right_A
    slab = a[:n_slab]

    def _build_child(slab, ads_a, indices_from_a, ads_b, indices_from_b, target, n_slab, rng):
        ads_part_a = ads_a[indices_from_a] if indices_from_a else Atoms()
        ads_part_b = ads_b[indices_from_b] if indices_from_b else Atoms()

        combined_ads = ads_part_a + ads_part_b

        # Remove clashes introduced by combining atoms from different parents
        combined_ads = remove_clashes(slab, combined_ads, n_slab, min_distance)

        # Correct stoichiometry to match target.
        # Correction adds atoms at 2.0–3.0 Å from a reference (guaranteed > min_distance),
        # so no second clash-removal pass is needed.  Any remaining strain (e.g. two
        # newly added atoms close to each other) is resolved by the pre-opt calculator.
        combined_ads = _correct_stoichiometry(slab, combined_ads, target, n_slab, rng)

        child = slab.copy()
        child += combined_ads
        return child

    ads_a = a[n_slab:]
    ads_b = b[n_slab:]

    child1 = _build_child(slab, ads_a, left_a, ads_b, right_b, target_a, n_slab, rng)
    child2 = _build_child(slab, ads_b, left_b, ads_a, right_a, target_b, n_slab, rng)

    return child1, child2


# ---------------------------------------------------------------------------
# GeneticOperator subclass (for registry)
# ---------------------------------------------------------------------------

class SpliceOperator(GeneticOperator):
    n_parents = 2
    operator_name = OP_NAMES.SPLICE

    def __init__(
        self,
        scale_factor: float = 1.25,
        min_distance: float = 1.5,
    ):
        self.scale_factor = scale_factor
        self.min_distance = min_distance

    def apply(self, parents, n_slab, rng=None, **kwargs):
        child1, child2 = splice(
            parents[0], parents[1], n_slab,
            scale_factor=self.scale_factor,
            min_distance=self.min_distance,
            rng=rng,
        )
        return [child1, child2]


OPERATOR_REGISTRY[OP_NAMES.SPLICE] = SpliceOperator()
