"""
gocia/population/population.py

Weighted selection and operator probability sampling for the GA loop.

Selection
---------
Parents are drawn by weighted random sampling without replacement from the
pool of selectable Individuals (status in {converged, isomer}).  Weights
come directly from Individual.weight:
  - Unique converged:  weight = 1.0
  - Isomers:           weight = ga.isomer_weight (default 0.01)
  - Desorbed/failed:   weight = 0.0  (excluded automatically)

Operator probabilities
----------------------
The operator to apply is sampled at each reproduction step.  Default
probabilities can be overridden in the config (not yet exposed in v1 —
hardcoded here as named constants for easy tuning).

Public API
----------
    select_parents(pool, operator, rng)  → list[Individual]
    sample_operator(rng)                 → str   (OPERATOR constant)
    selection_weights(pool)              → np.ndarray
"""

from __future__ import annotations

import numpy as np

from gocia.population.individual import Individual, OPERATOR


# ---------------------------------------------------------------------------
# Operator sampling probabilities
# ---------------------------------------------------------------------------

#: Default probability distribution over GA operators.
#: Must sum to 1.0.  Splice and merge are 2-parent ops; mutations are 1-parent.
OPERATOR_PROBS: dict[str, float] = {
    OPERATOR.SPLICE:          0.40,
    OPERATOR.MERGE:           0.30,
    OPERATOR.MUTATE_ADD:      0.10,
    OPERATOR.MUTATE_REMOVE:   0.10,
    OPERATOR.MUTATE_DISPLACE: 0.10,
}

_OP_NAMES  = list(OPERATOR_PROBS.keys())
_OP_PROBS  = [OPERATOR_PROBS[k] for k in _OP_NAMES]

assert abs(sum(_OP_PROBS) - 1.0) < 1e-9, "OPERATOR_PROBS must sum to 1.0"

#: Number of parents required per operator.
OPERATOR_N_PARENTS: dict[str, int] = {
    OPERATOR.SPLICE:          2,
    OPERATOR.MERGE:           2,
    OPERATOR.MUTATE_ADD:      1,
    OPERATOR.MUTATE_REMOVE:   1,
    OPERATOR.MUTATE_DISPLACE: 1,
}


def sample_operator(rng: np.random.Generator | None = None) -> str:
    """
    Sample a GA operator according to OPERATOR_PROBS.

    Parameters
    ----------
    rng:
        NumPy random generator.

    Returns
    -------
    str
        One of the OPERATOR constants.
    """
    if rng is None:
        rng = np.random.default_rng()
    return str(rng.choice(_OP_NAMES, p=_OP_PROBS))


# ---------------------------------------------------------------------------
# Selection weights
# ---------------------------------------------------------------------------

def selection_weights(pool: list[Individual]) -> np.ndarray:
    """
    Extract normalised selection weights from a population pool.

    Parameters
    ----------
    pool:
        List of selectable Individuals.  Individuals with weight=0 are
        effectively excluded (zero probability).

    Returns
    -------
    np.ndarray
        Normalised weight array of shape (len(pool),) summing to 1.0.
        If all weights are zero, returns a uniform distribution so selection
        does not crash (degenerate case).
    """
    weights = np.array([ind.weight for ind in pool], dtype=float)
    total = weights.sum()
    if total == 0.0:
        # Degenerate: uniform over all
        return np.ones(len(pool)) / len(pool)
    return weights / total


def select_parents(
    pool: list[Individual],
    operator: str,
    rng: np.random.Generator | None = None,
) -> list[Individual]:
    """
    Select parents for the given operator by weighted sampling without
    replacement.

    Parameters
    ----------
    pool:
        List of selectable Individuals (should already be filtered to
        is_selectable == True by the caller).
    operator:
        OPERATOR constant string.  Determines how many parents to draw.
    rng:
        NumPy random generator.

    Returns
    -------
    list[Individual]
        Drawn parents, length == OPERATOR_N_PARENTS[operator].

    Raises
    ------
    ValueError
        If the pool has fewer individuals than the operator requires.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_parents = OPERATOR_N_PARENTS[operator]

    if len(pool) < n_parents:
        raise ValueError(
            f"Operator '{operator}' requires {n_parents} parents but "
            f"the selection pool has only {len(pool)} individual(s)."
        )

    weights = selection_weights(pool)
    indices = rng.choice(len(pool), size=n_parents, replace=False, p=weights)
    return [pool[i] for i in indices]
