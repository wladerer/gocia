"""
gocia/population/initializer.py

Random initial population generation.

Builds gen_000 by placing adsorbates randomly on the slab, writing POSCAR
files, inserting Individual records into the DB, and writing PENDING sentinels.

The initial population can have variable composition: each Individual is
independently assigned a random stoichiometry drawn from the adsorbate
species defined in the config, subject to a per-species count range that
can be specified (defaults to 1–3 of each species).

Public API
----------
    build_initial_population(config, slab_info, db, run_dir) → list[Individual]
"""

from __future__ import annotations

import logging
from pathlib import Path
from collections import Counter

import numpy as np
from ase import Atoms
from ase.io import write

from gocia.population.individual import Individual, STATUS, OPERATOR
from gocia.database.status import write_sentinel
from gocia.structure.adsorbate import load_adsorbate
from gocia.structure.placement import place_adsorbate
from gocia.structure.slab import SlabInfo

logger = logging.getLogger(__name__)

# Default adsorbate count range per species for the initial population
_DEFAULT_MIN_COUNT = 1
_DEFAULT_MAX_COUNT = 3


def build_initial_population(
    config,             # GociaConfig
    slab_info: SlabInfo,
    db,                 # GociaDB (already connected)
    run_dir: Path,
    rng: np.random.Generator | None = None,
) -> list[Individual]:
    """
    Generate the initial random population and persist everything to disk and DB.

    For each Individual in the population:
      1. Draw a random adsorbate stoichiometry.
      2. Place adsorbates one at a time using place_adsorbate().
      3. Write the geometry as POSCAR in the structure directory.
      4. Compute a pre-submission fingerprint.
      5. Write the PENDING sentinel.
      6. Insert an Individual record into the DB.

    Parameters
    ----------
    config:
        Validated GociaConfig.
    slab_info:
        SlabInfo for the bare slab (from load_slab()).
    db:
        Connected GociaDB.  Records are inserted here.
    run_dir:
        Root run directory.  gen_000/ is created here.
    rng:
        NumPy random generator for reproducibility.

    Returns
    -------
    list[Individual]
        All newly created Individual objects (status=PENDING).
    """
    if rng is None:
        rng = np.random.default_rng()

    gen_dir = run_dir / "gen_000"
    gen_dir.mkdir(exist_ok=True)
    logger.info(f"Building initial population of {config.ga.population_size} structures")

    # Pre-load adsorbate geometries
    adsorbate_atoms: dict[str, Atoms] = {
        ads.symbol: load_adsorbate(ads)
        for ads in config.adsorbates
    }

    # Build chemical potential mapping for fitness calculation
    chemical_potentials: dict[str, float] = {
        ads.symbol: ads.chemical_potential
        for ads in config.adsorbates
    }

    individuals: list[Individual] = []

    for i in range(config.ga.population_size):
        struct_id = f"{i + 1:04d}"
        struct_dir = gen_dir / f"struct_{struct_id}"
        struct_dir.mkdir(exist_ok=True)

        # Draw random stoichiometry
        adsorbate_counts = _random_stoichiometry(config.adsorbates, rng)

        # Place adsorbates sequentially on the slab
        current_atoms = slab_info.atoms.copy()
        try:
            for symbol, count in adsorbate_counts.items():
                ads_template = adsorbate_atoms[symbol]
                ads_cfg = next(a for a in config.adsorbates if a.symbol == symbol)
                for _ in range(count):
                    current_atoms = place_adsorbate(
                        slab=current_atoms,
                        adsorbate=ads_template,
                        zmin=slab_info.sampling_zmin,
                        zmax=slab_info.sampling_zmax,
                        n_orientations=ads_cfg.n_orientations,
                        rng=rng,
                    )
        except RuntimeError as exc:
            logger.warning(
                f"  struct_{struct_id}: placement failed ({exc}). "
                "Trying with reduced stoichiometry."
            )
            # Retry with single adsorbate of first species
            first_sym = next(iter(adsorbate_counts))
            adsorbate_counts = {first_sym: 1}
            current_atoms = slab_info.atoms.copy()
            current_atoms = place_adsorbate(
                slab=current_atoms,
                adsorbate=adsorbate_atoms[first_sym],
                zmin=slab_info.sampling_zmin,
                zmax=slab_info.sampling_zmax,
                n_orientations=1,
                rng=rng,
            )

        # Write POSCAR
        poscar_path = struct_dir / "POSCAR"
        write(str(poscar_path), current_atoms, format="vasp")

        # Pre-submission fingerprint
        from gocia.structure.fingerprint import distance_histogram
        fp = distance_histogram(current_atoms)

        # Build Individual
        ind = Individual.from_init(
            generation=0,
            geometry_path=str(poscar_path),
            fingerprint=fp,
            extra_data={"adsorbate_counts": dict(adsorbate_counts)},
        )

        # Write PENDING sentinel
        write_sentinel(struct_dir, STATUS.PENDING)

        # Insert into DB
        db.insert(ind)
        individuals.append(ind)

        logger.debug(
            f"  struct_{struct_id}: "
            f"{dict(adsorbate_counts)} → {poscar_path}"
        )

    logger.info(f"Initial population created: {len(individuals)} structures in {gen_dir}")
    return individuals


def _random_stoichiometry(
    adsorbate_configs: list,
    rng: np.random.Generator,
    min_count: int = _DEFAULT_MIN_COUNT,
    max_count: int = _DEFAULT_MAX_COUNT,
) -> dict[str, int]:
    """
    Draw a random adsorbate stoichiometry.

    For each adsorbate species, draws a random integer count in
    [min_count, max_count].  At least one species always has count >= 1
    (prevents generating bare slabs).

    Parameters
    ----------
    adsorbate_configs:
        List of AdsorbateConfig from GociaConfig.
    rng:
        NumPy random generator.
    min_count, max_count:
        Inclusive range for per-species count.

    Returns
    -------
    dict[str, int]
        Symbol → count mapping with all counts >= min_count.
    """
    counts = {}
    for ads in adsorbate_configs:
        counts[ads.symbol] = int(rng.integers(min_count, max_count + 1))

    # Safety: ensure at least one adsorbate total
    if sum(counts.values()) == 0:
        first = adsorbate_configs[0].symbol
        counts[first] = 1

    return counts
