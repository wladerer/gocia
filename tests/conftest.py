"""
tests/conftest.py

Shared pytest fixtures for the GOCIA test suite.

All fixtures here are pure-geometry — no MACE, VASP, or network access is
required.  Tests that need those calculators should be marked with
@pytest.mark.mace or @pytest.mark.vasp and skipped in CI.

Fixture overview
----------------
Slabs
    bare_slab           Small Pt(111) 2x2 slab, bottom two layers frozen
    slab_config         SlabConfig pointing at the bare slab

Adsorbates
    o_atom              Single O atom (ASE Atoms)
    oh_molecule         OH molecule (ASE Atoms)
    adsorbate_configs   List of AdsorbateConfig for O and OH

Populations
    one_individual      A single pending Individual (gen 0, init)
    small_population    List of 5 Individuals at various statuses/energies

Database
    tmp_db              A GociaDB connected to a temp file with tables created
    populated_db        tmp_db pre-loaded with small_population

Config
    full_config         A complete GociaConfig built from minimal YAML
    run_dir             A tmp_path-based run directory with gocia.yaml written
"""

from __future__ import annotations

import json
import textwrap
import uuid
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# ASE import guard — ASE is a hard dependency so fail loudly if missing
# ---------------------------------------------------------------------------
try:
    import numpy as np
    from ase import Atoms
    from ase.build import fcc111, add_adsorbate
    from ase.constraints import FixAtoms
except ImportError as exc:
    pytest.exit(f"ASE is required to run the test suite: {exc}", returncode=1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pt111(layers: int = 4, vacuum: float = 10.0, size: tuple = (2, 2)) -> Atoms:
    """
    Build a small Pt(111) slab with the bottom half frozen.

    Returns an ASE Atoms object with:
      - selective dynamics equivalent (FixAtoms on lower layers)
      - enough vacuum for adsorbate placement
    """
    slab = fcc111("Pt", size=size + (layers,), vacuum=vacuum)
    slab.center(vacuum=vacuum, axis=2)

    # Freeze the bottom half of the slab (layers 0 and 1 in a 4-layer slab)
    n_frozen = len(slab) // 2
    z_sorted = sorted(set(slab.positions[:, 2]))
    frozen_z_cutoff = z_sorted[len(z_sorted) // 2]
    frozen_indices = [i for i, pos in enumerate(slab.positions) if pos[2] <= frozen_z_cutoff]
    slab.set_constraint(FixAtoms(indices=frozen_indices))
    return slab


def _make_oh() -> Atoms:
    """Build a simple OH molecule at a reasonable bond length."""
    oh = Atoms(
        "OH",
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.97]],
    )
    return oh


# ---------------------------------------------------------------------------
# Slab fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def bare_slab() -> Atoms:
    """
    A 2x2 Pt(111) 4-layer slab with bottom 2 layers frozen.
    Session-scoped: built once and shared across all tests.
    """
    return _make_pt111(layers=4, vacuum=10.0, size=(2, 2))


@pytest.fixture(scope="session")
def slab_with_o(bare_slab) -> Atoms:
    """Pt(111) slab with a single O atom adsorbed at an fcc hollow site."""
    slab = bare_slab.copy()
    add_adsorbate(slab, "O", height=1.2, position="fcc")
    return slab


@pytest.fixture(scope="session")
def slab_with_oh(bare_slab) -> Atoms:
    """Pt(111) slab with an OH molecule adsorbed."""
    slab = bare_slab.copy()
    oh = _make_oh()
    add_adsorbate(slab, oh, height=1.3, position="fcc")
    return slab


@pytest.fixture(scope="session")
def slab_z_bounds(bare_slab) -> tuple[float, float]:
    """
    (zmin, zmax) for the adsorbate sampling region, computed from the slab.
    zmin is just above the topmost slab atom; zmax adds 4 Å headroom.
    """
    top_z = bare_slab.positions[:, 2].max()
    return (top_z + 0.5, top_z + 4.0)


# ---------------------------------------------------------------------------
# Adsorbate fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def o_atom() -> Atoms:
    """Single oxygen atom."""
    return Atoms("O", positions=[[0.0, 0.0, 0.0]])


@pytest.fixture(scope="session")
def oh_molecule() -> Atoms:
    """OH molecule at a typical bond length."""
    return _make_oh()


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def slab_config_dict(bare_slab, slab_z_bounds, tmp_path_factory) -> dict:
    """
    A minimal slab config dict.  Writes the slab geometry to a temp file
    so SlabConfig.geometry points to a real path.
    """
    tmp = tmp_path_factory.mktemp("slab")
    slab_path = tmp / "slab.vasp"

    # Write a minimal POSCAR-style file via ASE
    from ase.io import write
    write(str(slab_path), bare_slab, format="vasp")

    zmin, zmax = slab_z_bounds
    return {
        "geometry": str(slab_path),
        "energy": -120.0,
        "sampling_zmin": float(zmin),
        "sampling_zmax": float(zmax),
        "chemical_potentials": {"Pt": -6.05},
    }


@pytest.fixture(scope="session")
def full_config_dict(slab_config_dict) -> dict:
    """A complete gocia.yaml-equivalent dict suitable for GociaConfig.model_validate."""
    return {
        "slab": slab_config_dict,
        "adsorbates": [
            {"symbol": "O",  "chemical_potential": -4.92, "n_orientations": 1},
            {"symbol": "OH", "chemical_potential": -3.75, "n_orientations": 4},
        ],
        "calculator_stages": [
            {"name": "mace_preopt", "type": "mace", "fmax": 0.10, "max_steps": 50},
            {"name": "vasp_fine",   "type": "vasp", "fmax": 0.02, "max_steps": 10,
             "incar": {"ENCUT": 400, "EDIFF": 1e-4}},
        ],
        "scheduler": {
            "type": "local",
            "nworkers": 2,
            "walltime": "00:10:00",
        },
        "ga": {
            "population_size": 5,
            "max_generations": 3,
            "min_generations": 1,
            "max_stall_generations": 2,
            "isomer_weight": 0.01,
        },
        "conditions": {
            "temperature": 298,   # int — should be coerced to float
            "pressure": 1.0,
            "potential": 0.0,
            "pH": 7,              # int — should be coerced to float
        },
    }


@pytest.fixture(scope="session")
def full_config(full_config_dict):
    """A validated GociaConfig object."""
    from gocia.config import GociaConfig
    return GociaConfig.model_validate(full_config_dict)


# ---------------------------------------------------------------------------
# Individual fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def one_individual() -> "Individual":
    """A single pending Individual at generation 0."""
    from gocia.population.individual import Individual
    return Individual(generation=0)


@pytest.fixture
def small_population() -> list:
    """
    A list of 5 Individuals representing a realistic mix of statuses.

    ind 0 — converged,  low energy  (best)
    ind 1 — converged,  mid energy
    ind 2 — isomer of ind 0,        low weight
    ind 3 — desorbed,               weight 0
    ind 4 — pending,                no energy yet
    """
    from gocia.population.individual import Individual, STATUS, OPERATOR

    base_id = uuid.uuid4().hex

    ind0 = Individual(
        id=base_id,
        generation=0,
        operator=OPERATOR.INIT,
        status=STATUS.CONVERGED,
        raw_energy=-125.0,
        grand_canonical_energy=-10.5,
        weight=1.0,
        extra_data={"adsorbate_counts": {"O": 2}},
    )
    ind1 = Individual(
        generation=0,
        operator=OPERATOR.INIT,
        status=STATUS.CONVERGED,
        raw_energy=-124.0,
        grand_canonical_energy=-9.8,
        weight=1.0,
        extra_data={"adsorbate_counts": {"O": 2}},
    )
    ind2 = Individual(
        generation=0,
        operator=OPERATOR.INIT,
        status=STATUS.ISOMER,
        raw_energy=-124.9,
        grand_canonical_energy=-10.4,
        weight=0.01,
        is_isomer=True,
        isomer_of=base_id,
        extra_data={"adsorbate_counts": {"O": 2}},
    )
    ind3 = Individual(
        generation=0,
        operator=OPERATOR.INIT,
        status=STATUS.DESORBED,
        raw_energy=-122.0,
        grand_canonical_energy=None,
        weight=0.0,
        desorption_flag=True,
        extra_data={"adsorbate_counts": {"O": 1}},
    )
    ind4 = Individual(
        generation=1,
        parent_ids=[ind0.id, ind1.id],
        operator=OPERATOR.SPLICE,
        status=STATUS.PENDING,
        weight=1.0,
    )
    return [ind0, ind1, ind2, ind3, ind4]


# ---------------------------------------------------------------------------
# Database fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db(tmp_path):
    """
    A GociaDB connected to a fresh temp SQLite file with tables created.
    Closed automatically after each test.
    """
    from gocia.database.db import GociaDB
    db_path = tmp_path / "gocia.db"
    db = GociaDB(db_path)
    db.connect()
    db.setup()
    yield db
    db.close()


@pytest.fixture
def populated_db(tmp_db, small_population):
    """tmp_db pre-loaded with the small_population fixture."""
    for ind in small_population:
        tmp_db.insert(ind)
    return tmp_db


# ---------------------------------------------------------------------------
# Filesystem fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def run_dir(tmp_path, full_config_dict):
    """
    A temporary run directory with gocia.yaml written.
    Mimics what `gocia init` would produce.
    """
    import yaml
    config_path = tmp_path / "gocia.yaml"
    config_path.write_text(yaml.dump(full_config_dict))
    return tmp_path


@pytest.fixture
def gen0_dir(run_dir) -> Path:
    """
    A gen_000 directory with 3 structure subdirectories, each with a
    PENDING sentinel file.  Simulates the state just after initial population
    generation.
    """
    from gocia.database.status import write_sentinel
    from gocia.population.individual import STATUS

    gen_dir = run_dir / "gen_000"
    gen_dir.mkdir()

    for i in range(1, 4):
        struct_dir = gen_dir / f"struct_{i:04d}"
        struct_dir.mkdir()
        write_sentinel(struct_dir, STATUS.PENDING)

    return gen_dir


@pytest.fixture
def struct_dir(tmp_path) -> Path:
    """A bare structure directory with no sentinel files."""
    d = tmp_path / "struct_test"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# Geometry helpers available to tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def make_slab():
    """
    Factory fixture: returns a function that builds a Pt(111) slab.
    Useful when a test needs multiple slabs with different parameters.

    Usage in a test::

        def test_something(make_slab):
            slab = make_slab(layers=3, size=(2, 2))
    """
    return _make_pt111


@pytest.fixture(scope="session")
def make_oh():
    """Factory fixture for building OH molecules."""
    return _make_oh
