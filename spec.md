# GOCIA — Grand Canonical Interface Optimization via Genetic Algorithm
## Architecture Summary & Technical Specification

---

## 1. Project Overview

GOCIA is a Python package for automated generation, optimization, and comparison of surface interface structures (slabs with adsorbates). It implements a genetic algorithm (GA) to sample composition space and minimize the grand canonical energy of a surface system under user-defined thermodynamic conditions. The workflow is designed to run on HPC clusters (Slurm or PBS) and is driven entirely by a single `gocia.yaml` configuration file.

**Core capabilities:**
- Generate initial populations of slab+adsorbate structures via random placement
- Evolve structures using graph-based splice, merge, and mutation operators
- Optimize structures through a staged calculator pipeline (MACE pre-opt → VASP coarse → VASP fine)
- Evaluate fitness using the Computational Hydrogen Electrode (CHE) framework
- Detect adsorbate desorption post-optimization via pluggable detector modules
- Detect and handle duplicate structures and near-duplicate isomers
- Track full genealogy of every structure through a SQLite database with a CLI and pandas interface
- Resume gracefully after HPC timeouts using sentinel files

---

## 2. Fitness Function

The fitness of each structure is its **grand canonical energy** under the CHE framework:

```
G = E_DFT - Σ_i [ n_i · μ_i(T, P, U, pH) ]
```

Where:
- `E_DFT` is the total DFT energy of the relaxed slab+adsorbate system
- `n_i` is the count of adsorbate species `i`
- `μ_i(T, P, U, pH)` is the chemical potential of species `i`, corrected for temperature, pressure, electrode potential (U vs RHE), and pH via CHE

The clean slab energy and all chemical potentials are specified in `gocia.yaml`. The raw DFT energy is stored separately from the corrected grand canonical energy in the database, allowing post-hoc re-ranking at different thermodynamic conditions without re-running calculations.

**Future extensions (not in scope for v1):** explicit solvation corrections (VASPsol), Pourbaix-style free energy corrections.

---

## 3. Directory Layout

```
run_dir/
├── gocia.yaml               # All run parameters
├── gocia.db                 # SQLite database (structures, runs, conditions)
├── gociastop                # Touch this file to trigger a graceful stop
├── gen_000/                 # Initial population
│   ├── struct_0001/
│   │   ├── POSCAR           # Input geometry
│   │   ├── CONTCAR          # Final converged geometry
│   │   ├── OUTCAR           # VASP output (if applicable)
│   │   ├── INCAR_1          # First VASP stage inputs
│   │   ├── INCAR_2          # Second VASP stage inputs (etc.)
│   │   ├── trajectory.h5    # All ionic steps, forces, stresses per stage
│   │   ├── PENDING          # Sentinel: not yet submitted
│   │   ├── SUBMITTED        # Sentinel: job submitted to scheduler
│   │   ├── RUNNING_1        # Sentinel: running calculator stage 1
│   │   ├── CONVERGED_1      # Sentinel: stage 1 complete
│   │   ├── RUNNING_2        # Sentinel: running calculator stage 2
│   │   ├── CONVERGED        # Sentinel: all stages complete
│   │   ├── DESORBED         # Sentinel: desorption detected post-opt
│   │   └── FAILED           # Sentinel: calculator error
│   └── struct_0002/
│       └── ...
├── gen_001/
│   └── ...
└── gen_002/
    └── ...
```

**Sentinel file convention:** only one sentinel file is present at a time. The main loop reads the sentinel file on restart to determine each structure's current state without querying the scheduler. This makes the loop robust to HPC session timeouts.

**trajectory.h5 layout:**
```
trajectory.h5
├── stage_mace_preopt/
│   ├── positions      (n_steps, n_atoms, 3)
│   ├── forces         (n_steps, n_atoms, 3)
│   ├── energies       (n_steps,)
│   └── stresses       (n_steps, 3, 3)
├── stage_vasp_coarse/
│   └── ...
└── stage_vasp_fine/
    └── ...
```

VASP input/output files (INCAR, KPOINTS, OUTCAR, CONTCAR, CHGCAR, etc.) are left as flat files in the structure folder as VASP produces them.

---

## 4. Package Structure

```
gocia/
├── README.md
├── pyproject.toml
├── gocia.yaml.example
│
├── tests/
│   ├── conftest.py               # Shared fixtures: toy slabs and adsorbate Atoms objects (no MACE/VASP required)
│   ├── test_graph_operators.py   # Splice/merge: stoichiometry preservation, clash removal
│   ├── test_desorption.py        # Distance-based desorption detection
│   ├── test_fingerprint.py       # Duplicate and isomer detection
│   ├── test_fitness.py           # CHE grand canonical energy, condition sweeps
│   ├── test_population.py        # Selection weights, isomer weighting
│   ├── test_placement.py         # Adsorbate placement, orientation sampling
│   └── test_database.py          # DB read/write, schema, run tracking
│
└── gocia/
    ├── __init__.py
    ├── config.py                 # Load + validate gocia.yaml → GociaConfig dataclass
    ├── cli.py                    # Click CLI: init, run, status, inspect, stop
    │
    ├── structure/
    │   ├── __init__.py
    │   ├── slab.py               # Load slab, selective dynamics handling, sampling region definition
    │   ├── adsorbate.py          # Load adsorbate geometries from file or coordinate list
    │   ├── placement.py          # ASE add_adsorbate wrapper, multi-orientation sampling + pre-opt ranking
    │   └── fingerprint.py        # Distance histogram fingerprint, SOAP wrapper, RMSD comparison
    │
    ├── population/
    │   ├── __init__.py
    │   ├── individual.py         # Individual pydantic model: atoms, metadata, genealogy, status
    │   ├── population.py         # Population: tournament selection, isomer weighting, weight updates
    │   └── initializer.py        # Random initial population generation
    │
    ├── operators/
    │   ├── __init__.py
    │   ├── base.py               # Abstract GeneticOperator base class
    │   ├── graph_splice.py       # Adjacency graph construction + graph-cut splice operator
    │   ├── graph_merge.py        # Merge two halves, distance-based clash removal, stoichiometry check
    │   └── mutation.py           # Random add / remove / displace adsorbate atoms
    │
    ├── calculator/
    │   ├── __init__.py
    │   ├── stage.py              # CalculatorStage dataclass: calculator type, ASE filter, fmax, max_steps
    │   ├── pipeline.py           # Run ordered list of CalculatorStages, write trajectory.h5, manage sentinels
    │   ├── mace_calc.py          # MACE-MP-0 setup + ASE optimizer wrapper
    │   └── vasp_calc.py          # VASP via ASE, default INCAR dict + user overrides per stage
    │
    ├── scheduler/
    │   ├── __init__.py
    │   ├── base.py               # Abstract Scheduler base class
    │   ├── slurm.py              # Slurm: sbatch submission, squeue monitoring, status parsing
    │   ├── pbs.py                # PBS: qsub submission, qstat monitoring, status parsing
    │   └── local.py              # Local runner for testing without HPC
    │
    ├── desorption/
    │   ├── __init__.py
    │   ├── base.py               # Abstract DesorptionDetector base class; declares stage = "post_opt"
    │   └── distance.py           # Naive distance cutoff detector (default, no extra dependencies)
    │
    ├── fitness/
    │   ├── __init__.py
    │   └── che.py                # CHE grand canonical energy; condition corrections for U, pH, T, P
    │
    ├── database/
    │   ├── __init__.py
    │   ├── schema.py             # SQLite schema definitions: structures, runs, conditions tables
    │   ├── db.py                 # DB connection, CRUD operations, pandas DataFrame interface
    │   └── status.py             # Sentinel file read / write / clear helpers
    │
    └── runner/
        ├── __init__.py
        └── loop.py               # Main GA loop: submit → monitor → collect → select → reproduce
```

---

## 5. Configuration: gocia.yaml

```yaml
slab:
  geometry: slab.vasp               # Path to slab geometry file (VASP format, selective dynamics pre-defined)
  energy: -42.31                    # DFT energy of clean slab (eV)
  sampling_zmin: 8.0                # Min z-height for adsorbate placement (Å)
  sampling_zmax: 12.0               # Max z-height for adsorbate placement (Å)
  chemical_potentials:              # Per-element potentials for surface stoichiometry changes (future)
    Pt: -6.05

adsorbates:
  - symbol: O
    chemical_potential: -4.92       # eV (referenced to standard state)
    n_orientations: 1               # Single atom: orientation sampling not needed
  - symbol: OH
    chemical_potential: -3.75
    geometry: OH.vasp               # Path to molecule geometry file (or omit for single atom)
    n_orientations: 6               # Number of orientations to trial at placement

calculator_stages:
  - name: mace_preopt
    type: mace
    fmax: 0.10
    max_steps: 300
  - name: vasp_coarse
    type: vasp
    fmax: 0.05
    max_steps: 100
    incar:
      ENCUT: 400
      EDIFF: 1.0e-4
      NSW: 100
      IBRION: 2
  - name: vasp_fine
    type: vasp
    fmax: 0.02
    max_steps: 200
    incar:
      ENCUT: 520
      EDIFF: 1.0e-6
      NSW: 200
      IBRION: 2

scheduler:
  type: slurm                       # slurm | pbs | local
  nworkers: 20                      # Max concurrent jobs
  walltime: "04:00:00"
  partition: gpu
  extra_headers:                    # Any extra lines to prepend to job scripts
    - "#SBATCH --mem=32G"

ga:
  population_size: 30
  max_generations: 50
  min_generations: 5
  max_stall_generations: 10         # Stop if best fitness unchanged for this many generations
  isomer_weight: 0.01               # Selection weight assigned to isomers (vs 1.0 for unique structures)

conditions:
  temperature: 298.15               # K (accepts int or float)
  pressure: 1.0                     # atm
  potential: 0.0                    # V vs RHE
  pH: 0                             # (accepts int or float)
```

---

## 6. Pydantic Configuration Models

All models live in `gocia/config.py`. They are intentionally shallow — type hints and defaults only, no custom validators.

```python
class AdsorbateConfig(BaseModel):
    symbol: str
    chemical_potential: float
    geometry: str | None = None
    n_orientations: int = 6

class SlabConfig(BaseModel):
    geometry: str
    energy: float
    sampling_zmin: float
    sampling_zmax: float
    chemical_potentials: dict[str, float] = {}

class CalculatorStageConfig(BaseModel):
    name: str
    type: str                        # "mace" | "vasp"
    fmax: float = 0.05
    max_steps: int = 500
    incar: dict | None = None        # VASP only

class SchedulerConfig(BaseModel):
    type: str                        # "slurm" | "pbs" | "local"
    nworkers: int
    walltime: str
    partition: str | None = None
    extra_headers: list[str] = []

class GAConfig(BaseModel):
    population_size: int
    max_generations: int
    min_generations: int
    max_stall_generations: int
    isomer_weight: float = 0.01

class ConditionsConfig(BaseModel):
    temperature: float = 298.15
    pressure: float = 1.0
    potential: float = 0.0
    pH: float = 0.0

class GociaConfig(BaseModel):
    slab: SlabConfig
    adsorbates: list[AdsorbateConfig]
    calculator_stages: list[CalculatorStageConfig]
    scheduler: SchedulerConfig
    ga: GAConfig
    conditions: ConditionsConfig
```

Note: pydantic v2 automatically coerces `int → float` for all `float` fields, so user inputs like `pH: 7` or `temperature: 298` are handled without additional validators.

---

## 7. Individual Model

Lives in `gocia/population/individual.py`.

```python
class Individual(BaseModel):
    id: str                          # UUID
    generation: int
    parent_ids: list[str] = []
    operator: str = "init"           # "init" | "splice" | "merge" | "mutate_add" | "mutate_remove" | "mutate_displace"
    status: str = "pending"          # See status lifecycle below
    raw_energy: float | None = None
    grand_canonical_energy: float | None = None
    weight: float = 1.0
    fingerprint: list[float] | None = None
    geometry_path: str | None = None # Path to final CONTCAR / converged geometry
    desorption_flag: bool = False

    class Config:
        extra = "allow"              # Forward-compatible with future fields
```

**Status lifecycle:**
```
pending → submitted → running_stage_{n} → converged_stage_{n} → ... → converged
                                                                      → desorbed
                                                                      → failed
converged → duplicate
converged → isomer
```

Each status maps 1:1 to a sentinel filename in the structure directory (e.g. status `running_stage_1` → file `RUNNING_1`, status `converged` → file `CONVERGED`).

---

## 8. Database Schema

Lives in `gocia/database/schema.py`. Three tables.

**`structures` table**
| Column | Type | Notes |
|---|---|---|
| id | TEXT PK | UUID |
| generation | INTEGER | |
| parent_ids | TEXT | JSON-encoded list |
| operator | TEXT | |
| status | TEXT | |
| raw_energy | REAL | DFT total energy (eV) |
| grand_canonical_energy | REAL | CHE-corrected (eV), at run conditions |
| weight | REAL | Selection weight |
| fingerprint | TEXT | JSON-encoded float list |
| geometry_path | TEXT | Path to final geometry |
| desorption_flag | INTEGER | 0 or 1 |
| is_isomer | INTEGER | 0 or 1 |
| isomer_of | TEXT | ID of the representative structure |

**`runs` table**
| Column | Type | Notes |
|---|---|---|
| id | INTEGER PK | Auto-increment |
| timestamp | TEXT | ISO 8601 |
| generation_start | INTEGER | |
| generation_end | INTEGER | null if still running |
| temperature | REAL | K |
| pressure | REAL | atm |
| potential | REAL | V vs RHE |
| pH | REAL | |
| notes | TEXT | Optional user annotation |

**`conditions` table** (for named condition sets used in post-hoc analysis)
| Column | Type | Notes |
|---|---|---|
| id | INTEGER PK | |
| name | TEXT | e.g. "acidic_low_U" |
| temperature | REAL | |
| pressure | REAL | |
| potential | REAL | |
| pH | REAL | |

---

## 9. GA Operators

### Graph-Based Splice and Merge

Implemented in `gocia/operators/graph_splice.py` and `graph_merge.py`.

**Graph construction:**
- Nodes: adsorbate atoms only (frozen slab atoms are excluded from the graph but retained in the structure)
- Edges: pairs of atoms within a distance threshold (sum of covalent radii × scale factor, typically 1.2)
- Result: an adjacency graph over the mobile adsorbate layer

**Splice (two parents → two offspring):**
1. Build the adsorbate connectivity graph for each parent
2. Identify a geometric cut plane (or graph partition boundary) that divides the sampling region
3. Assign each adsorbate node to a half based on its centroid position
4. Swap the two halves between parents
5. Check stoichiometry: atom counts of each species must be preserved
6. Apply distance-based clash removal at the boundary: if two atoms are closer than a minimum distance threshold, remove the one with the higher energy contribution (or the one from the inserted half, as a simple tiebreaker)
7. If stoichiometry is violated after clash removal, add or remove atoms of the appropriate species at valid sites

**Merge** follows the same logic with a single offspring combining the best-scoring half from each parent.

**Mutation** (`gocia/operators/mutation.py`):
- `mutate_add`: place one adsorbate atom/molecule at a random valid site using ASE placement
- `mutate_remove`: remove one randomly selected adsorbate atom/molecule
- `mutate_displace`: displace one adsorbate atom/molecule to a new random site

All operators preserve stoichiometry. A test suite in `tests/test_graph_operators.py` verifies atom counts before and after every operation using pure-geometry fixtures (no MACE or VASP required).

---

## 10. Calculator Pipeline

Implemented in `gocia/calculator/pipeline.py`.

A pipeline is an ordered list of `CalculatorStage` objects defined in `gocia.yaml`. Each stage:
1. Reads the geometry from the previous stage's output (or initial placement for stage 1)
2. Runs the ASE optimizer with the specified calculator, filter, `fmax`, and `max_steps`
3. Writes ionic step data (positions, forces, energies, stresses) to `trajectory.h5` under a group named after the stage
4. Updates the sentinel file to `CONVERGED_{n}` on success or `FAILED` on error
5. Passes the relaxed geometry to the next stage

**VASP stages** use a default INCAR dictionary merged with user overrides from `gocia.yaml`. Multiple VASP stages allow a coarse-to-fine convergence strategy. VASP input/output files are left as flat files in the structure directory.

**MACE stage** uses MACE-MP-0 (universal potential) via the `mace-torch` package. This stage can be used alone for prototyping without VASP.

**ASE filters** (e.g. `FrechetCellFilter`, `StrainFilter`, or unconstrained positions) can be specified per stage in the config.

---

## 11. Adsorbate Placement

Implemented in `gocia/structure/placement.py`.

Uses ASE's `add_adsorbate` and `Atoms` manipulation tools. For each placement:
1. Sample a random (x, y) position within the surface unit cell and within the z sampling region defined in the slab config
2. For molecules: trial `n_orientations` random rotations
3. Run a MACE pre-opt (or a fast single-point energy evaluation) for each orientation
4. Select the orientation with the lowest energy as the placed geometry

Reference geometries for molecules are provided as either a path to a geometry file (VASP, XYZ, etc.) or a list of coordinates in `gocia.yaml`. Internal degrees of freedom are unconstrained during optimization.

---

## 12. Fingerprinting and Duplicate Detection

Implemented in `gocia/structure/fingerprint.py`.

**Pre-submission (fast):** sorted interatomic distance histogram over the adsorbate layer. O(N²) but cheap. Used to avoid submitting obvious duplicates.

**Post-relaxation (accurate):** RMSD after optimal alignment (using ASE or spglib), with optional SOAP descriptor comparison via DScribe for more robust isomer detection.

**Duplicate:** fingerprint distance below a tight threshold → status set to `duplicate`, structure logged but excluded from selection.

**Isomer:** fingerprint distance below a looser threshold but above the duplicate threshold → status set to `isomer`, structure kept in selection pool with `weight = ga.isomer_weight` (default 0.01). The `isomer_of` field in the DB records the representative structure's ID.

Both checks run pre-submission and post-relaxation. All structures (including duplicates and isomers) are retained in the database and on disk.

---

## 13. Desorption Detection

Implemented in `gocia/desorption/`. Runs post-optimization.

**Abstract base class** (`base.py`) defines the interface:
```python
class DesorptionDetector:
    stage = "post_opt"
    def detect(self, atoms: Atoms, slab: Atoms) -> bool:
        raise NotImplementedError
```

**Distance detector** (`distance.py`): checks if any adsorbate atom is more than a cutoff distance from the nearest slab atom. Configurable per adsorbate species. No extra dependencies.

**Planned (not v1):** Bader charge connectivity graph detector, requiring CHGCAR from VASP.

When desorption is detected:
- Status set to `desorbed`, sentinel file `DESORBED` written
- Structure logged in the database with `desorption_flag = True`
- Structure **not** re-queued and **not** used in selection
- Used as a thermodynamic reference point; its energy is recorded

---

## 14. CLI

Implemented in `gocia/cli.py` using Click.

| Command | Description |
|---|---|
| `gocia init` | Validate `gocia.yaml`, create directory structure, initialize `gocia.db`, generate a fully commented `gocia.yaml` template if no config found |
| `gocia run` | Start or resume the main GA loop. Reads sentinel files to reconstruct state on restart |
| `gocia status` | Print current generation, number of converged/running/pending/failed structures, best grand canonical energy |
| `gocia inspect` | Query the database; returns a pandas-friendly table. Re-ranks by grand canonical energy at specified conditions by default |
| `gocia stop` | Write a `gociastop` file; the main loop exits gracefully after the current generation completes |

**`gocia inspect` flags:**
```bash
gocia inspect --generation 5           # Filter by generation
gocia inspect --status converged       # Filter by status
gocia inspect --potential -0.5 --pH 7  # Re-rank at these conditions
gocia inspect --no-rerank              # Skip condition re-ranking, use stored energies
gocia inspect --top 10                 # Return top N by fitness
gocia inspect --output results.csv     # Export to CSV
```

---

## 15. Main GA Loop

Implemented in `gocia/runner/loop.py`.

```
1. Load config and initialize DB (or resume from existing DB)
2. Log this run in the `runs` table
3. If gen_000 does not exist: generate initial population via random placement
4. LOOP per generation:
   a. Check for gociastop file → exit gracefully if present
   b. Walk current generation folder, read sentinel files, update DB
   c. Submit pending structures up to nworkers limit
   d. Poll scheduler for running jobs; update sentinels and DB on completion
   e. For newly completed structures:
      - Run desorption detection
      - Run post-relaxation fingerprint check (duplicate / isomer)
      - Compute grand canonical fitness
      - Update weight in DB
   f. Check convergence criteria:
      - min_generations reached AND (max_generations reached OR max_stall_generations exceeded)
   g. Select parents via weighted selection from converged pool
   h. Apply GA operators (splice / merge / mutate) to generate offspring
   i. Run pre-submission fingerprint check on offspring
   j. Create next generation directory, write offspring geometries and PENDING sentinels
   k. Insert offspring into DB with genealogy metadata
5. On exit: update `runs` table with final generation number
```

**Convergence criteria (all must be satisfied to stop):**
- Current generation ≥ `ga.min_generations`
- Current generation ≥ `ga.max_generations` OR best grand canonical energy has not improved for `ga.max_stall_generations` consecutive generations

---

## 16. Key Dependencies

| Package | Role |
|---|---|
| `ase` | Atoms objects, calculators, optimizers, placement tools |
| `pydantic` (v2) | Config and Individual model validation |
| `click` | CLI |
| `mace-torch` | MACE-MP-0 universal pre-optimizer |
| `h5py` | trajectory.h5 read/write |
| `numpy` | Array operations |
| `pandas` | Database query interface |
| `dscribe` | SOAP descriptors (optional, for post-relaxation fingerprinting) |
| `pytest` | Test suite |

**No SGE support.** Schedulers: Slurm and PBS only, plus a local runner for testing.

---

## 17. Testing Strategy

All tests in `tests/` are runnable without MACE or VASP. Fixtures in `conftest.py` provide:
- A small toy FCC(111) slab (ASE Atoms object, ~20 atoms, with selective dynamics tags)
- A set of pre-placed adsorbate Atoms objects (O atom, OH molecule)
- A small pre-relaxed population of Individual objects

**Test coverage targets:**
- `test_graph_operators.py`: atom count invariance before/after splice, merge, and all mutation types; clash removal does not create unphysical geometries; stoichiometry correction edge cases
- `test_desorption.py`: distance detector correctly flags desorbed vs. adsorbed geometries
- `test_fingerprint.py`: identical structures → duplicate; slightly displaced → isomer; very different → unique
- `test_fitness.py`: CHE energy calculation at multiple (U, pH) conditions; int inputs for pH/T coerced correctly
- `test_population.py`: isomer weight applied correctly; selection probabilities sum to 1
- `test_placement.py`: placed atoms are within sampling z-range; orientation with lowest energy is selected
- `test_database.py`: structure insert/update/query round-trips; run tracking; pandas interface returns correct dtypes

---

## 18. Future Extensions (Not in v1 Scope)

- **Bader charge connectivity desorption detector** (requires CHGCAR)
- **VASPsol** implicit solvation corrections
- **Pourbaix-style free energy corrections**
- **Surface stoichiometry mutations** (already architected for via `SlabConfig.chemical_potentials`)
- **SOAP/RMSD-based fingerprinting** (DScribe dependency, optional from day one)
- **Fine-tuned MACE models** trained on trajectory.h5 data accumulated during runs
- **Job array submission** (Slurm/PBS arrays) as an alternative to individual submissions
- **Web dashboard** for run monitoring (the SQLite DB makes this straightforward)
