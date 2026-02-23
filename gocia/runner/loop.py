"""
gocia/runner/loop.py

Main GA loop for GOCIA.

Entry point: run(config, run_dir)

Loop structure (per the spec, Section 15)
------------------------------------------
1.  Load config → initialise or resume DB and scheduler.
2.  Log this run in the `runs` table.
3.  If gen_000 missing → build_initial_population().
4.  Per generation:
    a.  Check gociastop → exit gracefully.
    b.  Scan generation sentinel files → sync DB status.
    c.  Submit PENDING structures up to nworkers.
    d.  Poll scheduler → update sentinels + DB for finished jobs.
    e.  Process newly converged structures:
          - desorption detection
          - post-relaxation fingerprint (duplicate / isomer)
          - CHE fitness
          - DB weight update
    f.  Check stop criteria.
    g.  If stopping → close out and return.
    h.  Select parents, sample operator, apply operator → offspring.
    i.  Pre-submission fingerprint dedup on offspring.
    j.  Write offspring to next gen_dir, insert into DB.
5.  Update `runs` table with final generation.

Restart behaviour
-----------------
Because every state transition is recorded in both sentinel files and the DB,
calling run() on an already-started run directory is safe and idempotent.
The loop will:
  - Skip structure directories that already have a terminal sentinel.
  - Re-submit any structure stuck in RUNNING_N (the pipeline handles restart).
  - Resume from the current generation rather than regenerating gen_000.

Graceful stop
-------------
Touching `<run_dir>/gociastop` causes the loop to exit cleanly after the
current generation's submissions are done and before the next generation is
created.  The runs table is updated, all sentinels are left intact.
"""

from __future__ import annotations

import logging
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from ase.io import read, write

from gocia.calculator.stage import build_pipeline
from gocia.config import GociaConfig
from gocia.database.db import GociaDB
from gocia.database.status import (
    read_sentinel,
    scan_generation,
    write_sentinel,
)
from gocia.desorption.distance import DistanceDesorptionDetector
from gocia.fitness.che import grand_canonical_energy
from gocia.population.individual import OPERATOR, STATUS, Individual
from gocia.population.initializer import build_initial_population
from gocia.population.population import (
    OPERATOR_N_PARENTS,
    sample_operator,
    select_parents,
)
from gocia.scheduler.base import JobStatus, build_scheduler
from gocia.structure.fingerprint import (
    classify_structure,
    distance_histogram,
    find_closest,
)
from gocia.structure.slab import load_slab

logger = logging.getLogger(__name__)

# How long to sleep between scheduler polls (seconds)
POLL_INTERVAL = 30


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def run(
    config: GociaConfig,
    run_dir: Path,
    rng: np.random.Generator | None = None,
) -> None:
    """
    Start or resume a GOCIA genetic algorithm run.

    Parameters
    ----------
    config:
        Validated GociaConfig loaded from gocia.yaml.
    run_dir:
        Root directory of the run.  Must already contain gocia.yaml.
        Created if it does not exist.
    rng:
        NumPy random generator.  A default_rng() is created if None.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    if rng is None:
        rng = np.random.default_rng()

    logger.info(f"GOCIA run starting in {run_dir}")

    # --- Infrastructure setup ---
    slab_info = load_slab(
        config.slab.geometry,
        sampling_zmin=config.slab.sampling_zmin,
        sampling_zmax=config.slab.sampling_zmax,
    )
    bare_slab = slab_info.atoms

    stages = build_pipeline(config.calculator_stages)
    scheduler = build_scheduler(config.scheduler)
    desorption_detector = DistanceDesorptionDetector(cutoff=4.0)

    chemical_potentials = {
        ads.symbol: ads.chemical_potential for ads in config.adsorbates
    }

    db_path = run_dir / "gocia.db"
    with GociaDB(db_path) as db:
        db.setup()

        run_id = db.start_run(
            generation_start=_current_generation(run_dir),
            temperature=config.conditions.temperature,
            pressure=config.conditions.pressure,
            potential=config.conditions.potential,
            pH=config.conditions.pH,
        )

        # --- Initial population ---
        gen0_dir = run_dir / "gen_000"
        if not gen0_dir.exists():
            logger.info("gen_000 not found — building initial population")
            build_initial_population(config, slab_info, db, run_dir, rng)

        # --- Main generation loop ---
        current_gen = _current_generation(run_dir)
        stall_count = 0
        best_fitness_seen = float("inf")

        # active_jobs: struct_dir_name → scheduler job_id
        active_jobs: dict[str, str] = {}

        while True:
            gen_dir = run_dir / f"gen_{current_gen:03d}"
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Generation {current_gen}  |  {gen_dir.name}")
            logger.info(f"{'=' * 60}")

            # a. Graceful stop check
            if _stop_requested(run_dir):
                logger.info("gociastop detected — exiting gracefully.")
                break

            # b. Sync sentinel files → DB
            _sync_sentinels_to_db(gen_dir, db)

            # c/d. Submit + poll until generation is fully converged
            _run_generation(
                gen_dir=gen_dir,
                db=db,
                scheduler=scheduler,
                stages=stages,
                active_jobs=active_jobs,
                config=config,
                slab_info=slab_info,
            )

            # e. Process newly converged structures
            _process_converged(
                gen_dir=gen_dir,
                db=db,
                bare_slab=bare_slab,
                desorption_detector=desorption_detector,
                chemical_potentials=chemical_potentials,
                config=config,
                slab_info=slab_info,
            )

            # f. Convergence check
            counts = db.count_by_status(generation=current_gen)
            n_converged = counts.get(STATUS.CONVERGED, 0) + counts.get(STATUS.ISOMER, 0)
            logger.info(
                f"  Generation {current_gen} summary: "
                + ", ".join(f"{s}={n}" for s, n in sorted(counts.items()))
            )

            best = db.best(n=1)
            current_best = best[0].grand_canonical_energy if best else float("inf")

            if current_best < best_fitness_seen - 1e-6:
                best_fitness_seen = current_best
                stall_count = 0
            else:
                stall_count += 1

            logger.info(
                f"  Best G = {current_best:.4f} eV  |  "
                f"Stall = {stall_count}/{config.ga.max_stall_generations}"
            )

            if _should_stop(current_gen, stall_count, config):
                logger.info("Convergence criteria met — stopping.")
                break

            # g. Stop check again (user may have touched gociastop mid-generation)
            if _stop_requested(run_dir):
                logger.info("gociastop detected — exiting gracefully.")
                break

            # h/i/j. Generate offspring → next generation
            next_gen = current_gen + 1
            next_gen_dir = run_dir / f"gen_{next_gen:03d}"

            _generate_offspring(
                next_gen=next_gen,
                next_gen_dir=next_gen_dir,
                db=db,
                config=config,
                slab_info=slab_info,
                rng=rng,
            )

            current_gen = next_gen

        # --- Finalise ---
        db.end_run(run_id, generation_end=current_gen)
        logger.info(f"\nRun complete. Final generation: {current_gen}")
        logger.info(db.summary())


# ---------------------------------------------------------------------------
# Generation runner: submit → poll → wait
# ---------------------------------------------------------------------------


def _run_generation(
    gen_dir: Path,
    db: GociaDB,
    scheduler,
    stages,
    active_jobs: dict[str, str],
    config: GociaConfig,
    slab_info,
) -> None:
    """
    Submit all PENDING structures in gen_dir and poll until all are terminal.

    Blocks until every structure in the generation has reached a terminal
    status (CONVERGED, DESORBED, FAILED, DUPLICATE, ISOMER).
    """
    while True:
        _sync_sentinels_to_db(gen_dir, db)

        # Count non-terminal structures
        all_inds = db.get_generation(_gen_number(gen_dir))
        non_terminal = [i for i in all_inds if not STATUS.is_terminal(i.status)]

        if not non_terminal:
            break

        # Submit PENDING structures (not yet queued) and re-submit any
        # SUBMITTED structures whose jobs died before the pipeline started
        # (i.e. never wrote running_stage_1).  SUBMITTED jobs that are still
        # active are tracked in active_jobs — exclude those.
        n_running = len(active_jobs)
        active_struct_names = set(active_jobs.keys())
        pending = [
            i
            for i in non_terminal
            if i.status in (STATUS.PENDING, STATUS.SUBMITTED)
            and _struct_dir_from_ind(i, gen_dir).name not in active_struct_names
        ]

        for ind in pending:
            if n_running >= config.scheduler.nworkers:
                break

            struct_dir = _struct_dir_from_ind(ind, gen_dir)
            job_id = _submit_structure(
                ind=ind,
                struct_dir=struct_dir,
                stages=stages,
                scheduler=scheduler,
                config=config,
                config_path=(gen_dir.parent / "gocia.yaml").resolve(),
            )

            ind_submitted = ind.with_status(STATUS.SUBMITTED)
            db.update_status(ind_submitted)
            write_sentinel(struct_dir, STATUS.SUBMITTED)

            active_jobs[struct_dir.name] = job_id
            n_running += 1
            logger.info(f"  Submitted {struct_dir.name} → job {job_id}")

        # Poll active jobs
        if active_jobs:
            job_statuses = scheduler.status(list(active_jobs.values()))
            finished = {
                name: jid
                for name, jid in active_jobs.items()
                if job_statuses.get(jid)
                in (JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED)
            }
            for name in finished:
                del active_jobs[name]
                struct_dir = gen_dir / name
                # Sync sentinel written by pipeline
                _sync_one_sentinel(struct_dir, db)

        if non_terminal:
            logger.debug(
                f"  {len(non_terminal)} structure(s) still running — "
                f"sleeping {POLL_INTERVAL}s"
            )
            time.sleep(POLL_INTERVAL)


def _submit_structure(
    ind: Individual,
    struct_dir: Path,
    stages,
    scheduler,
    config: GociaConfig,
    config_path: Path | None = None,
) -> str:
    """Write a job script and submit it. Returns job_id."""
    struct_dir = struct_dir.resolve()

    # Resolve config path: prefer explicit argument, then walk up to run root.
    # run root = struct_dir.parent.parent  (run_dir/gen_NNN/struct_NNNN)
    if config_path is None:
        config_path = (struct_dir.parent.parent / "gocia.yaml").resolve()

    body = f"cd {struct_dir}\ngocia _run-pipeline {struct_dir} --config {config_path.resolve()}\n"
    job_name = f"gocia_{struct_dir.parent.name}_{struct_dir.name}"
    return scheduler.submit_structure(
        job_name=job_name,
        body=body,
        work_dir=struct_dir,
    )


# ---------------------------------------------------------------------------
# Post-convergence processing
# ---------------------------------------------------------------------------


def _process_converged(
    gen_dir: Path,
    db: GociaDB,
    bare_slab,
    desorption_detector,
    chemical_potentials: dict[str, float],
    config: GociaConfig,
    slab_info,
) -> None:
    """
    For each newly converged structure: desorption, fingerprint, fitness.

    'Newly converged' means status=CONVERGED and grand_canonical_energy is None
    (i.e. not yet processed in a previous restart of this generation).
    """
    gen_n = _gen_number(gen_dir)
    converged = [
        i
        for i in db.get_by_status(STATUS.CONVERGED)
        if i.generation == gen_n and i.grand_canonical_energy is None
    ]

    if not converged:
        return

    # Fetch fingerprint pool from the whole DB for duplicate detection
    existing_fps = db.fingerprints()

    for ind in converged:
        struct_dir = (
            Path(ind.geometry_path).parent if ind.geometry_path else gen_dir / "unknown"
        )
        contcar = struct_dir / "CONTCAR"

        if not contcar.exists():
            logger.warning(f"  {struct_dir.name}: CONTCAR missing — marking FAILED")
            db.update_status(ind.mark_failed())
            write_sentinel(struct_dir, STATUS.FAILED)
            continue

        atoms = read(str(contcar), format="vasp")

        # --- Desorption ---
        if desorption_detector.detect(atoms, bare_slab):
            logger.info(f"  {struct_dir.name}: desorbed")
            db.update_status(ind.mark_desorbed())
            write_sentinel(struct_dir, STATUS.DESORBED)
            continue

        # --- Post-relaxation fingerprint ---
        fp = distance_histogram(atoms)

        # Exclude self from the existing pool (it was inserted at PENDING)
        other_fps = [(fid, ffp) for fid, ffp in existing_fps if fid != ind.id]
        other_fp_lists = [ffp for _, ffp in other_fps]

        classification = classify_structure(
            fp,
            other_fp_lists,
            duplicate_threshold=0.01,
            isomer_threshold=0.10,
        )

        if classification == "duplicate":
            closest = find_closest(fp, other_fps)
            of_id = closest[0] if closest else None
            logger.info(f"  {struct_dir.name}: duplicate of {of_id}")
            updated = ind.mark_duplicate()
            db.update(updated)
            write_sentinel(struct_dir, STATUS.DUPLICATE)
            # Update existing_fps to include this one anyway (prevents
            # cascading duplicates if many near-identical structures land
            # in the same generation)
            existing_fps.append((ind.id, fp))
            continue

        if classification == "isomer":
            closest = find_closest(fp, other_fps)
            of_id = closest[0] if closest else None
            logger.info(f"  {struct_dir.name}: isomer of {of_id}")
            updated = ind.mark_isomer(
                of_id=of_id, isomer_weight=config.ga.isomer_weight
            )
            db.update(updated)
            write_sentinel(struct_dir, STATUS.ISOMER)
            existing_fps.append((ind.id, fp))
        else:
            # Unique — update fingerprint in DB
            updated = ind.with_fingerprint(fp)
            db.update(updated)
            existing_fps.append((ind.id, fp))

        # --- CHE fitness ---
        adsorbate_counts = ind.extra_data.get("adsorbate_counts", {})
        if not adsorbate_counts:
            # Try to infer from atom counts in CONTCAR
            n_slab = slab_info.n_slab_atoms
            syms = atoms.get_chemical_symbols()[n_slab:]
            adsorbate_counts = dict(Counter(syms))

        try:
            gce = grand_canonical_energy(
                raw_energy=_read_final_energy(struct_dir),
                adsorbate_counts=adsorbate_counts,
                chemical_potentials=chemical_potentials,
                potential=config.conditions.potential,
                pH=config.conditions.pH,
                temperature=config.conditions.temperature,
                pressure=config.conditions.pressure,
            )
        except (KeyError, ValueError) as exc:
            logger.warning(f"  {struct_dir.name}: fitness failed ({exc})")
            gce = float("nan")

        raw_e = _read_final_energy(struct_dir)
        final_ind = db.get(ind.id)
        db.update_energy(final_ind.with_energy(raw=raw_e, grand_canonical=gce))

        logger.info(f"  {struct_dir.name}: G = {gce:.4f} eV (raw = {raw_e:.4f} eV)")


# ---------------------------------------------------------------------------
# Offspring generation
# ---------------------------------------------------------------------------


def _generate_offspring(
    next_gen: int,
    next_gen_dir: Path,
    db: GociaDB,
    config: GociaConfig,
    slab_info,
    rng: np.random.Generator,
) -> None:
    """
    Select parents, apply operators, fingerprint-check offspring, write to disk.
    """
    next_gen_dir.mkdir(exist_ok=True)

    selectable = db.get_selectable()
    if not selectable:
        logger.warning("No selectable individuals — cannot generate offspring.")
        return

    existing_fps = db.fingerprints()
    n_offspring = config.ga.population_size

    offspring_individuals: list[Individual] = []
    attempts = 0
    max_attempts = n_offspring * 5

    while len(offspring_individuals) < n_offspring and attempts < max_attempts:
        attempts += 1

        # Sample operator and parents
        op_name = sample_operator(rng)
        try:
            parents = select_parents(selectable, op_name, rng)
        except ValueError:
            continue

        # Load parent geometries
        parent_atoms = []
        for p in parents:
            if p.geometry_path is None:
                break
            contcar = Path(p.geometry_path).parent / "CONTCAR"
            if not contcar.exists():
                break
            parent_atoms.append(read(str(contcar), format="vasp"))

        if len(parent_atoms) != len(parents):
            continue

        # Apply operator
        try:
            children_atoms = _apply_operator(
                op_name=op_name,
                parent_atoms=parent_atoms,
                n_slab=slab_info.n_slab_atoms,
                rng=rng,
            )
        except Exception as exc:
            logger.debug(f"  Operator '{op_name}' failed: {exc}")
            continue

        # Process each child
        for child_atoms in children_atoms:
            if len(offspring_individuals) >= n_offspring:
                break

            # Pre-submission fingerprint dedup
            fp = distance_histogram(child_atoms)
            existing_fp_lists = [ffp for _, ffp in existing_fps]
            classification = classify_structure(
                fp,
                existing_fp_lists,
                duplicate_threshold=0.01,
                isomer_threshold=0.10,
            )
            if classification == "duplicate":
                logger.debug("  Pre-submission: offspring is duplicate — skipping")
                continue

            # Assign struct ID and write to disk
            struct_id = f"{len(offspring_individuals) + 1:04d}"
            struct_dir = next_gen_dir / f"struct_{struct_id}"
            struct_dir.mkdir(exist_ok=True)

            poscar_path = struct_dir / "POSCAR"
            write(str(poscar_path), child_atoms, format="vasp")

            # Build Individual with genealogy
            n_slab = slab_info.n_slab_atoms
            syms = child_atoms.get_chemical_symbols()[n_slab:]
            adsorbate_counts = dict(Counter(syms))

            ind = Individual.from_parents(
                generation=next_gen,
                parents=parents,
                operator=op_name,
                geometry_path=str(poscar_path),
                fingerprint=fp,
                extra_data={"adsorbate_counts": adsorbate_counts},
            )

            write_sentinel(struct_dir, STATUS.PENDING)
            db.insert(ind)
            offspring_individuals.append(ind)

            # Add to in-memory fp pool to prevent duplicates within this batch
            existing_fps.append((ind.id, fp))

    if len(offspring_individuals) < n_offspring:
        logger.warning(
            f"  Could only generate {len(offspring_individuals)}/{n_offspring} "
            f"offspring after {max_attempts} attempts."
        )

    logger.info(
        f"  Generation {next_gen}: "
        f"{len(offspring_individuals)} offspring written to {next_gen_dir.name}"
    )


def _apply_operator(
    op_name: str,
    parent_atoms: list,
    n_slab: int,
    rng: np.random.Generator,
) -> list:
    """Dispatch to the appropriate operator function."""
    if op_name == OPERATOR.SPLICE:
        from gocia.operators.graph_splice import splice

        c1, c2 = splice(parent_atoms[0], parent_atoms[1], n_slab, rng=rng)
        return [c1, c2]

    if op_name == OPERATOR.MERGE:
        from gocia.operators.graph_merge import merge

        return [merge(parent_atoms[0], parent_atoms[1], n_slab, rng=rng)]

    if op_name == OPERATOR.MUTATE_ADD:
        from gocia.operators.mutation import mutate_add

        atoms = parent_atoms[0]
        # Pick a random species to add
        syms = atoms.get_chemical_symbols()[n_slab:]
        if not syms:
            return []
        sym = str(rng.choice(syms))
        top_z = atoms.positions[:n_slab, 2].max()
        x = float(rng.uniform(0, atoms.cell[0, 0]))
        y = float(rng.uniform(0, atoms.cell[1, 1]))
        z = float(rng.uniform(top_z + 1.5, top_z + 3.5))
        return [mutate_add(atoms, n_slab, sym, (x, y, z))]

    if op_name == OPERATOR.MUTATE_REMOVE:
        from gocia.operators.mutation import mutate_remove

        atoms = parent_atoms[0]
        syms = list(set(atoms.get_chemical_symbols()[n_slab:]))
        if not syms:
            return []
        sym = str(rng.choice(syms))
        return [mutate_remove(atoms, n_slab, sym, rng=rng)]

    if op_name == OPERATOR.MUTATE_DISPLACE:
        from gocia.operators.mutation import mutate_displace

        atoms = parent_atoms[0]
        syms = list(set(atoms.get_chemical_symbols()[n_slab:]))
        if not syms:
            return []
        sym = str(rng.choice(syms))
        top_z = atoms.positions[:n_slab, 2].max()
        x = float(rng.uniform(0, atoms.cell[0, 0]))
        y = float(rng.uniform(0, atoms.cell[1, 1]))
        z = float(rng.uniform(top_z + 1.5, top_z + 3.5))
        return [mutate_displace(atoms, n_slab, sym, (x, y, z), rng=rng)]

    raise ValueError(f"Unknown operator: {op_name}")


# ---------------------------------------------------------------------------
# Convergence helpers
# ---------------------------------------------------------------------------


def _should_stop(
    current_gen: int,
    stall_count: int,
    config: GociaConfig,
) -> bool:
    """
    Return True if all convergence criteria are met.

    Rules (all must be true):
      - current_gen >= min_generations
      - current_gen >= max_generations  OR  stall_count >= max_stall_generations
    """
    if current_gen < config.ga.min_generations:
        return False
    if current_gen >= config.ga.max_generations:
        return True
    if stall_count >= config.ga.max_stall_generations:
        return True
    return False


def _stop_requested(run_dir: Path) -> bool:
    return (run_dir / "gociastop").exists()


# ---------------------------------------------------------------------------
# Filesystem / DB helpers
# ---------------------------------------------------------------------------


def _current_generation(run_dir: Path) -> int:
    """Find the highest gen_NNN directory that exists."""
    gen_dirs = sorted(run_dir.glob("gen_???"))
    if not gen_dirs:
        return 0
    return int(gen_dirs[-1].name.split("_")[1])


def _gen_number(gen_dir: Path) -> int:
    return int(gen_dir.name.split("_")[1])


def _sync_sentinels_to_db(gen_dir: Path, db: GociaDB) -> None:
    """Walk all struct_* dirs in gen_dir, read sentinels, update DB."""
    if not gen_dir.exists():
        return
    sentinel_map = scan_generation(gen_dir)
    gen_n = _gen_number(gen_dir)
    all_inds = {ind.id: ind for ind in db.get_generation(gen_n)}

    for name, status in sentinel_map.items():
        if status is None:
            continue  # no sentinel file yet — directory just created
        # Find the Individual whose geometry_path lives in this dir
        matching = [
            ind
            for ind in all_inds.values()
            if ind.geometry_path and (gen_dir / name).as_posix() in ind.geometry_path
        ]
        for ind in matching:
            if ind.status != status:
                db.update_status(ind.with_status(status))


def _sync_one_sentinel(struct_dir: Path, db: GociaDB) -> None:
    """Read the sentinel for one struct_dir and update its DB record."""
    status = read_sentinel(struct_dir)
    if status is None:
        return
    gen_dir = struct_dir.parent
    gen_n = _gen_number(gen_dir)
    for ind in db.get_generation(gen_n):
        if ind.geometry_path and struct_dir.as_posix() in ind.geometry_path:
            if ind.status != status:
                db.update_status(ind.with_status(status))
            break


def _struct_dir_from_ind(ind: Individual, gen_dir: Path) -> Path:
    """Derive struct_dir from Individual.geometry_path or invent one."""
    if ind.geometry_path:
        return Path(ind.geometry_path).parent
    # Fallback: use UUID prefix as dir name
    return gen_dir / f"struct_{ind.id[:8]}"


def _read_final_energy(struct_dir: Path) -> float:
    """Read the FINAL_ENERGY file written by pipeline.py."""
    energy_file = struct_dir / "FINAL_ENERGY"
    if energy_file.exists():
        try:
            return float(energy_file.read_text().strip())
        except ValueError:
            pass
    return float("nan")
