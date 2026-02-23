"""
gocia/cli.py

Command-line interface for GOCIA.

Commands
--------
  gocia init      Validate gocia.yaml, create run directory structure,
                  initialise gocia.db.  Prints a template config if none exists.
  gocia run       Start or resume the main GA loop.
  gocia status    Print a one-screen status summary of the current run.
  gocia inspect   Query the database with optional filtering and re-ranking.
  gocia stop      Write a gociastop file; the loop exits after the current generation.

Hidden commands (called by job scripts, not intended for direct user use)
  gocia _run-pipeline   Run the calculator pipeline for one structure directory.

Usage
-----
    gocia init [--config gocia.yaml] [--run-dir .]
    gocia run  [--config gocia.yaml] [--run-dir .]
    gocia status [--run-dir .]
    gocia inspect [--run-dir .] [--generation N] [--status converged]
                  [--potential U] [--pH V] [--no-rerank] [--top N] [--output file.csv]
    gocia stop [--run-dir .]
"""

from __future__ import annotations

import sys
import logging
import shutil
from pathlib import Path

import click

# ---------------------------------------------------------------------------
# Logging setup — configured once at CLI entry, not at import time
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        level=level,
        stream=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Shared options
# ---------------------------------------------------------------------------

_config_option = click.option(
    "--config", "-c",
    default="gocia.yaml",
    show_default=True,
    type=click.Path(exists=False, dir_okay=False),
    help="Path to the gocia.yaml configuration file.",
)

_run_dir_option = click.option(
    "--run-dir", "-d",
    default=".",
    show_default=True,
    type=click.Path(file_okay=False),
    help="Root directory of the GOCIA run.",
)

_verbose_option = click.option(
    "--verbose", "-v",
    is_flag=True,
    default=False,
    help="Enable debug logging.",
)


# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(package_name="gocia")
def cli() -> None:
    """
    GOCIA — Genetic algorithm for surface adsorbate structure search.

    Start with `gocia init` to validate your config and create the run
    directory, then `gocia run` to start the genetic algorithm.
    """


# ---------------------------------------------------------------------------
# gocia init
# ---------------------------------------------------------------------------

@cli.command("init")
@_config_option
@_run_dir_option
@_verbose_option
def cmd_init(config: str, run_dir: str, verbose: bool) -> None:
    """
    Validate gocia.yaml, create the run directory, and initialise gocia.db.

    If no config file is found, prints a fully commented template to stdout
    and exits with code 1.  Capture it to create your config:

        gocia init > gocia.yaml
        # then edit gocia.yaml and run:
        gocia init
    """
    _setup_logging(verbose)
    config_path = Path(config)
    run_dir_path = Path(run_dir)

    # If no config exists or is empty (e.g. created by shell redirect before
    # this process ran): print template to stdout and exit with code 1 so the
    # caller can capture it.  The canonical usage is:
    #   gocia init > gocia.yaml
    # The shell creates an empty gocia.yaml before this process starts, so we
    # must treat a zero-byte file the same as a missing file.
    if not config_path.exists() or config_path.stat().st_size == 0:
        click.echo(_CONFIG_TEMPLATE, nl=False)
        raise SystemExit(1)

    # Validate config
    try:
        from gocia.config import load_config
        cfg = load_config(str(config_path))
    except Exception as exc:
        click.echo(f"Error: config validation failed:\n  {exc}", err=True)
        raise SystemExit(1)

    # Create run directory structure
    run_dir_path.mkdir(parents=True, exist_ok=True)

    # Initialise database
    from gocia.database.db import GociaDB
    db_path = run_dir_path / "gocia.db"
    with GociaDB(db_path) as db:
        db.setup()

    click.echo(f"✓ Config valid: {config_path}")
    click.echo(f"✓ Run directory ready: {run_dir_path.resolve()}")
    click.echo(f"✓ Database initialised: {db_path}")
    click.echo()
    click.echo("Next step:")
    click.echo(f"  gocia run --config {config_path} --run-dir {run_dir_path}")


# ---------------------------------------------------------------------------
# gocia run
# ---------------------------------------------------------------------------

@cli.command("run")
@_config_option
@_run_dir_option
@_verbose_option
@click.option("--seed", type=int, default=None,
              help="Random seed for reproducibility.")
def cmd_run(config: str, run_dir: str, verbose: bool, seed: int | None) -> None:
    """
    Start or resume the main GA loop.

    Reads sentinel files to reconstruct state after HPC timeouts.
    Safe to call multiple times on the same run directory.
    """
    _setup_logging(verbose)
    config_path = Path(config)
    run_dir_path = Path(run_dir)

    if not config_path.exists():
        click.echo(f"Error: config file not found: {config_path}", err=True)
        raise SystemExit(1)

    try:
        from gocia.config import load_config
        cfg = load_config(str(config_path))
    except Exception as exc:
        click.echo(f"Error: config validation failed:\n  {exc}", err=True)
        raise SystemExit(1)

    import numpy as np
    rng = np.random.default_rng(seed)

    from gocia.runner.loop import run
    try:
        run(cfg, run_dir_path, rng=rng)
    except KeyboardInterrupt:
        click.echo("\nInterrupted by user.", err=True)
        raise SystemExit(130)
    except Exception as exc:
        logging.getLogger(__name__).exception("Fatal error in GA loop")
        click.echo(f"\nFatal error: {exc}", err=True)
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# gocia status
# ---------------------------------------------------------------------------

@cli.command("status")
@_run_dir_option
@click.option("--generation", "-g", type=int, default=None,
              help="Show status for a specific generation only.")
def cmd_status(run_dir: str, generation: int | None) -> None:
    """
    Print a one-screen status summary of the current run.

    Shows the current generation, counts by status, and the best grand
    canonical energy found so far.
    """
    _setup_logging(verbose=False)
    run_dir_path = Path(run_dir)
    db_path = run_dir_path / "gocia.db"

    if not db_path.exists():
        click.echo(f"No database found at {db_path}. Have you run `gocia init`?", err=True)
        raise SystemExit(1)

    from gocia.database.db import GociaDB
    with GociaDB(db_path) as db:
        click.echo(db.summary())

        if generation is not None:
            counts = db.count_by_status(generation=generation)
            click.echo(f"\nGeneration {generation} breakdown:")
            for status, n in sorted(counts.items()):
                click.echo(f"  {status:<25} {n}")

        best = db.best(n=3)
        if best:
            click.echo(f"\nTop structures by grand canonical energy:")
            for i, ind in enumerate(best, 1):
                gce = f"{ind.grand_canonical_energy:.4f} eV" if ind.grand_canonical_energy else "N/A"
                click.echo(
                    f"  {i}. {ind.id[:8]}  gen={ind.generation:3d}  "
                    f"G={gce}  op={ind.operator}"
                )

        # Check for gociastop
        if (run_dir_path / "gociastop").exists():
            click.echo("\n⚠  gociastop file present — loop will exit after current generation.")


# ---------------------------------------------------------------------------
# gocia inspect
# ---------------------------------------------------------------------------

def _format_inspect_table(df, run_dir: Path):
    """
    Produce a clean, terminal-friendly DataFrame for `gocia inspect`.

    Transformations applied (raw data is preserved in --output csv):
    - id            → 8-char prefix (enough to be unique in a typical run)
    - parent_ids    → comma-separated 8-char prefixes, or '-' if initial
    - geometry_path → relative struct path (gen_NNN/struct_NNNN) or '-'
    - extra_data    → unpacked to 'adsorbates' column (e.g. "O:3")
    - fingerprint   → dropped (50-element vector is unreadable in a table)
    - desorption_flag / is_isomer / isomer_of → dropped (redundant with status)
    - created_at    → dropped
    - updated_at    → renamed 'updated' and truncated to HH:MM:SS
    - Column order  → gen | struct | status | raw_e | gce | operator | parents | adsorbates
    """
    import pandas as pd

    out = pd.DataFrame()

    out["gen"] = df["generation"]

    # Short struct label from geometry_path
    def _struct_label(path):
        if not path or (isinstance(path, float)):
            return "-"
        p = Path(str(path))
        # Take gen_NNN/struct_NNNN — last two parts of the path minus filename
        parts = p.parts
        # geometry_path ends in POSCAR; parent is struct dir, grandparent is gen dir
        try:
            return f"{parts[-3]}/{parts[-2]}"
        except IndexError:
            return p.parent.name

    out["struct"] = df["geometry_path"].apply(_struct_label)

    out["status"] = df["status"]

    out["raw_e"] = df["raw_energy"].map(
        lambda v: f"{v:.4f}" if pd.notna(v) else "-"
    )
    out["gce"] = df["grand_canonical_energy"].map(
        lambda v: f"{v:.4f}" if pd.notna(v) else "-"
    )

    out["operator"] = df["operator"].fillna("-")

    def _short_parents(pids):
        if not pids or (isinstance(pids, float)):
            return "-"
        return ", ".join(str(p)[:8] for p in pids)

    out["parents"] = df["parent_ids"].apply(_short_parents)

    def _adsorbates(extra):
        if not extra or (isinstance(extra, float)):
            return "-"
        counts = extra.get("adsorbate_counts", {})
        if not counts:
            return "-"
        return " ".join(f"{sym}:{n}" for sym, n in sorted(counts.items()))

    out["adsorbates"] = df["extra_data"].apply(_adsorbates)

    def _time(ts):
        if not ts or (isinstance(ts, float)):
            return "-"
        # Keep only HH:MM:SS from "YYYY-MM-DD HH:MM:SS"
        return str(ts).split(" ")[-1].split(".")[0]

    out["updated"] = df["updated_at"].apply(_time)

    return out


@cli.command("inspect")
@_run_dir_option
@click.option("--generation", "-g", type=int, default=None,
              help="Filter to a specific generation.")
@click.option("--status", "-s", type=str, default=None,
              help="Filter by status (e.g. converged, failed, desorbed).")
@click.option("--potential", type=float, default=None,
              help="Electrode potential (V vs RHE) for re-ranking.")
@click.option("--pH", "ph", type=float, default=None,
              help="pH for re-ranking.")
@click.option("--temperature", type=float, default=None,
              help="Temperature (K) for re-ranking.")
@click.option("--no-rerank", "no_rerank", is_flag=True, default=False,
              help="Skip re-ranking; use stored grand canonical energies.")
@click.option("--top", type=int, default=None,
              help="Return only the top N structures by fitness.")
@click.option("--output", type=click.Path(dir_okay=False), default=None,
              help="Export results to a CSV file.")
def cmd_inspect(
    run_dir: str,
    generation: int | None,
    status: str | None,
    potential: float | None,
    ph: float | None,
    temperature: float | None,
    no_rerank: bool,
    top: int | None,
    output: str | None,
) -> None:
    """
    Query the database with optional filtering and condition re-ranking.

    By default re-ranks structures at the stored run conditions.
    Pass --potential / --pH to re-rank at different conditions.
    Pass --no-rerank to skip re-ranking entirely.

    Examples:

    \b
        gocia inspect --generation 5
        gocia inspect --status converged --top 10
        gocia inspect --potential -0.5 --pH 7 --output results.csv
    """
    _setup_logging(verbose=False)
    run_dir_path = Path(run_dir)
    db_path = run_dir_path / "gocia.db"

    if not db_path.exists():
        click.echo(f"No database found at {db_path}.", err=True)
        raise SystemExit(1)

    try:
        import pandas as pd
    except ImportError:
        click.echo("pandas is required for `gocia inspect`. Install with: pip install pandas", err=True)
        raise SystemExit(1)

    from gocia.database.db import GociaDB
    with GociaDB(db_path) as db:
        # Retrieve as DataFrame
        df = db.to_dataframe(generation=generation, status=status)

        if df.empty:
            click.echo("No structures match the given filters.")
            return

        # Re-ranking
        if not no_rerank:
            # Load config for chemical potentials if needed
            config_path = run_dir_path / "gocia.yaml"
            if config_path.exists() and (potential is not None or ph is not None):
                try:
                    from gocia.config import load_config
                    cfg = load_config(str(config_path))
                    chem_pots = {ads.symbol: ads.chemical_potential for ads in cfg.adsorbates}
                    U = potential if potential is not None else cfg.conditions.potential
                    pH_val = ph if ph is not None else cfg.conditions.pH
                    T = temperature if temperature is not None else cfg.conditions.temperature

                    df = db.rerank(
                        potential=U,
                        pH=pH_val,
                        temperature=T,
                        adsorbate_chemical_potentials=chem_pots,
                    )
                    click.echo(
                        f"Re-ranked at U={U} V vs RHE, pH={pH_val}, T={T} K",
                        err=True,
                    )
                except Exception as exc:
                    click.echo(f"Warning: re-ranking failed ({exc}). Showing stored energies.", err=True)

        # Top N filter
        if top is not None:
            df = df.sort_values("grand_canonical_energy").head(top)

        # Output
        if output:
            df.to_csv(output, index=False)
            click.echo(f"Saved {len(df)} rows to {output}")
        else:
            display_df = _format_inspect_table(df, run_dir_path)
            term_width = shutil.get_terminal_size((160, 40)).columns
            with pd.option_context(
                "display.max_rows", 200,
                "display.max_columns", 30,
                "display.width", term_width,
                "display.max_colwidth", 24,
            ):
                click.echo(display_df.to_string(index=False))
            click.echo(f"\n{len(df)} structure(s) shown.")


# ---------------------------------------------------------------------------
# gocia stop
# ---------------------------------------------------------------------------

@cli.command("stop")
@_run_dir_option
def cmd_stop(run_dir: str) -> None:
    """
    Request a graceful stop of the GA loop.

    Writes a 'gociastop' file in the run directory.  The loop will finish
    the current generation and exit cleanly before starting the next one.
    """
    run_dir_path = Path(run_dir)
    stop_file = run_dir_path / "gociastop"
    stop_file.touch()
    click.echo(f"✓ Graceful stop requested. Written: {stop_file}")
    click.echo("  The loop will exit after the current generation completes.")


# ---------------------------------------------------------------------------
# gocia _run-pipeline  (hidden: called from job scripts)
# ---------------------------------------------------------------------------

@cli.command("_run-pipeline", hidden=True)
@click.argument("struct_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--config", "-c", default="gocia.yaml",
              type=click.Path(exists=True, dir_okay=False),
              help="Path to gocia.yaml.")
@_verbose_option
def cmd_run_pipeline(struct_dir: str, config: str, verbose: bool) -> None:
    """
    Run the calculator pipeline for a single structure directory.

    Called from HPC job scripts — not intended for direct user use.
    Exits 0 on success, 1 on failure (pipeline writes FAILED sentinel).
    """
    _setup_logging(verbose)
    log = logging.getLogger(__name__)

    try:
        from gocia.config import load_config
        from gocia.calculator.stage import build_pipeline
        from gocia.calculator.pipeline import run_pipeline
        from ase.io import read

        cfg = load_config(config)
        stages = build_pipeline(cfg.calculator_stages)

        struct_dir_path = Path(struct_dir)
        poscar = struct_dir_path / "POSCAR"
        if not poscar.exists():
            click.echo(f"Error: POSCAR not found in {struct_dir}", err=True)
            raise SystemExit(1)

        atoms = read(str(poscar), format="vasp")
        h5_path = struct_dir_path / "trajectory.h5"

        # n_slab is simply the atom count of the bare slab geometry file.
        # Read it directly — no need for load_slab() which requires z-bounds.
        slab_geom = Path(cfg.slab.geometry)
        if not slab_geom.is_absolute():
            # Resolve relative to the directory containing gocia.yaml
            slab_geom = (Path(config).parent / slab_geom).resolve()
        n_slab = len(read(str(slab_geom), format="vasp"))

        run_pipeline(
            atoms=atoms,
            stages=stages,
            struct_dir=struct_dir_path,
            h5_path=h5_path,
            mace_model=getattr(cfg, "mace_model", "medium"),
            mace_device=getattr(cfg, "mace_device", "auto"),
            slab_energy=cfg.slab.energy,
            n_slab=n_slab,
        )

    except SystemExit:
        raise
    except Exception as exc:
        log.exception(f"Pipeline failed for {struct_dir}")
        click.echo(f"Pipeline error: {exc}", err=True)
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# gocia.yaml template
# ---------------------------------------------------------------------------

_CONFIG_TEMPLATE = """\
# gocia.yaml — GOCIA configuration file
# Generated by `gocia init`.  Edit before running `gocia run`.
#
# All fields marked (required) must be filled in.
# Fields with defaults can be left as-is for a first run.

# ---------------------------------------------------------------------------
# Slab
# ---------------------------------------------------------------------------
slab:
  geometry: slab.vasp                 # (required) Path to VASP POSCAR/CONTCAR
  energy: -125.0                      # (required) DFT energy of the bare slab (eV)
  sampling_zmin: 10.0                 # (required) Min z for adsorbate placement (Å)
  sampling_zmax: 15.0                 # (required) Max z for adsorbate placement (Å)

# ---------------------------------------------------------------------------
# Adsorbates
# ---------------------------------------------------------------------------
adsorbates:
  - symbol: O
    chemical_potential: -4.92         # (required) μ referenced to standard state (eV)
    n_orientations: 1                 # Orientations to trial at placement (1 for atoms)
    # geometry:                       # Optional: path to molecule geometry file
    # coordinates:                    # Or inline: [[x, y, z], ...]

  - symbol: OH
    chemical_potential: -3.75
    n_orientations: 6
    # geometry: oh.xyz

# ---------------------------------------------------------------------------
# Calculator stages
# ---------------------------------------------------------------------------
calculator_stages:
  - name: mace_preopt
    type: mace
    fmax: 0.10                        # Force convergence threshold (eV/Å)
    max_steps: 300

  # - name: vasp_coarse               # Uncomment to add a VASP stage
  #   type: vasp
  #   fmax: 0.05
  #   max_steps: 100
  #   incar:
  #     ENCUT: 400
  #     EDIFF: 1.0e-5
  #     ISMEAR: 1
  #     SIGMA: 0.1

  # - name: vasp_fine
  #   type: vasp
  #   fmax: 0.02
  #   max_steps: 100
  #   incar:
  #     ENCUT: 520
  #     EDIFF: 1.0e-6

# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------
scheduler:
  type: local                         # local | slurm | pbs
  nworkers: 4                         # Max concurrent jobs
  walltime: "01:00:00"               # HH:MM:SS or D-HH:MM:SS

  # Uncomment and fill in for Slurm/PBS:
  # resources:
  #   nodes: 1
  #   tasks_per_node: 16
  #   mem: "32G"
  #   account: "myproject"
  #   partition: "regular"
  # extra_directives:
  #   - "--mail-type=FAIL"

# ---------------------------------------------------------------------------
# Genetic algorithm
# ---------------------------------------------------------------------------
ga:
  population_size: 20
  max_generations: 50
  min_generations: 5
  max_stall_generations: 10          # Stop if best energy unchanged for N generations
  isomer_weight: 0.01                # Selection weight for near-duplicate structures

# ---------------------------------------------------------------------------
# Thermodynamic conditions
# ---------------------------------------------------------------------------
conditions:
  temperature: 298.15                # K
  pressure: 1.0                      # atm
  potential: 0.0                     # V vs RHE
  pH: 0.0
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    cli()


if __name__ == "__main__":
    main()
