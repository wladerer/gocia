"""
gocia/database/merge.py

Merge two or more GOCIA run databases into a single combined database.

Algorithm
---------
1. Load all source databases and their configs.
2. Validate that slab energies and chemical potentials are consistent
   across runs — warn loudly if they differ beyond a tolerance.
3. Collect all converged/isomer structures from each run.
4. Deduplicate by fingerprint: if two structures from different runs have
   a cosine distance below --fingerprint-threshold, keep the one with the
   lower GCE and mark the other as a cross-run duplicate.
5. Assign a run_id tag in each structure's extra_data so the source run
   is traceable in inspect/plot output.
6. Rewrite geometry_path to absolute paths so the merged DB is portable.
7. Re-number generations sequentially: run_A keeps its original generation
   numbers; run_B's generations are offset by (run_A.max_gen + 1), etc.
   The original generation is preserved in extra_data["source_generation"].
8. Write a new merged.db (does not modify any source database).

Non-converged structures (desorbed, failed, pending) are excluded by
default since they carry no energy data and inflate the merged DB.  Pass
--include-all to keep them.

Public API
----------
    merge_runs(source_dirs, output_dir, fingerprint_threshold, include_all)
      -> MergeResult
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Statuses that carry energy data and are useful in a merged analysis
SELECTABLE = {"converged", "isomer"}
TERMINAL   = {"converged", "isomer", "desorbed", "failed", "duplicate"}

# Tolerance for slab energy / chemical potential consistency checks (eV)
ENERGY_CONSISTENCY_TOL = 0.01


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class MergeResult:
    """Summary of a completed merge operation."""
    output_db:          Path
    n_sources:          int
    n_input:            int       # total structures considered
    n_merged:           int       # structures written to output DB
    n_deduplicated:     int       # structures dropped as cross-run duplicates
    n_skipped:          int       # non-terminal structures skipped
    warnings:           list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Merged {self.n_sources} run(s) → {self.output_db}",
            f"  Input structures : {self.n_input}",
            f"  Written          : {self.n_merged}",
            f"  Deduplicated     : {self.n_deduplicated}",
            f"  Skipped          : {self.n_skipped}",
        ]
        if self.warnings:
            lines.append("  Warnings:")
            for w in self.warnings:
                lines.append(f"    ⚠  {w}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Config consistency check
# ---------------------------------------------------------------------------

def _check_consistency(configs: list, warnings: list[str]) -> None:
    """
    Compare slab energies and chemical potentials across runs.
    Appends human-readable warnings to `warnings` for any inconsistencies.
    Does not raise — the user decides whether to proceed.
    """
    ref_cfg = configs[0]
    ref_slab_e = ref_cfg.slab.energy
    ref_mu = {ads.symbol: ads.chemical_potential for ads in ref_cfg.adsorbates}

    for i, cfg in enumerate(configs[1:], start=2):
        # Slab energy
        delta_slab = abs(cfg.slab.energy - ref_slab_e)
        if delta_slab > ENERGY_CONSISTENCY_TOL:
            warnings.append(
                f"Run {i} slab energy ({cfg.slab.energy:.4f} eV) differs from "
                f"run 1 ({ref_slab_e:.4f} eV) by {delta_slab:.4f} eV. "
                "GCE comparisons across runs may be unreliable."
            )

        # Chemical potentials
        run_mu = {ads.symbol: ads.chemical_potential for ads in cfg.adsorbates}
        all_syms = set(ref_mu) | set(run_mu)
        for sym in all_syms:
            if sym not in run_mu:
                warnings.append(
                    f"Run {i} is missing chemical potential for '{sym}' "
                    f"(present in run 1)."
                )
            elif sym not in ref_mu:
                warnings.append(
                    f"Run 1 is missing chemical potential for '{sym}' "
                    f"(present in run {i})."
                )
            else:
                delta_mu = abs(run_mu[sym] - ref_mu[sym])
                if delta_mu > ENERGY_CONSISTENCY_TOL:
                    warnings.append(
                        f"Run {i} μ({sym}) = {run_mu[sym]:.4f} eV differs from "
                        f"run 1 ({ref_mu[sym]:.4f} eV) by {delta_mu:.4f} eV."
                    )


# ---------------------------------------------------------------------------
# Fingerprint deduplication
# ---------------------------------------------------------------------------

def _cosine_distance(a: list[float], b: list[float]) -> float:
    """
    Cosine distance between two fingerprint vectors.
    Returns a value in [0, 1]: 0 = identical, 1 = orthogonal.
    Returns 1.0 if either vector is all-zero or mismatched length.
    """
    va = np.array(a, dtype=float)
    vb = np.array(b, dtype=float)
    if len(va) != len(vb):
        return 1.0
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 1.0
    return float(1.0 - np.dot(va, vb) / (norm_a * norm_b))


def _deduplicate(rows: list[dict], threshold: float) -> tuple[list[dict], int]:
    """
    Remove cross-run duplicate structures.

    Two structures are considered duplicates if their fingerprint cosine
    distance is below `threshold`.  When duplicates are found, keep the
    one with the lower GCE (or lower raw_energy if GCE is absent).  Only
    compares structures from *different* runs (same-run duplicates are
    already handled by the GA loop's internal duplicate filter).

    Parameters
    ----------
    rows : list[dict]
        Each dict is a row with keys matching the structures table plus
        a 'run_id' key.
    threshold : float
        Cosine distance below which two structures are considered the same.

    Returns
    -------
    (kept_rows, n_dropped)
    """
    if not rows:
        return rows, 0

    # Index rows by run_id
    kept   = []   # indices of rows to keep
    dropped = set()

    for i, row_i in enumerate(rows):
        if i in dropped:
            continue
        fp_i = row_i.get("fingerprint")
        if not fp_i:
            kept.append(i)
            continue

        for j in range(i + 1, len(rows)):
            if j in dropped:
                continue
            row_j = rows[j]

            # Only check cross-run pairs
            if row_i["run_id"] == row_j["run_id"]:
                continue

            fp_j = row_j.get("fingerprint")
            if not fp_j:
                continue

            dist = _cosine_distance(fp_i, fp_j)
            if dist < threshold:
                # Keep the lower-energy one
                e_i = row_i.get("grand_canonical_energy") or row_i.get("raw_energy") or 0.0
                e_j = row_j.get("grand_canonical_energy") or row_j.get("raw_energy") or 0.0
                if e_i <= e_j:
                    dropped.add(j)
                    log.debug(
                        f"Dedup: kept {str(row_i['id'])[:8]} (run {row_i['run_id']}), "
                        f"dropped {str(row_j['id'])[:8]} (run {row_j['run_id']}), "
                        f"dist={dist:.4f}"
                    )
                else:
                    dropped.add(i)
                    log.debug(
                        f"Dedup: kept {str(row_j['id'])[:8]} (run {row_j['run_id']}), "
                        f"dropped {str(row_i['id'])[:8]} (run {row_i['run_id']}), "
                        f"dist={dist:.4f}"
                    )
                    break  # row_i is dropped; move to next i

        if i not in dropped:
            kept.append(i)

    kept_rows = [rows[i] for i in kept]
    return kept_rows, len(dropped)


# ---------------------------------------------------------------------------
# Main merge function
# ---------------------------------------------------------------------------

def merge_runs(
    source_dirs: list[Path],
    output_dir: Path,
    fingerprint_threshold: float = 0.01,
    include_all: bool = False,
) -> MergeResult:
    """
    Merge multiple GOCIA runs into a single database.

    Parameters
    ----------
    source_dirs : list[Path]
        Run root directories, each containing gocia.db and gocia.yaml.
    output_dir : Path
        Directory where the merged gocia.db will be written.
        Created if it does not exist.
    fingerprint_threshold : float
        Cosine distance below which two structures from different runs are
        considered duplicates.  Default 0.01 (tight).  Increase to 0.05
        for more aggressive deduplication.
    include_all : bool
        If True, include desorbed/failed/pending structures too.
        Default False — only converged and isomer.

    Returns
    -------
    MergeResult
    """
    import json as _json
    from gocia.config import load_config
    from gocia.database.db import GociaDB

    warnings: list[str] = []

    # ── 1. Load all sources ──────────────────────────────────────────────
    dfs      = []
    configs  = []
    run_dirs = []

    for src in source_dirs:
        src = Path(src).resolve()
        db_path  = src / "gocia.db"
        cfg_path = src / "gocia.yaml"

        if not db_path.exists():
            raise FileNotFoundError(f"gocia.db not found in {src}")
        if not cfg_path.exists():
            raise FileNotFoundError(f"gocia.yaml not found in {src}")

        cfg = load_config(cfg_path)
        configs.append(cfg)
        run_dirs.append(src)

        with GociaDB(db_path) as db:
            df = db.to_dataframe()
        dfs.append(df)
        log.info(f"Loaded {len(df)} structures from {src}")

    # ── 2. Consistency check ─────────────────────────────────────────────
    _check_consistency(configs, warnings)
    for w in warnings:
        log.warning(w)

    # ── 3. Collect rows ──────────────────────────────────────────────────
    keep_statuses = TERMINAL if include_all else SELECTABLE
    all_rows: list[dict] = []
    n_skipped = 0
    gen_offset = 0

    for run_idx, (df, src_dir) in enumerate(zip(dfs, run_dirs)):
        run_id   = src_dir.name      # e.g. "run_A", "pt111_O_low_P"
        run_label = f"run_{run_idx + 1}" if run_id == "." else run_id

        # Filter to desired statuses
        mask = df["status"].isin(keep_statuses)
        n_skipped += int((~mask).sum())
        subset = df[mask].copy()

        if subset.empty:
            warnings.append(f"{src_dir}: no structures with status in {keep_statuses}.")
            gen_offset += 0
            continue

        max_gen = int(subset["generation"].max())

        for _, row in subset.iterrows():
            # Absolute geometry path
            geom = row["geometry_path"]
            if geom and not Path(geom).is_absolute():
                geom = str((src_dir / geom).resolve())

            # Augment extra_data with provenance
            extra = dict(row["extra_data"]) if row["extra_data"] else {}
            extra["run_id"]            = run_label
            extra["source_dir"]        = str(src_dir)
            extra["source_generation"] = int(row["generation"])

            all_rows.append({
                "id":                     row["id"],
                "generation":             int(row["generation"]) + gen_offset,
                "parent_ids":             row["parent_ids"],
                "operator":               row["operator"],
                "status":                 row["status"],
                "raw_energy":             row["raw_energy"],
                "grand_canonical_energy": row["grand_canonical_energy"],
                "weight":                 row["weight"],
                "fingerprint":            row["fingerprint"],
                "geometry_path":          geom,
                "desorption_flag":        bool(row["desorption_flag"]),
                "is_isomer":              bool(row["is_isomer"]),
                "isomer_of":              row["isomer_of"],
                "extra_data":             extra,
                "run_id":                 run_label,
            })

        gen_offset += max_gen + 1

    n_input = len(all_rows)

    # ── 4. Deduplicate ───────────────────────────────────────────────────
    all_rows, n_deduped = _deduplicate(all_rows, fingerprint_threshold)
    log.info(f"Deduplicated {n_deduped} cross-run duplicate(s)")

    # ── 5. Write merged DB ───────────────────────────────────────────────
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_db = output_dir / "gocia.db"

    if output_db.exists():
        # Back up before overwriting
        import shutil
        backup = output_db.with_suffix(".db.bak")
        shutil.copy2(output_db, backup)
        log.info(f"Backed up existing {output_db} → {backup}")

    import math as _math

    def _s(v):
        """NaN-safe string-or-None."""
        if v is None:
            return None
        try:
            if _math.isnan(float(v)):
                return None
        except (TypeError, ValueError):
            pass
        return str(v)

    def _f(v):
        """NaN-safe float-or-None."""
        if v is None:
            return None
        try:
            f = float(v)
            return None if _math.isnan(f) else f
        except (TypeError, ValueError):
            return None

    from gocia.population.individual import Individual
    with GociaDB(output_db) as db:
        db.setup()
        for row in all_rows:
            ind = Individual(
                id=row["id"],
                generation=int(row["generation"]),
                parent_ids=row["parent_ids"] or [],
                operator=_s(row["operator"]),
                status=row["status"],
                raw_energy=_f(row["raw_energy"]),
                grand_canonical_energy=_f(row["grand_canonical_energy"]),
                weight=_f(row["weight"]) or 1.0,
                fingerprint=row["fingerprint"],
                geometry_path=_s(row["geometry_path"]),
                desorption_flag=bool(row["desorption_flag"]),
                is_isomer=bool(row["is_isomer"]),
                isomer_of=_s(row["isomer_of"]),
                extra_data=row["extra_data"],
            )
            db.insert(ind)

    n_merged = len(all_rows)
    log.info(f"Wrote {n_merged} structures to {output_db}")

    return MergeResult(
        output_db=output_db,
        n_sources=len(source_dirs),
        n_input=n_input,
        n_merged=n_merged,
        n_deduplicated=n_deduped,
        n_skipped=n_skipped,
        warnings=warnings,
    )
