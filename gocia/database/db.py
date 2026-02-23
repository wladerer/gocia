"""
gocia/database/db.py

Database connection, CRUD operations, and pandas interface for gocia.db.

The GociaDB class is the single point of contact between the rest of the
codebase and the SQLite database.  It wraps an sqlite3 connection and
provides typed methods for inserting and updating Individual records, logging
runs, and querying the population as pandas DataFrames.

Usage
-----
    from gocia.database.db import GociaDB
    from gocia.population.individual import Individual

    db = GociaDB("gocia.db")
    db.setup()   # create tables if needed (idempotent)

    # Insert a new individual
    ind = Individual(generation=0)
    db.insert(ind)

    # Update status after job submission
    ind = ind.with_status("submitted")
    db.update_status(ind)

    # Query as a pandas DataFrame
    df = db.to_dataframe()
    df = db.to_dataframe(generation=1, status="converged")

    # Re-rank at different CHE conditions
    df = db.rerank(potential=-0.5, pH=7,
                   adsorbate_chemical_potentials={"O": -4.92, "OH": -3.75})
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from gocia.database.schema import (
    CURRENT_SCHEMA_VERSION,
    create_tables,
    get_schema_version,
    set_schema_version,
)
from gocia.population.individual import Individual, STATUS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _encode(value: list | dict | None) -> str | None:
    """Encode a Python list or dict to a JSON string for storage."""
    if value is None:
        return None
    return json.dumps(value)


def _decode_list(value: str | None) -> list:
    """Decode a JSON string to a list; return empty list if None."""
    if not value:
        return []
    return json.loads(value)


def _decode_dict(value: str | None) -> dict:
    """Decode a JSON string to a dict; return empty dict if None."""
    if not value:
        return {}
    return json.loads(value)


def _row_to_individual(row: sqlite3.Row) -> Individual:
    """Convert a sqlite3.Row from the structures table to an Individual."""
    return Individual(
        id=row["id"],
        generation=row["generation"],
        parent_ids=_decode_list(row["parent_ids"]),
        operator=row["operator"],
        status=row["status"],
        raw_energy=row["raw_energy"],
        grand_canonical_energy=row["grand_canonical_energy"],
        weight=row["weight"],
        fingerprint=_decode_list(row["fingerprint"]) or None,
        geometry_path=row["geometry_path"],
        desorption_flag=bool(row["desorption_flag"]),
        is_isomer=bool(row["is_isomer"]),
        isomer_of=row["isomer_of"],
        extra_data=_decode_dict(row["extra_data"]),
    )


# ---------------------------------------------------------------------------
# GociaDB
# ---------------------------------------------------------------------------

class GociaDB:
    """
    Thread-friendly SQLite interface for GOCIA.

    Opens one connection per GociaDB instance.  For concurrent access
    (e.g. gocia status running alongside gocia run), SQLite's WAL mode
    (enabled in schema.py) allows one writer + many readers simultaneously.

    Parameters
    ----------
    path:
        Path to the gocia.db file.  Created if it does not exist.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._conn: sqlite3.Connection | None = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Open the database connection."""
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON;")

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Cursor]:
        """
        Context manager that yields a cursor inside a transaction.
        Commits on exit; rolls back on exception.
        """
        conn = self._require_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def _require_connection(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError(
                "Database is not connected.  Call GociaDB.connect() first, "
                "or use GociaDB as a context manager."
            )
        return self._conn

    def __enter__(self) -> "GociaDB":
        self.connect()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """
        Create tables and indexes if they do not exist.

        Safe to call on an existing database.  Also handles simple schema
        version checks for future migrations.
        """
        conn = self._require_connection()
        create_tables(conn)
        version = get_schema_version(conn)
        if version == 0:
            # Fresh database: stamp with current version
            set_schema_version(conn, CURRENT_SCHEMA_VERSION)
        elif version < CURRENT_SCHEMA_VERSION:
            raise RuntimeError(
                f"gocia.db schema version {version} is older than the current "
                f"version {CURRENT_SCHEMA_VERSION}.  Run `gocia migrate` to "
                "upgrade.  (Back up gocia.db first!)"
            )

    # ------------------------------------------------------------------
    # Individual CRUD
    # ------------------------------------------------------------------

    def insert(self, ind: Individual) -> None:
        """
        Insert a new Individual into the structures table.

        Raises
        ------
        sqlite3.IntegrityError
            If an Individual with the same id already exists.
        """
        with self.transaction() as cur:
            cur.execute(
                """
                INSERT INTO structures (
                    id, generation, parent_ids, operator, status,
                    raw_energy, grand_canonical_energy, weight,
                    fingerprint, geometry_path,
                    desorption_flag, is_isomer, isomer_of, extra_data
                ) VALUES (
                    :id, :generation, :parent_ids, :operator, :status,
                    :raw_energy, :grand_canonical_energy, :weight,
                    :fingerprint, :geometry_path,
                    :desorption_flag, :is_isomer, :isomer_of, :extra_data
                )
                """,
                {
                    "id": ind.id,
                    "generation": ind.generation,
                    "parent_ids": _encode(ind.parent_ids),
                    "operator": ind.operator,
                    "status": ind.status,
                    "raw_energy": ind.raw_energy,
                    "grand_canonical_energy": ind.grand_canonical_energy,
                    "weight": ind.weight,
                    "fingerprint": _encode(ind.fingerprint),
                    "geometry_path": ind.geometry_path,
                    "desorption_flag": int(ind.desorption_flag),
                    "is_isomer": int(ind.is_isomer),
                    "isomer_of": ind.isomer_of,
                    "extra_data": _encode(ind.extra_data),
                },
            )

    def update(self, ind: Individual) -> None:
        """
        Replace all mutable fields for an existing Individual.

        Raises
        ------
        KeyError
            If no Individual with ind.id exists in the database.
        """
        with self.transaction() as cur:
            cur.execute(
                """
                UPDATE structures SET
                    status                  = :status,
                    raw_energy              = :raw_energy,
                    grand_canonical_energy  = :grand_canonical_energy,
                    weight                  = :weight,
                    fingerprint             = :fingerprint,
                    geometry_path           = :geometry_path,
                    desorption_flag         = :desorption_flag,
                    is_isomer               = :is_isomer,
                    isomer_of               = :isomer_of,
                    extra_data              = :extra_data
                WHERE id = :id
                """,
                {
                    "id": ind.id,
                    "status": ind.status,
                    "raw_energy": ind.raw_energy,
                    "grand_canonical_energy": ind.grand_canonical_energy,
                    "weight": ind.weight,
                    "fingerprint": _encode(ind.fingerprint),
                    "geometry_path": ind.geometry_path,
                    "desorption_flag": int(ind.desorption_flag),
                    "is_isomer": int(ind.is_isomer),
                    "isomer_of": ind.isomer_of,
                    "extra_data": _encode(ind.extra_data),
                },
            )
            if cur.rowcount == 0:
                raise KeyError(f"No Individual with id '{ind.id}' found in database.")

    def update_status(self, ind: Individual) -> None:
        """Lightweight update — only writes the status field."""
        with self.transaction() as cur:
            cur.execute(
                "UPDATE structures SET status = :status WHERE id = :id",
                {"status": ind.status, "id": ind.id},
            )

    def update_energy(self, ind: Individual) -> None:
        """Lightweight update — only writes energy fields and weight."""
        with self.transaction() as cur:
            cur.execute(
                """
                UPDATE structures SET
                    raw_energy             = :raw_energy,
                    grand_canonical_energy = :grand_canonical_energy,
                    weight                 = :weight
                WHERE id = :id
                """,
                {
                    "raw_energy": ind.raw_energy,
                    "grand_canonical_energy": ind.grand_canonical_energy,
                    "weight": ind.weight,
                    "id": ind.id,
                },
            )

    def get(self, ind_id: str) -> Individual | None:
        """
        Fetch a single Individual by id.

        Returns None if not found.
        """
        conn = self._require_connection()
        row = conn.execute(
            "SELECT * FROM structures WHERE id = ?", (ind_id,)
        ).fetchone()
        return _row_to_individual(row) if row else None

    def get_generation(self, generation: int) -> list[Individual]:
        """Return all Individuals in a given generation."""
        conn = self._require_connection()
        rows = conn.execute(
            "SELECT * FROM structures WHERE generation = ? ORDER BY rowid",
            (generation,),
        ).fetchall()
        return [_row_to_individual(r) for r in rows]

    def get_by_status(self, status: str) -> list[Individual]:
        """Return all Individuals with a given status."""
        conn = self._require_connection()
        rows = conn.execute(
            "SELECT * FROM structures WHERE status = ? ORDER BY generation, rowid",
            (status,),
        ).fetchall()
        return [_row_to_individual(r) for r in rows]

    def get_selectable(self) -> list[Individual]:
        """
        Return all Individuals eligible for selection (weight > 0, status
        is converged or isomer).
        """
        conn = self._require_connection()
        rows = conn.execute(
            """
            SELECT * FROM structures
            WHERE weight > 0
              AND status IN ('converged', 'isomer')
            ORDER BY grand_canonical_energy ASC
            """,
        ).fetchall()
        return [_row_to_individual(r) for r in rows]

    def count_by_status(self, generation: int | None = None) -> dict[str, int]:
        """
        Return a dict mapping status → count.

        If generation is given, count only within that generation.
        """
        conn = self._require_connection()
        if generation is not None:
            rows = conn.execute(
                """
                SELECT status, COUNT(*) as n FROM structures
                WHERE generation = ?
                GROUP BY status
                """,
                (generation,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT status, COUNT(*) as n FROM structures GROUP BY status"
            ).fetchall()
        return {r["status"]: r["n"] for r in rows}

    def best(self, n: int = 1) -> list[Individual]:
        """
        Return the n Individuals with the lowest grand canonical energy
        among selectable structures.
        """
        conn = self._require_connection()
        rows = conn.execute(
            """
            SELECT * FROM structures
            WHERE weight > 0
              AND status IN ('converged', 'isomer')
              AND grand_canonical_energy IS NOT NULL
            ORDER BY grand_canonical_energy ASC
            LIMIT ?
            """,
            (n,),
        ).fetchall()
        return [_row_to_individual(r) for r in rows]

    def fingerprints(self) -> list[tuple[str, list[float]]]:
        """
        Return (id, fingerprint) pairs for all Individuals that have a
        fingerprint stored.  Used by the duplicate detector.
        """
        conn = self._require_connection()
        rows = conn.execute(
            "SELECT id, fingerprint FROM structures WHERE fingerprint IS NOT NULL"
        ).fetchall()
        return [(r["id"], _decode_list(r["fingerprint"])) for r in rows]

    # ------------------------------------------------------------------
    # Run tracking
    # ------------------------------------------------------------------

    def start_run(
        self,
        generation_start: int,
        temperature: float,
        pressure: float,
        potential: float,
        pH: float,
        notes: str = "",
    ) -> int:
        """
        Log the start of a `gocia run` invocation.

        Returns
        -------
        int
            The run id (auto-incremented).
        """
        with self.transaction() as cur:
            cur.execute(
                """
                INSERT INTO runs (
                    started_at, generation_start,
                    temperature, pressure, potential, pH, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (_now_utc(), generation_start, temperature, pressure, potential, pH, notes),
            )
            return cur.lastrowid  # type: ignore[return-value]

    def end_run(self, run_id: int, generation_end: int) -> None:
        """Record the end of a run invocation."""
        with self.transaction() as cur:
            cur.execute(
                """
                UPDATE runs SET ended_at = ?, generation_end = ?
                WHERE id = ?
                """,
                (_now_utc(), generation_end, run_id),
            )

    def get_runs(self) -> list[dict]:
        """Return all run records as a list of dicts."""
        conn = self._require_connection()
        rows = conn.execute("SELECT * FROM runs ORDER BY id").fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Named condition sets
    # ------------------------------------------------------------------

    def save_conditions(
        self,
        name: str,
        temperature: float,
        pressure: float,
        potential: float,
        pH: float,
    ) -> None:
        """
        Insert or replace a named condition set.

        Used by `gocia inspect` to persist condition presets.
        """
        with self.transaction() as cur:
            cur.execute(
                """
                INSERT INTO conditions (name, temperature, pressure, potential, pH)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    temperature = excluded.temperature,
                    pressure    = excluded.pressure,
                    potential   = excluded.potential,
                    pH          = excluded.pH
                """,
                (name, temperature, pressure, potential, pH),
            )

    def get_conditions(self, name: str) -> dict | None:
        """Fetch a named condition set.  Returns None if not found."""
        conn = self._require_connection()
        row = conn.execute(
            "SELECT * FROM conditions WHERE name = ?", (name,)
        ).fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # Pandas interface
    # ------------------------------------------------------------------

    def to_dataframe(
        self,
        generation: int | None = None,
        status: str | None = None,
    ):
        """
        Return the structures table as a pandas DataFrame.

        Parameters
        ----------
        generation:
            If given, filter to this generation only.
        status:
            If given, filter to this status only.

        Returns
        -------
        pandas.DataFrame
            One row per structure.  JSON columns (parent_ids, fingerprint,
            extra_data) are decoded to Python objects.

        Raises
        ------
        ImportError
            If pandas is not installed.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "pandas is required for GociaDB.to_dataframe(). "
                "Install it with: pip install pandas"
            ) from exc

        conn = self._require_connection()

        clauses = []
        params: list = []
        if generation is not None:
            clauses.append("generation = ?")
            params.append(generation)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"SELECT * FROM structures {where} ORDER BY generation, rowid"

        df = pd.read_sql_query(query, conn, params=params)

        # Decode JSON columns
        df["parent_ids"] = df["parent_ids"].apply(
            lambda v: json.loads(v) if v else []
        )
        df["fingerprint"] = df["fingerprint"].apply(
            lambda v: json.loads(v) if v else None
        )
        df["extra_data"] = df["extra_data"].apply(
            lambda v: json.loads(v) if v else {}
        )

        # Boolean columns
        df["desorption_flag"] = df["desorption_flag"].astype(bool)
        df["is_isomer"] = df["is_isomer"].astype(bool)

        return df

    def rerank(
        self,
        potential: float,
        pH: float,
        adsorbate_chemical_potentials: dict[str, float],
        temperature: float = 298.15,
        pressure: float = 1.0,
        generation: int | None = None,
    ):
        """
        Return a DataFrame of selectable structures re-ranked by grand
        canonical energy at the given CHE conditions.

        This does NOT modify the database.  It recomputes the grand canonical
        energy on-the-fly using the stored raw_energy and adsorbate counts
        from extra_data.

        Parameters
        ----------
        potential:
            Electrode potential in V vs RHE.
        pH:
            Solution pH.
        adsorbate_chemical_potentials:
            Mapping of adsorbate symbol → standard chemical potential (eV).
        temperature:
            Temperature in K (used for kT*ln(P) pressure corrections).
        pressure:
            Pressure in atm.
        generation:
            If given, restrict to this generation.

        Returns
        -------
        pandas.DataFrame
            Sorted by reranked_gce ascending.  Includes a new column
            'reranked_gce' with the recomputed grand canonical energy.
        """
        # Import here to avoid circular imports at module load time
        from gocia.fitness.che import grand_canonical_energy

        df = self.to_dataframe(generation=generation)

        # Keep only selectable structures with a raw energy
        df = df[
            df["status"].isin([STATUS.CONVERGED, STATUS.ISOMER])
            & df["raw_energy"].notna()
        ].copy()

        def _recompute(row):
            adsorbate_counts = row["extra_data"].get("adsorbate_counts", {})
            return grand_canonical_energy(
                raw_energy=row["raw_energy"],
                adsorbate_counts=adsorbate_counts,
                chemical_potentials=adsorbate_chemical_potentials,
                potential=potential,
                pH=pH,
                temperature=temperature,
                pressure=pressure,
            )

        df["reranked_gce"] = df.apply(_recompute, axis=1)
        df = df.sort_values("reranked_gce").reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Convenience summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """
        Return a human-readable summary of the current run state.
        Used by `gocia status`.
        """
        conn = self._require_connection()

        total = conn.execute("SELECT COUNT(*) FROM structures").fetchone()[0]
        counts = self.count_by_status()
        best_list = self.best(n=1)
        best_gce = (
            f"{best_list[0].grand_canonical_energy:.4f} eV"
            if best_list else "N/A"
        )

        gen_max = conn.execute(
            "SELECT MAX(generation) FROM structures"
        ).fetchone()[0]

        lines = [
            f"Structures total : {total}",
            f"Current generation: {gen_max}",
            f"Best G (CHE)     : {best_gce}",
            "Status breakdown:",
        ]
        for status, count in sorted(counts.items()):
            lines.append(f"  {status:<25} {count}")

        return "\n".join(lines)
