"""
gocia/database/schema.py

SQLite schema for the GOCIA run database (gocia.db).

Three tables
------------
structures
    One row per Individual.  Central table for the entire run history.
    Raw and grand-canonical energies are stored separately so the population
    can be re-ranked at arbitrary (U, pH, T, P) without re-running jobs.

runs
    One row per `gocia run` invocation.  Records which conditions were active
    during selection at each invocation so the full optimisation history is
    auditable.

conditions
    Named condition sets for post-hoc analysis via `gocia inspect`.
    Populated explicitly by the user or by gocia inspect flags; not used
    during the GA run itself.

Usage
-----
    from gocia.database.schema import create_tables
    import sqlite3

    conn = sqlite3.connect("gocia.db")
    create_tables(conn)
"""

from __future__ import annotations

import sqlite3

# ---------------------------------------------------------------------------
# DDL statements
# ---------------------------------------------------------------------------

_CREATE_STRUCTURES = """
CREATE TABLE IF NOT EXISTS structures (
    -- Identity
    id                      TEXT PRIMARY KEY,
    generation              INTEGER NOT NULL,

    -- Genealogy
    parent_ids              TEXT    NOT NULL DEFAULT '[]',   -- JSON list of IDs
    operator                TEXT    NOT NULL DEFAULT 'init',

    -- Pipeline state
    status                  TEXT    NOT NULL DEFAULT 'pending',

    -- Energetics (eV)
    raw_energy              REAL,           -- total DFT / MACE energy
    grand_canonical_energy  REAL,           -- CHE-corrected, at run conditions

    -- Selection
    weight                  REAL    NOT NULL DEFAULT 1.0,

    -- Fingerprint for duplicate detection
    fingerprint             TEXT,           -- JSON list of floats

    -- Filesystem
    geometry_path           TEXT,           -- absolute path to final CONTCAR

    -- Desorption
    desorption_flag         INTEGER NOT NULL DEFAULT 0,  -- 0 or 1

    -- Isomer bookkeeping
    is_isomer               INTEGER NOT NULL DEFAULT 0,  -- 0 or 1
    isomer_of               TEXT,           -- ID of representative structure

    -- Overflow metadata (per-stage energies, custom flags, etc.)
    extra_data              TEXT    NOT NULL DEFAULT '{}', -- JSON dict

    -- Audit timestamps (UTC ISO-8601)
    created_at              TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at              TEXT    NOT NULL DEFAULT (datetime('now'))
);
"""

_CREATE_STRUCTURES_INDEXES = [
    # Fast lookup by generation (used every loop iteration)
    "CREATE INDEX IF NOT EXISTS idx_structures_generation ON structures (generation);",
    # Fast lookup by status (used to find pending / converged structures)
    "CREATE INDEX IF NOT EXISTS idx_structures_status ON structures (status);",
    # Fast lookup of selectable structures
    "CREATE INDEX IF NOT EXISTS idx_structures_weight ON structures (weight);",
]

_CREATE_RUNS = """
CREATE TABLE IF NOT EXISTS runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Timing
    started_at          TEXT    NOT NULL DEFAULT (datetime('now')),  -- UTC ISO-8601
    ended_at            TEXT,                                        -- NULL while running

    -- Generation range covered by this invocation
    generation_start    INTEGER NOT NULL,
    generation_end      INTEGER,           -- NULL while running

    -- Conditions active during selection pressure for this invocation
    temperature         REAL    NOT NULL DEFAULT 298.15,  -- K
    pressure            REAL    NOT NULL DEFAULT 1.0,     -- atm
    potential           REAL    NOT NULL DEFAULT 0.0,     -- V vs RHE
    pH                  REAL    NOT NULL DEFAULT 0.0,

    -- Optional user annotation (e.g. "resumed after node crash")
    notes               TEXT    NOT NULL DEFAULT ''
);
"""

_CREATE_CONDITIONS = """
CREATE TABLE IF NOT EXISTS conditions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT    NOT NULL UNIQUE,  -- e.g. "acidic_low_U"
    temperature     REAL    NOT NULL DEFAULT 298.15,
    pressure        REAL    NOT NULL DEFAULT 1.0,
    potential       REAL    NOT NULL DEFAULT 0.0,
    pH              REAL    NOT NULL DEFAULT 0.0
);
"""

# Trigger: keep updated_at current on every UPDATE to structures
_CREATE_UPDATED_AT_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS trg_structures_updated_at
AFTER UPDATE ON structures
BEGIN
    UPDATE structures SET updated_at = datetime('now') WHERE id = NEW.id;
END;
"""

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_tables(conn: sqlite3.Connection) -> None:
    """
    Create all GOCIA tables and indexes if they do not already exist.

    Safe to call on an existing database — uses IF NOT EXISTS throughout.

    Parameters
    ----------
    conn:
        An open sqlite3 connection.  The caller is responsible for committing
        or rolling back.
    """
    cursor = conn.cursor()

    # Enable WAL mode for better concurrent read performance
    # (useful when gocia status / gocia inspect runs alongside gocia run)
    cursor.execute("PRAGMA journal_mode=WAL;")

    # Enforce foreign key constraints
    cursor.execute("PRAGMA foreign_keys=ON;")

    # Create tables
    cursor.execute(_CREATE_STRUCTURES)
    cursor.execute(_CREATE_RUNS)
    cursor.execute(_CREATE_CONDITIONS)

    # Create indexes
    for idx_sql in _CREATE_STRUCTURES_INDEXES:
        cursor.execute(idx_sql)

    # Create trigger
    cursor.execute(_CREATE_UPDATED_AT_TRIGGER)

    conn.commit()


def get_schema_version(conn: sqlite3.Connection) -> int:
    """
    Return the current SQLite user_version pragma.

    This is a lightweight way to track schema migrations without a full
    migration framework.  Start at 0 (SQLite default); increment when
    making breaking schema changes.
    """
    return conn.execute("PRAGMA user_version;").fetchone()[0]


def set_schema_version(conn: sqlite3.Connection, version: int) -> None:
    """Set the schema version pragma."""
    # PRAGMA user_version cannot be parameterised — safe because version is an int
    conn.execute(f"PRAGMA user_version = {version};")
    conn.commit()


CURRENT_SCHEMA_VERSION = 1
