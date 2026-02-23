"""
tests/test_database.py

Tests for:
  - SQLite schema creation (schema.py)
  - Sentinel file read/write/scan (status.py)
  - Individual CRUD operations (db.py)
  - Run tracking (db.py)
  - Pandas interface and re-ranking (db.py)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from gocia.population.individual import Individual, STATUS, OPERATOR


# ===========================================================================
# schema.py
# ===========================================================================

class TestSchema:

    def test_create_tables_is_idempotent(self, tmp_path):
        """Calling create_tables twice on the same DB should not raise."""
        from gocia.database.schema import create_tables
        conn = sqlite3.connect(str(tmp_path / "gocia.db"))
        create_tables(conn)
        create_tables(conn)   # second call — must not fail
        conn.close()

    def test_expected_tables_exist(self, tmp_path):
        from gocia.database.schema import create_tables
        conn = sqlite3.connect(str(tmp_path / "gocia.db"))
        create_tables(conn)
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "structures" in tables
        assert "runs" in tables
        assert "conditions" in tables
        conn.close()

    def test_expected_indexes_exist(self, tmp_path):
        from gocia.database.schema import create_tables
        conn = sqlite3.connect(str(tmp_path / "gocia.db"))
        create_tables(conn)
        indexes = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        assert "idx_structures_generation" in indexes
        assert "idx_structures_status" in indexes
        conn.close()

    def test_schema_version_set_on_fresh_db(self, tmp_db):
        from gocia.database.schema import get_schema_version, CURRENT_SCHEMA_VERSION
        conn = tmp_db._conn
        assert get_schema_version(conn) == CURRENT_SCHEMA_VERSION

    def test_wal_mode_enabled(self, tmp_db):
        mode = tmp_db._conn.execute("PRAGMA journal_mode;").fetchone()[0]
        assert mode == "wal"


# ===========================================================================
# status.py  (sentinel files)
# ===========================================================================

class TestSentinelFiles:

    def test_write_and_read_simple_statuses(self, struct_dir):
        from gocia.database.status import write_sentinel, read_sentinel

        for status in (STATUS.PENDING, STATUS.SUBMITTED, STATUS.CONVERGED,
                       STATUS.DESORBED, STATUS.FAILED, STATUS.DUPLICATE, STATUS.ISOMER):
            write_sentinel(struct_dir, status)
            assert read_sentinel(struct_dir) == status

    def test_write_and_read_stage_statuses(self, struct_dir):
        from gocia.database.status import write_sentinel, read_sentinel

        for n in (1, 2, 3):
            write_sentinel(struct_dir, STATUS.running_stage(n))
            assert read_sentinel(struct_dir) == STATUS.running_stage(n)

            write_sentinel(struct_dir, STATUS.converged_stage(n))
            assert read_sentinel(struct_dir) == STATUS.converged_stage(n)

    def test_write_replaces_previous_sentinel(self, struct_dir):
        from gocia.database.status import write_sentinel, read_sentinel, _all_sentinel_files

        write_sentinel(struct_dir, STATUS.PENDING)
        write_sentinel(struct_dir, STATUS.SUBMITTED)

        assert read_sentinel(struct_dir) == STATUS.SUBMITTED
        # Exactly one sentinel file should remain
        assert len(_all_sentinel_files(struct_dir)) == 1

    def test_read_returns_none_for_empty_dir(self, struct_dir):
        from gocia.database.status import read_sentinel
        assert read_sentinel(struct_dir) is None

    def test_read_returns_none_for_missing_dir(self, tmp_path):
        from gocia.database.status import read_sentinel
        assert read_sentinel(tmp_path / "nonexistent") is None

    def test_clear_sentinels(self, struct_dir):
        from gocia.database.status import write_sentinel, clear_sentinels, read_sentinel

        write_sentinel(struct_dir, STATUS.CONVERGED)
        n = clear_sentinels(struct_dir)
        assert n == 1
        assert read_sentinel(struct_dir) is None

    def test_sentinel_exists(self, struct_dir):
        from gocia.database.status import write_sentinel, sentinel_exists

        write_sentinel(struct_dir, STATUS.CONVERGED)
        assert sentinel_exists(struct_dir, STATUS.CONVERGED)
        assert not sentinel_exists(struct_dir, STATUS.PENDING)

    def test_write_raises_for_missing_dir(self, tmp_path):
        from gocia.database.status import write_sentinel
        with pytest.raises(FileNotFoundError):
            write_sentinel(tmp_path / "ghost", STATUS.PENDING)

    def test_scan_generation(self, gen0_dir):
        from gocia.database.status import scan_generation

        result = scan_generation(gen0_dir)
        assert len(result) == 3
        for name, status in result.items():
            assert status == STATUS.PENDING
            assert name.startswith("struct_")

    def test_scan_generation_mixed_statuses(self, gen0_dir):
        from gocia.database.status import scan_generation, write_sentinel

        dirs = sorted(gen0_dir.iterdir())
        write_sentinel(dirs[0], STATUS.CONVERGED)
        write_sentinel(dirs[1], STATUS.RUNNING_1 if hasattr(STATUS, "RUNNING_1")
                       else STATUS.running_stage(1))

        result = scan_generation(gen0_dir)
        statuses = list(result.values())
        assert STATUS.CONVERGED in statuses
        assert STATUS.running_stage(1) in statuses


# ===========================================================================
# db.py — Individual CRUD
# ===========================================================================

class TestIndividualCRUD:

    def test_insert_and_get(self, tmp_db, one_individual):
        tmp_db.insert(one_individual)
        fetched = tmp_db.get(one_individual.id)
        assert fetched is not None
        assert fetched.id == one_individual.id
        assert fetched.generation == one_individual.generation
        assert fetched.status == STATUS.PENDING

    def test_insert_duplicate_raises(self, tmp_db, one_individual):
        tmp_db.insert(one_individual)
        with pytest.raises(Exception):   # sqlite3.IntegrityError
            tmp_db.insert(one_individual)

    def test_get_nonexistent_returns_none(self, tmp_db):
        assert tmp_db.get("definitely_not_a_real_id") is None

    def test_update_status(self, tmp_db, one_individual):
        tmp_db.insert(one_individual)
        updated = one_individual.with_status(STATUS.SUBMITTED)
        tmp_db.update_status(updated)

        fetched = tmp_db.get(one_individual.id)
        assert fetched.status == STATUS.SUBMITTED

    def test_update_energy(self, tmp_db, one_individual):
        tmp_db.insert(one_individual)
        updated = one_individual.with_energy(raw=-125.0, grand_canonical=-10.5)
        tmp_db.update_energy(updated)

        fetched = tmp_db.get(one_individual.id)
        assert fetched.raw_energy == pytest.approx(-125.0)
        assert fetched.grand_canonical_energy == pytest.approx(-10.5)

    def test_update_nonexistent_raises(self, tmp_db, one_individual):
        with pytest.raises(KeyError):
            tmp_db.update(one_individual)

    def test_get_generation(self, populated_db, small_population):
        gen0 = populated_db.get_generation(0)
        # small_population has 4 gen-0 individuals and 1 gen-1
        gen0_ids = {ind.id for ind in gen0}
        for ind in small_population:
            if ind.generation == 0:
                assert ind.id in gen0_ids

    def test_get_by_status(self, populated_db):
        converged = populated_db.get_by_status(STATUS.CONVERGED)
        assert all(ind.status == STATUS.CONVERGED for ind in converged)
        assert len(converged) == 2  # ind0 and ind1 in small_population

    def test_get_selectable_excludes_desorbed_and_pending(self, populated_db):
        selectable = populated_db.get_selectable()
        for ind in selectable:
            assert ind.status in (STATUS.CONVERGED, STATUS.ISOMER)
            assert ind.weight > 0

    def test_best_returns_lowest_gce(self, populated_db):
        best = populated_db.best(n=1)
        assert len(best) == 1
        assert best[0].grand_canonical_energy == pytest.approx(-10.5)

    def test_best_n_returns_n_results(self, populated_db):
        best = populated_db.best(n=2)
        assert len(best) == 2
        # Should be sorted ascending
        assert best[0].grand_canonical_energy <= best[1].grand_canonical_energy

    def test_count_by_status(self, populated_db):
        counts = populated_db.count_by_status()
        assert counts[STATUS.CONVERGED] == 2
        assert counts[STATUS.ISOMER] == 1
        assert counts[STATUS.DESORBED] == 1
        assert counts[STATUS.PENDING] == 1

    def test_count_by_status_filtered_by_generation(self, populated_db):
        counts = populated_db.count_by_status(generation=1)
        assert counts.get(STATUS.PENDING, 0) == 1
        assert STATUS.CONVERGED not in counts

    def test_fingerprints_returns_stored(self, tmp_db):
        ind = Individual(
            generation=0,
            fingerprint=[0.1, 0.2, 0.3],
        )
        tmp_db.insert(ind)
        fps = tmp_db.fingerprints()
        assert len(fps) == 1
        stored_id, stored_fp = fps[0]
        assert stored_id == ind.id
        assert stored_fp == pytest.approx([0.1, 0.2, 0.3])

    def test_roundtrip_parent_ids(self, tmp_db, small_population):
        # ind4 has two parent IDs
        offspring = small_population[4]
        tmp_db.insert(offspring)
        fetched = tmp_db.get(offspring.id)
        assert fetched.parent_ids == offspring.parent_ids
        assert len(fetched.parent_ids) == 2

    def test_roundtrip_extra_data(self, tmp_db):
        ind = Individual(
            generation=0,
            extra_data={"adsorbate_counts": {"O": 2, "OH": 1}, "custom": 42},
        )
        tmp_db.insert(ind)
        fetched = tmp_db.get(ind.id)
        assert fetched.extra_data["adsorbate_counts"] == {"O": 2, "OH": 1}
        assert fetched.extra_data["custom"] == 42

    def test_roundtrip_boolean_flags(self, tmp_db):
        ind = Individual(
            generation=0,
            desorption_flag=True,
            is_isomer=True,
            isomer_of="some_parent_id",
        )
        tmp_db.insert(ind)
        fetched = tmp_db.get(ind.id)
        assert fetched.desorption_flag is True
        assert fetched.is_isomer is True
        assert fetched.isomer_of == "some_parent_id"


# ===========================================================================
# db.py — Run tracking
# ===========================================================================

class TestRunTracking:

    def test_start_run_returns_id(self, tmp_db):
        run_id = tmp_db.start_run(
            generation_start=0,
            temperature=298.15,
            pressure=1.0,
            potential=0.0,
            pH=7.0,
        )
        assert isinstance(run_id, int)
        assert run_id >= 1

    def test_end_run_records_generation(self, tmp_db):
        run_id = tmp_db.start_run(0, 298.15, 1.0, 0.0, 7.0)
        tmp_db.end_run(run_id, generation_end=5)

        runs = tmp_db.get_runs()
        assert len(runs) == 1
        assert runs[0]["generation_end"] == 5
        assert runs[0]["ended_at"] is not None

    def test_multiple_runs_tracked(self, tmp_db):
        for i in range(3):
            run_id = tmp_db.start_run(i * 10, 298.15, 1.0, float(i) * -0.1, 7.0)
            tmp_db.end_run(run_id, (i + 1) * 10)

        runs = tmp_db.get_runs()
        assert len(runs) == 3

    def test_conditions_saved_with_run(self, tmp_db):
        tmp_db.start_run(0, 350.0, 2.0, -0.5, 3.0)
        runs = tmp_db.get_runs()
        assert runs[0]["temperature"] == pytest.approx(350.0)
        assert runs[0]["potential"] == pytest.approx(-0.5)
        assert runs[0]["pH"] == pytest.approx(3.0)

    def test_save_and_get_named_conditions(self, tmp_db):
        tmp_db.save_conditions("acidic_low_U", 298.15, 1.0, -0.5, 3.0)
        cond = tmp_db.get_conditions("acidic_low_U")
        assert cond is not None
        assert cond["potential"] == pytest.approx(-0.5)
        assert cond["pH"] == pytest.approx(3.0)

    def test_save_conditions_upsert(self, tmp_db):
        """Saving the same name twice should overwrite, not duplicate."""
        tmp_db.save_conditions("test", 298.15, 1.0, 0.0, 7.0)
        tmp_db.save_conditions("test", 350.0, 1.0, -0.3, 2.0)
        cond = tmp_db.get_conditions("test")
        assert cond["temperature"] == pytest.approx(350.0)
        assert cond["potential"] == pytest.approx(-0.3)

    def test_get_nonexistent_conditions_returns_none(self, tmp_db):
        assert tmp_db.get_conditions("does_not_exist") is None


# ===========================================================================
# db.py — Pandas interface
# ===========================================================================

class TestPandasInterface:

    pytest.importorskip("pandas", reason="pandas required for DataFrame tests")

    def test_to_dataframe_returns_all(self, populated_db, small_population):
        df = populated_db.to_dataframe()
        assert len(df) == len(small_population)

    def test_to_dataframe_filtered_by_generation(self, populated_db):
        df = populated_db.to_dataframe(generation=0)
        assert all(df["generation"] == 0)

    def test_to_dataframe_filtered_by_status(self, populated_db):
        df = populated_db.to_dataframe(status=STATUS.CONVERGED)
        assert all(df["status"] == STATUS.CONVERGED)
        assert len(df) == 2

    def test_to_dataframe_boolean_columns(self, populated_db):
        df = populated_db.to_dataframe()
        assert df["desorption_flag"].dtype == bool
        assert df["is_isomer"].dtype == bool

    def test_to_dataframe_parent_ids_decoded(self, populated_db, small_population):
        df = populated_db.to_dataframe()
        # ind4 has two parent IDs
        offspring_row = df[df["operator"] == OPERATOR.SPLICE]
        assert len(offspring_row) == 1
        parent_ids = offspring_row.iloc[0]["parent_ids"]
        assert isinstance(parent_ids, list)
        assert len(parent_ids) == 2

    def test_to_dataframe_extra_data_decoded(self, populated_db):
        df = populated_db.to_dataframe()
        converged = df[df["status"] == STATUS.CONVERGED]
        for _, row in converged.iterrows():
            assert isinstance(row["extra_data"], dict)

    def test_summary_contains_expected_fields(self, populated_db):
        summary = populated_db.summary()
        assert "Structures total" in summary
        assert "Best G" in summary
        assert STATUS.CONVERGED in summary


# ===========================================================================
# db.py — GociaDB context manager
# ===========================================================================

class TestContextManager:

    def test_context_manager_connects_and_closes(self, tmp_path):
        from gocia.database.db import GociaDB
        db_path = tmp_path / "ctx.db"
        with GociaDB(db_path) as db:
            db.setup()
            assert db._conn is not None
        assert db._conn is None

    def test_require_connection_raises_when_disconnected(self, tmp_path):
        from gocia.database.db import GociaDB
        db = GociaDB(tmp_path / "x.db")
        with pytest.raises(RuntimeError, match="not connected"):
            db._require_connection()
