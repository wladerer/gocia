from __future__ import annotations

import numpy as np
import pytest


class TestCHEFitness:

    def test_zero_adsorbates_returns_raw_energy(self):
        from gocia.fitness.che import grand_canonical_energy

        raw = -100.0
        result = grand_canonical_energy(
            raw_energy=raw,
            adsorbate_counts={},
            chemical_potentials={},
            potential=0.0,
            pH=0.0,
        )
        assert result == pytest.approx(raw)

    def test_single_adsorbate_subtracted(self):
        from gocia.fitness.che import grand_canonical_energy

        raw = -100.0
        mu_o = -4.92
        result = grand_canonical_energy(
            raw_energy=raw,
            adsorbate_counts={"O": 1},
            chemical_potentials={"O": mu_o},
            potential=0.0,
            pH=0.0,
        )
        assert result == pytest.approx(raw - mu_o)

    def test_multiple_adsorbates_all_subtracted(self):
        from gocia.fitness.che import grand_canonical_energy

        raw = -100.0
        result = grand_canonical_energy(
            raw_energy=raw,
            adsorbate_counts={"O": 2, "OH": 1},
            chemical_potentials={"O": -4.92, "OH": -3.75},
            potential=0.0,
            pH=0.0,
        )
        expected = raw - (2 * -4.92) - (1 * -3.75)
        assert result == pytest.approx(expected)

    def test_potential_shifts_energy(self):
        """Non-zero U should change the grand canonical energy via CHE."""
        from gocia.fitness.che import grand_canonical_energy

        raw = -100.0
        mu_oh = -3.75

        result_0v = grand_canonical_energy(
            raw_energy=raw,
            adsorbate_counts={"OH": 1},
            chemical_potentials={"OH": mu_oh},
            potential=0.0,
            pH=0.0,
        )
        result_1v = grand_canonical_energy(
            raw_energy=raw,
            adsorbate_counts={"OH": 1},
            chemical_potentials={"OH": mu_oh},
            potential=-1.0,
            pH=0.0,
        )
        assert result_0v != pytest.approx(result_1v)

    def test_pH_shifts_energy(self):
        """Non-zero pH should shift the CHE correction."""
        from gocia.fitness.che import grand_canonical_energy

        raw = -100.0
        result_ph0 = grand_canonical_energy(
            raw_energy=raw,
            adsorbate_counts={"OH": 1},
            chemical_potentials={"OH": -3.75},
            potential=0.0,
            pH=0.0,
        )
        result_ph7 = grand_canonical_energy(
            raw_energy=raw,
            adsorbate_counts={"OH": 1},
            chemical_potentials={"OH": -3.75},
            potential=0.0,
            pH=7.0,
        )
        assert result_ph0 != pytest.approx(result_ph7)

    def test_int_inputs_accepted(self):
        """pH and potential given as ints should not raise."""
        from gocia.fitness.che import grand_canonical_energy

        result = grand_canonical_energy(
            raw_energy=-100.0,
            adsorbate_counts={"O": 1},
            chemical_potentials={"O": -4.92},
            potential=0,   # int
            pH=7,          # int
        )
        assert isinstance(result, float)

    def test_unknown_adsorbate_in_counts_raises(self):
        """An adsorbate in counts with no matching chemical potential should raise."""
        from gocia.fitness.che import grand_canonical_energy

        with pytest.raises((KeyError, ValueError)):
            grand_canonical_energy(
                raw_energy=-100.0,
                adsorbate_counts={"X": 1},   # no μ for X
                chemical_potentials={"O": -4.92},
                potential=0.0,
                pH=0.0,
            )

    def test_reranking_consistency(self):
        """
        Re-evaluating at the same conditions as stored should reproduce the
        stored grand canonical energy.
        """
        from gocia.fitness.che import grand_canonical_energy

        params = dict(
            raw_energy=-100.0,
            adsorbate_counts={"O": 2},
            chemical_potentials={"O": -4.92},
            potential=-0.5,
            pH=3.0,
            temperature=298.15,
            pressure=1.0,
        )
        g1 = grand_canonical_energy(**params)
        g2 = grand_canonical_energy(**params)
        assert g1 == pytest.approx(g2)

