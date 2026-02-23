from __future__ import annotations

import numpy as np
import pytest


class TestDistanceDesorptionDetector:

    def test_adsorbed_atom_not_flagged(self, bare_slab, slab_with_o):
        from gocia.desorption.distance import DistanceDesorptionDetector

        detector = DistanceDesorptionDetector(cutoff=4.0)
        assert not detector.detect(slab_with_o, bare_slab)

    def test_faraway_atom_flagged_as_desorbed(self, bare_slab):
        from gocia.desorption.distance import DistanceDesorptionDetector
        from ase import Atoms

        slab = bare_slab.copy()
        # Place O atom 20 Å above the surface
        top_z = slab.positions[:, 2].max()
        o = Atoms("O", positions=[[1.0, 1.0, top_z + 20.0]])
        slab_with_far_o = slab + o

        detector = DistanceDesorptionDetector(cutoff=4.0)
        assert detector.detect(slab_with_far_o, bare_slab)

    def test_bare_slab_not_flagged(self, bare_slab):
        from gocia.desorption.distance import DistanceDesorptionDetector

        detector = DistanceDesorptionDetector(cutoff=4.0)
        assert not detector.detect(bare_slab, bare_slab)

    def test_cutoff_respected(self, bare_slab):
        from gocia.desorption.distance import DistanceDesorptionDetector
        from ase import Atoms

        slab = bare_slab.copy()
        top_z = slab.positions[:, 2].max()
        # O atom exactly at cutoff + ε
        o = Atoms("O", positions=[[1.0, 1.0, top_z + 3.5]])
        slab_close = slab + o

        detector_tight = DistanceDesorptionDetector(cutoff=3.0)
        detector_loose = DistanceDesorptionDetector(cutoff=5.0)

        assert detector_tight.detect(slab_close, bare_slab)
        assert not detector_loose.detect(slab_close, bare_slab)

