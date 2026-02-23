from __future__ import annotations

import numpy as np
import pytest


class TestDistanceHistogramFingerprint:

    def test_identical_structures_give_identical_fingerprints(self, slab_with_o):
        from gocia.structure.fingerprint import distance_histogram

        fp1 = distance_histogram(slab_with_o)
        fp2 = distance_histogram(slab_with_o.copy())
        assert np.allclose(fp1, fp2, atol=1e-8)

    def test_different_structures_give_different_fingerprints(
        self, slab_with_o, slab_with_oh
    ):
        from gocia.structure.fingerprint import distance_histogram

        fp1 = distance_histogram(slab_with_o)
        fp2 = distance_histogram(slab_with_oh)
        assert not np.allclose(fp1, fp2, atol=1e-4)

    def test_fingerprint_is_list_of_floats(self, slab_with_o):
        from gocia.structure.fingerprint import distance_histogram

        fp = distance_histogram(slab_with_o)
        assert isinstance(fp, list)
        assert all(isinstance(v, float) for v in fp)

    def test_fingerprint_length_consistent(self, slab_with_o, slab_with_oh):
        """Two fingerprints from the same function must have the same length."""
        from gocia.structure.fingerprint import distance_histogram

        fp1 = distance_histogram(slab_with_o)
        fp2 = distance_histogram(slab_with_oh)
        assert len(fp1) == len(fp2)

    def test_small_displacement_gives_small_fingerprint_distance(self, slab_with_o):
        from gocia.structure.fingerprint import distance_histogram, fingerprint_distance

        displaced = slab_with_o.copy()
        displaced.positions[-1] += [0.05, 0.05, 0.05]   # tiny nudge

        fp1 = distance_histogram(slab_with_o)
        fp2 = distance_histogram(displaced)
        dist = fingerprint_distance(fp1, fp2)
        assert dist < 0.1, f"Expected small distance for tiny displacement, got {dist}"

    def test_large_displacement_gives_large_fingerprint_distance(
        self, slab_with_o, slab_with_oh
    ):
        from gocia.structure.fingerprint import distance_histogram, fingerprint_distance

        fp1 = distance_histogram(slab_with_o)
        fp2 = distance_histogram(slab_with_oh)
        dist = fingerprint_distance(fp1, fp2)
        assert dist > 0.1, f"Expected large distance for different structures, got {dist}"


class TestDuplicateAndIsomerDetection:

    def test_exact_duplicate_detected(self, slab_with_o):
        from gocia.structure.fingerprint import classify_structure, distance_histogram

        existing_fps = [distance_histogram(slab_with_o)]
        new_fp = distance_histogram(slab_with_o.copy())

        result = classify_structure(
            new_fp, existing_fps,
            duplicate_threshold=0.01,
            isomer_threshold=0.1,
        )
        assert result == "duplicate"

    def test_near_duplicate_classified_as_isomer(self, slab_with_o):
        from gocia.structure.fingerprint import classify_structure, distance_histogram

        existing_fps = [distance_histogram(slab_with_o)]

        slightly_displaced = slab_with_o.copy()
        slightly_displaced.positions[-1] += [0.1, 0.0, 0.0]
        new_fp = distance_histogram(slightly_displaced)

        result = classify_structure(
            new_fp, existing_fps,
            duplicate_threshold=0.01,
            isomer_threshold=0.5,
        )
        assert result in ("isomer", "duplicate")

    def test_unique_structure_classified_as_unique(self, slab_with_o, slab_with_oh):
        from gocia.structure.fingerprint import classify_structure, distance_histogram

        existing_fps = [distance_histogram(slab_with_o)]
        new_fp = distance_histogram(slab_with_oh)

        result = classify_structure(
            new_fp, existing_fps,
            duplicate_threshold=0.01,
            isomer_threshold=0.1,
        )
        assert result == "unique"

    def test_empty_existing_population_always_unique(self, slab_with_o):
        from gocia.structure.fingerprint import classify_structure, distance_histogram

        fp = distance_histogram(slab_with_o)
        result = classify_structure(fp, [], duplicate_threshold=0.01, isomer_threshold=0.1)
        assert result == "unique"

