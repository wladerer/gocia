"""
gocia/desorption/base.py

Abstract base class for adsorbate desorption detectors.

All detectors share the same interface: they receive a relaxed
slab+adsorbate structure and the bare slab, and return True if any
adsorbate has desorbed.

Detector lifecycle
------------------
Detectors run post-optimisation (after all calculator stages complete).
They never run during optimisation — that would require coupling to the
calculator loop in ways that are hard to generalise.

When desorption is detected:
  - The structure is flagged with status=DESORBED, weight=0
  - Its DFT energy is still recorded as a thermodynamic reference
  - It is not re-queued

Adding a new detector
---------------------
1. Subclass DesorptionDetector and implement detect()
2. Declare stage = "post_opt" (only option for now)
3. Register in DETECTOR_REGISTRY

Usage
-----
    from gocia.desorption.base import DETECTOR_REGISTRY
    detector = DETECTOR_REGISTRY["distance"]
    is_desorbed = detector.detect(relaxed_atoms, bare_slab)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ase import Atoms


class DesorptionDetector(ABC):
    """
    Abstract base class for desorption detectors.

    All detectors run post-optimisation and return a boolean.
    """

    #: When in the pipeline this detector runs.  Currently always "post_opt".
    stage: str = "post_opt"

    #: Short name used in DETECTOR_REGISTRY and gocia.yaml config.
    detector_name: str = ""

    @abstractmethod
    def detect(self, atoms: Atoms, slab: Atoms) -> bool:
        """
        Determine whether any adsorbate has desorbed.

        Parameters
        ----------
        atoms:
            The relaxed slab+adsorbate Atoms object (post-optimisation).
        slab:
            The bare slab Atoms object (same as the initial slab, used as a
            reference for identifying slab vs adsorbate atoms and for
            computing distances to the surface).

        Returns
        -------
        bool
            True if desorption is detected, False otherwise.
        """
        ...

    def __call__(self, atoms: Atoms, slab: Atoms) -> bool:
        """Convenience: allows calling the detector directly."""
        return self.detect(atoms, slab)


# ---------------------------------------------------------------------------
# Detector registry
# ---------------------------------------------------------------------------

DETECTOR_REGISTRY: dict[str, DesorptionDetector] = {}
