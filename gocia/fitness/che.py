"""
gocia/fitness/che.py

Computational Hydrogen Electrode (CHE) grand canonical energy.

Theory
------
The grand canonical energy of a surface configuration under the CHE framework
is:

    G(U, pH, T, P) = E_DFT - Σ_i [ n_i · μ_i(U, pH, T, P) ]

where:
  - E_DFT is the total DFT (or MACE) energy of the relaxed slab+adsorbate
  - n_i is the count of adsorbate species i
  - μ_i is the chemical potential of species i corrected for electrochemical
    conditions

CHE potential correction
------------------------
Under the CHE, the free energy of a proton-electron pair (H⁺ + e⁻) at
electrode potential U vs RHE and pH is referenced to ½H₂:

    μ(H⁺ + e⁻) = ½ μ(H₂) - eU

The RHE convention already folds pH into the reference, so at standard
conditions (pH=0, U=0 vs RHE) the correction vanishes.  For non-zero pH
and U:

    ΔG_CHE = -n_e · (eU + k_B T ln(10) · pH)

where n_e is the number of proton-electron pairs transferred.

In this implementation, the number of electrons transferred per adsorbate
molecule is specified in the CHE_ELECTRON_COUNT registry below.  For simple
atomic adsorbates like O (formed by O²⁻ + 2(H⁺+e⁻) → O* + H₂O), n_e = 2.
For OH, n_e = 1.  Species not in the registry default to n_e = 0 (no
electrochemical correction).

The registry can be extended for new adsorbates without changing the core
function.

Pressure correction
-------------------
A simple ideal-gas pressure correction is applied for gaseous references:

    Δμ(T, P) = k_B T ln(P / P_ref)

where P_ref = 1 atm.  This affects the chemical potential of adsorbates
referenced to gas-phase species.  Currently disabled by default (pressure
correction only applied when temperature > 0 and pressure != 1.0).

Usage
-----
    from gocia.fitness.che import grand_canonical_energy, compute_fitness

    G = grand_canonical_energy(
        raw_energy=-125.0,
        adsorbate_counts={"O": 2, "OH": 1},
        chemical_potentials={"O": -4.92, "OH": -3.75},
        potential=-0.5,
        pH=7.0,
        temperature=298.15,
        pressure=1.0,
    )
"""

from __future__ import annotations

import math
from typing import Any

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

#: Boltzmann constant in eV/K
K_B: float = 8.617333262e-5

#: Reference pressure in atm
P_REF: float = 1.0

# ---------------------------------------------------------------------------
# Electron count registry
# ---------------------------------------------------------------------------
# Maps adsorbate symbol → number of (H⁺ + e⁻) pairs transferred during
# adsorption from the gas phase reference.
#
# These values follow the standard CHE convention for oxygen reduction /
# oxygen evolution intermediates on metal surfaces.
#
# To add new adsorbates: just extend this dict.  Unknown species default to 0.

CHE_ELECTRON_COUNT: dict[str, int] = {
    "O":   2,   # ½O₂ + 2(H⁺+e⁻) → O* + H₂O
    "OH":  1,   # ½O₂ + H⁺ + e⁻  → OH*
    "OOH": 3,   # O₂  + 3(H⁺+e⁻) → OOH*
    "H":   1,   # H⁺  + e⁻        → H*
    # Extend as needed, e.g.:
    # "N":  3,
    # "NH": 2,
    # "NH2": 1,
    # "NH3": 0,
}


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def grand_canonical_energy(
    raw_energy: float,
    adsorbate_counts: dict[str, int],
    chemical_potentials: dict[str, float],
    potential: float = 0.0,
    pH: float = 0.0,
    temperature: float = 298.15,
    pressure: float = 1.0,
) -> float:
    """
    Compute the CHE grand canonical energy of a surface configuration.

    Parameters
    ----------
    raw_energy:
        Total DFT (or MACE) energy of the relaxed slab+adsorbate system (eV).
    adsorbate_counts:
        Mapping of adsorbate symbol → integer count, e.g. {"O": 2, "OH": 1}.
        An empty dict returns raw_energy unchanged.
    chemical_potentials:
        Mapping of adsorbate symbol → standard chemical potential (eV).
        Every species present in adsorbate_counts must have an entry here.
    potential:
        Electrode potential in V vs RHE.  Negative values stabilise oxidised
        surfaces (more negative U → larger -eU correction).
        Accepts int (coerced to float).
    pH:
        Solution pH.  Under the CHE with RHE reference, pH enters as
        k_B T ln(10) · pH per transferred electron.
        Accepts int (coerced to float).
    temperature:
        Temperature in K.  Used for the pH correction and pressure term.
        Default 298.15 K.
    pressure:
        Partial pressure of the reference gas species in atm.  Only affects
        the grand canonical energy when pressure ≠ 1.0.  Default 1.0 atm.

    Returns
    -------
    float
        Grand canonical energy G in eV.

    Raises
    ------
    KeyError
        If a species in adsorbate_counts has no entry in chemical_potentials.

    Examples
    --------
    >>> grand_canonical_energy(-100.0, {"O": 1}, {"O": -4.92}, 0.0, 0.0)
    -95.08
    >>> grand_canonical_energy(-100.0, {}, {}, 0.0, 0.0)
    -100.0
    """
    # Coerce int inputs — pydantic handles this at the config level, but
    # this function may be called directly in notebooks or scripts
    potential = float(potential)
    pH = float(pH)
    temperature = float(temperature)
    pressure = float(pressure)

    if not adsorbate_counts:
        return float(raw_energy)

    # Validate all species are covered before computing anything
    missing = set(adsorbate_counts) - set(chemical_potentials)
    if missing:
        raise KeyError(
            f"No chemical potential defined for adsorbate species: {sorted(missing)}. "
            "Add them to the 'adsorbates' section of gocia.yaml."
        )

    # CHE correction per electron transferred:
    #   ΔG = -e(U + k_B T ln(10) · pH)
    # Per species: multiply by n_e (electron count for that species)
    kt_ln10_pH = K_B * temperature * math.log(10) * pH
    correction_per_electron = -(potential + kt_ln10_pH)  # eV per electron

    # Pressure correction: k_B T ln(P / P_ref)
    # Only meaningful when pressure ≠ 1.0 atm
    pressure_correction = 0.0
    if abs(pressure - P_REF) > 1e-10 and temperature > 0:
        pressure_correction = K_B * temperature * math.log(pressure / P_REF)

    # Sum over all adsorbate species
    grand_canonical = float(raw_energy)
    for symbol, count in adsorbate_counts.items():
        if count == 0:
            continue

        mu = chemical_potentials[symbol]
        n_e = CHE_ELECTRON_COUNT.get(symbol, 0)

        # Electrochemical correction for this species
        electrochemical = n_e * correction_per_electron

        # Effective chemical potential at these conditions
        mu_effective = mu + electrochemical + pressure_correction

        grand_canonical -= count * mu_effective

    return grand_canonical


# ---------------------------------------------------------------------------
# Convenience wrapper for the GA loop
# ---------------------------------------------------------------------------

def compute_fitness(
    individual,
    slab_energy: float,
    chemical_potentials: dict[str, float],
    potential: float,
    pH: float,
    temperature: float = 298.15,
    pressure: float = 1.0,
) -> float:
    """
    Compute the grand canonical fitness for an Individual.

    Extracts adsorbate_counts from individual.extra_data["adsorbate_counts"]
    and calls grand_canonical_energy.  Returns None if the individual has no
    raw_energy (not yet converged).

    This is a thin wrapper used by the GA loop — it handles the adsorbate
    count lookup so the loop doesn't need to know about extra_data layout.

    Parameters
    ----------
    individual:
        An Individual instance with raw_energy set and extra_data containing
        "adsorbate_counts".
    slab_energy:
        DFT energy of the clean bare slab (eV).  Currently unused in the
        CHE fitness (the slab is a constant reference), but retained for
        future surface stoichiometry support.
    chemical_potentials:
        Symbol → μ mapping from GoicaConfig.
    potential, pH, temperature, pressure:
        Thermodynamic conditions.

    Returns
    -------
    float or None
        Grand canonical energy in eV, or None if individual.raw_energy is None.
    """
    if individual.raw_energy is None:
        return None

    adsorbate_counts = individual.extra_data.get("adsorbate_counts", {})

    return grand_canonical_energy(
        raw_energy=individual.raw_energy,
        adsorbate_counts=adsorbate_counts,
        chemical_potentials=chemical_potentials,
        potential=potential,
        pH=pH,
        temperature=temperature,
        pressure=pressure,
    )


# ---------------------------------------------------------------------------
# Condition sweep utility
# ---------------------------------------------------------------------------

def sweep_conditions(
    raw_energy: float,
    adsorbate_counts: dict[str, int],
    chemical_potentials: dict[str, float],
    potentials: list[float],
    pHs: list[float],
    temperature: float = 298.15,
    pressure: float = 1.0,
) -> list[dict[str, Any]]:
    """
    Evaluate grand canonical energy over a grid of (potential, pH) values.

    Useful for generating volcano/Pourbaix-style plots post-hoc.

    Parameters
    ----------
    raw_energy:
        Total DFT energy of the structure (eV).
    adsorbate_counts:
        Adsorbate stoichiometry.
    chemical_potentials:
        Standard chemical potentials (eV).
    potentials:
        List of electrode potentials to evaluate (V vs RHE).
    pHs:
        List of pH values to evaluate.
    temperature, pressure:
        Fixed thermodynamic conditions for the sweep.

    Returns
    -------
    list[dict]
        Each dict has keys: "potential", "pH", "grand_canonical_energy".
    """
    results = []
    for U in potentials:
        for ph in pHs:
            G = grand_canonical_energy(
                raw_energy=raw_energy,
                adsorbate_counts=adsorbate_counts,
                chemical_potentials=chemical_potentials,
                potential=U,
                pH=ph,
                temperature=temperature,
                pressure=pressure,
            )
            results.append({"potential": U, "pH": ph, "grand_canonical_energy": G})
    return results
