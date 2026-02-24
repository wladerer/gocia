from ase.io import write
from ase.build import fcc100, molecule
from ase.constraints import FixAtoms
from mace.calculators import mace_mp


calc = mace_mp(model="medium", device="cuda")

# ------------------------------------------------------------------
# Gas-phase references
# ------------------------------------------------------------------

# O reference: ½ E(O₂)
o2 = molecule("O2")
o2.calc = calc
e_o2 = o2.get_potential_energy()
mu_O = e_o2 / 2

# CO reference: E(CO)
co = molecule("CO")
co.calc = calc
e_co = co.get_potential_energy()
mu_CO = e_co  # or e_co - mu_O depending on your thermodynamic cycle

# H₂ reference: E(H₂)
h2 = molecule("H2")
h2.calc = calc
e_h2 = h2.get_potential_energy()
mu_H = e_h2 / 2  # ½ E(H₂), chemical potential of one H atom

# ------------------------------------------------------------------
# Cu(100) slab: 4-layer 6×6 supercell, 15.336 Å × 15.336 Å
# a_Cu derived from cell dimension: a = 15.336 / (6 / sqrt(2)) ≈ 3.614 Å
# ------------------------------------------------------------------

import numpy as np

a_Cu = 15.336 / (6.0 / np.sqrt(2))  # lattice constant consistent with supercell

slab = fcc100(
    "Cu",
    size=(6, 6, 4),  # 6×6 in-plane, 4 layers
    a=a_Cu,
    vacuum=10.0,  # 10 Å vacuum on each side
    periodic=True,
)

# Verify / enforce exact cell dimensions
cell = slab.get_cell()
cell[0] = [15.336, 0.0, 0.0]
cell[1] = [0.0, 15.336, 0.0]
slab.set_cell(cell, scale_atoms=True)

# Fix bottom two layers to mimic bulk
z_positions = slab.get_positions()[:, 2]
z_sorted = np.unique(np.round(z_positions, 3))
fixed_z = z_sorted[:2]  # two lowest layers
fixed_mask = [atom.index for atom in slab if round(atom.position[2], 3) in fixed_z]
slab.set_constraint(FixAtoms(indices=fixed_mask))

slab.calc = calc
e_slab = slab.get_potential_energy()

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------

print(f"E(O2)    = {e_o2:.6f}  eV")
print(f"mu_O     = {mu_O:.6f}  eV  [½ E(O2)]")
print(f"E(CO)    = {e_co:.6f}  eV")
print(f"mu_CO    = {mu_CO:.6f}  eV")
print(f"E(H2)    = {e_h2:.6f}  eV")
print(f"mu_H     = {mu_H:.6f}  eV  [½ E(H2)]")
print(f"E(slab)  = {e_slab:.6f}  eV  [Cu(100) 6×6×4]")


# ------------------------------------------------------------------
# Uncomment to save slab for your own purposes
# ------------------------------------------------------------------
#
#write('slab.vasp',slab)


