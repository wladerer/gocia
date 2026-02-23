from ase.build import molecule
from mace.calculators import mace_mp

calc = mace_mp(model="medium", device="cuda")

# O reference: ½ E(O₂)
o2 = molecule('O2')
o2.calc = calc
e_o2 = o2.get_potential_energy()
mu_O = e_o2 / 2

# CO reference: E(CO) - mu_O  (i.e. referenced to gas-phase CO and O)
co = molecule('CO')
co.calc = calc
e_co = co.get_potential_energy()
mu_CO = e_co  # or e_co - mu_O depending on your thermodynamic cycle

print(f"mu_O = {mu_O}")
print(f"mu_CO = {mu_CO}")
