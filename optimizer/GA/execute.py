from ase.ga.data import DataConnection
from ase.optimize import BFGS
db = DataConnection(db_file)
while db.get_number_of_unrelaxed_candidates() > 0:
atoms = db.get_an_unrelaxed_candidate()
atoms.set_calculator(calc_scf())
atoms.info['key_value_pairs']['raw_score'] 
= -atoms.get_potential_energy()
db.add_relaxed_step(atoms)
from random import random
from ase.calculators.emt import EMT
for i in range(population_size * n_generation):
# 交叉
atoms1, atoms2 = population.get_two_candidates()
atoms3, desc = pairing.get_new_individual([atoms1, atoms2])
if atoms3 is None: continue
db.add_unrelaxed_candidate(atoms3, description=desc)
# 突然変異
if random() < mutation_probability:
atoms3_mut, desc = mutations.get_new_individual([atoms3])
if atoms3_mut is not None:
db.add_unrelaxed_step(atoms3_mut, description=desc)
atoms3 = atoms3_mut
# 子個体の構造緩和
atoms3.set_calculator(calc_scf()) 
atoms3.info['key_value_pairs']['raw_score'] = -atoms3.get_potential_energy()
db.add_relaxed_step(atoms3)
population.update()
print(f'{i}th structure relaxation completed')