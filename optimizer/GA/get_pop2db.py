from ase.ga.data import PrepareDB

db_file = 'ga_name.db'
db = PrepareDB(db_file_name=db_file,
                simulation_cell=slab,
                stoichiometry=atom_numbers,
                population_size=population_size)
for atoms in starting_population:
db.add_unrelaxed_candidate(atoms)
