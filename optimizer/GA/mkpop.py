import os
import numpy as np
from ase.atoms import Atoms
from ase.ga.startgenerator import StartGenerator
from ase.ga.utilities import closest_distances_generator

L1 = [2.504, 0.000, 0.000]
L2 = [-1.252, 2.169, 0.000]
L3 = [0.000, 0.000, c]

atom_numbers = 2 * [5] + 2 * [7]
blocks = ['B'] * 2 + ['N'] * 2
slab = Atoms(pbc=True, cell=np.array([L1, L2, L3]))
blmin = closest_distances_generator(atom_numbers=atom_numbers, 
                                    ratio_of_covalent_radii=0.5)
sg = StartGenerator(slab, blocks, blmin)
population_size = N
starting_population = [sg.get_new_candidate() for i in range(population_size)]
