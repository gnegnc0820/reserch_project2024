import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from ase import Atoms
from ase.build import bulk
from ase.optimize import BFGS, GPMin
from ase.constraints import UnitCellFilter
from ase.visualize import view
from ase.io.trajectory import Trajectory
from ase.calculators.espresso import Espresso
b_pp = "B.pbe-n-kjpaw_psl.1.0.0.UPF"
n_pp = "N.pbe-n-kjpaw_psl.1.0.0.UPF"
def calc_scf():
cutoff:[eV] 
pseudopotentials = {'B':b_pp,'N':n_pp}
cmd = 'mpirun -np 6 pw.x < espresso.pwi > espresso.pwo'
input_data = {
'control':{
'pseudo_dir':'/home/user/qe/pseudo',
'calculation':'scf', 
'restart_mode':'from_scratch', 
'outdir':'./work', 
'prefix':'BN', 
#'etot_conv_thr':0.00001, 
#'forc_conv_thr':0.0001,
'nstep':500
},
'system':{
'ibrav':0,
'nat':4,
'ntyp':2,
'ecutwfc':43,
'occupations':'smearing',
'smearing':'m-p',
'degauss':0.01
},
'electrons':{
'mixing_beta':0.5,
'conv_thr':0.000001,
'diagonalization':'cg'
}
}
calc = Espresso(command=cmd,
pseudopotentials=pseudopotentials,
kpts=(6, 6, 6, 0, 0, 0),
tprnfor=True,#ASE を使った構造最適化時に必要
wf_collect=True,
input_data=input_data)
return calc

from random import random
from ase.ga.data import DataConnection
from ase.ga.population import Population
from ase.ga.standard_comparators 
import InteratomicDistanceComparator
from ase.ga.cutandsplicepairing import CutAndSplicePairing
from ase.ga.offspring_creator import OperationSelector
from ase.ga.standardmutations 
import (MirrorMutation, RattleMutation)
mutation_probability = 0.3
n_generation = 50
comperator = InteratomicDistanceComparator(n_top=len(atom_numbers), 
pair_cor_cum_diff=0.015,
pair_cor_max=0.7,
dE=0.02, mic=False)
pairing = CutAndSplicePairing(slab, len(atom_numbers), blmin)
mutations = OperationSelector([1., 1.], 
[MirrorMutation(blmin, len(atom_numbers)),
RattleMutation(blmin, len(atom_numbers))])
population = Population(data_connection=db,
population_size=population_size,
comparator=comperator)