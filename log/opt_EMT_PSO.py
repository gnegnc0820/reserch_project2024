# import libs
import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt

# import ase
from ase import Atoms
from ase.optimize import BFGS
from ase.visualize import view
from ase.io.trajectory import Trajectory
from ase.calculators.espresso import Espresso
from ase.calculators.emt import EMT
import nglview as nv

# improt optimizer
from optimizer.PSO import PSO
from optimizer.Optimizer import Optimizer
from optimizer.Env import Env

#教科書の例にならい、pw91の擬ポテンシャルを使います。
#!wget http://nninc.cnf.cornell.edu/psp_files/O.pw91-van_ak.UPF
#!wget http://nninc.cnf.cornell.edu/psp_files/H.pw91-van_ak.UPF
o_pp = "pseudo/O.pw91-van_ak.UPF"
h_pp = "pseudo/H.pw91-van_ak.UPF"

c_pp = "pseudo/C.pw91-van_ak.UPF"
n_pp = "pseudo/N.pw91-van_ak.UPF"
b_pp = "pseudo/B.pz-vbc.UPF"

material = "C8"
atoms_n = 8

def calc_dft(cutoff=300.0,Mk=3):
    # QEでDFT計算をする. cutoff:[eV] 
    eV_to_Ry =  1/13.605698#[eV/Ry] 
    pseudopotentials = {
        'O':o_pp,'H':h_pp,'C':c_pp,'N':n_pp,'B':b_pp
    }
    #実行環境に合わせて、必要に応じて下記を変えてください。
    cmd = 'pw.x -in espresso.pwi > espresso.pwo'

    input_data = {
        'control':{'pseudo_dir':'./pseudo'},
        'system': {
            'ecutwfc': cutoff*eV_to_Ry,      #[eV]->[Ry]
            'ecutrho': cutoff*4*eV_to_Ry},   #ecutwfc * 4
        'disk_io': 'low'}

    calc = Espresso(command=cmd,
                    pseudopotentials=pseudopotentials,
                    kpts=(Mk, Mk, Mk),
                    tprnfor=True,#ASEを使った構造最適化時に必要
                    input_data=input_data)
    return calc

def make_model(molc,L,positions):
    # 座標を指定して分子モデルを作成する
    # 原点に１つ、対称位置に原子を２つ配置して作成
    model = Atoms(molc, 
                positions=positions,
                cell=L,
                pbc=[True, True, True])
    return model

def calc_length(a,b):
    #結合長を計算する
    return np.linalg.norm((a-b))

def calc_angle(center,a0,b0):
    #結合角を計算する。centerに中心の原子の座標を指定。
    a = a0 - center
    b = b0 - center
    cos = np.inner(a,b)/(np.linalg.norm(a) * np.linalg.norm(b))
    deg = np.rad2deg(np.arccos(np.clip(cos, -1.0, 1.0)))
    return deg

def f_obj(x,data=None,visualize=False):
    # return np.sum(np.array(x)**2),data
    #分子モデルを作成
    # molecule = make_model(molc='OH2', L=10.0, a=1.2, b=0.5)
    positions = [tuple(x[i:i + 3]) for i in range(0, len(x), 3)]
    # molecule = make_model(molc='OH2', L=[10,10,10],positions=positions)
    molecule = make_model(molc=material, L=data,positions=positions)
    # molecule.pbc = [True, True, True]
    #必要に応じて可視化して確認
    if visualize:
        
        # molecule.pbc = [True, True, False]
        view(molecule)
        # molecule.pbc = [True, True, True]

    molecule.set_calculator(EMT())
    energy = molecule.get_potential_energy()
    print(f"Total energy:{energy}")
    return energy,data

# # PSO 開始
max_vec_size = [3.56,3.56,3.56]
dim = 3 * atoms_n
UB = np.array(max_vec_size*(dim//3))
LB = np.array([0 for i in range(len(UB))])

env = Env(f_obj=f_obj, data=max_vec_size)
opt = Optimizer(N=30, T=100, LB=LB, UB=UB, Dim=dim, env=env)
pso = PSO()

Best_F, Best_P, env = pso.exp(opt)
print(f"best(PSO) = {Best_F}")
hist_P, hist_F = pso.getHistory()

res_pso = []
for p in hist_P:
    res_pso.append(opt.F_obj(p))

plt.plot(hist_F)
plt.show()

f_obj(hist_P[-1],data = max_vec_size, visualize=True)
