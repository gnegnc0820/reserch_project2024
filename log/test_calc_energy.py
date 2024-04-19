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

#教科書の例にならい、pw91の擬ポテンシャルを使います。
#!wget http://nninc.cnf.cornell.edu/psp_files/O.pw91-van_ak.UPF
#!wget http://nninc.cnf.cornell.edu/psp_files/H.pw91-van_ak.UPF
o_pp = "pseudo/O.pw91-van_ak.UPF"
h_pp = "pseudo/H.pw91-van_ak.UPF"

def calc_dft(cutoff=300.0,Mk=3):
    # QEでDFT計算をする. cutoff:[eV] 
    eV_to_Ry =  1/13.605698#[eV/Ry] 
    pseudopotentials = {'O':o_pp,'H':h_pp}
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

def make_model(molc,L,a,b=0):
    # 座標を指定して分子モデルを作成する
    # 原点に１つ、対称位置に原子を２つ配置して作成
    model = Atoms(molc, 
                positions=[(0,0,0),(a,b,0),(-a,b,0)],
                cell=[L,L,L],
                pbc=[1,1,1])
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

#分子モデルを作成
molecule = make_model(molc='OH2', L=10.0, a=1.2, b=0.5)

#原子と座標を確認
for i in range(len(molecule)):
    print(f'elem : {molecule.symbols[i]}, position : {molecule.positions[i]}')
# ###出力結果###  
#   #原点がO(酸素)なのが確認できます
#   elem : O, position :[0. 0. 0.]
#   elem : H, position :[1.2 0.5 0. ]
#   elem : H, position :[-1.2  0.5  0. ]

#初期構造の結合長、結合角も確認
pos=molecule.get_positions()
print(f'Length : {calc_length(pos[0],pos[1])}, {calc_length(pos[0],pos[2])} ')
print(f'Angle  : {calc_angle(pos[0],pos[1],pos[2])}' )
# ###出力結果###
#   Length : 1.3, 1.3 
#   Angle  : 134.76027010391914

molecule.pbc = [True, True, True]
#必要に応じて可視化して確認
view(molecule ,viewer="ase")
# molecule.set_calculator(calc_dft())
molecule.set_calculator(EMT())
print(f"Total energy:{molecule.get_potential_energy()}")