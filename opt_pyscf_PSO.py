# import libs
import os
import re
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import defaultdict

# # import ase
# from ase import Atoms
# from ase.optimize import BFGS
# from ase.visualize import view
# from ase.io.trajectory import Trajectory
# from ase.calculators.espresso import Espresso
# from ase.calculators.emt import EMT
# import nglview as nv

# import pyscf
from pyscf import pbc # type: ignore
from pyscf.pbc import gto, dft # type: ignore

# improt optimizer
from optimizer.PSO import PSO
from optimizer.Optimizer import Optimizer
from optimizer.Env import Env

import os

def save_file_at_dir(dir_path, filename, file_content, mode='w'):
    os.makedirs(dir_path, exist_ok=True)
    with open(os.path.join(dir_path, filename), mode) as f:
        f.write(file_content)

def get_posistion_range_hex(a,c):
    # ひし形範囲の取りうる最大値
    gamma = 120
    return [
        a*np.sin((180-gamma)*np.pi),
        a*np.cos((180-gamma)*np.pi),
        c
    ]

def hex_cell_vec(a,c):
    cell = [
        [a, 0.0, 0.0],
        [a / 2, a * (3 ** 0.5) / 2, 0.0],
        [0.0, 0.0, c]
    ]
    return cell
    
def parse_chemical_formula(formula):
    # 化学式を解析して元素とその個数を抽出
    element_counts = defaultdict(int)
    
    # 正規表現で化学式を解析するパターン
    pattern = r'([A-Z][a-z]?)(\d*)'
    
    matches = re.finditer(pattern, formula)
    
    for match in matches:
        element = match.group(1)
        count = match.group(2)
        
        if count == '':
            count = 1  # 個数が明示されていない場合は1とみなす
        else:
            count = int(count)
        
        element_counts[element] += count
    
    # 元素の文字列の配列を生成
    element_array = []
    for element, count in element_counts.items():
        element_array.extend([element] * count)
    
    return element_array

def set_atom(cell, materials, positions):
    atom = ""
    for i in range(len(materials)):
        # atoms.append([materials[i], positions[i]])
        atom = atom + f"{materials[i]} {positions[i][0]:.16f} {positions[i][1]:.16f} {positions[i][2]:.16f}\n"

    cell.atom = atom
    return cell,atom

def make_model(cell_vec):
    cell = pbc.gto.Cell()
    cell.a = cell_vec

    # cell.omega_cutoff = 50
    cell.ke_cutoff = 50
    cell.precision = 1e-6

    # ToDo 設定方法
    # cell.kpts = cell.make_kpts([4,4,4])

    # ToDo 基底関数など適切なもの
    # 学習用などに用いられるもので精度が低い？
    cell.basis = "sto3g"
    cell.pseudo = {'C': 'gthbp'}
    # cell.build()

    return cell
    
def f_obj(x,data=None,visualize=False,output=None):
    #分子モデルを作成
    positions = [x[i:i + 3] for i in range(0, len(x), 3)]
    global materials
    cell,atom = set_atom(cell=data,materials=materials,positions=positions)
    mf = cell.build()
    
    # DFT計算を設定（RKS: Restricted Kohn-Sham法）
    mf = dft.RKS(cell)

    # 交換・相関汎関数の指定
    mf.xc = 'PBE'  # PBE交換相関汎関数を使用

    # SCF収束基準やその他のオプションを設定（必要に応じて）
    mf.conv_tol = 1e-6  # SCF収束の許容誤差

    # # デフォルトの最大scf計算反復回数
    # mf.max_cycle = 50
    # # DIIS法の利用
    # mf.diis = True
    # 混合パラメータの調整
    # mf.mix = 0.2  # 混合パラメータを変更
    # mf.verbose = 4  # 詳細な情報を表示
    
    # mf.max_cycle = 
    # ニュートン法に変更
    mf.newton().run()
    energy = mf.kernel()
    # energy = np.sum(np.array(x)**2)
    
    # energy = mf.e_tot    # mf.kernelの返り値
    if output:
        save_file_at_dir("out/",f"{output.t}.{output.i}.out",
            f"Total energy:{energy}\n" + atom
        )
    # print(f"Total energy:{energy}")
    
    return energy,data

if __name__ == "__main__":

    material = "C2"
    a,c = 2.46, 6.71
    a,c = a*1.2, c*1.2
    N = 30
    T = 50
    # # 立方晶
    # cell_vec = np.diag(max_vec_size)
    # # 六方晶
    cell_vec = hex_cell_vec(10,20)

    # global cell_vertices
    # cell_vertices = calculate_lattice_vertices(cell_vec)

    global materials
    materials = parse_chemical_formula(material)
    # ['C', 'C', 'C', 'C']

    atoms_n = len(materials)

    # # PSO 環境設定
    # max_vec_size = [3.56,3.56,3.56]
    
    max_vec_size = get_posistion_range_hex(a=a,c=c)

    dim = 3 * atoms_n
    UB = np.array(max_vec_size*(dim//3))
    LB = np.array([0 for i in range(len(UB))])


    cell = make_model(
        cell_vec=cell_vec
    )

    env = Env(f_obj=f_obj, data = cell)
    opt = Optimizer(N=N, T=T, LB=LB, UB=UB, Dim=dim, env=env)
    pso = PSO()

    # 問題の出力
    print(f"optimize:{materials}")
    print(f" atoms_n:{atoms_n}")
    print(f"    cell:{cell_vec}")
    print(f"N={N}, T={T}")


    # # 実行
    Best_F, Best_P, env = pso.exp(opt)
    print(f"best(PSO) = {Best_F}")
    hist_P, hist_F = pso.getHistory()

    res_pso = []
    for p in hist_P:
        res_pso.append(opt.F_obj(p))

    plt.plot(hist_F)
    plt.show()
