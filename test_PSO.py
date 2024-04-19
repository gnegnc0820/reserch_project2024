import numpy as np

from optimizer.PSO import PSO
from optimizer.Optimizer import Optimizer
from optimizer.Env import Env

# # PSO
max_vec_size = 10.0
dim = 2
LB = [-max_vec_size for i in range(dim)]
UB = [ max_vec_size for i in range(dim)]

def f_obj(x, data=None):
    return np.sum(x**2), None


env = Env(f_obj=f_obj, data=None)
opt = Optimizer(N=30, T=50, LB=LB, UB=UB, Dim=dim, env=env)
pso = PSO()

Best_F, Best_P, env = pso.exp(opt)
print(f"best(PSO) = {Best_F}")
hist_P, hist_F = pso.getHistory()

res_pso = []
for p in hist_P:
    res_pso.append(-opt.F_obj(p))