import numpy as np
from .Env import Env


class Optimizer:
    def __init__(self, N, T, LB, UB, Dim, env:Env, x0=[]) -> None:
        
        self.N = N
        self.T = T
        self.LB = np.array(LB)
        self.UB = np.array(UB)
        self.Dim = Dim
        self.env = env
        self.F_obj = self.env.exp
        
        self.X = self.init(N, Dim, UB, LB)
        self.Xnew = np.copy(self.X)
        self.Ffun = np.full(N, np.inf)

        # self.Best_P = np.zeros(Dim)
        self.Best_P = np.copy(self.X[0])
        self.Best_F = np.inf
        self.hist_P = np.zeros((T, Dim))
        self.hist_F = np.zeros(T)

        if(len(x0) != 0):
            if len(x0[0] != Dim):
                print(f"x0 must be a vector of length Dim")
            else:
                print("use x0 as initial point")
                self.X[1,:] = x0

        # return opt
    
    def init(self, N, Dim, UB, LB):
        X = np.zeros((N,Dim))
        
        if(Dim == 1):
            X = np.random.rand(N)*(UB-LB)+LB
            X = np.ravel(X)         # 1次元化
        
        # 各変数のLBとUBが異なる場合
        print(UB)
        if(Dim > 1):
            for i in range(Dim):
                Ub_i = UB[i]
                Lb_i = LB[i]
                X[:,i] = np.random.rand(N)*(Ub_i-Lb_i)+Lb_i
        
        return X

