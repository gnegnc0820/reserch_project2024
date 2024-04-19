import numpy as np

from numpy.random import rand, randint

from .Optimizer import Optimizer

class OUTPUT:
    def __init__(self, t,i) -> None:
        self.t = t
        self.i = i

class PSO:
        
    def exp(self, opt:Optimizer, w=0.9, c1=0.9, c2=0.9):
        # w=0.9, c1=0.5, c2=0.3
        self.hist_group = []
        
        v = np.zeros((opt.N,opt.Dim))
        # opt.Xnew = np.zeros((opt.N,opt.Dim))

        # イテレーションに応じて実行
        for t in range(opt.T):
            v = w * v + c1 * (opt.Best_P-opt.X)*rand() + c2 * (opt.Xnew-opt.X)*rand()
            opt.Xnew += v
            
            opt.X = np.clip(opt.Xnew, a_min=opt.LB, a_max=opt.UB)

            for i in range(opt.N):
                opt.env.output = OUTPUT(t,i)
                opt.Ffun[i] = opt.F_obj(X=opt.X[i])
                if opt.Ffun[i] < opt.Best_F:
                    opt.Best_F = opt.Ffun[i]
                    opt.Best_P = opt.X[i]
            
            opt.hist_P[t] = opt.Best_P
            opt.hist_F[t] = opt.Best_F

            self.hist_group.append(np.copy(opt.X))

        self.opt = opt
        return opt.Best_F, opt.Best_P, opt.env
    
    def getHistory(self):
        return self.opt.hist_P, self.opt.hist_F
    
    def get_hist_group(self):
        return self.hist_group

class PSO_lattice_clip:
    
    def exp(self, opt:Optimizer, w=0.9, c1=0.9, c2=0.9):
        # w=0.9, c1=0.5, c2=0.3
        self.hist_group = []
        
        v = np.zeros((opt.N,opt.Dim))
        # opt.Xnew = np.zeros((opt.N,opt.Dim))

        # イテレーションに応じて実行
        for t in range(opt.T):
            v = w * v + c1 * (opt.Best_P-opt.X)*rand() + c2 * (opt.Xnew-opt.X)*rand()
            opt.Xnew += v
            
            # opt.X = np.clip(opt.Xnew, a_min=opt.LB, a_max=opt.UB)
            # 解候補ごとに格子内にクリップ
            for i in range(len(opt.X)):
                x = opt.X[i]
                positions = [x[i:i + 3] for i in range(0, len(x), 3)]
                
                print(positions)
                positions = [opt.env.clip(d) for d in positions]
                positions = np.vstack(positions)
                print(positions)
                opt.X[i] = np.array(positions)


            for i in range(opt.N):
                opt.Ffun[i] = opt.F_obj(opt.X[i])
                if opt.Ffun[i] < opt.Best_F:
                    opt.Best_F = opt.Ffun[i]
                    opt.Best_P = opt.X[i]
            
            opt.hist_P[t] = opt.Best_P
            opt.hist_F[t] = opt.Best_F

            self.hist_group.append(np.copy(opt.X))

        self.opt = opt
        return opt.Best_F, opt.Best_P, opt.env
    
    def getHistory(self):
        return self.opt.hist_P, self.opt.hist_F
    
    def get_hist_group(self):
        return self.hist_group


class PSO_lbest:
        
    def exp(self, opt:Optimizer, w=0.9, c1=0.9, c2=0.9):
        # w=0.9, c1=0.5, c2=0.3
        self.hist_group = []
        
        opt.Best_F = np.full(opt.N, np.inf)
        # opt.Best_P = opt.X.copy()
        opt.Best_P = opt.X
        
        v = np.zeros((opt.N,opt.Dim))
        # opt.Xnew = np.zeros((opt.N,opt.Dim))

        # イテレーションに応じて実行
        for t in range(opt.T):
            v = w * v + c1 * (opt.Best_P-opt.X)*rand() + c2 * (opt.Xnew-opt.X)*rand()
            opt.Xnew += v
            
            opt.X = np.clip(opt.Xnew, a_min=opt.LB, a_max=opt.UB)

            # 以下を修正
            for i in range(opt.N):
                neighbor_n = 5
                sharing_step = 10

                for i in range(opt.N):                
                    opt.Ffun[i] = opt.F_obj(opt.X[i])
                    
                    if opt.Ffun[i] < opt.Best_F[i]:
                        opt.Best_F[i] = opt.Ffun[i]
                        opt.Best_P[i] = opt.X[i]
            
                lbest_P = opt.Best_P.copy()
                lbest_F = opt.Best_F.copy()
                # 循環配列としてみたとき近傍2の配列の最大値を取得する
                # if t==0:
                #     opt.Best_F = np.tile(max(opt.Best_F),(opt.N,1))
                #     opt.Best_P = np.tile(opt.Best_P[np.argmax(opt.Best_F)],(opt.N,1))
                for i in range(opt.N):
                    for t in [-1,1]:
                        for j in range(1,neighbor_n+1):
                            if lbest_F[i] > opt.Best_F[(i-j*t)%opt.N]:
                                lbest_F[i] = opt.Best_F[(i-j*t)%opt.N]
                                lbest_P[i] = opt.Best_P[(i-j*t)%opt.N]
                opt.Best_P = lbest_P
                opt.Best_F = lbest_F
            
            # opt.hist_P[t] = opt.Best_P
            # opt.hist_F[t] = opt.Best_F
            
            opt.hist_F[t] = min(opt.Best_F)
            opt.hist_P[t] = opt.Best_P[np.argmin(opt.Best_F)]

            self.hist_group.append(np.copy(opt.X))

        self.opt = opt
        return opt.Best_F, opt.Best_P, opt.env
    
    def getHistory(self):
        return self.opt.hist_P, self.opt.hist_F
    
    def get_hist_group(self):
        return self.hist_group