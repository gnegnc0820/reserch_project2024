import numpy as np

# 関数オブジェクトとデータを持つ
class Env:
    def __init__(self, f_obj, data=None) -> None:
        self.f_obj = f_obj
        self.data = data
        
        self.best_score = np.inf
        self.best_data = None
        self.output = None
            
    def exp(self, X):
        value, next_data = self.f_obj(X, data=self.data, output=self.output)
        self.data = next_data
        if self.best_score < value:
            self.best_score = value
            self.best_data = self.data
            
        return value
    