import numpy as np

class GradientDescent:
    def __init__(self):
        self.m = 1
        self.b = 1

    def fit(self, X, y, alpha=0.001, batch_size=8):
        samples = X.shape[0]
        for i in range(0, samples, batch_size):
            diff = np.mean(y[i:i+batch_size] - X[i:i+batch_size])
            step_size_m = 
