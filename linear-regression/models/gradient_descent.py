import numpy as np
np.random.seed(42)

class GradientDescentLR:
    def __init__(self):
        pass

    def fit(self, X, y, alpha=0.001, batch_size=8, epochs=1000, tol=1e-6):
        
        samples = X.shape[0]
        self.dimn = X.shape[1]
        self.m = np.random.random((1, self.dimn))  # Initial slope
        self.b = 0.0  # Initial intercept
        for epoch in range(epochs):
            m_prev = self.m
            b_prev = self.b

            for i in range(0, samples, batch_size):
                # print(X[i:i+batch_size].T.shape)
                # print(self.m.shape)
                y_pred = self.m @ X[i:i+batch_size].T + self.b
                # print(y_pred.shape)
                # print(y[i:i+batch_size].reshape(batch_size, -1).shape)
                error = y[i:i+batch_size].reshape(-1, batch_size) - y_pred
                # print(error.shape)
                # print(X[i:i+batch_size].shape)

                step_size_m = -2 * np.mean(error @ X[i:i+batch_size])
                step_size_b = -2 * np.mean(error)

                self.m -= alpha * step_size_m
                self.b -= alpha * step_size_b

            # Check convergence
            if abs(np.mean(self.m - m_prev)) < tol and abs(self.b - b_prev) < tol:
                print(f"Early stopping at epoch {epoch}")
                break

    def predict(self, X):
        return self.m @ X.T + self.b



def main():
    X = np.random.random((16,4))
    y = np.random.random((16,)) 
    model = GradientDescentLR()
    model.fit(X, y)
    print(f"m: {model.m}")
    print(f"b: {model.b}")

if __name__ == "__main__":
    main()