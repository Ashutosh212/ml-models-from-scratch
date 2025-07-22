import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class LogisticRegression():
    def __init__(self):
        pass

    def fit(self, X, y, alpha=0.001, batch_size=8, epochs=1000, tol=1e-6):
        self.samples = X.shape[0]
        self.dimn = X.shape[1]
        X = np.concatenate((X, np.ones((self.samples, 1))), axis=1)

        self.weight = np.random.random((1, self.dimn+1))

        for epoch in range(epochs):
            
            weight_prev = self.weight

            for i in range(0, self.samples, batch_size):
                # step: (y_i - y^_i)x_i  derivative of likelihood of parameter w.r.t to parameter
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                pred = sigmoid(self.weight @ X_batch.T).reshape(batch_size, -1)
                step = (y_batch - pred) @ X_batch
                batch_step = np.mean(step, axis=0) 

                # We are going to plus the step to the weight because we want to move in the direction of 
                # gradient because we want to maximize likelihood unlike linear regression where we want 
                # to minimize the MSE
                self.weight = self.weight + alpha * batch_step

            # Check convergence
            if abs(np.mean(self.weight - weight_prev)) < tol:
                print(f"Early stopping at epoch {epoch}")
                break
        print(f"1000 Epochs completed")

    def predict(self, X):
        X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        return sigmoid(self.weight @ X.T).reshape(X.shape[0], -1)


def main():
    # sigmoid test
    # x = np.zeros((4,1))
    # print(sigmoid(x))
    X = np.random.random((16,4))
    y = np.random.random((16,))
    model = LogisticRegression()
    model.fit(X, y)
    print(model.predict(X).shape)

if __name__ == "__main__":
    main()