import numpy as np

class LinearRegression:
    def __init__(self, alpha=1e-3, iters=1000):
        self.alpha = alpha
        self.iters = iters
        self.w = None
        self.b = None

    def _init_params(self):
        """Initialize weights and bias to zeros."""
        self.w = np.zeros(self.n)
        self.b = 0

    def update_param(self, dw, db):
        """Update weights and bias using gradients."""
        self.w -= self.alpha * dw
        self.b -= self.alpha * db

    def predict(self, X):
        """Return predicted values for input X."""
        return np.dot(X, self.w) + self.b

    def gradients(self, X, y, y_pred):
        """Compute gradients for weights and bias."""
        e = y_pred - y
        dw = (1 / self.m) * np.dot(X.T, e)
        db = (1 / self.m) * np.sum(e)
        return dw, db

    def fit(self, X, y):
        """Train the model using gradient descent."""
        self.m, self.n = X.shape
        self._init_params()

        for _ in range(self.iters):
            y_pred = self.predict(X)
            dw, db = self.gradients(X, y, y_pred)
            self.update_param(dw, db)

    def final_predict(self, X):
        """Return final predictions after training."""
        return self.predict(X)
