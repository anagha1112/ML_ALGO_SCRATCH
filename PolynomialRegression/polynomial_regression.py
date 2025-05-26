import numpy as np

class PolynomialRegression:

    def __init__(self, degree, alpha=1e-3, iters=1000):
        """Initialize polynomial degree, learning rate, and iteration count."""
        self.alpha = alpha
        self.iters = iters
        self.degree = degree
        self.w = None
        self.b = None

    def _polynomial_features(self, X):
        """Generate polynomial features up to the given degree."""
        return np.hstack([X ** i for i in range(1, self.degree + 1)])

    def _normalize(self, X):
        """Normalize features using mean and standard deviation (used in training)."""
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0) + 1e-8  # Add epsilon to avoid division by zero
        return (X - self.mean) / self.std

    def _normalize_with_stats(self, X):
        """Normalize using stored mean and std (used in prediction)."""
        return (X - self.mean) / self.std

    def gradients(self, X, y, y_pred):
        """Compute gradients of weights and bias."""
        err = y - y_pred
        dw = -(1 / self.m) * np.dot(X.T, err)
        db = -(1 / self.m) * np.sum(err)
        return dw, db

    def update_params(self, dw, db):
        """Update weights and bias using gradient descent."""
        self.w -= self.alpha * dw
        self.b -= self.alpha * db

    def predict(self, X):
        """Compute predictions using current weights and bias."""
        return np.dot(X, self.w) + self.b

    def fit(self, X, y):
        """Train the model using polynomial features and gradient descent."""
        X_poly = self._polynomial_features(X)
        X_poly_norm = self._normalize(X_poly)

        self.m, self.n = X_poly_norm.shape
        self.w = np.zeros(self.n)
        self.b = 0

        for _ in range(self.iters):
            y_pred = self.predict(X_poly_norm)
            dw, db = self.gradients(X_poly_norm, y, y_pred)
            self.update_params(dw, db)

    def final_predict(self, X):
        """Generate predictions on new data using trained model."""
        X_poly = self._polynomial_features(X)
        X_poly_norm = self._normalize_with_stats(X_poly)
        return self.predict(X_poly_norm)
