import numpy as np
class LogisticRegression:
    def __init__(self,alpha=1e-3,iters=1000):
        self.alpha =alpha
        self.iters = iters
        self.w = None
        self.b = None

    def _sigmoid(self,z):

        return 1 / (1 + np.exp(-z))
    
    def _gradient(self, X, y, y_pred):
        
        err = y_pred - y

        dw = (1 / self.m)* np.dot(err, X)
        db = (1 / self.m)* np.sum(err)

        return dw, db

    def _update_param(self, dw, db):
        self.w -= self.alpha*dw
        self.b -= self.alpha*db

    def predict(self, X):

        z = np.dot(X, self.w) + self.b
        return self._sigmoid(z)


    def fit(self, X, y):

        self.m, self.n = X.shape
        self.w = np.zeros(self.n)
        self.b = 0

    
        for _ in range(self.iters):

            y_pred = self.predict(X)
            dw, db = self._gradient(X, y, y_pred)

            self._update_param(dw, db)


    def final_predict(self, X):
        
        return self.predict(X)

