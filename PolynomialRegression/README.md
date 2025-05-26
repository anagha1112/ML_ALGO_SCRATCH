
# 📈 Polynomial Regression from Scratch (with Gradient Descent)

This project implements Polynomial Regression using gradient descent in Python without using any machine learning libraries. It's a great way to understand how models can fit non-linear data by transforming features.

## 🚀 What is Polynomial Regression?

Polynomial Regression is an extension of Linear Regression that models the non-linear relationship between the independent variable `X` and the dependent variable `y` by introducing polynomial terms (e.g., X², X³, ...).

---
    🎯 Example: Predicting house price where growth increases non-linearly with size.

---

The formula is:

            Prediction(ŷ) = w₁·X + w₂·X² + w₃·X³ + ... + b

Where:

- `X`= features (input)
- `w` = weights (slope)
- `b` = bias (intercept)
- `ŷ` = predicted output

## 🧠 How Does Gradient Descent Work?

Gradient Descent is a method to update weights and bias to minimize prediction error.

Steps:

1. Create polynomial features from original input.

2. Normalize features to ensure stable learning.

3. Make a prediction using current weights and bias.

4. Calculate the error and find gradients.

5. Update weights and bias to reduce the error.

6. Repeat the process for multiple iterations.



---

## 🧾 Code Explanation

### 1. Initialization – `__init__()`

```python
def __init__(self, degree=2, alpha=1e-3, iters=1000):
    self.degree = degree
    self.alpha = alpha
    self.iters = iters
    self.w = None
    self.b = None


```

- `alpha` : Controls how big the steps are when updating weights (learning rate).

- `iters` : Number of times we go through the training data (epochs).

- `degree`: Degree of the polynomial (e.g., 2 for quadratic).

- `w and b` : are initialized later.

### 2. Generate Polynomial Features – `_polynomial_features()`

```python
def _polynomial_features(self, X):
    return np.hstack([X ** i for i in range(1, self.degree + 1)])


```

- Converts a single input feature `X` into multiple columns: `[X, X², X³, ..., X^degree]`.

### 3. Normalize Features – `_normalize()` and `_normalize_with_stats()`

```python
def _normalize(self, X):
    self.mean = np.mean(X, axis=0)
    self.std = np.std(X, axis=0) + 1e-8
    return (X - self.mean) / self.std


```

```python
def _normalize_with_stats(self, X):
    return (X - self.mean) / self.std

```

- Ensures all features are on a similar scale, helping gradient descent converge faster.

### 4. Compute Gradients – `gradients()`

```python
def gradients(self, X, y, y_pred):
    err = y - y_pred
    dw = -(1 / self.m) * np.dot(X.T, err)
    db = -(1 / self.m) * np.sum(err)
    return dw, db


```

- Calculates how much to adjust the weights (dw) and bias (db).

### 5. Update Parameters – `update_param()`

```python
def update_param(self,dw,db):
    self.w -= self.alpha*dw
    self.b -= self.alpha*db


```

- Update rule: subtract the gradient times the learning rate from the weights/bias.

### 6. predict – `predict()`

```python
def predict(self, X):
    return np.dot(X, self.w) + self.b


```

- Uses the current weights and bias to predict `ŷ`.


### 7. Train the model – `fit()`

```python
def fit(self, X, y):
        
    X_poly = self._polynomial_features(X)
    X_poly_norm = self._normalize(X_poly)
    self.m, self.n = X_poly_norm.shape
    self.w = np.zeros(self.n)
    self.b = 0

    for _ in range(self.iters):
        y_pred = self.predict(X_poly_norm)
        dw, db = self.gradients(X_poly_norm, y, y_pred)
        self.update_param(dw, db)


```

- Prepares the polynomial features, normalizes them, and trains using gradient descent.


### 8. To predict on New Data - `final_predict()`

```python
  def final_predict(self, X):
    X_poly = self._polynomial_features(X)
    X_poly_norm = self._normalize_with_stats(X_poly)
    return self.predict(X_poly_norm)


```

- Transforms and normalizes new input data using stored stats, then predicts output.

## 📦 How to Use This Class ?

Here’s how to use the  `PolynomialRegression` class in a script:

```python
import numpy as np
import matplotlib.pyplot as plt
from polynomial_regression import PolynomialRegression

# Sample data
X = np.array([[1], [2], [3], [4], [5], [6], [7]])
y = np.array([1000, 1200, 1800, 3000, 5000, 8000, 12000])

# Initialize and train model
pr = PolynomialRegression(degree=2, alpha=0.01, iters=1000)
pr.fit(X, y)

# Predict

y_pred = pr.final_predict(X)

# Plot
plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, y_pred, color='orange', label='Predicted (poly)')
plt.title('Polynomial Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()


```
## Test Your Understanding

- Try changing the degree of the polynomial (e.g., 2, 3, 5).

- See how training behaves with different learning rates and iterations.

- Try fitting non-linear real-world datasets.

- Add RMSE (root mean square error) calculation to evaluate performance.

## 📚 Recommended Topics to Learn Next

- Feature Normalization

- Train-Test Split

- Overfitting and Underfitting

- Cross-validation

- Regularization (Ridge/Lasso)

## ❤️ Why Build This?

Implementing Polynomial Regression from scratch deepens your understanding of how models learn from non-linear data. It’s a fundamental step toward mastering more complex algorithms in machine learning.

## 🔧 Requirements

- Python 3.x

- NumPy

- Matplotlib

You can install NumPy using:

```bash
pip install numpy

```

## 📄 

This project is open-source and intended for educational use. Feel free to experiment and improve it!
