
# 📈 Linear Regression from Scratch (with Gradient Descent)

This project implements Linear Regression using gradient descent in Python without using any machine learning libraries. It’s a great starting point for beginners to understand how machine learning models learn from data.

## 🚀 What is Linear Regression?

Linear Regression is a basic supervised learning algorithm used for predicting a continuous value. It tries to find a straight line (or hyperplane in higher dimensions) that best fits the data.

---
    🎯 Example: Predicting the price of a house based on its size.
---

The formula is:

            ## Prediction(ŷ) = w⋅X + b

Where:

- `X`= features (input)
- `w` = weights (slope)
- `b` = bias (intercept)
- `ŷ` = predicted output

## 🧠 How Does Gradient Descent Work?

Gradient Descent is a method to update weights and bias to minimize prediction error.

Steps:

1. Make a prediction using current weights and bias.

2. Calculate the error between predicted and actual value.

3. Find the gradients (slope of error).

4. Update weights and bias in the opposite direction of the gradient.



---

## 🧾 Code Explanation

### 1. Initialization – `__init__()`

```python
def __init__(self, alpha=1e-3, iters=1000):
    self.alpha = alpha
    self.iters = iters
    self.w = None
    self.b = None

```

- `alpha` : Controls how big the steps are when updating weights (learning rate).

- `iters` : Number of times we go through the training data (epochs).

- `w and b` : are initialized later.

### 2. Initialize Parameters – `_init_params()`

```python
def _init_params(self):
    self.w = np.zeros(self.n)
    self.b = 0


```

- We start with weights as zeros for all features.

- Bias is also zero initially.

### 3. Update Parameters – `update_param()`

```python
def update_param(self,dw,db):
    self.w -= self.alpha*dw
    self.b -= self.alpha*db



```

- Update rule: subtract the gradient times the learning rate from the weights/bias.

### 4. predict – `predict()`

```python
ddef predict(self, X):
       
        return np.dot(X, self.w) + self.b


```

- Uses the formula Prediction(ŷ) = w⋅X + b

### 5. to get the gradients – `gradients()`

```python
ddef gradients(self, X, y, y_pred):
        """Compute gradients for weights and bias."""
        e = y_pred - y
        dw = (1 / self.m) * np.dot(X.T, e)
        db = (1 / self.m) * np.sum(e)
        return dw, db



```

- `dw` : Gradient of weights

- `db` : Gradient of bias

- Calculated from the error between prediction and actual output.

### 6. Update Parameters – `update_param()`

```python
def update_param(self,dw,db):
    self.w -= self.alpha*dw
    self.b -= self.alpha*db



```

- Update rule: subtract the gradient times the learning rate from the weights/bias.

### 7. Train the model – `fit()`

```python
def fit(self, X, y):
        
        self.m, self.n = X.shape
        self._init_params()

        for _ in range(self.iters):
            y_pred = self.predict(X)
            dw, db = self.gradients(X, y, y_pred)
            self.update_param(dw, db)

```

- Prepares the model, then runs gradient descent `iters` times.


### 8. To predict on New Data - `final_predict()`

```python
  def final_predict(self, X):
        
        return self.predict(X)

```

- After training, you can use this to predict outputs for new data.

## 📦 How to Use This Class ?

Here’s how to use the  `LinearRegression` class in a script:

```python
from sklearn.model_selection import train_test_split
from sklearn import datasets
from linear_regression import LinearRegression  # Your custom class
import numpy as np


# RMSE calculation
def rmse(y, y_pred):
    return np.sqrt(np.mean((y_pred - y) ** 2))

X, y = datasets.make_regression(n_samples=1000, n_features=1, noise=20, random_state=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Initialize and train model
lr = LinearRegression(alpha=0.01, iters=1000)
lr.fit(X_train, y_train)

# Predict on test data
y_pred = lr.final_predict(X_test)

# Evaluate performance
print(f"Root Mean Squared Error (RMSE): {rmse(y_test, y_pred):.4f}")


```
## Test Your Understanding

- Try using this model with different learning rates and iteration counts.

- Modify the input `X` to include more features (columns).

- Try plotting predictions with matplotlib to visualize how the line fits.

## 📚 Recommended Topics to Learn Next

- Mean Squared Error (MSE)

- Feature Scaling

- Overfitting and Underfitting

- Multiple Linear Regression

- Regularization (L1/L2)

## ❤️ Why Build This?

Understanding how linear regression works without libraries helps build your intuition on how machine learning really works under the hood.

## 🔧 Requirements

- Python 3.x

- NumPy

You can install NumPy using:

```bash
pip install numpy

```

## 📄 

This project is open-source for educational use.
