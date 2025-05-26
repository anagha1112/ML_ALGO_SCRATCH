import matplotlib.pyplot as plt
import numpy as np

from polynomial_regression import PolynomialRegression  

# Sample data

X = np.array([[1], [2], [3], [4], [5], [6], [7]])


y = np.array([1000, 1200, 1800, 3000, 5000, 8000, 12000])
# Model
pr = PolynomialRegression(degree=2, alpha=0.01, iters=1000)
pr.fit(X, y)

# Prediction

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
