import numpy as np
import matplotlib.pyplot as plt
from logistic_regression import LogisticRegression 

# Sample Data
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

# Initialize logistic regression model
model = LogisticRegression(alpha=0.1, iters=1000)

# Fit the model
model.fit(X, y)

# Predict probabilities
X_test = np.linspace(1, 8, 100).reshape(-1, 1)
y_pred = model.predict(X_test)


# Plot the sigmoid curve
plt.scatter(X, y, color='red', label='Original Data')
plt.plot(X_test, y_pred, color='blue', label='Sigmoid Curve')
plt.axhline(y=0.5, color='gray', linestyle='--', label='Decision Boundary')
plt.xlabel("Hours Studied")
plt.title("Logistic Regression: Pass or Fail")
plt.legend()
plt.grid(True)
plt.show()

# Optional: print predicted class for new values
print("Prediction for 3.5 hours studied:", model.predict(np.array([[3.5]])))
print("Prediction for 6.5 hours studied:", model.predict(np.array([[6.5]])))
