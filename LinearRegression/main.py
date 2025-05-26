from sklearn.model_selection import train_test_split
from sklearn import datasets
from linear_regression import LinearRegression  # Your custom class
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# RMSE calculation
def rmse(y, y_pred):
    return np.sqrt(np.mean((y_pred - y) ** 2))

# Load and generate regression data
X, y = datasets.make_regression(n_samples=1000, n_features=1, noise=20, random_state=1)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Initialize and train model
lr = LinearRegression(alpha=0.01, iters=1000)
lr.fit(X_train, y_train)

# Predict on test data
y_pred = lr.final_predict(X_test)

# Evaluate performance
print(f"Root Mean Squared Error (RMSE): {rmse(y_test, y_pred):.4f}")



# X and y are your data arrays
sns.regplot(x=X.flatten(), y=y, color="green", line_kws={"color": "red"})

plt.xlabel("Feature (X)")
plt.ylabel("Target (y)")
plt.title("Linear Regression ")
plt.show()    #plotting the best fit line