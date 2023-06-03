import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Linear regression model
X_linear = np.array([1, 2, 3, 4, 5]).reshape((-1, 1))  # Independent variable
y_linear = np.array([2, 4, 6, 8, 10])  # Dependent variable

# Create and fit the linear regression model
linear_model = LinearRegression()
linear_model.fit(X_linear, y_linear)

# Print the linear regression model coefficients
print("Linear Regression Model Coefficients:")
print("Intercept:", linear_model.intercept_)
print("Slope:", linear_model.coef_)

# Multiple regression model
X_multiple = np.array([[1, 3], [2, 5], [3, 7], [4, 9], [5, 11]])  # Independent variables
y_multiple = np.array([2, 4, 6, 8, 10])  # Dependent variable

# Create and fit the multiple regression model
multiple_model = LinearRegression()
multiple_model.fit(X_multiple, y_multiple)

# Print the multiple regression model coefficients
print("\nMultiple Regression Model Coefficients:")
print("Intercept:", multiple_model.intercept_)
print("Coefficients:", multiple_model.coef_)
