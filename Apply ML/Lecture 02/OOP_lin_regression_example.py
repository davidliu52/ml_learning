from sklearn.linear_model import LinearRegression
import numpy as np
from matplotlib import pyplot as plt

"""
--- LINEAR REGRESSION: SCALAR LEAST-SQUARES
"""


# some little helper
def plot(x, y, x_pred, y_pred):
    # plot the training data and the model predictions
    plt.figure()
    plt.scatter(x, y, marker='.')
    plt.scatter(x_pred, y_pred, color='red', marker='x')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(['data', 'prediction'])
    plt.show()


# Training data
x = np.arange(0, 10, .1)
y = x + 0.15 * np.random.randn(100)

# Query points (where we want to evaluate the model)
x_pred = np.array([0.5, 1.7, 9.3, 12.4])

"""
Functional programming: 

1. define a function that finds the coefficients to the linear regression model
2. define a function that evaluates the model at query points

"""


def fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    # assuming x and y being 1-dim np.arrays of length N (number of training data samples)

    # number of training samples
    N = x.shape[0]

    # normal form
    # A = [N, sum(x); sum(x), sum(x^2)]
    # b = [sum(y); sum(y*x)]
    A = np.array([[N, np.sum(x)], [np.sum(x), np.sum(x ** 2)]])
    b = np.expand_dims(np.array([np.sum(y), np.sum(y * x)]), axis=-1)

    # solve Ax = b
    theta = np.linalg.solve(A, b).flatten()
    print(f'theta0 = {theta[0]}, theta1 = {theta[1]}')

    return (theta[0], theta[1])


def predict(theta: tuple, x: np.ndarray) -> np.ndarray:
    # expects model parameters (theta_0, theta_1) and query points x

    # evaluate model at query points
    y_eval = theta[0] + theta[1] * x

    return y_eval


# fit model and make a prediction
theta = fit(x=x, y=y)
y_hat = predict(theta=theta, x=x_pred)

# plot the training data and the model predictions
plot(x, y, x_pred, y_hat)

"""
Object-oriented programming: 

- resolve the issue of needing to store the model parameters (weights) as a variable
- unite the function (fit, prediction) with the weights in one place
"""


class LinRegressor:

    def __init__(self):
        self.theta: tuple
        self.x_train: np.ndarray
        self.y_train: np.ndarray
        self.N_train: int  # number of training samples

    def fit(self, x, y):
        self.x_train = x
        self.y_train = y
        self.N_train = self.x_train.shape[0]

        # normal form: A = [N, sum(x); sum(x), sum(x^2)]; b = [sum(y); sum(y*x)]
        A = np.array([[self.N_train, np.sum(self.x_train)], [np.sum(self.x_train), np.sum(self.x_train ** 2)]])
        b = np.expand_dims(np.array([np.sum(self.y_train), np.sum(self.y_train * self.x_train)]), axis=-1)

        # solve Ax = b
        self.theta = np.linalg.solve(A, b).flatten()
        print(f'theta0 = {self.theta[0]}, theta1 = {self.theta[1]}')

    def predict(self, x):
        return self.theta[0] + self.theta[1] * x


# fit and evaluate lin. regress. model using OOP
regressor = LinRegressor()
regressor.fit(x=x, y=y)
y_hat = regressor.predict(x=x_pred)
plot(x, y, x_pred, y_hat)
