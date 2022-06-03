from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    f = lambda x: (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
    X = np.random.uniform(low=-1.2, high=2, size=n_samples)
    eps = np.random.normal(loc=0, scale=noise, size=n_samples)

    true_y = f(X)
    y = true_y + eps

    train_X, train_Y, test_X, test_Y = split_train_test(X=pd.DataFrame(X), y=pd.Series(y), train_proportion=2 / 3)
    train_X = np.array(train_X[0])
    test_X = np.array(test_X[0])
    train_Y = np.array(train_Y)
    test_Y = np.array(test_Y)

    # todo uncomment
    go.Figure([
        go.Scatter(x=X, y=true_y, mode='markers', name=r'True model'),
        go.Scatter(x=train_X, y=train_Y, mode='markers', name=r'Train set'),
        go.Scatter(x=test_X, y=test_Y, mode='markers', name=r'Test set', )]) \
        .update_layout(
        title=rf"$\text{{f(x)=(x+3)(x+2)(x+1)(x-1)(x-2), sample count = {n_samples}, noise = {noise}}}$",
        xaxis=dict(title="x"), yaxis=dict(title="f(x)")).show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    degrees = np.arange(11)
    train_score, validation_score = [], []
    for k in degrees:
        k_train_score, k_validation_score = cross_validate(estimator=PolynomialFitting(k=k), X=train_X, y=train_Y,
                                                           scoring=mean_square_error, cv=5)
        train_score.append(k_train_score)
        validation_score.append(k_validation_score)

    go.Figure([
        go.Scatter(x=degrees, y=train_score, mode='markers + lines', name=r'Train score'),
        go.Scatter(x=degrees, y=validation_score, mode='markers + lines', name=r'Validation score')]) \
        .update_layout(
        title=rf"$\text{{Cross-validated mean square error as a function of the polynom's degree, sample count = {n_samples}, noise = {noise}}}$",
        xaxis=dict(title="Degree"), yaxis=dict(title="Score")).show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    best_k = np.argmin(validation_score)
    model = PolynomialFitting(k=best_k)
    model.fit(train_X, train_Y)
    best_k_test_err = model.loss(train_X, train_Y)

    print(f"Best k is {best_k} and the test loss is {best_k_test_err}.\n"
          f"Validation error for the same K is {validation_score[best_k]}")


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    valudes for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions

    X, y = datasets.load_diabetes(return_X_y=True, as_frame=True)

    train_X, train_Y, test_X, test_Y = split_train_test(X=X, y=y, train_proportion=n_samples / len(X))

    # # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    # raise NotImplementedError()
    #
    # # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    # raise NotImplementedError()


if __name__ == '__main__':
    # todo write verbal answers in word for all questions after finished
    np.random.seed(0)

    # todo uncomment
    # # Q1-3
    # select_polynomial_degree(n_samples=100, noise=5)
    # # Q4
    # select_polynomial_degree(n_samples=100, noise=0)
    # # Q5
    # select_polynomial_degree(n_samples=1500, noise=10)

    select_regularization_parameter()
