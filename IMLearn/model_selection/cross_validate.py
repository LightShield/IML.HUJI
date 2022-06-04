from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """

    split_indices = np.arange(len(X))
    chuncks_indices = [split_indices[i:i + cv] for i in range(0, len(split_indices), cv)]

    validation_loss_sum, train_loss_sum = 0, 0
    for chunk in chuncks_indices:
        validation_X = X[chunk]
        validation_y = y[chunk]
        train_X = np.delete(arr=X, obj=np.array(chunk), axis=0)
        train_y = np.delete(arr=y, obj=np.array(chunk))

        estimator.fit(train_X, train_y)

        validation_loss_sum += scoring(validation_y, estimator.predict(validation_X))
        train_loss_sum += scoring(train_y, estimator.predict(train_X))
    return train_loss_sum / cv, validation_loss_sum / cv
