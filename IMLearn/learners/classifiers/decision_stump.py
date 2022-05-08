from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
from ...metrics import misclassification_error
import numpy as np
from itertools import product


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """

    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        feature_count = X.shape[1]
        possible_signs = [-1, 1]

        thresholds = np.zeros(feature_count)
        threshold_errors = np.zeros(feature_count)
        actual_sign = np.zeros(feature_count)

        for i in range(feature_count):
            threshold_i, error_i = np.zeros(len(possible_signs)), np.zeros(len(possible_signs))
            for j, sign in enumerate(possible_signs):
                threshold_i[j], error_i[j] = self._find_threshold(values=X[i], labels=y, sign=sign)

            i_index_with_min_error = np.argmin(error_i)
            thresholds[i], threshold_errors[i], actual_sign[i] = \
                threshold_i[i_index_with_min_error], \
                error_i[i_index_with_min_error], \
                possible_signs[i_index_with_min_error]

        self.j_ = np.argmin(threshold_errors)
        self.threshold_ = thresholds[self.j_]
        self.sign_ = actual_sign[self.j_]

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        return np.where(X[:, self.j_] >= self.threshold_, self.sign_, -self.sign_)

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        losses_per_threshold = np.zeros(values.shape[0])
        for i, value in enumerate(values):
            new_y = np.where(values >= value, sign, -sign)
            losses_per_threshold[i] = misclassification_error(y_true=labels, y_pred=new_y)

        threshold_index = np.argmin(losses_per_threshold)

        thr = values[threshold_index]
        thr_err = losses_per_threshold[threshold_index]

        return thr, thr_err

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return misclassification_error(y_true=y, y_pred=self.predict(X=X))
