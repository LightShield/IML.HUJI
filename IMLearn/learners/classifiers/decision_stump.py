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

        min_thr, min_err, thr_sign, min_j = np.inf, np.inf, self.sign_, 0
        for j, sign in product(range(feature_count), possible_signs):
            thr, thr_err = self._find_threshold(values=X[:, j], labels=y, sign=sign)
            if thr_err < min_err:
                min_thr, min_err, thr_sign, min_j = thr, thr_err, sign, j
        self.j_ = min_j
        self.threshold_ = min_thr
        self.sign_ = thr_sign

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
        sorted_indexs = np.argsort(values)
        values, labels = values[sorted_indexs], labels[sorted_indexs]
        loss_if_threshold_is_min_val = np.sum(labels != sign)

        # after the inital loss, for each shift of the threshold by one value
        # we either increment the loss if the sign was originally equal to the label and we "broke it"
        # or we decrement our loss if the sign was originally different from the label and we "fixed it"
        cumulative_incremental_loss = np.cumsum(labels * sign)
        lossess = np.append(loss_if_threshold_is_min_val,
                            loss_if_threshold_is_min_val + cumulative_incremental_loss[:-1])

        min_loss_ind = np.argmin(lossess)
        return values[min_loss_ind], lossess[min_loss_ind]

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
