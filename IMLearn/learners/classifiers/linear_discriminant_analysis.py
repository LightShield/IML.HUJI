from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """

        uniques = np.unique(ar=y, return_counts=True)
        self.classes_ = uniques[0]
        self.pi_ = uniques[1] / len(y)

        classes_count = len(self.classes_)
        samples_count = X.shape[0]
        features_count = X.shape[1]
        n = uniques[1]

        self.mu_ = np.zeros([classes_count, features_count])
        for k in range(classes_count):
            index_of_k = np.nonzero(np.array(y == self.classes_[k]))[0]
            samples_where_class_is_k = np.take(X, index_of_k, axis=0)
            self.mu_[k] = np.sum(samples_where_class_is_k, axis=0) / n[k]

        mat_in_cov_parenthesis = X - self.mu_[y]
        self.cov_ = (mat_in_cov_parenthesis.T @ mat_in_cov_parenthesis) / (samples_count - classes_count)
        self._cov_inv = inv(self.cov_)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        return np.argmax(self.likelihood(X), axis=1)

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        samples_count = X.shape[0]
        features_count = X.shape[1]
        classes_count = self.classes_.shape[0]

        likelihoods_denominator = np.sqrt(np.power(2 * np.pi, features_count) * det(self.cov_))
        likelihoods_numerator = np.zeros([samples_count, classes_count])

        for k in range(classes_count):
            adjusted_X = X - self.mu_[k]
            likelihoods_numerator[:, k] = np.apply_along_axis(lambda xi: np.exp(-0.5 * xi @ self._cov_inv @ xi.T) * self.pi_[k],
                                                              axis=1,
                                                              arr=adjusted_X)

        return likelihoods_numerator / likelihoods_denominator


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
        from ...metrics import misclassification_error
        return misclassification_error(y_true=y, y_pred=self.predict(X=X))
