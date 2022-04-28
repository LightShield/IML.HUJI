from typing import NoReturn
from ...base import BaseEstimator
import numpy as np


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

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
        features_count = X.shape[1]
        n = uniques[1]

        self.mu_ = np.zeros([classes_count, features_count])
        self.vars_ = np.zeros([classes_count, features_count])

        for k in range(classes_count):
            index_of_k = np.nonzero(np.array(y == self.classes_[k]))[0]
            samples_where_class_is_k = np.take(X, index_of_k, axis=0)
            self.mu_[k] = np.sum(samples_where_class_is_k, axis=0) / n[k]
            self.vars_[k] = np.sum(np.power(samples_where_class_is_k - self.mu_[k], 2), axis=0) / (n[k] - 1)

        # assumption of i.i.d, so cov is diagonal
        self.cov_ = np.array([np.diag(self.vars_[k]) for k in range(classes_count)])

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

        classes_count = len(self.classes_)
        samples_count = X.shape[0]
        features_count = X.shape[1]

        likelihoods = np.zeros((samples_count, classes_count))
        denominator_const = np.power(2 * np.pi, 0.5 * features_count)

        sigs = np.sqrt(self.vars_)
        denominators = np.apply_along_axis(np.prod, axis=1, arr=sigs) * denominator_const

        for k in range(classes_count):
            exp_deg_parenthesis = (X - self.mu_[k]) / sigs[k]
            for i in range(samples_count):
                numerator = np.exp(-0.5 * np.dot(exp_deg_parenthesis[i], exp_deg_parenthesis[i]))
                likelihoods[i, k] = numerator / denominators[k]

        return likelihoods

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
