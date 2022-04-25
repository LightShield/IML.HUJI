from __future__ import annotations
import numpy as np
from numpy.linalg import inv, det, slogdet


class UnivariateGaussian:
    """
    Class for univariate Gaussian Distribution Estimator
    """

    def __init__(self, biased_var: bool = False) -> UnivariateGaussian:
        """
        Estimator for univariate Gaussian mean and variance parameters

        Parameters
        ----------
        biased_var : bool, default=False
            Should fitted estimator of variance be a biased or unbiased estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `UnivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `UnivariateGaussian.fit`
            function.

        var_: float
            Estimated variance initialized as None. To be set in `UnivariateGaussian.fit`
            function.
        """
        self.biased_ = biased_var
        self.fitted_, self.mu_, self.var_ = False, None, None

    def fit(self, X: np.ndarray) -> UnivariateGaussian:
        """
        Estimate Gaussian expectation and variance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.var_` attributes according to calculated estimation (where
        estimator is either biased or unbiased). Then sets `self.fitted_` attribute to `True`
        """
        ### formulas taken from lab1 ###
        self.mu_ = np.mean(X)

        var_numerator = np.sum(np.power((X - self.mu_), 2))
        var_denominator = X.size if self.biased_ else (X.size - 1)

        self.var_ = var_numerator / var_denominator

        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, var_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")

        ### formula taken from https://en.wikipedia.org/wiki/Probability_density_function#Families_of_densities ###

        density_numerator = np.exp(-0.5 * (np.power(X - self.mu_, 2) / self.var_))
        density_denominator = np.sqrt(2 * np.pi * self.var_)

        return density_numerator / density_denominator

    @staticmethod
    def log_likelihood(mu: float, sigma: float, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        sigma : float
            Variance of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        ### formula taken from under "MLE of μ" at http://jrmeyer.github.io/machinelearning/2017/08/18/mle.html
        return - (X.size / 2) * np.log(2 * np.pi * sigma) - np.sum(np.power(X - mu, 2)) / (2 * sigma)




class MultivariateGaussian:
    """
    Class for multivariate Gaussian Distribution Estimator
    """

    def __init__(self):
        """
        Initialize an instance of multivariate Gaussian estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `MultivariateGaussian.fit` function.

        mu_: ndarray of shape (n_features,)
            Estimated expectation initialized as None. To be set in `MultivariateGaussian.fit`
            function.

        cov_: ndarray of shape (n_features, n_features)
            Estimated covariance initialized as None. To be set in `MultivariateGaussian.fit`
            function.
        """
        self.mu_, self.cov_ = None, None
        self.fitted_ = False

    def fit(self, X: np.ndarray) -> MultivariateGaussian:
        """
        Estimate Gaussian expectation and covariance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Training data

        Returns
        -------
        self : returns an instance of self

        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to calculated estimation.
        Then sets `self.fitted_` attribute to `True`
        """
        self.mu_ = np.apply_along_axis(func1d=np.mean, axis=0, arr=X).T

        zero_meaned_X = X - self.mu_
        samples_count = X.shape[0]

        cov_numerator = zero_meaned_X.T @ zero_meaned_X
        cov_denominator = samples_count - 1
        self.cov_ = cov_numerator / cov_denominator

        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray):
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, cov_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """

        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html
        d = X.shape[1]
        zero_meaned_X = X - self.mu_
        inverted_cov = inv(self.cov_)

        univar_action = lambda x_i: np.exp(-0.5 * (x_i @ inverted_cov * x_i.T))

        pdf_enumerator = np.apply_along_axis(univar_action, 1, zero_meaned_X)
        pdf_denominator = np.sqrt(np.power(2 * np.pi, d) * det(self.cov_))

        return pdf_enumerator / pdf_denominator

    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: np.ndarray, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : ndarray of shape (n_features,)
            Expectation of Gaussian
        cov : ndarray of shape (n_features, n_features)
            covariance matrix of Gaussian
        X : ndarray of shape (n_samples, n_features)
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated over all input data and under given parameters of Gaussian
        """
        # taken from Q9 in the theoretical part
        m = X.shape[0]
        d = X.shape[1]
        zero_meaned_X = X - mu

        return - 0.5 * \
               (d * m * np.log(2 * np.pi) +
                m * np.log(det(cov)) +
                np.sum(zero_meaned_X @ inv(cov) * zero_meaned_X))
