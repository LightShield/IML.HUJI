from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt

###Constants
ID = "id"
PRICE = "price"
ZIP = "zipcode"
BEDROOMS = "bedrooms"
BATHROOMS = "bathrooms"
FLOORS = "floors"
YEAR_BUILT = "yr_built"
YEAR_RENOVATED = "yr_renovated"
DATE = "date"
LATITUDE = "lat"
LONGITUDE = "long"

CATEGORICAL_FEATURES = [ZIP, YEAR_RENOVATED]
NON_NEG_FEATURES = [BEDROOMS, BATHROOMS, FLOORS, YEAR_BUILT]
# id - irrelevant for prices, lati\long - already present in zip code, which is categorical
DROP_FEATURES = [ID, LATITUDE, LONGITUDE]

pio.templates.default = "simple_white"


def remove_duplicates(raw_data: pd.DataFrame):
    raw_data.drop_duplicates(subset=ID, inplace=True)


def remove_irelevant_features(raw_data: pd.DataFrame):
    raw_data.drop(columns=DROP_FEATURES, inplace=True)


def handle_categorical_features(raw_data: pd.DataFrame):
    return pd.get_dummies(raw_data, prefix=CATEGORICAL_FEATURES, prefix_sep=" = ", columns=CATEGORICAL_FEATURES)


def remove_invalid_data(raw_data: pd.DataFrame):
    # ignore data where value shouldn't be negative, in order of appearance in the csv
    for feature in NON_NEG_FEATURES:
        raw_data = raw_data[raw_data[feature] > 0]
    # ignore when values are missing
    return raw_data.dropna()


def reformat_data(raw_data: pd.DataFrame):
    raw_dates = pd.to_datetime(raw_data.date)
    raw_data[DATE] = raw_dates.dt.strftime('%Y.%m').astype(float)

    raw_data[ZIP] = raw_data[ZIP].astype(int)
    raw_data[YEAR_RENOVATED] = raw_data[YEAR_RENOVATED].astype(int)

    return raw_data


def pre_process_data(raw_data: pd.DataFrame):
    remove_duplicates(raw_data)
    remove_irelevant_features(raw_data)
    raw_data = remove_invalid_data(raw_data)
    raw_data = reformat_data(raw_data)
    return handle_categorical_features(raw_data)


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    raw_data = pd.read_csv(filename)
    raw_data = pre_process_data(raw_data)
    responses = raw_data[PRICE]
    raw_data.drop(PRICE, axis=1, inplace=True)
    return raw_data, responses


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    feature_std = np.sqrt(X.var())
    y_std = np.std(y)

    should_plot_feature = lambda feature: True if ("zip" not in feature and "yr" not in feature) else False

    for feature in X.columns:
        if should_plot_feature(feature):
            feature_response_cov = np.cov(X[feature], y)[0][1]
            price_cov = feature_response_cov / (feature_std[feature] * y_std)
            print(f"For feature {feature} Price-Cov is {price_cov}")

            plt.clf()
            plt.scatter(x=X[feature], y=y)
            plt.title(f"Price-Covariance of {feature} and response = {price_cov}")
            plt.xlabel(feature)
            plt.ylabel('Response')
            plt.savefig(f"{output_path}/{feature}_PriceCov")


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    data, pred = load_data("../datasets/house_prices.csv")
    print(data[data[BATHROOMS] > 20])
    # Question 2 - Feature evaluation with respect to response
    #todo remove outputpath before submission
    # feature_evaluation(X=data, y=pred, output_path="C:/Users/light/OneDrive/Desktop/School/CS/IML/IML.HUJI/temp")

    # # Question 3 - Split samples into training- and testing sets.
    split_train_test(X=data, y=pred)
    # # Question 4 - Fit model over increasing percentages of the overall training data
    # # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    # #   1) Sample p% of the overall training data
    # #   2) Fit linear model (including intercept) over sampled set
    # #   3) Test fitted model over test set
    # #   4) Store average and variance of loss over test set
    # # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    # raise NotImplementedError()
