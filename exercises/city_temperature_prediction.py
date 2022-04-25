import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt

pio.templates.default = "simple_white"

###Constants
DATE = "Date"
COUNTRY = "Country"
CITY = "City"
DAY = "Day"
MONTH = "Month"
YEAR = "Year"
TEMP = "Temp"
DAY_OF_YEAR = "DayOfYear"  # added fature

ISRAEL = "Israel"

MIN_VALID_TEMP = -20

CATEGORICAL_FEATURES = []
NON_NEG_FEATURES = [DAY, MONTH, YEAR]
# Seems to me that we cen derive Day\Month\Year from "Date", but in accordance to what was asked in the ex
# (That we use specific data such as "Month") - there is nothing to drop
DROP_FEATURES = []
DUPLICATE_INDICATOR_FEATURE = [DATE, CITY]


def remove_duplicates(raw_data: pd.DataFrame):
    raw_data.drop_duplicates(subset=DUPLICATE_INDICATOR_FEATURE, inplace=True)


def remove_irelevant_features(raw_data: pd.DataFrame):
    raw_data.drop(columns=DROP_FEATURES, inplace=True)


def handle_categorical_features(raw_data: pd.DataFrame):
    return pd.get_dummies(raw_data, prefix=CATEGORICAL_FEATURES, prefix_sep=" = ", columns=CATEGORICAL_FEATURES)


def remove_invalid_data(raw_data: pd.DataFrame):
    # ignore data where value shouldn't be negative, in order of appearance in the csv
    for feature in NON_NEG_FEATURES:
        raw_data = raw_data[raw_data[feature] > 0]
    # can't measure in the future
    raw_data = raw_data[raw_data[DATE] < pd.to_datetime("now")]
    # temp outliers
    raw_data = raw_data[raw_data[TEMP] > MIN_VALID_TEMP]
    # ignore when values are missing
    return raw_data.dropna()


def reformat_data(raw_data: pd.DataFrame):
    # Intentionally left blank
    return raw_data


def pre_process_data(raw_data: pd.DataFrame):
    remove_duplicates(raw_data)
    remove_irelevant_features(raw_data)
    raw_data = remove_invalid_data(raw_data)
    raw_data = reformat_data(raw_data)
    return handle_categorical_features(raw_data)


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    raw_data = pd.read_csv(filename, parse_dates=[DATE])
    raw_data = pre_process_data(raw_data)
    responses = raw_data[TEMP]
    raw_data.drop(TEMP, axis=1, inplace=True)
    raw_data[DAY_OF_YEAR] = raw_data[DATE].dt.dayofyear
    return raw_data, responses


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    data, responses = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    israel_data = data
    israel_data[TEMP] = responses.values
    israel_data = data.loc[data[COUNTRY] == ISRAEL, :]

    plt.clf()
    plt.scatter(x=israel_data[DAY_OF_YEAR], y=israel_data[TEMP], c=israel_data[YEAR])
    plt.xlabel(DAY_OF_YEAR)
    plt.ylabel(TEMP)
    plt.title(f"{TEMP} in {ISRAEL} according to {DAY_OF_YEAR}")
    plt.show()

    month_and_temp = israel_data[[MONTH, TEMP]]
    aggregated_std = month_and_temp.groupby(MONTH).agg("std")
    months = np.arange(1, 13, 1)
    plt.clf()
    plt.bar(x=months, height=np.array(aggregated_std).reshape(-1))
    plt.xlabel("Month in the year")
    plt.ylabel(f"STD of {TEMP}")
    plt.title(f"Standard deviation in {ISRAEL}'s {TEMP} according to {MONTH}")
    plt.show()

    # Question 3 - Exploring differences between countries
    grouped_data = data.groupby([COUNTRY, MONTH])

    aggregated_data = grouped_data[TEMP].agg(["mean", "std"]).reset_index()

    temp_plot = px.line(aggregated_data, x=MONTH, y="mean", error_y="std", line_group=COUNTRY, color=COUNTRY)
    temp_plot.update_layout(title="Temperature per Month for each country",
                              xaxis_title=MONTH,
                              yaxis_title=TEMP)
    temp_plot.show()

    # Question 4 - Fitting model for different values of `k`
    israel_responses = israel_data[TEMP]
    israel_data = israel_data.drop(columns=TEMP)

    train_X, train_Y, test_X, test_Y = split_train_test(israel_data[DAY_OF_YEAR], israel_responses)

    loss = []
    degrees = range(1, 11)
    for degree in degrees:
        model = PolynomialFitting(degree)
        model.fit(train_X, train_Y)
        degree_loss = np.round(model.loss(test_X, test_Y), 2)
        loss.append(degree_loss)
        print(f"for degree k = {degree} the loss on test set is {degree_loss}")

    plt.clf()
    plt.errorbar(x=degrees, y=loss)
    plt.title("Loss as a function of the polynomial degree")
    plt.xlabel("Degree")
    plt.ylabel("Loss")
    plt.show()

    # Question 5 - Evaluating fitted model on different countries
    best_k = loss.index(min(loss)) + 1
    model = PolynomialFitting(best_k)
    model.fit(train_X, train_Y)

    countries = set(data[COUNTRY])
    countries.remove(ISRAEL)
    countries = list(countries)

    country_loss = []
    print(countries)
    for country in countries:
        country_data = data[data[COUNTRY] == country]
        country_loss.append(model.loss(X=country_data[DAY_OF_YEAR], y=country_data[TEMP]))

    plt.clf()
    plt.bar(x=countries, height=country_loss)
    plt.title(f"Loss of {ISRAEL} trained model on other countries")
    plt.xlabel("Country")
    plt.ylabel("Loss")
    plt.show()
