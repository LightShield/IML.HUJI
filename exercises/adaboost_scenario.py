import numpy as np
from typing import Tuple
from IMLearn.learners.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    iterations = 250
    ada = AdaBoost(wl=DecisionStump, iterations=iterations)
    ada.fit(X=train_X, y=train_y)

    train_err, test_err = [], []
    iterations_range = np.arange(iterations)
    for i in iterations_range:
        train_err.append(ada.partial_loss(X=train_X, y=train_y, T=i))
        test_err.append(ada.partial_loss(X=test_X, y=test_y, T=i))

    go.Figure([
        go.Scatter(x=iterations_range, y=train_err, mode='markers + lines', name=r'Train loss'),
        go.Scatter(x=iterations_range, y=test_err, mode='markers + lines', name=r'Test loss')]) \
        .update_layout(title=rf"$\text{{Loss as a function of adaboost iteration. noise = {noise}}}$",
                       xaxis=dict(title="Iteration number"), yaxis=dict(title="Loss")).show()


    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    all_X = np.concatenate((train_X, test_X), axis=0)
    all_Y = np.concatenate((train_y, test_y), axis=0)

    plot_shapes = np.zeros(all_Y.shape)
    plot_shapes[test_X.shape[0]:] = 1

    fig = make_subplots(rows=2, cols=2, subplot_titles=[rf"$\textbf{{{t} Iterations}}$" for t in T],
                        horizontal_spacing=0.01, vertical_spacing=.03)
    for i, t in enumerate(T):
        t_predict = lambda X: ada.partial_predict(X=X, T=t)
        fig.add_traces([decision_surface(t_predict, lims[0], lims[1], showscale=False),
                        go.Scatter(x=all_X[:, 0], y=all_X[:, 1], mode="markers", showlegend=False,
                                   marker=dict(color=all_Y+1, symbol=plot_shapes, colorscale=[custom[0], custom[-1]],
                                               line=dict(color="black", width=1)))],
                       rows=(i // 2) + 1, cols=(i % 2) + 1)

    fig.update_layout(title=rf"$\textbf{{Decision Boundaries Of Model With different Iterations count. noise = {noise}}}$",
                      margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False).show()


    # Question 3: Decision surface of best performing ensemble
    best_iteration = np.argmin(test_err)

    fig = make_subplots(rows=1, cols=1,
                        horizontal_spacing=0.01, vertical_spacing=.03)
    i_predict = lambda X: ada.partial_predict(X=X, T=best_iteration)
    fig.add_traces([decision_surface(i_predict, lims[0], lims[1], showscale=False),
                    go.Scatter(x=all_X[:, 0], y=all_X[:, 1], mode="markers", showlegend=False,
                               marker=dict(color=all_Y + 1, symbol=plot_shapes, colorscale=[custom[0], custom[-1]],
                                           line=dict(color="black", width=1)))])

    fig.update_layout(
        title=rf"$\textbf{{Decision Boundaries Of Lowest Test Error - Iteration = {best_iteration}, Test Loss = {ada.partial_loss(X=test_X, y=test_y, T=best_iteration)}. noise = {noise}}}$",
        margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False).show()

    # Question 4: Decision surface with weighted samples
    weights = ada.D_ / np.max(ada.D_) * 5
    fig = make_subplots(rows=1, cols=1,
                        horizontal_spacing=0.01, vertical_spacing=.03)
    fig.add_traces([decision_surface(ada.predict, lims[0], lims[1], showscale=False),
                    go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                               marker=dict(color=train_y + 1, symbol=plot_shapes, size=weights, colorscale=[custom[0], custom[-1]],
                                           line=dict(color="black", width=1)))])

    fig.update_layout(
        title=rf"$\textbf{{Decision Boundaries With Weights of Training Data. noise = {noise}}}$",
        margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False).show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(noise=0)
    fit_and_evaluate_adaboost(noise=0.4)
