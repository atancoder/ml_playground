import os

# need to import utils
import sys

from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# Get the parent directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(script_dir, "..")
sys.path.insert(0, parent_dir)
from utils import plot_regression_line, score


def linear_regression_model(training_data, training_labels, test_data, test_labels):
    model = LinearRegression()
    model.fit(training_data, training_labels)
    score(model.score, training_data, training_labels, test_data, test_labels)
    plot_regression_line(training_data, training_labels, model.predict)


def quadratic_regression_model(training_data, training_labels, test_data, test_labels):
    pipeline = Pipeline(
        [("poly", PolynomialFeatures(degree=2)), ("clf", LinearRegression())]
    )
    pipeline.fit(training_data, training_labels)
    score(pipeline.score, training_data, training_labels, test_data, test_labels)
    plot_regression_line(training_data, training_labels, pipeline.predict)


def gaussian_regression_model(
    training_data, training_labels, test_data, test_labels, gamma
):
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("kernel", RBFSampler(gamma=gamma)),
            ("clf", LinearRegression()),
        ]
    )
    pipeline.fit(training_data, training_labels)
    score(pipeline.score, training_data, training_labels, test_data, test_labels)
    plot_regression_line(training_data, training_labels, pipeline.predict)


def gaussian_regularization_regression_model(
    training_data, training_labels, test_data, test_labels, gamma, alpha
):
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("kernel", RBFSampler(gamma=gamma)),
            ("clf", Ridge(alpha=alpha)),
        ]
    )
    pipeline.fit(training_data, training_labels)
    score(pipeline.score, training_data, training_labels, test_data, test_labels)
    plot_regression_line(training_data, training_labels, pipeline.predict)
