import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC

from ml_playground.utils import plot_decision_boundary, score


def linear_lr_model(training_data, training_labels, test_data, test_labels, plot=False):
    # linear model
    clf = LogisticRegression()
    clf.fit(training_data, training_labels)
    weights = clf.coef_[0]
    intercept = clf.intercept_[0]
    print("Weights: ", weights)
    print("Intercept: ", intercept)
    slope = -weights[0] / weights[1]
    new_intercept = -intercept / weights[1]
    print(f"Formula is y = {slope}x + {new_intercept}")

    score(clf.score, training_data, training_labels, test_data, test_labels)
    if plot:
        plot_decision_boundary(training_data, training_labels, clf.predict)


def quadratic_lr_model(
    training_data, training_labels, test_data, test_labels, plot=False
):
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2)),
            ("clf", LogisticRegression()),
        ]
    )
    pipeline.fit(training_data, training_labels)
    weights = pipeline.named_steps["clf"].coef_[0]
    intercept = pipeline.named_steps["clf"].intercept_[0]
    print("Weights: ", weights)
    print("Intercept: ", intercept)
    score(pipeline.score, training_data, training_labels, test_data, test_labels)
    if plot:
        plot_decision_boundary(training_data, training_labels, pipeline.predict)


def gaussian_lr_model(
    training_data, training_labels, test_data, test_labels, gamma, plot=False
):
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("kernel", RBFSampler(gamma=gamma)),
            ("clf", LogisticRegression(penalty="l2")),
        ]
    )
    pipeline.fit(training_data, training_labels)
    score(pipeline.score, training_data, training_labels, test_data, test_labels)
    if plot:
        plot_decision_boundary(training_data, training_labels, pipeline.predict)


def linear_svm_model(
    training_data, training_labels, test_data, test_labels, plot=False
):
    model = SVC(kernel="linear")
    model.fit(training_data, training_labels)
    score(model.score, training_data, training_labels, test_data, test_labels)
    if plot:
        plot_decision_boundary(training_data, training_labels, model.predict)


def quadratic_svm_model(
    training_data, training_labels, test_data, test_labels, plot=False
):
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2)),
            ("clf", SVC(kernel="linear")),
        ]
    )
    pipeline.fit(training_data, training_labels)
    score(pipeline.score, training_data, training_labels, test_data, test_labels)
    if plot:
        plot_decision_boundary(training_data, training_labels, pipeline.predict)


def gaussian_svm_model(
    training_data, training_labels, test_data, test_labels, gamma, plot=False
):
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("kernel", RBFSampler(gamma=gamma)),
            ("clf", SVC(kernel="linear")),
        ]
    )
    pipeline.fit(training_data, training_labels)
    score(pipeline.score, training_data, training_labels, test_data, test_labels)
    if plot:
        plot_decision_boundary(training_data, training_labels, pipeline.predict)
