import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Get the parent directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(script_dir, "..")
sys.path.insert(0, parent_dir)
from utils import clf_score, plot_decision_boundary


def linear_lr_model(training_data, training_labels, test_data, test_labels, plot=True):
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

    clf_score(clf.predict, training_data, training_labels, test_data, test_labels)
    if plot:
        plot_decision_boundary(training_data, training_labels, clf.predict)


def quadratic_lr_model(
    training_data, training_labels, test_data, test_labels, plot=True
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
    clf_score(pipeline.predict, training_data, training_labels, test_data, test_labels)
    if plot:
        plot_decision_boundary(training_data, training_labels, pipeline.predict)


def gaussian_lr_model(
    training_data, training_labels, test_data, test_labels, gamma, plot=True
):
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("kernel", RBFSampler(gamma=gamma)),
            ("clf", LogisticRegression(penalty="l2")),
        ]
    )
    pipeline.fit(training_data, training_labels)
    clf_score(pipeline.predict, training_data, training_labels, test_data, test_labels)
    if plot:
        plot_decision_boundary(training_data, training_labels, pipeline.predict)


def linear_svm_model(training_data, training_labels, test_data, test_labels, plot=True):
    model = SVC(kernel="linear")
    model.fit(training_data, training_labels)
    clf_score(model.predict, training_data, training_labels, test_data, test_labels)
    if plot:
        plot_decision_boundary(training_data, training_labels, model.predict)


def quadratic_svm_model(
    training_data, training_labels, test_data, test_labels, plot=True
):
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2)),
            ("clf", SVC(kernel="linear")),
        ]
    )
    pipeline.fit(training_data, training_labels)
    clf_score(pipeline.predict, training_data, training_labels, test_data, test_labels)
    if plot:
        plot_decision_boundary(training_data, training_labels, pipeline.predict)


def gaussian_svm_model(
    training_data, training_labels, test_data, test_labels, gamma, plot=True
):
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("kernel", RBFSampler(gamma=gamma)),
            ("clf", SVC(kernel="linear")),
        ]
    )
    pipeline.fit(training_data, training_labels)
    clf_score(pipeline.predict, training_data, training_labels, test_data, test_labels)
    if plot:
        plot_decision_boundary(training_data, training_labels, pipeline.predict)


def decision_tree_model(
    training_data, training_labels, test_data, test_labels, plot=True
):
    model = DecisionTreeClassifier()
    model.fit(training_data, training_labels)
    clf_score(model.predict, training_data, training_labels, test_data, test_labels)
    if plot:
        tree.plot_tree(model)
        plt.show()
        plot_decision_boundary(training_data, training_labels, model.predict)


def random_forest_model(
    training_data, training_labels, test_data, test_labels, plot=True
):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(training_data, training_labels)
    clf_score(model.predict, training_data, training_labels, test_data, test_labels)
    if plot:
        plot_decision_boundary(training_data, training_labels, model.predict)


def xgboost_tree_model(
    training_data, training_labels, test_data, test_labels, plot=True
):
    model = xgb.XGBClassifier(n_estimators=100)
    model.fit(training_data, training_labels)
    clf_score(model.predict, training_data, training_labels, test_data, test_labels)
    if plot:
        plot_decision_boundary(training_data, training_labels, model.predict)
