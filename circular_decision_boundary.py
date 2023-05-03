# mypy: ignore-errors
import random
from functools import partial
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

"""
Logistic Regression does a fairly decent job at classifying the data. But once I tailor the test data
to be more specific to the X, Y boundaries, the model does not do as well. 
The linear model doesn't really fit the right decision boundary for the data.
"""

X_MIN_BOUNDARY = 3
X_MAX_BOUNDARY = 8
Y_MIN_BOUNDARY = 2
Y_MAX_BOUNDARY = 5


def in_range(x, y) -> bool:
    return (
        X_MIN_BOUNDARY <= x <= X_MAX_BOUNDARY and Y_MIN_BOUNDARY <= y <= Y_MAX_BOUNDARY
    )


def generate_train_data() -> Any:
    data = []
    labels = []
    for _ in range(10000):
        x = random.randint(0, 30)
        y = random.randint(0, 10)
        data.append((x, y))
        if in_range(x, y):
            labels.append(1)
        else:
            labels.append(0)
    return data, labels


def generate_test_data() -> Any:
    # test data is more specific to tailor around X, Y boundaries
    data = []
    labels = []
    for _ in range(100):
        x = random.randint(X_MIN_BOUNDARY, X_MAX_BOUNDARY)
        y = random.randint(Y_MIN_BOUNDARY, Y_MAX_BOUNDARY)
        data.append((x, y))
        if in_range(x, y):
            labels.append(1)
        else:
            labels.append(0)
    return data, labels


def plot_decision_boundary(training_data, training_labels, predict_fn):
    # Plot the data points
    X = np.array(training_data)
    y = np.array(training_labels)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis")
    x1_min, x1_max = X[:, 0].min() - 5, X[:, 0].max() + 5
    x2_min, x2_max = X[:, 1].min() - 5, X[:, 1].max() + 5
    xx1, xx2 = np.meshgrid(
        np.linspace(x1_min, x1_max, 100), np.linspace(x2_min, x2_max, 100)
    )
    Z = predict_fn(np.c_[xx1.ravel(), xx2.ravel()])
    Z = Z.reshape(xx1.shape)
    plt.contour(xx1, xx2, Z, colors="k", levels=[0])

    plt.show()


def score(score_fn, training_data, training_labels, test_data, test_labels):
    training_accuracy = score_fn(training_data, training_labels)
    test_accuracy = score_fn(test_data, test_labels)
    print("Training Accuracy: ", training_accuracy)
    print("Test Accuracy: ", test_accuracy)


def linear_lr_model(plot=False):
    training_data, training_labels = generate_train_data()
    test_data, test_labels = generate_test_data()

    # linear model
    clf = LogisticRegression()
    clf.fit(training_data, training_labels)
    weights = clf.coef_[0]
    intercept = clf.intercept_[0]
    print("Weights: ", weights)
    print("Intercept: ", intercept)

    score(clf.score, training_data, training_labels, test_data, test_labels)
    if plot:
        plot_decision_boundary(training_data, training_labels, clf.predict)


def quadratic_lr_model(plot=False):
    training_data, training_labels = generate_train_data()
    test_data, test_labels = generate_test_data()

    def elevate_data(data):
        return [(x, y, x**2, y**2) for x, y in data]
        # adding x*y here actually hurts the results

    def new_predict(clf, data):
        new_data = elevate_data(data)
        return clf.predict(new_data)

    # nonlinear model
    new_training_data = elevate_data(training_data)
    new_test_data = elevate_data(test_data)
    clf = LogisticRegression()
    clf.fit(new_training_data, training_labels)
    weights = clf.coef_[0]
    intercept = clf.intercept_[0]
    print("Weights: ", weights)
    print("Intercept: ", intercept)

    score(clf.score, new_training_data, training_labels, new_test_data, test_labels)
    if plot:
        plot_decision_boundary(
            training_data, training_labels, partial(new_predict, clf)
        )


def gaussian_lr_model(plot=False):
    training_data, training_labels = generate_train_data()
    test_data, test_labels = generate_test_data()
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("kernel", RBFSampler(gamma=5)),
            ("clf", LogisticRegression(penalty="l2")),
        ]
    )
    pipeline.fit(training_data, training_labels)
    score(pipeline.score, training_data, training_labels, test_data, test_labels)
    if plot:
        plot_decision_boundary(training_data, training_labels, pipeline.predict)


def linear_svm_model(plot=False):
    training_data, training_labels = generate_train_data()
    test_data, test_labels = generate_test_data()

    model = SVC(kernel="linear")
    model.fit(training_data, training_labels)
    score(model.score, training_data, training_labels, test_data, test_labels)
    if plot:
        plot_decision_boundary(training_data, training_labels, model.predict)


def quadratic_svm_model(plot=False):
    training_data, training_labels = generate_train_data()
    test_data, test_labels = generate_test_data()

    def elevate_data(data):
        return [(x, y, x**2, y**2) for x, y in data]

    def new_predict(clf, data):
        new_data = elevate_data(data)
        return clf.predict(new_data)

    # nonlinear model
    new_training_data = elevate_data(training_data)
    new_test_data = elevate_data(test_data)
    model = SVC(kernel="linear")
    model.fit(new_training_data, training_labels)
    score(model.score, new_training_data, training_labels, new_test_data, test_labels)
    if plot:
        plot_decision_boundary(
            training_data, training_labels, partial(new_predict, model)
        )


def gaussian_svm_model(plot=False):
    training_data, training_labels = generate_train_data()
    test_data, test_labels = generate_test_data()

    model = SVC(kernel="rbf")
    model.fit(training_data, training_labels)
    score(model.score, training_data, training_labels, test_data, test_labels)
    if plot:
        plot_decision_boundary(training_data, training_labels, model.predict)


# linear_lr_model(plot=True)
quadratic_lr_model(plot=True)
# gaussian_lr_model(plot=True)
# linear_svm_model(plot=True)
# quadratic_svm_model(plot=True)  # this is the best classifier
# gaussian_svm_model(plot=True)
