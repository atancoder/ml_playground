import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score


def plot_decision_boundary(X, y, predict_fn):
    x1 = X[:, 0]
    x2 = X[:, 1]
    # Plot the data points
    plt.scatter(x1, x2, c=y, cmap="bwr")

    x1_min, x1_max = min(x1), max(x1)
    x2_min, x2_max = min(x2), max(x2)
    x1_linspace = np.linspace(x1_min - 5, x1_max + 5, num=100)
    x2_linspace = np.linspace(x2_min - 5, x2_max + 5, num=100)

    # meshgrid gives us 2 2D arrays, where we can get a coordinate
    # i,j with X[i][j], Y[i][j]
    X, Y = np.meshgrid(x1_linspace, x2_linspace)

    # We have to unravel into a rows X 2 matrix for predictions
    coordinates = []
    for i in range(len(X)):
        for j in range(len(X[0])):
            coordinates.append((X[i][j], Y[i][j]))
    coordinates = np.array(coordinates)
    Z = predict_fn(coordinates)

    # Reshape Z into a meshgrid matrix
    Z = Z.reshape(len(X), len(X[0]))

    # plot contour
    plt.contour(X, Y, Z, levels=[0], colors="black")

    # Make a prediction on a bunch of training points. Project points with Z(x) = 1 onto the 2D plot

    plt.show()

def plot_regression_line(training_data, training_labels, predict_fn):
    # Plot the data points
    X = np.array(training_data)
    y = np.array(training_labels)
    plt.scatter(X, y)
    x1_min = X.min() - 5
    x1_max = X.max() + 5
    xx1 = np.linspace(x1_min, x1_max, 100)
    plt.plot(xx1, predict_fn(xx1.reshape(len(xx1), 1)), c="red")
    plt.show()


def clf_score(model_predict_fn, training_data, training_labels, test_data, test_labels):
    # We have to create our own score b/c the model doesn't naturally apply the threshold
    print("Outputting scores")
    score_types = [accuracy_score, precision_score, recall_score]
    for score_type in score_types:
        training_score = score_type(training_labels, model_predict_fn(training_data))
        test_score = score_type(test_labels, model_predict_fn(test_data))
        print(f"Training {score_type.__name__}: ", training_score)
        print(f"Test {score_type.__name__}: ", test_score)

def reg_score(score_fn, training_data, training_labels, test_data, test_labels):
    training_accuracy = score_fn(training_data, training_labels)
    test_accuracy = score_fn(test_data, test_labels)
    print("Training Accuracy: ", training_accuracy)
    print("Test Accuracy: ", test_accuracy)