import matplotlib.pyplot as plt
import numpy as np


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
