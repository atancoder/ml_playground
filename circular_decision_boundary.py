import random

from models import (
    gaussian_lr_model,
    gaussian_svm_model,
    linear_lr_model,
    linear_svm_model,
    quadratic_lr_model,
    quadratic_svm_model,
)

X_MIN_BOUNDARY = 3
X_MAX_BOUNDARY = 8
Y_MIN_BOUNDARY = 2
Y_MAX_BOUNDARY = 5

"""
Conclusions:
- Adding Standard Scaler can mess up the results
- Adding x*y to quadratic features slightly reduces accuracy
- Choice of gamma matters for Gaussian RBF kernel accuracy
"""


def in_range(x, y) -> bool:
    return (
        X_MIN_BOUNDARY <= x <= X_MAX_BOUNDARY and Y_MIN_BOUNDARY <= y <= Y_MAX_BOUNDARY
    )


def generate_train_data():
    data = []
    labels = []
    for _ in range(1000):
        x = random.uniform(0, 10)
        y = random.uniform(0, 10)
        data.append((x, y))
        if in_range(x, y):
            labels.append(1)
        else:
            labels.append(0)
    return data, labels


def generate_test_data():
    # test data is more specific to tailor around X, Y boundaries
    data = []
    labels = []
    for _ in range(100):
        x = random.uniform(X_MIN_BOUNDARY, X_MAX_BOUNDARY)
        y = random.uniform(Y_MIN_BOUNDARY, Y_MAX_BOUNDARY)
        data.append((x, y))
        if in_range(x, y):
            labels.append(1)
        else:
            labels.append(0)
    return data, labels


training_data, training_labels = generate_train_data()
test_data, test_labels = generate_test_data()

# linear_lr_model(training_data, training_labels, test_data, test_labels, plot=True)
quadratic_lr_model(training_data, training_labels, test_data, test_labels, plot=True)
# gaussian_lr_model(
#     training_data, training_labels, test_data, test_labels, gamma=5, plot=True
# )

# returns nothing b/c it's not linearly separable
# linear_svm_model(training_data, training_labels, test_data, test_labels, plot=True)
quadratic_svm_model(training_data, training_labels, test_data, test_labels, plot=True)
# gaussian_svm_model(
#     training_data, training_labels, test_data, test_labels, gamma=0.5, plot=True
# )
