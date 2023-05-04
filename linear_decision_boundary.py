import random

from clf_models import (
    gaussian_lr_model,
    gaussian_svm_model,
    linear_lr_model,
    linear_svm_model,
    quadratic_lr_model,
    quadratic_svm_model,
)

"""
Quadratic fits into hyperbola
"""


SLOPE, INTERCEPT = 1, 2


def in_range(x, y) -> bool:
    return y < SLOPE * x + INTERCEPT


def generate_train_data():
    data = []
    labels = []
    for _ in range(10000):
        x = random.uniform(0, 10)
        y = random.uniform(0, 10)
        data.append((x, y))
        if in_range(x, y):
            labels.append(1)
        else:
            labels.append(0)
    return data, labels


training_data, training_labels = generate_train_data()
test_data, test_labels = generate_train_data()

# linear_lr_model(training_data, training_labels, test_data, test_labels, plot=True)
# quadratic_lr_model(training_data, training_labels, test_data, test_labels, plot=True)
# gaussian_lr_model(
#     training_data, training_labels, test_data, test_labels, gamma=0.1, plot=True
# )

# linear_svm_model(training_data, training_labels, test_data, test_labels, plot=True)
# quadratic_svm_model(training_data, training_labels, test_data, test_labels, plot=True)
# gaussian_svm_model(
#     training_data, training_labels, test_data, test_labels, gamma=0.5, plot=True
# )
