import random

from models import (
    gaussian_lr_model,
    gaussian_svm_model,
    linear_lr_model,
    linear_svm_model,
    quadratic_lr_model,
    quadratic_svm_model,
)

X1_MIN_BOUNDARY = 3
X1_MAX_BOUNDARY = 8
Y1_MIN_BOUNDARY = 2
Y1_MAX_BOUNDARY = 5

X2_MIN_BOUNDARY = 10
X2_MAX_BOUNDARY = 12
Y2_MIN_BOUNDARY = 5
Y2_MAX_BOUNDARY = 13


def in_range(x, y) -> bool:
    return (
        X1_MIN_BOUNDARY <= x <= X1_MAX_BOUNDARY
        and Y1_MIN_BOUNDARY <= y <= Y1_MAX_BOUNDARY
    ) or (
        X2_MIN_BOUNDARY <= x <= X2_MAX_BOUNDARY
        and Y2_MIN_BOUNDARY <= y <= Y2_MAX_BOUNDARY
    )


def generate_train_data():
    data = []
    labels = []
    for _ in range(1000):
        x = random.uniform(0, 20)
        y = random.uniform(0, 20)
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
        x = random.uniform(X1_MIN_BOUNDARY, X2_MAX_BOUNDARY)
        y = random.uniform(Y1_MIN_BOUNDARY, Y2_MAX_BOUNDARY)
        data.append((x, y))
        if in_range(x, y):
            labels.append(1)
        else:
            labels.append(0)
    return data, labels


training_data, training_labels = generate_train_data()
test_data, test_labels = generate_test_data()

# linear_lr_model(training_data, training_labels, test_data, test_labels, plot=True)
# quadratic_lr_model(training_data, training_labels, test_data, test_labels, plot=True)
gaussian_lr_model(
    training_data, training_labels, test_data, test_labels, gamma=6, plot=True
)

# returns nothing b/c it's not linearly separable
# linear_svm_model(training_data, training_labels, test_data, test_labels, plot=True)
# quadratic_svm_model(training_data, training_labels, test_data, test_labels, plot=True)
gaussian_svm_model(
    training_data, training_labels, test_data, test_labels, gamma=6, plot=True
)
