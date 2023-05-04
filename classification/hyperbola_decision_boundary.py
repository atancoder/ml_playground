import random

from .clf_models import (
    gaussian_lr_model,
    gaussian_svm_model,
    linear_lr_model,
    linear_svm_model,
    quadratic_lr_model,
    quadratic_svm_model,
)


def in_range(x, y) -> bool:
    # y = +- sqrt(x^2 - 1)
    inside = x**2 - 1
    if inside < 0:
        return False
    return y < inside**0.5 and y > -(inside**0.5)


def generate_train_data():
    data = []
    labels = []
    for _ in range(1000):
        x = random.uniform(-10, 10)
        y = random.uniform(-10, 10)
        data.append((x, y))
        if in_range(x, y):
            labels.append(1)
        else:
            labels.append(0)
    return data, labels


training_data, training_labels = generate_train_data()
test_data, test_labels = generate_train_data()

# linear_lr_model(training_data, training_labels, test_data, test_labels, plot=True)
quadratic_lr_model(training_data, training_labels, test_data, test_labels, plot=True)
# gaussian_lr_model(
#     training_data, training_labels, test_data, test_labels, gamma=6, plot=True
# )

# # returns nothing b/c it's not linearly separable
# linear_svm_model(training_data, training_labels, test_data, test_labels, plot=True)
quadratic_svm_model(training_data, training_labels, test_data, test_labels, plot=True)
# gaussian_svm_model(
#     training_data, training_labels, test_data, test_labels, gamma=6, plot=True
# )
