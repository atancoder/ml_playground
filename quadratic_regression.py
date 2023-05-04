import random

from regression_models import (
    gaussian_regression_model,
    linear_regression_model,
    quadratic_regression_model,
)

second_degree = -1
first_degree = 7
zero_degree = 0


def generate_label(x):
    variance = random.uniform(-1, 1)
    return second_degree * x**2 + first_degree * x + zero_degree + variance


def generate_train_data():
    data = []
    labels = []
    for _ in range(1000):
        x = random.uniform(0, 10)
        data.append((x,))
        labels.append(generate_label(x))
    return data, labels


training_data, training_labels = generate_train_data()
test_data, test_labels = generate_train_data()
# linear_regression_model(training_data, training_labels, test_data, test_labels)
quadratic_regression_model(training_data, training_labels, test_data, test_labels)
# gaussian_regression_model(
#     training_data, training_labels, test_data, test_labels, gamma=1
# )
