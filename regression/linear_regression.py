import random

from models import *

SLOPE = 3
INTERCEPT = 2


def generate_label(x):
    variance = random.uniform(-1, 1)
    return SLOPE * x + INTERCEPT + variance


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

data_args = (training_data, training_labels, test_data, test_labels)

# linear_regression_model(*data_args)
# quadratic_regression_model(*data_args)
# gaussian_regression_model(
#     *data_args, gamma=0.1
# )
gaussian_regularization_regression_model(*data_args, gamma=0.1, alpha=0.001)
