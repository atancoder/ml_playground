import math
import random

from models import *

gamma = 0.5


def generate_label(x):
    return x**gamma


def generate_train_data():
    data = []
    labels = []
    for _ in range(1000):
        x = random.uniform(0, 10)
        data.append((x,))
        labels.append(generate_label(x))
    return data, labels


def generate_test_data():
    data = []
    labels = []
    for _ in range(100):
        x = random.uniform(0, 20)
        data.append((x,))
        labels.append(generate_label(x))
    return data, labels


training_data, training_labels = generate_train_data()
test_data, test_labels = generate_test_data()

data_args = (training_data, training_labels, test_data, test_labels)

linear_regression_model(*data_args)
quadratic_regression_model(*data_args)
gaussian_regression_model(*data_args, gamma=1)
gaussian_regularization_regression_model(*data_args, gamma=0.1, alpha=0.1)
power_law_model(*data_args)
