import math
import random

import numpy as np
from models import *

SCALAR = 10


def generate_label(x):
    return math.log(SCALAR * x)


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

# linear_regression_model(*data_args)
# quadratic_regression_model(*data_args)
# gaussian_regression_model(*data_args, gamma=1)
# gaussian_regularization_regression_model(*data_args, gamma=0.1, alpha=0.1)

# Transform features into log
np_training_data = np.array(training_data)
np_test_data = np.array(test_data)
linear_regression_model(
    np.log(training_data), training_labels, np.log(np_test_data), test_labels
)
