import random

from models import *

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

data_args = (training_data, training_labels, test_data, test_labels)

# linear_lr_model(*data_args)
# quadratic_lr_model(*data_args)
# gaussian_lr_model(*data_args, gamma=0.1)

# linear_svm_model(*data_args)
# quadratic_svm_model(*data_args)
# gaussian_svm_model(*data_args, gamma=0.5)
decision_tree_model(*data_args)
random_forest_model(*data_args)
