import random

from models import *


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

data_args = (training_data, training_labels, test_data, test_labels)


# linear_lr_model(*data_args)
# quadratic_lr_model(*data_args)
# gaussian_lr_model(*data_args, gamma=6)

# # returns nothing b/c it's not linearly separable
# linear_svm_model(*data_args)
# quadratic_svm_model(*data_args)
# gaussian_svm_model(*data_args, gamma=6)
decision_tree_model(*data_args)
random_forest_model(*data_args)
