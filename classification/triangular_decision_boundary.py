import random

from models import *

A_SLOPE, A_INTERCEPT = 2, -1
B_SLOPE, B_INTERCEPT = -2, 10


def in_range(x, y) -> bool:
    return y < A_SLOPE * x + A_INTERCEPT and y < B_SLOPE * x + B_INTERCEPT


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


# def generate_test_data():
#     # test data is more specific to tailor around X, Y boundaries
#     data = []
#     labels = []
#     for _ in range(100):
#         x = random.randint(X_MIN_BOUNDARY, X_MAX_BOUNDARY)
#         y = random.randint(Y_MIN_BOUNDARY, Y_MAX_BOUNDARY)
#         data.append((x, y))
#         if in_range(x, y):
#             labels.append(1)
#         else:
#             labels.append(0)
#     return data, labels

training_data, training_labels = generate_train_data()
test_data, test_labels = generate_train_data()

data_args = (training_data, training_labels, test_data, test_labels)

# linear_lr_model(*data_args)
# quadratic_lr_model(*data_args)
# gaussian_lr_model(*data_args, gamma=3)

# returns nothing b/c it's not linearly separable
# linear_svm_model(*data_args)
# quadratic_svm_model(*data_args)
# gaussian_svm_model(*data_args, gamma=0.5)
decision_tree_model(*data_args)
random_forest_model(*data_args)
