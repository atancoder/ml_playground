import random

from models import *
from neural_network_models import *

X_MIN_BOUNDARY = 3
X_MAX_BOUNDARY = 8
Y_MIN_BOUNDARY = 2
Y_MAX_BOUNDARY = 5


def in_range(x, y) -> bool:
    return (
        X_MIN_BOUNDARY <= x <= X_MAX_BOUNDARY and Y_MIN_BOUNDARY <= y <= Y_MAX_BOUNDARY
    )


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
    return np.array(data), np.array(labels).reshape(-1, 1)


def generate_test_data():
    # test data is more specific to tailor around X, Y boundaries
    data = []
    labels = []
    for _ in range(100):
        x = random.uniform(X_MIN_BOUNDARY, X_MAX_BOUNDARY)
        y = random.uniform(Y_MIN_BOUNDARY, Y_MAX_BOUNDARY)
        data.append((x, y))
        if in_range(x, y):
            labels.append(1)
        else:
            labels.append(0)
    return np.array(data), np.array(labels).reshape(-1, 1)


training_data, training_labels = generate_train_data()
test_data, test_labels = generate_test_data()

data_args = (training_data, training_labels, test_data, test_labels)
# linear_lr_model(*data_args)
# quadratic_lr_model(*data_args)
# gaussian_lr_model(
#     *data_args, gamma=5
# )

# returns nothing b/c it's not linearly separable
# linear_svm_model(*data_args)
# quadratic_svm_model(*data_args)
# gaussian_svm_model(
#     *data_args, gamma=0.5
# )
# decision_tree_model(*data_args)
# random_forest_model(*data_args)
xgboost_tree_model(*data_args)
# relu_neural_net(*data_args)
