# mypy: ignore-errors
import random
from functools import partial
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

A_SLOPE, A_INTERCEPT = 0.5, -1
B_SLOPE, B_INTERCEPT = -2, 10


def in_range(x, y) -> bool:
    return y <= A_SLOPE * x + A_INTERCEPT and y < B_SLOPE * x + B_INTERCEPT


def generate_train_data() -> Any:
    data = []
    labels = []
    for _ in range(1000):
        x = random.randint(0, 30)
        y = random.randint(0, 10)
        data.append((x, y))
        if in_range(x, y):
            labels.append(1)
        else:
            labels.append(0)
    return data, labels


# def generate_test_data() -> Any:
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
