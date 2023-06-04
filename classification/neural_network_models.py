from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers.legacy import Adam

from utils import plot_decision_boundary, score


def predict_with_threshold(model, input_data, threshold=0.5):
    return (model.predict(input_data) > threshold).astype(int)


def relu_neural_net(training_data, training_labels, test_data, test_labels, plot=True):
    training_data = training_data.reshape(-1, 2)
    training_labels = training_labels.reshape(-1, 1)
    test_data = test_data.reshape(-1, 2)
    test_labels = test_labels.reshape(-1, 1)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(units=5, activation="relu"),
            tf.keras.layers.Dense(units=1, activation="sigmoid"),
        ]
    )
    model.compile(
        loss=BinaryCrossentropy(),
        optimizer=Adam(learning_rate=0.01),
    )
    model.fit(training_data, training_labels, epochs=100)
    print_weights(model)
    score(
        partial(predict_with_threshold, model),
        training_data,
        training_labels,
        test_data,
        test_labels,
    )

    if plot:
        plot_decision_boundary(
            training_data, training_labels, partial(predict_with_threshold, model)
        )


def print_weights(model):
    for i, layer in enumerate(model.layers):
        weights = layer.get_weights()
        print(f"Layer {i} weights: \n{weights[0]}")
        print(f"Layer {i} biases: \n{weights[1]}")
