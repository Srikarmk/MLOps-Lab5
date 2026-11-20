"""Fashion MNIST: 60,000 training and 10,000 test images of clothing, 28x28 grayscale, 10 classes."""

import keras
import numpy as np

VALIDATION_FRACTION = 0.2
SPLIT_SEED = 0  # fixed, so every run and every seed uses the same validation set


def load_fashion_mnist():
    """Download Fashion MNIST (once, to ~/.keras/datasets) and scale the pixels to [0, 1]."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    return (x_train.astype("float32") / 255.0, y_train), (x_test.astype("float32") / 255.0, y_test)


def split_validation(x, y, fraction=VALIDATION_FRACTION, seed=SPLIT_SEED):
    """Hold out the same fraction of every class as a validation set. Returns (x_train, y_train), (x_val, y_val).

    The search picks hyperparameters on it and the final models pick their epoch on it, so the test set
    is only used once, for the final comparison.
    """
    rng = np.random.default_rng(seed)
    val_idx = np.concatenate([
        rng.choice(np.flatnonzero(y == c), size=round(fraction * np.sum(y == c)), replace=False)
        for c in np.unique(y)
    ])
    val_mask = np.zeros(len(y), dtype=bool)
    val_mask[val_idx] = True
    return (x[~val_mask], y[~val_mask]), (x[val_mask], y[val_mask])
