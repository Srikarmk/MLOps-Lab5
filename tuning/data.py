"""CIFAR-10: 50,000 training and 10,000 test photos, 32x32 colour, 10 classes (airplane, car, bird, cat, deer,
dog, frog, horse, ship, truck)."""

import keras
import numpy as np

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
VALIDATION_FRACTION = 0.2
SPLIT_SEED = 0  # fixed, so every run and every seed uses the same validation set

# Keras downloads CIFAR-10 from the University of Toronto, which can be very slow. TensorFlow's Google Cloud
# bucket has the same archive: same SHA-256 as Keras checks, saved under the name Keras looks for, so
# keras.datasets.cifar10.load_data() then finds it and doesn't download anything.
MIRROR = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/cifar-10-python.tar.gz"
KERAS_NAME = "cifar-10-batches-py-target"
SHA256 = "6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce"


def load_cifar10():
    """Download CIFAR-10 (once, 170 MB, to ~/.keras/datasets) and scale the pixels to [0, 1].

    Keras returns the labels as a column (shape (n, 1)); they are flattened to shape (n,).
    """
    keras.utils.get_file(fname=KERAS_NAME, origin=MIRROR, extract=True, file_hash=SHA256)
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    return ((x_train.astype("float32") / 255.0, y_train.ravel()),
            (x_test.astype("float32") / 255.0, y_test.ravel()))


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
