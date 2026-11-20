"""Fashion MNIST: 60,000 training and 10,000 test images of clothing, 28x28 grayscale, 10 classes."""

import keras


def load_fashion_mnist():
    """Download Fashion MNIST (once, to ~/.keras/datasets) and scale the pixels to [0, 1]."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    return (x_train.astype("float32") / 255.0, y_train), (x_test.astype("float32") / 255.0, y_test)
