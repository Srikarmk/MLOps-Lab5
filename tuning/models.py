"""The baseline model, and the hypermodel whose hyperparameters Keras Tuner searches."""

import keras


def build_baseline():
    """The lab's hand-picked model: one hidden layer of 512 units and a learning rate of 0.001."""
    model = keras.Sequential([
        keras.Input(shape=(28, 28)),
        keras.layers.Flatten(name="flatten"),
        keras.layers.Dense(512, activation="relu", name="hidden"),
        keras.layers.Dropout(0.2, name="dropout"),
        keras.layers.Dense(10, activation="softmax", name="output"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss=keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])
    return model


def model_builder(hp):
    """The same model with two hyperparameters left to the tuner: the hidden layer's units and the learning rate."""
    units = hp.Int("units", min_value=32, max_value=512, step=32)
    learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
    model = keras.Sequential([
        keras.Input(shape=(28, 28)),
        keras.layers.Flatten(name="flatten"),
        keras.layers.Dense(units, activation="relu", name="hidden"),
        keras.layers.Dropout(0.2, name="dropout"),
        keras.layers.Dense(10, activation="softmax", name="output"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])
    return model
