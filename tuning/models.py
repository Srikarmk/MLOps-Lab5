"""The baseline model, and the hypermodel whose hyperparameters Keras Tuner searches."""

import keras
import keras_tuner as kt

MAX_LAYERS = 3
BATCH_SIZES = [32, 64, 128, 256]
BASELINE_BATCH_SIZE = 32  # Keras's default, which the original lab used


def build_mlp(layers, learning_rate):
    """A dense network for 28x28 images: `layers` is a list of (units, dropout) pairs, one per hidden layer."""
    model = keras.Sequential([keras.Input(shape=(28, 28)), keras.layers.Flatten(name="flatten")])
    for i, (units, dropout) in enumerate(layers, start=1):
        model.add(keras.layers.Dense(units, activation="relu", name=f"hidden_{i}"))
        model.add(keras.layers.Dropout(dropout, name=f"dropout_{i}"))
    model.add(keras.layers.Dense(10, activation="softmax", name="output"))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])
    return model


def build_baseline():
    """The lab's hand-picked model: one hidden layer of 512 units, dropout 0.2, learning rate 0.001."""
    return build_mlp([(512, 0.2)], 1e-3)


class FashionHyperModel(kt.HyperModel):
    """The search space: the network's depth and shape, its learning rate, and the batch size it trains with.

    - layers: 1 to 3 hidden layers. Each layer's units and dropout only exist when the layer does
      (a conditional hyperparameter), so the tuner never records values for a layer that isn't there.
    - learning_rate: anywhere from 0.0001 to 0.01, sampled on a log scale so each power of ten is equally likely.
    - batch_size: tuned in fit(), because it is a setting of training rather than of the model.
    """

    def build(self, hp):
        n_layers = hp.Int("layers", min_value=1, max_value=MAX_LAYERS)
        layers = []
        for i in range(1, MAX_LAYERS + 1):
            with hp.conditional_scope("layers", list(range(i, MAX_LAYERS + 1))):
                if i <= n_layers:
                    layers.append((hp.Int(f"units_{i}", min_value=32, max_value=512, step=32),
                                   hp.Float(f"dropout_{i}", min_value=0.0, max_value=0.5, step=0.1)))
        learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")
        return build_mlp(layers, learning_rate)

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(*args, batch_size=hp.Choice("batch_size", BATCH_SIZES), **kwargs)


def describe(hps):
    """The best hyperparameters as a plain dict, with only the layers that exist."""
    values = {"layers": hps.get("layers"), "learning_rate": round(hps.get("learning_rate"), 6),
              "batch_size": hps.get("batch_size")}
    for i in range(1, hps.get("layers") + 1):
        values[f"units_{i}"] = hps.get(f"units_{i}")
        values[f"dropout_{i}"] = round(hps.get(f"dropout_{i}"), 2)
    return values
