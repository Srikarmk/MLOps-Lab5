"""The baseline CNN, and the hypermodel whose hyperparameters Keras Tuner searches."""

import keras
import keras_tuner as kt

INPUT_SHAPE = (32, 32, 3)
MIN_BLOCKS, MAX_BLOCKS = 2, 4  # four 2x2 poolings take 32x32 down to 2x2
FILTERS = [16, 32, 64]  # 128 filters with 5x5 kernels trained ~8x slower per epoch than the baseline on a laptop CPU
BATCH_SIZES = [32, 64, 128, 256]
BASELINE_BATCH_SIZE = 32  # Keras's default


def build_cnn(filters, kernel_size, dense_units, dropout, learning_rate, augment):
    """A small convolutional network for 32x32 colour images.

    - `filters`: one entry per block; each block is a ReLU convolution ("same" padding) and a 2x2 max pooling
    - then a ReLU dense layer of `dense_units`, dropout, and the 10-class softmax
    - `augment`: random horizontal flips and shifts of up to 10% of the image, only while training
    """
    model = keras.Sequential([keras.Input(shape=INPUT_SHAPE)])
    if augment:
        model.add(keras.layers.RandomFlip("horizontal", name="augment_flip"))
        model.add(keras.layers.RandomTranslation(0.1, 0.1, name="augment_shift"))
    for i, n in enumerate(filters, start=1):
        model.add(keras.layers.Conv2D(n, kernel_size, padding="same", activation="relu", name=f"conv_{i}"))
        model.add(keras.layers.MaxPooling2D(name=f"pool_{i}"))
    model.add(keras.layers.Flatten(name="flatten"))
    model.add(keras.layers.Dense(dense_units, activation="relu", name="dense"))
    model.add(keras.layers.Dropout(dropout, name="dropout"))
    model.add(keras.layers.Dense(10, activation="softmax", name="output"))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])
    return model


BASELINE = {"blocks": 3, "filters_1": 32, "filters_2": 64, "filters_3": 64, "kernel_size": 3, "dense_units": 64,
            "dropout": 0.0, "augment": False, "learning_rate": 0.001, "batch_size": BASELINE_BATCH_SIZE}


def build_baseline():
    """The hand-picked model, after TensorFlow's CNN tutorial: convolutions of 32, 64 and 64 filters (3x3),
    a dense layer of 64, no dropout or augmentation, Adam at its default learning rate of 0.001."""
    return build_cnn([32, 64, 64], 3, 64, 0.0, 1e-3, augment=False)


class CNNHyperModel(kt.HyperModel):
    """The search space: the network's depth and width, regularization, learning rate and batch size.

    - blocks: 2 to 4 convolution blocks. Each block's filters only exist when the block does
      (a conditional hyperparameter), so the tuner never records values for a block that isn't there.
    - kernel_size (3 or 5), dense_units (32 to 256), dropout (0 to 0.5) and augment (on or off).
    - learning_rate: anywhere from 0.0001 to 0.01, sampled on a log scale so each power of ten is equally likely.
    - batch_size: tuned in fit(), because it is a setting of training rather than of the model.

    The baseline is one point in this space, so the search can always find it.

    Every trial starts from the same random state (`seed`), so a trial's result depends only on its
    hyperparameters. Without this, the weights a trial starts from depend on what ran before it in the
    process, and background threads (such as the TensorBoard callback's) made repeated searches differ.
    """

    def __init__(self, seed=42, **kwargs):
        super().__init__(**kwargs)
        self.seed = seed

    def build(self, hp):
        keras.utils.set_random_seed(self.seed)
        n_blocks = hp.Int("blocks", min_value=MIN_BLOCKS, max_value=MAX_BLOCKS)
        filters = []
        for i in range(1, MAX_BLOCKS + 1):
            with hp.conditional_scope("blocks", list(range(max(i, MIN_BLOCKS), MAX_BLOCKS + 1))):
                if i <= n_blocks:
                    filters.append(hp.Choice(f"filters_{i}", FILTERS))
        return build_cnn(
            filters,
            kernel_size=hp.Choice("kernel_size", [3, 5]),
            dense_units=hp.Int("dense_units", min_value=32, max_value=256, step=32),
            dropout=hp.Float("dropout", min_value=0.0, max_value=0.5, step=0.1),
            learning_rate=hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log"),
            augment=hp.Boolean("augment"),
        )

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(*args, batch_size=hp.Choice("batch_size", BATCH_SIZES), **kwargs)


def describe(hps):
    """The hyperparameters as a plain dict, in the same order as BASELINE, with only the blocks that exist."""
    values = {"blocks": hps.get("blocks")}
    for i in range(1, hps.get("blocks") + 1):
        values[f"filters_{i}"] = hps.get(f"filters_{i}")
    values.update({"kernel_size": hps.get("kernel_size"), "dense_units": hps.get("dense_units"),
                   "dropout": round(hps.get("dropout"), 2), "augment": hps.get("augment"),
                   "learning_rate": round(hps.get("learning_rate"), 6), "batch_size": hps.get("batch_size")})
    return values
