"""The baseline and the search space: depth, conditional blocks, augmentation, a log-scale learning rate, batch
size in fit()."""

import keras_tuner as kt
import numpy as np
import pytest

from tuning.models import BASELINE, BATCH_SIZES, CNNHyperModel, build_baseline, describe


def layer_summary(model):
    return [(type(layer).__name__, getattr(layer, "filters", None), getattr(layer, "kernel_size", None),
             getattr(layer, "units", None)) for layer in model.layers]


def fixed(values):
    hp = kt.HyperParameters()
    for name, value in values.items():
        hp.Fixed(name, value)
    return hp


def test_baseline_is_the_tutorial_cnn():
    model = build_baseline()
    convs = [layer for layer in model.layers if layer.name.startswith("conv_")]
    assert [layer.filters for layer in convs] == [32, 64, 64]
    assert {layer.kernel_size for layer in convs} == {(3, 3)}
    assert model.get_layer("dense").units == 64
    assert model.get_layer("dropout").rate == 0.0
    assert not any(layer.name.startswith("augment_") for layer in model.layers)
    assert float(model.optimizer.learning_rate.numpy()) == pytest.approx(1e-3)


def test_the_baseline_is_a_point_in_the_search_space():
    hp = fixed(BASELINE)
    model = CNNHyperModel().build(hp)
    assert layer_summary(model) == layer_summary(build_baseline())
    assert describe(hp) == BASELINE


@pytest.mark.parametrize("blocks", [2, 3, 4])
def test_block_hyperparameters_exist_only_for_blocks_the_network_has(blocks):
    hp = fixed({"blocks": blocks, "batch_size": 64})
    model = CNNHyperModel().build(hp)

    assert len([layer for layer in model.layers if layer.name.startswith("conv_")]) == blocks
    assert {k for k in describe(hp) if k.startswith("filters_")} == {f"filters_{i}" for i in range(1, blocks + 1)}


@pytest.mark.parametrize("augment", [False, True])
def test_augmentation_layers_exist_only_when_augment_is_on(augment):
    model = CNNHyperModel().build(fixed({"augment": augment}))
    names = {layer.name for layer in model.layers}
    assert {n for n in names if n.startswith("augment_")} == ({"augment_flip", "augment_shift"} if augment else set())


def test_augmentation_changes_training_images_but_not_predictions():
    model = CNNHyperModel().build(fixed({"augment": True}))
    images = np.random.default_rng(0).random((4, 32, 32, 3), dtype=np.float32)
    flip, shift = model.get_layer("augment_flip"), model.get_layer("augment_shift")
    np.testing.assert_array_equal(shift(flip(images, training=False), training=False), images)
    assert not np.allclose(shift(flip(images, training=True), training=True), images)


def test_learning_rate_is_searched_on_a_log_scale_and_batch_size_in_fit():
    hp = kt.HyperParameters()
    hypermodel = CNNHyperModel()
    hypermodel.build(hp)
    learning_rate = hp._hps["learning_rate"][0]
    assert (learning_rate.min_value, learning_rate.max_value, learning_rate.sampling) == (1e-4, 1e-2, "log")

    class RecordingModel:
        def fit(self, *args, **kwargs):
            return kwargs

    hp.Fixed("batch_size", 128)
    assert hypermodel.fit(hp, RecordingModel(), epochs=1)["batch_size"] == 128
    assert hp.get("batch_size") in BATCH_SIZES


def test_every_build_starts_from_the_hypermodels_seed():
    hp = fixed({"blocks": 2})

    def weights(seed):
        return CNNHyperModel(seed=seed).build(hp).get_layer("conv_1").get_weights()[0]

    np.testing.assert_array_equal(weights(1), weights(1))
    assert not np.array_equal(weights(1), weights(2))
