"""The baseline and the search space: depth, conditional layers, a log-scale learning rate, batch size in fit()."""

import keras_tuner as kt
import numpy as np
import pytest

from tuning.models import BATCH_SIZES, FashionHyperModel, build_baseline, describe


def test_baseline_is_the_original_labs_model():
    model = build_baseline()
    hidden = [layer for layer in model.layers if layer.name.startswith("hidden_")]
    assert [layer.units for layer in hidden] == [512]
    assert model.get_layer("dropout_1").rate == 0.2
    assert float(model.optimizer.learning_rate.numpy()) == pytest.approx(1e-3)


@pytest.mark.parametrize("layers", [1, 2, 3])
def test_layer_hyperparameters_exist_only_for_layers_the_network_has(layers):
    hp = kt.HyperParameters()
    hp.Fixed("layers", layers)
    hp.Fixed("batch_size", 64)
    model = FashionHyperModel().build(hp)

    assert len([layer for layer in model.layers if layer.name.startswith("hidden_")]) == layers
    values = describe(hp)
    assert {k for k in values if k.startswith("units_")} == {f"units_{i}" for i in range(1, layers + 1)}
    assert {k for k in values if k.startswith("dropout_")} == {f"dropout_{i}" for i in range(1, layers + 1)}


def test_learning_rate_is_searched_on_a_log_scale_and_batch_size_in_fit():
    hp = kt.HyperParameters()
    hypermodel = FashionHyperModel()
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
    hp = kt.HyperParameters()
    hp.Fixed("layers", 1)

    def weights(seed):
        return FashionHyperModel(seed=seed).build(hp).get_layer("hidden_1").get_weights()[0]

    np.testing.assert_array_equal(weights(1), weights(1))
    assert not np.array_equal(weights(1), weights(2))
