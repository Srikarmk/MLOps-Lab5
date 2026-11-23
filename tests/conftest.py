"""Shared test setup: quiet TensorFlow, deterministic ops, and a small slice of Fashion MNIST so tests run fast."""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from tuning import data  # noqa: E402
from tuning.lab import make_reproducible  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def deterministic():
    make_reproducible(0)


@pytest.fixture(scope="session")
def fashion():
    return data.load_fashion_mnist()


@pytest.fixture
def small_data(monkeypatch, fashion):
    """Make the lab load 2,000 training and 500 test images instead of 60,000 and 10,000."""
    (x, y), (x_test, y_test) = fashion
    rng = np.random.default_rng(0)
    keep = rng.choice(len(y), 2000, replace=False)
    small = ((x[keep], y[keep]), (x_test[:500], y_test[:500]))
    monkeypatch.setattr("tuning.lab.load_fashion_mnist", lambda: small)
    monkeypatch.setattr("tuning.compare.load_fashion_mnist", lambda: small)
    return small
