"""Shared test setup: quiet TensorFlow, deterministic ops, and a small slice of CIFAR-10 so tests run fast."""

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
def cifar():
    return data.load_cifar10()


@pytest.fixture
def small_data(monkeypatch, cifar):
    """Make the lab load 2,000 training and 500 test images instead of 50,000 and 10,000."""
    (x, y), (x_test, y_test) = cifar
    rng = np.random.default_rng(0)
    keep = rng.choice(len(y), 2000, replace=False)
    small = ((x[keep], y[keep]), (x_test[:500], y_test[:500]))
    monkeypatch.setattr("tuning.lab.load_cifar10", lambda: small)
    monkeypatch.setattr("tuning.compare.load_cifar10", lambda: small)
    monkeypatch.setattr("tuning.augment.load_cifar10", lambda: small)
    return small
