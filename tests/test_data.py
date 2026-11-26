"""The validation split: the same images every time, 20% of every class, never overlapping the training part."""

import numpy as np

from tuning.data import split_validation


def test_split_is_stratified_disjoint_and_fixed(cifar):
    (x, y), _ = cifar
    ids = np.arange(len(y))  # split the row numbers instead of the images, to see which rows went where
    (train_ids, train_y), (val_ids, val_y) = split_validation(ids, y)

    assert len(train_ids) == 40_000 and len(val_ids) == 10_000
    assert set(np.bincount(val_y)) == {1_000}  # 20% of each of the 10 classes
    assert not set(train_ids) & set(val_ids)
    assert (train_y == y[train_ids]).all() and (val_y == y[val_ids]).all()

    (_, _), (again, _) = split_validation(ids, y)
    np.testing.assert_array_equal(np.sort(val_ids), np.sort(again))
