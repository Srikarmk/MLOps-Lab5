"""The lab's safeguards and statistics, and a small end-to-end run of python -m tuning."""

import json
import statistics
import sys
from types import SimpleNamespace

import pytest

from tuning import augment, lab


def fake_tuner(*trials):
    return SimpleNamespace(oracle=SimpleNamespace(trials={t.trial_id: t for t in trials}))


def trial(trial_id, score=0.9, status="COMPLETED", message=None, **values):
    return SimpleNamespace(trial_id=trial_id, score=score, status=status, message=message,
                           hyperparameters=SimpleNamespace(values=values))


def test_a_search_with_a_failed_trial_or_no_trials_is_refused():
    assert lab.check_search(fake_tuner(trial("0"), trial("1"))) == 2
    with pytest.raises(SystemExit, match="1 of 2 trials failed"):
        lab.check_search(fake_tuner(trial("0"), trial("1", status="FAILED", message="boom")))
    with pytest.raises(SystemExit, match="did not finish cleanly"):
        lab.check_search(fake_tuner())


def test_missing_tensorboard_stops_before_any_training(monkeypatch):
    monkeypatch.setitem(sys.modules, "tensorboard", None)  # makes `import tensorboard` fail
    with pytest.raises(SystemExit, match="TensorBoard is not installed"):
        lab.require_tensorboard()


def test_top_configurations_keeps_full_length_trials_once_each_best_first():
    tuner = fake_tuner(
        trial("a", 0.95, units=64, **{"tuner/epochs": 2}),   # judged after 2 epochs: not comparable, left out
        trial("b", 0.88, units=128, **{"tuner/epochs": 10}),
        trial("c", 0.90, units=128, **{"tuner/epochs": 10}),  # same configuration as b, better score: kept
        trial("d", 0.89, units=256, **{"tuner/epochs": 10}),
        trial("e", 0.80, units=512, **{"tuner/epochs": 10}),
    )
    top = lab.top_configurations(tuner, 2, max_epochs=10)
    assert [t.trial_id for t in top] == ["c", "d"]


def test_paired_difference_and_its_interval():
    tuned = [{"test_accuracy": a} for a in [0.90, 0.80, 0.85]]
    baseline = [{"test_accuracy": a} for a in [0.88, 0.79, 0.86]]
    result = lab.paired_difference(tuned, baseline)
    diffs = [0.02, 0.01, -0.01]
    half = 4.303 * statistics.stdev(diffs) / 3 ** 0.5  # t with 2 degrees of freedom
    assert result["mean"] == pytest.approx(statistics.mean(diffs), abs=1e-4)
    assert result["ci95"] == pytest.approx([statistics.mean(diffs) - half, statistics.mean(diffs) + half], abs=1e-4)
    assert (result["tuned_better_in"], result["seeds"]) == (2, 3)


def test_quick_run_is_complete_and_reproducible(small_data, tmp_path):
    first = lab.main(["--quick", "--runs-dir", str(tmp_path / "a")])
    second = lab.main(["--quick", "--runs-dir", str(tmp_path / "b")])
    for result in (first, second):
        result["search"].pop("seconds")
    assert first == second

    saved = json.loads(next((tmp_path / "a").glob("*/results.json")).read_text())
    assert set(saved) >= {"search", "recheck", "baseline", "tuned", "tuned_minus_baseline"}
    candidates = saved["recheck"]["candidates"]
    best = max(candidates, key=lambda c: c["recheck_mean"])
    assert saved["tuned"]["hyperparameters"] == best["hyperparameters"]  # chosen by re-checked mean
    assert len(saved["tuned"]["runs"]) == len(saved["baseline"]["runs"]) == 2

    # the augmentation check rebuilds the chosen model exactly: with augmentation as the lab chose it, it
    # repeats the lab's tuned run for the same seed
    check = augment.main(["--quick", "--runs-dir", str(tmp_path / "a")])
    as_chosen = "augmented" if saved["tuned"]["hyperparameters"]["augment"] else "not_augmented"
    assert check[as_chosen]["runs"][0] == saved["tuned"]["runs"][0]
    assert "augmented_minus_baseline" in check
