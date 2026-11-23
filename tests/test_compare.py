"""The strategy comparison: independent seeds, a real Bayesian optimization, equal epoch budgets."""

import json

from tuning import compare
from tuning.models import FashionHyperModel


def test_default_search_seeds_cannot_share_draws():
    # Keras Tuner's draws for seed s are s, s+1, s+2, ...; one search makes a few hundred
    seeds = sorted(compare.DEFAULT_SEEDS)
    assert all(b - a >= 1_000 for a, b in zip(seeds, seeds[1:]))


def test_bayesian_optimization_uses_its_model_within_the_budget(tmp_path):
    tuner = compare.make_tuner("bayesian", FashionHyperModel(), seed=1, max_epochs=10, trials=12,
                               directory=tmp_path, initial_points=4)
    # Keras Tuner's default would be 3 random trials per hyperparameter: more than all 12 trials
    assert tuner.oracle.num_initial_points == 4


def test_best_so_far_curve():
    log = [{"epochs": 2, "best_val_accuracy": 0.8}, {"epochs": 2, "best_val_accuracy": 0.7},
           {"epochs": 10, "best_val_accuracy": 0.9}]
    assert compare.best_so_far(log) == [[2, 0.8], [4, 0.8], [14, 0.9]]


def test_quick_comparison_gives_every_strategy_the_same_budget(small_data, tmp_path):
    output = compare.main(["--quick", "--runs-dir", str(tmp_path)])
    epochs = {s["strategy"]: s["epochs"] for s in output["searches"]}
    assert set(epochs) == set(compare.STRATEGIES)
    assert epochs["random"] == epochs["bayesian"] == epochs["hyperband"]
    run_dir = next(tmp_path.glob("*-compare"))
    assert json.loads((run_dir / "comparison.json").read_text())["summary"].keys() == set(compare.STRATEGIES)
    assert (run_dir / "strategy_comparison.png").stat().st_size > 10_000
