"""Compare Keras Tuner's three search strategies on the same search space with the same budget of training epochs.

    python -m tuning.compare              the full comparison (about 35 minutes on a laptop CPU)
    python -m tuning.compare --quick      a quick check that everything runs (about 1 minute)

For each search seed it runs Hyperband first and counts the epochs it actually trains. Random search and
Bayesian optimization then get the same number of epochs: as many full-length trials as fit. Every
search's chosen configuration is then retrained with several seeds, because a search's best score
comes from one lucky run (see the README's results).

Writes runs/<date-time>-compare/comparison.json and strategy_comparison.png.
"""

import argparse
import json
import statistics
import time
import warnings
from datetime import datetime
from pathlib import Path

import keras
import keras_tuner as kt
from sklearn.exceptions import ConvergenceWarning

from tuning.data import load_fashion_mnist, split_validation
from tuning.lab import BASELINE, check_search, make_reproducible, require_tensorboard, train_and_evaluate
from tuning.models import BASELINE_BATCH_SIZE, FashionHyperModel, build_baseline, describe

STRATEGIES = ["hyperband", "random", "bayesian"]  # Hyperband first: its epoch count sets the others' budget
LABELS = {"hyperband": "Hyperband", "random": "Random search", "bayesian": "Bayesian optimization"}
COLORS = {"hyperband": "#1b7837", "random": "#7b3294", "bayesian": "#e66101"}
# Keras Tuner draws hyperparameter values from a counter that starts at the seed and goes up by one per
# draw, so seeds 42 and 44 give the same values shifted by two draws. One search makes a few hundred
# draws; seeds 10,000 apart never share any.
DEFAULT_SEEDS = [42, 10042, 20042]

# Bayesian optimization's Gaussian process warns when a length scale reaches its bound. It still fits;
# the warning repeats on every trial and drowns the output.
warnings.filterwarnings("ignore", category=ConvergenceWarning)


class CountingHyperModel(FashionHyperModel):
    """The lab's hypermodel, recording how many epochs each trial actually trains and its best validation accuracy.

    Keras Tuner stores only a trial's best epoch, and Hyperband trials resume from earlier ones, so the
    epochs spent can't be read back from the trials afterwards.
    """

    def __init__(self, seed):
        super().__init__(seed=seed)
        self.log = []

    def fit(self, hp, model, *args, **kwargs):
        history = super().fit(hp, model, *args, **kwargs)
        val = history.history["val_accuracy"]
        self.log.append({"epochs": len(val), "best_val_accuracy": round(max(val), 4)})
        return history


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="python -m tuning.compare", description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                        help="search seeds, far apart so the searches are independent (default 42 10042 20042)")
    parser.add_argument("--max-epochs", type=int, default=10,
                        help="the most epochs one trial trains for, in every strategy (default 10)")
    parser.add_argument("--retrain-seeds", type=int, nargs="+", default=[42, 43, 44],
                        help="seeds each chosen configuration is retrained with (default 42 43 44)")
    parser.add_argument("--final-max-epochs", type=int, default=30, help="epoch limit when retraining (default 30)")
    parser.add_argument("--patience", type=int, default=3, help="early-stopping patience when retraining (default 3)")
    parser.add_argument("--bayesian-initial-points", type=int, default=4,
                        help="random trials before Bayesian optimization starts using its model (default 4)")
    parser.add_argument("--quick", action="store_true", help="a fast check: one seed, 2-epoch trials, one retrain")
    parser.add_argument("--runs-dir", default="runs", help="where run folders are created (default runs/)")
    args = parser.parse_args(argv)
    if args.quick:
        args.seeds, args.max_epochs, args.retrain_seeds, args.final_max_epochs = [42], 2, [42], 2
    return args


def make_tuner(strategy, hypermodel, seed, max_epochs, trials, directory, initial_points):
    common = {
        "objective": "val_accuracy",
        "seed": seed,
        "directory": str(directory),
        "project_name": f"{strategy}-seed{seed}",
        "overwrite": True,
        "max_retries_per_trial": 0,
        "max_consecutive_failed_trials": 1,
    }
    if strategy == "hyperband":
        return kt.Hyperband(hypermodel, max_epochs=max_epochs, factor=3, **common)
    if strategy == "random":
        return kt.RandomSearch(hypermodel, max_trials=trials, **common)
    # Keras Tuner's default is 3 random trials per hyperparameter before the Gaussian process is used:
    # 27 for this 9-hyperparameter space, more than the whole budget, so the default never leaves random search
    return kt.BayesianOptimization(hypermodel, max_trials=trials, num_initial_points=min(initial_points, trials),
                                   **common)


def best_so_far(log):
    """The search's progress: after each trial, the epochs spent so far and the best validation accuracy so far."""
    curve, spent, best = [], 0, 0.0
    for trial in log:
        spent += trial["epochs"]
        best = max(best, trial["best_val_accuracy"])
        curve.append([spent, best])
    return curve


def run_search(strategy, seed, args, train, val, directory, trials=None):
    make_reproducible(seed)
    hypermodel = CountingHyperModel(seed)
    tuner = make_tuner(strategy, hypermodel, seed, args.max_epochs, trials, directory, args.bayesian_initial_points)
    fit_kwargs = {} if strategy == "hyperband" else {"epochs": args.max_epochs}
    started = time.time()
    tuner.search(*train, validation_data=val, verbose=0, **fit_kwargs,
                 callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)])
    seconds = round(time.time() - started)
    check_search(tuner)
    best_hps = tuner.get_best_hyperparameters()[0]
    return tuner, best_hps, {
        "strategy": strategy,
        "seed": seed,
        "trials": len(hypermodel.log),
        "epochs": sum(t["epochs"] for t in hypermodel.log),
        "seconds": seconds,
        "search_best_val_accuracy": max(t["best_val_accuracy"] for t in hypermodel.log),
        "chosen": describe(best_hps),
        "curve": best_so_far(hypermodel.log),
    }


def retrain(build, batch_size, args, train, val, test, log_dir):
    runs = [train_and_evaluate(build, batch_size, seed, train, val, test, args.final_max_epochs, args.patience,
                               log_dir / f"seed{seed}")
            for seed in args.retrain_seeds]
    return {
        "retrained_val_accuracy": round(statistics.mean(r["val_accuracy"] for r in runs), 4),
        "retrained_test_accuracy": round(statistics.mean(r["test_accuracy"] for r in runs), 4),
        "retrain_runs": runs,
    }


def plot(searches, path):
    """Best validation accuracy so far against epochs spent, one line per search. Bayesian optimization is
    dashed: its first random trials are the same as random search's, so the two lines start on top of each other."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for strategy in STRATEGIES:
        for i, s in enumerate(x for x in searches if x["strategy"] == strategy):
            epochs, best = zip(*s["curve"])
            ax.step(epochs, best, where="post", color=COLORS[strategy], alpha=0.85, linewidth=1.6,
                    linestyle="--" if strategy == "bayesian" else "-", label=LABELS[strategy] if i == 0 else None)
    ax.set_xlabel("Training epochs spent by the search")
    ax.set_ylabel("Best validation accuracy so far")
    ax.set_title("Search strategies at the same epoch budget (one line per search seed)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    require_tensorboard()
    run_dir = Path(args.runs_dir) / (datetime.now().strftime("%Y%m%d-%H%M%S") + "-compare")
    print(f"Run folder: {run_dir}")
    (x_full, y_full), test = load_fashion_mnist()
    train, val = split_validation(x_full, y_full)

    searches = []
    for seed in args.seeds:
        budget = None
        for strategy in STRATEGIES:
            # Random and Bayesian trials train up to max_epochs each, so this many of them spend about
            # what Hyperband spent with the same seed
            trials = None if strategy == "hyperband" else max(1, round(budget / args.max_epochs))
            print(f"\n==> {LABELS[strategy]}, search seed {seed}" + (f": {trials} trials" if trials else ""))
            tuner, best_hps, search = run_search(strategy, seed, args, train, val, run_dir / "searches", trials)
            budget = budget or search["epochs"]
            search.update(retrain(lambda s: FashionHyperModel(seed=s).build(best_hps), best_hps.get("batch_size"),
                                  args, train, val, test, run_dir / "tensorboard" / f"{strategy}-seed{seed}"))
            print(f"  {search['trials']} trials, {search['epochs']} epochs, {search['seconds']}s. "
                  f"Search's best val_accuracy {search['search_best_val_accuracy']:.4f}; "
                  f"retrained: val {search['retrained_val_accuracy']:.4f}, test {search['retrained_test_accuracy']:.4f}")
            searches.append(search)

    print("\n==> Baseline, retrained the same way")
    baseline = retrain(lambda s: build_baseline(), BASELINE_BATCH_SIZE, args, train, val, test, run_dir / "tensorboard" / "baseline")

    summary = {}
    for strategy in STRATEGIES:
        group = [s for s in searches if s["strategy"] == strategy]
        summary[strategy] = {
            key: round(statistics.mean(s[key] for s in group), 4)
            for key in ["trials", "epochs", "search_best_val_accuracy", "retrained_val_accuracy",
                        "retrained_test_accuracy"]
        }

    run_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "settings": {"seeds": args.seeds, "max_epochs": args.max_epochs, "retrain_seeds": args.retrain_seeds,
                     "bayesian_initial_points": args.bayesian_initial_points,
                     "final_max_epochs": args.final_max_epochs, "patience": args.patience},
        "summary": summary,
        "baseline": {"hyperparameters": BASELINE, **baseline},
        "searches": searches,
    }
    (run_dir / "comparison.json").write_text(json.dumps(output, indent=2))
    plot(searches, run_dir / "strategy_comparison.png")

    n = len(args.seeds)
    print(f"\n==> Averages over {n} search seed{'s' * (n > 1)} (retrained with {len(args.retrain_seeds)} seeds each)")
    print(f"{'Strategy':22} {'trials':>6} {'epochs':>6} {'search best':>11} {'retrained val':>13} {'test':>7}")
    for strategy in STRATEGIES:
        s = summary[strategy]
        print(f"{LABELS[strategy]:22} {s['trials']:>6g} {s['epochs']:>6.0f} {s['search_best_val_accuracy']:>11.4f} "
              f"{s['retrained_val_accuracy']:>13.4f} {s['retrained_test_accuracy']:>7.4f}")
    print(f"{'Baseline':22} {'':>6} {'':>6} {'':>11} {baseline['retrained_val_accuracy']:>13.4f} "
          f"{baseline['retrained_test_accuracy']:>7.4f}")
    print(f"\nSaved {run_dir / 'comparison.json'} and {run_dir / 'strategy_comparison.png'}")
    return output


if __name__ == "__main__":
    main()
