"""The lab end to end: search the network's shape and training settings with Hyperband, then compare the tuned
model with the baseline fairly.

    python -m tuning                      the full lab (about 10 minutes on a laptop CPU)
    python -m tuning --quick              a quick check that everything runs (about 1 minute)

The search's top configurations are re-checked with fresh seeds before one is chosen, because the
search's best score is the luckiest of many single runs. The chosen model and the baseline are then
trained the same way (until validation accuracy stops improving, keeping the best epoch) with several
seeds, and compared on the test set only at the end. Each run writes to its own
folder, runs/<date-time>/: the tuner's trials, TensorBoard logs and results.json.
"""

import argparse
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

import keras
import keras_tuner as kt
import tensorflow as tf

from tuning.data import load_fashion_mnist, split_validation
from tuning.models import BASELINE_BATCH_SIZE, FashionHyperModel, build_baseline, describe

BASELINE = {"layers": 1, "learning_rate": 0.001, "batch_size": BASELINE_BATCH_SIZE, "units_1": 512, "dropout_1": 0.2}
RECHECK_SEED_OFFSET = 100  # re-check seeds are seed+100, seed+101, ...: different from the search's and the final models'

# Two-sided 95% critical values of Student's t, by degrees of freedom, for the paired-difference interval
T_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="python -m tuning", description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=42, help="seed for the search; final models use seed, seed+1, ... (default 42)")
    parser.add_argument("--max-epochs", type=int, default=10,
                        help="Hyperband's max_epochs: the most epochs any one trial trains for (default 10)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="the search's best configurations to re-check before choosing one (default 5)")
    parser.add_argument("--recheck-seeds", type=int, default=3,
                        help="fresh seeds each of those is retrained with, under the search's conditions (default 3)")
    parser.add_argument("--repeats", type=int, default=5, help="seeds each final model is trained with (default 5)")
    parser.add_argument("--final-max-epochs", type=int, default=30,
                        help="epoch limit for the final models, which stop early (default 30)")
    parser.add_argument("--patience", type=int, default=3,
                        help="epochs without a better validation accuracy before a final model stops (default 3)")
    parser.add_argument("--quick", action="store_true",
                        help="a fast check: --max-epochs 2 --top-k 2 --recheck-seeds 2 --repeats 2 --final-max-epochs 2")
    parser.add_argument("--runs-dir", default="runs", help="where run folders are created (default runs/)")
    args = parser.parse_args(argv)
    if args.quick:
        args.max_epochs, args.top_k, args.recheck_seeds, args.repeats, args.final_max_epochs = 2, 2, 2, 2, 2
    return args


def make_reproducible(seed):
    """Same seed, same numbers: seeds Python, NumPy and TensorFlow, and makes TensorFlow's ops deterministic."""
    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def require_tensorboard():
    """The TensorBoard callback needs the tensorboard package. Without it every fit fails, so stop now."""
    try:
        import tensorboard  # noqa: F401
    except ImportError:
        raise SystemExit("TensorBoard is not installed. Run: pip install -r requirements.txt") from None


def check_search(tuner):
    """Refuse to go on unless every trial finished.

    Keras Tuner skips failed trials and get_best_hyperparameters() still returns something even if none
    finished, so a broken search would otherwise produce a "best" model that was never evaluated.
    """
    trials = list(tuner.oracle.trials.values())
    failed = [t for t in trials if t.status != "COMPLETED"]
    if not trials or failed:
        details = "; ".join(f"trial {t.trial_id}: {t.status} {t.message or ''}".strip() for t in failed[:3])
        raise SystemExit(f"The search did not finish cleanly ({len(failed)} of {len(trials)} trials failed): {details}")
    return len(trials)


def train_and_evaluate(build, batch_size, seed, train, val, test, max_epochs, patience, log_dir):
    """Train one model until validation accuracy stops improving, keep its best epoch, and score it on the test set.

    `build(seed)` returns a new compiled model whose random initial weights come from `seed`.
    """
    keras.backend.clear_session()  # start from a clean graph, so every seed builds the same model
    keras.utils.set_random_seed(seed)
    model = build(seed)
    stop = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=patience, restore_best_weights=True)
    history = model.fit(*train, validation_data=val, epochs=max_epochs, batch_size=batch_size, verbose=0,
                        callbacks=[stop, keras.callbacks.TensorBoard(str(log_dir))])
    val_curve = history.history["val_accuracy"]
    best_epoch = max(range(len(val_curve)), key=val_curve.__getitem__)
    test_scores = model.evaluate(*test, return_dict=True, verbose=0)
    return {
        "seed": seed,
        "epochs_trained": len(val_curve),
        "best_epoch": best_epoch + 1,
        "val_accuracy": round(val_curve[best_epoch], 4),
        "test_accuracy": round(float(test_scores["accuracy"]), 4),
        "test_loss": round(float(test_scores["loss"]), 4),
    }


def top_configurations(tuner, k, max_epochs):
    """The k best distinct configurations among trials that trained for the full max_epochs.

    Hyperband judges most trials after a few epochs; only its full-length trials compare like for like
    with each other and with a full-length re-check. A configuration can appear in several trials (one
    per Hyperband round), so each is kept once, with its best score.
    """
    best = {}
    for trial in tuner.oracle.trials.values():
        values = trial.hyperparameters.values
        if values.get("tuner/epochs", max_epochs) != max_epochs:
            continue
        key = json.dumps({k: v for k, v in values.items() if not k.startswith("tuner/")}, sort_keys=True)
        if key not in best or trial.score > best[key].score:
            best[key] = trial
    return sorted(best.values(), key=lambda t: t.score, reverse=True)[:k]


def recheck(hps, seeds, train, val, max_epochs):
    """Retrain one configuration with each seed under the search's own conditions, and return each run's
    best validation accuracy: the same number the search scored it by, just not the luckiest one."""
    scores = []
    for seed in seeds:
        keras.backend.clear_session()
        model = FashionHyperModel(seed=seed).build(hps)
        history = model.fit(*train, validation_data=val, epochs=max_epochs, batch_size=hps.get("batch_size"),
                            verbose=0, callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)])
        scores.append(round(max(history.history["val_accuracy"]), 4))
    return scores


def summarize(runs):
    accuracies = [r["test_accuracy"] for r in runs]
    return {
        "test_accuracy_mean": round(statistics.mean(accuracies), 4),
        "test_accuracy_sd": round(statistics.stdev(accuracies), 4) if len(accuracies) > 1 else 0.0,
        "val_accuracy_mean": round(statistics.mean(r["val_accuracy"] for r in runs), 4),
        "best_epoch_mean": round(statistics.mean(r["best_epoch"] for r in runs), 1),
        "runs": runs,
    }


def paired_difference(tuned_runs, baseline_runs):
    """Tuned minus baseline test accuracy, seed by seed, with a 95% t interval for the mean difference."""
    diffs = [t["test_accuracy"] - b["test_accuracy"] for t, b in zip(tuned_runs, baseline_runs)]
    mean = statistics.mean(diffs)
    result = {"mean": round(mean, 4), "per_seed": [round(d, 4) for d in diffs],
              "tuned_better_in": sum(d > 0 for d in diffs), "seeds": len(diffs)}
    if len(diffs) > 1:
        half = T_95.get(len(diffs) - 1, 1.96) * statistics.stdev(diffs) / len(diffs) ** 0.5
        result["ci95"] = [round(mean - half, 4), round(mean + half, 4)]
    return result


def main(argv=None):
    args = parse_args(argv)
    require_tensorboard()
    make_reproducible(args.seed)
    run_dir = Path(args.runs_dir) / datetime.now().strftime("%Y%m%d-%H%M%S")
    board = run_dir / "tensorboard"
    print(f"Run folder: {run_dir}")

    (x_full, y_full), test = load_fashion_mnist()
    train, val = split_validation(x_full, y_full)
    print(f"{len(train[1])} training, {len(val[1])} validation and {len(test[1])} test images")

    print("\n==> Hyperband search over depth, units, dropout, learning rate and batch size")
    tuner = kt.Hyperband(
        FashionHyperModel(seed=args.seed),
        objective="val_accuracy",
        max_epochs=args.max_epochs,
        factor=3,
        seed=args.seed,
        directory=str(run_dir),
        project_name="hyperband",
        overwrite=True,  # never resume a search left over from another run
        max_retries_per_trial=0,
        max_consecutive_failed_trials=1,  # the first failure stops the search
    )
    tuner.search_space_summary()
    started = time.time()
    tuner.search(*train, validation_data=val, verbose=0,
                 callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=5),
                            keras.callbacks.TensorBoard(str(board / "search"))])
    search_seconds = round(time.time() - started)
    trials = check_search(tuner)
    best_trial = tuner.oracle.get_best_trials()[0]
    print(f"{trials} trials in {search_seconds}s. Best (val_accuracy {best_trial.score:.4f}): "
          f"{format_hps(describe(best_trial.hyperparameters))}")

    recheck_seeds = [args.seed + RECHECK_SEED_OFFSET + i for i in range(args.recheck_seeds)]
    candidates = top_configurations(tuner, args.top_k, args.max_epochs)
    print(f"\n==> Re-check the top {len(candidates)} full-length configurations with seeds {recheck_seeds}")
    rechecked = []
    for rank, trial in enumerate(candidates, start=1):
        scores = recheck(trial.hyperparameters, recheck_seeds, train, val, args.max_epochs)
        rechecked.append({
            "search_rank": rank,
            "hyperparameters": describe(trial.hyperparameters),
            "search_score": round(float(trial.score), 4),
            "recheck_scores": scores,
            "recheck_mean": round(statistics.mean(scores), 4),
            "_hps": trial.hyperparameters,
        })
        print(f"  #{rank}: search {trial.score:.4f}, re-checked {statistics.mean(scores):.4f} {scores}  "
              f"{format_hps(describe(trial.hyperparameters))}")
    chosen = max(rechecked, key=lambda c: c["recheck_mean"])
    best_hps = chosen.pop("_hps")
    for c in rechecked:
        c.pop("_hps", None)
    by_recheck = sorted(rechecked, key=lambda c: c["recheck_mean"], reverse=True)
    for recheck_rank, c in enumerate(by_recheck, start=1):
        c["recheck_rank"] = recheck_rank
    optimism = statistics.mean(c["search_score"] - c["recheck_mean"] for c in rechecked)
    print(f"Chosen: search's #{chosen['search_rank']}, the best re-checked mean ({chosen['recheck_mean']:.4f}). "
          f"The search's scores were {optimism:+.4f} above the re-checks on average; "
          f"its #1 by {rechecked[0]['search_score'] - rechecked[0]['recheck_mean']:+.4f}.")

    seeds = [args.seed + i for i in range(args.repeats)]
    hyperparameters = {"baseline": BASELINE, "tuned": describe(best_hps)}
    models = {
        "baseline": (lambda seed: build_baseline(), BASELINE_BATCH_SIZE),
        "tuned": (lambda seed: FashionHyperModel(seed=seed).build(best_hps), best_hps.get("batch_size")),
    }
    results = {}
    for name, (build, batch_size) in models.items():
        print(f"\n==> Train the {name} model with seeds {seeds}")
        runs = []
        for seed in seeds:
            run = train_and_evaluate(build, batch_size, seed, train, val, test, args.final_max_epochs, args.patience,
                                     board / "final" / f"{name}-seed{seed}")
            print(f"  seed {seed}: best epoch {run['best_epoch']:>2} of {run['epochs_trained']:>2}, "
                  f"val_accuracy {run['val_accuracy']:.4f}, test_accuracy {run['test_accuracy']:.4f}")
            runs.append(run)
        results[name] = summarize(runs)

    difference = paired_difference(results["tuned"]["runs"], results["baseline"]["runs"])
    output = {
        "seed": args.seed,
        "max_epochs": args.max_epochs,
        "split": {"train": len(train[1]), "validation": len(val[1]), "test": len(test[1])},
        "search": {"trials": trials, "seconds": search_seconds, "best_val_accuracy": round(float(best_trial.score), 4)},
        "recheck": {"seeds": recheck_seeds, "candidates": rechecked, "chosen_search_rank": chosen["search_rank"],
                    "mean_search_minus_recheck": round(optimism, 4)},
        "baseline": {"hyperparameters": hyperparameters["baseline"], **results["baseline"]},
        "tuned": {"hyperparameters": hyperparameters["tuned"], **results["tuned"]},
        "tuned_minus_baseline": difference,
    }
    (run_dir / "results.json").write_text(json.dumps(output, indent=2))

    print(f"\n==> Results on the test set ({len(seeds)} seeds each)")
    for name in models:
        r = results[name]
        print(f"{name.capitalize():9} test_accuracy={r['test_accuracy_mean']:.4f} ± {r['test_accuracy_sd']:.4f}  "
              f"(best epoch {r['best_epoch_mean']:g} on average)  {format_hps(hyperparameters[name])}")
    ci = difference.get("ci95")
    print(f"Tuned - baseline: {difference['mean']:+.4f}"
          + (f", 95% interval [{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci else "")
          + f"; tuned better with {difference['tuned_better_in']} of {difference['seeds']} seeds")
    print(f"\nSaved {run_dir / 'results.json'}")
    print(f"View the training curves: tensorboard --logdir {board}")
    return output


def format_hps(hps):
    return ", ".join(f"{k}={v}" for k, v in hps.items())
