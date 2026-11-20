"""The lab end to end: search hyperparameters with Hyperband, then compare the tuned model with the baseline fairly.

    python -m tuning                      the full lab (about 10 minutes on a laptop CPU)
    python -m tuning --quick              a quick check that everything runs (about 1 minute)

Both models are trained the same way (until validation accuracy stops improving, keeping the best
epoch) with several seeds, and compared on the test set only at the end. Each run writes to its own
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
from tuning.models import build_baseline, model_builder

# Two-sided 95% critical values of Student's t, by degrees of freedom, for the paired-difference interval
T_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="python -m tuning", description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=42, help="seed for the search; final models use seed, seed+1, ... (default 42)")
    parser.add_argument("--max-epochs", type=int, default=10,
                        help="Hyperband's max_epochs: the most epochs any one trial trains for (default 10)")
    parser.add_argument("--repeats", type=int, default=5, help="seeds each final model is trained with (default 5)")
    parser.add_argument("--final-max-epochs", type=int, default=30,
                        help="epoch limit for the final models, which stop early (default 30)")
    parser.add_argument("--patience", type=int, default=3,
                        help="epochs without a better validation accuracy before a final model stops (default 3)")
    parser.add_argument("--quick", action="store_true",
                        help="a fast check: --max-epochs 2 --repeats 2 --final-max-epochs 2")
    parser.add_argument("--runs-dir", default="runs", help="where run folders are created (default runs/)")
    args = parser.parse_args(argv)
    if args.quick:
        args.max_epochs, args.repeats, args.final_max_epochs = 2, 2, 2
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


def train_and_evaluate(build, seed, train, val, test, max_epochs, patience, log_dir):
    """Train one model until validation accuracy stops improving, keep its best epoch, and score it on the test set."""
    keras.backend.clear_session()  # start from a clean graph, so every seed builds the same model
    keras.utils.set_random_seed(seed)
    model = build()
    stop = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=patience, restore_best_weights=True)
    history = model.fit(*train, validation_data=val, epochs=max_epochs, verbose=0,
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

    print("\n==> Hyperband search over units and learning rate")
    tuner = kt.Hyperband(
        model_builder,
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
    best_hps = tuner.get_best_hyperparameters()[0]
    best_trial = tuner.oracle.get_best_trials()[0]
    print(f"{trials} trials in {search_seconds}s. Best: units={best_hps.get('units')}, "
          f"learning_rate={best_hps.get('learning_rate')}, val_accuracy={best_trial.score:.4f}")

    seeds = [args.seed + i for i in range(args.repeats)]
    models = {
        "baseline": build_baseline,
        "tuned": lambda: tuner.hypermodel.build(best_hps),
    }
    results = {}
    for name, build in models.items():
        print(f"\n==> Train the {name} model with seeds {seeds}")
        runs = []
        for seed in seeds:
            run = train_and_evaluate(build, seed, train, val, test, args.final_max_epochs, args.patience,
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
        "search": {"trials": trials, "seconds": search_seconds, "best_val_accuracy": round(float(best_trial.score), 4),
                   "best_hyperparameters": {"units": best_hps.get("units"), "learning_rate": best_hps.get("learning_rate")}},
        "baseline": {"units": 512, "learning_rate": 0.001, **results["baseline"]},
        "tuned": {**output_hps(best_hps), **results["tuned"]},
        "tuned_minus_baseline": difference,
    }
    (run_dir / "results.json").write_text(json.dumps(output, indent=2))

    print(f"\n==> Results on the test set ({len(seeds)} seeds each)")
    for name in models:
        r, hps = results[name], output[name]
        print(f"{name.capitalize():9} units={hps['units']:<4} learning_rate={hps['learning_rate']:<7} "
              f"test_accuracy={r['test_accuracy_mean']:.4f} ± {r['test_accuracy_sd']:.4f}  "
              f"(best epoch {r['best_epoch_mean']:g} on average)")
    ci = difference.get("ci95")
    print(f"Tuned - baseline: {difference['mean']:+.4f}"
          + (f", 95% interval [{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci else "")
          + f"; tuned better with {difference['tuned_better_in']} of {difference['seeds']} seeds")
    print(f"\nSaved {run_dir / 'results.json'}")
    print(f"View the training curves: tensorboard --logdir {board}")
    return output


def output_hps(hps):
    return {"units": hps.get("units"), "learning_rate": hps.get("learning_rate")}
