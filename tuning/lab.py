"""The lab end to end: train the baseline, search hyperparameters with Hyperband, retrain the best and compare.

    python -m tuning                      the full lab (about 8 minutes on a laptop CPU)
    python -m tuning --max-epochs 2 --epochs 2    a quick check that everything runs (about 1 minute)

Each run writes to its own folder, runs/<date-time>/: the tuner's trials, TensorBoard logs and results.json.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import keras
import keras_tuner as kt
import tensorflow as tf

from tuning.data import load_fashion_mnist
from tuning.models import build_baseline, model_builder

VALIDATION_SPLIT = 0.2


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="python -m tuning", description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=42, help="seed for weights, shuffling and the tuner (default 42)")
    parser.add_argument("--epochs", type=int, default=10, help="epochs for the baseline and final models (default 10)")
    parser.add_argument("--max-epochs", type=int, default=10,
                        help="Hyperband's max_epochs: the most epochs any one trial trains for (default 10)")
    parser.add_argument("--runs-dir", default="runs", help="where run folders are created (default runs/)")
    return parser.parse_args(argv)


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


def describe(model, layer_name, scores):
    return {
        "units": int(model.get_layer(layer_name).units),
        "learning_rate": round(float(model.optimizer.learning_rate.numpy()), 6),
        "test_accuracy": round(float(scores["accuracy"]), 4),
        "test_loss": round(float(scores["loss"]), 4),
    }


def main(argv=None):
    args = parse_args(argv)
    require_tensorboard()
    make_reproducible(args.seed)
    run_dir = Path(args.runs_dir) / datetime.now().strftime("%Y%m%d-%H%M%S")
    board = run_dir / "tensorboard"
    print(f"Run folder: {run_dir}")

    (x_train, y_train), (x_test, y_test) = load_fashion_mnist()

    print("\n==> Baseline: 512 units, learning rate 0.001")
    baseline = build_baseline()
    baseline.fit(x_train, y_train, epochs=args.epochs, validation_split=VALIDATION_SPLIT, verbose=2,
                 callbacks=[keras.callbacks.TensorBoard(str(board / "baseline"))])
    baseline_result = describe(baseline, "dense_1", baseline.evaluate(x_test, y_test, return_dict=True, verbose=0))

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
    tuner.search(x_train, y_train, validation_split=VALIDATION_SPLIT, verbose=0,
                 callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=5),
                            keras.callbacks.TensorBoard(str(board / "search"))])
    search_seconds = round(time.time() - started)
    trials = check_search(tuner)
    best_hps = tuner.get_best_hyperparameters()[0]
    best_trial = tuner.oracle.get_best_trials()[0]
    print(f"{trials} trials in {search_seconds}s. Best: units={best_hps.get('units')}, "
          f"learning_rate={best_hps.get('learning_rate')}, val_accuracy={best_trial.score:.4f}")

    print("\n==> Retrain the best hyperparameters")
    tuned = tuner.hypermodel.build(best_hps)
    tuned.fit(x_train, y_train, epochs=args.epochs, validation_split=VALIDATION_SPLIT, verbose=2,
              callbacks=[keras.callbacks.TensorBoard(str(board / "tuned"))])
    tuned_result = describe(tuned, "tuned_dense_1", tuned.evaluate(x_test, y_test, return_dict=True, verbose=0))

    results = {
        "seed": args.seed,
        "epochs": args.epochs,
        "max_epochs": args.max_epochs,
        "search": {"trials": trials, "seconds": search_seconds, "best_val_accuracy": round(float(best_trial.score), 4)},
        "baseline": baseline_result,
        "tuned": tuned_result,
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))

    print("\n==> Results on the test set")
    for name, result in [("Baseline", baseline_result), ("Tuned", tuned_result)]:
        print(f"{name:9} units={result['units']:<4} learning_rate={result['learning_rate']:<7} "
              f"test_accuracy={result['test_accuracy']:.4f} test_loss={result['test_loss']:.4f}")
    print(f"\nSaved {run_dir / 'results.json'}")
    print(f"View the training curves: tensorboard --logdir {board}")
    return results
