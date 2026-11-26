"""Does augmentation help the lab's chosen model once it trains as long as the final models do?

    python -m tuning.augment                       the latest lab run's chosen configuration (about 15 minutes)
    python -m tuning.augment --run runs/<date-time>   a specific lab run
    python -m tuning.augment --quick               a quick check that everything runs (about 30 seconds)

The search judges trials after at most 10 epochs, and Hyperband judges most of them after 2 or 4. Random flips
and shifts make the training images harder, so augmented trials look worse that early and rarely survive.
This retrains the lab's chosen configuration the same way as its final models (until validation accuracy
stops improving, up to 30 epochs), once with augmentation off and once with it on, with the same seeds, and
compares the two on the test set. When the lab trained its final models the same way, its baseline runs
with the same seeds are compared too.
Writes augment.json into the lab run's folder.
"""

import argparse
import json
import statistics
from pathlib import Path

import keras_tuner as kt

from tuning.data import load_cifar10, split_validation
from tuning.lab import make_reproducible, paired_difference, require_tensorboard, summarize, train_and_evaluate
from tuning.models import CNNHyperModel


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="python -m tuning.augment", description=__doc__.splitlines()[0])
    parser.add_argument("--run", help="a lab run folder with results.json (default: the latest one in --runs-dir)")
    parser.add_argument("--runs-dir", default="runs", help="where lab runs are (default runs/)")
    parser.add_argument("--repeats", type=int, default=3, help="seeds each version is trained with (default 3)")
    parser.add_argument("--final-max-epochs", type=int, default=30, help="epoch limit (default 30)")
    parser.add_argument("--patience", type=int, default=3,
                        help="stop after this many epochs without a better validation accuracy (default 3)")
    parser.add_argument("--quick", action="store_true", help="a fast check: one seed, 2 epochs")
    args = parser.parse_args(argv)
    if args.quick:
        args.repeats, args.final_max_epochs = 1, 2
    return args


def latest_lab_run(runs_dir):
    """The newest run folder that has the lab's results.json (strategy comparisons have comparison.json instead)."""
    runs = sorted(p.parent for p in Path(runs_dir).glob("*/results.json"))
    if not runs:
        raise SystemExit(f"No lab run with results.json in {runs_dir}/. Run the lab first: python -m tuning")
    return runs[-1]


def fixed_hyperparameters(values):
    """HyperParameters that make CNNHyperModel.build return exactly the network `values` describes."""
    hp = kt.HyperParameters()
    for name, value in values.items():
        hp.Fixed(name, value)
    return hp


def compare(first_runs, second_runs):
    """First minus second test accuracy, paired by seed, with a 95% interval."""
    d = paired_difference(first_runs, second_runs)
    return {"mean": d["mean"], "ci95": d.get("ci95"), "per_seed": d["per_seed"],
            "first_better_in": d["tuned_better_in"], "seeds": d["seeds"]}


def describe_difference(label, d):
    ci = f", 95% interval [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]" if d["ci95"] else ""
    return f"{label}: {d['mean']:+.4f}{ci}; better with {d['first_better_in']} of {d['seeds']} seeds"


def main(argv=None):
    args = parse_args(argv)
    require_tensorboard()
    run_dir = Path(args.run) if args.run else latest_lab_run(args.runs_dir)
    lab = json.loads((run_dir / "results.json").read_text())
    chosen = lab["tuned"]["hyperparameters"]
    exact = lab["tuned"].get("exact_hyperparameters")
    if exact is None:
        print("This lab run has no exact_hyperparameters (it predates them); using the rounded values, so the "
              "runs without augmentation won't match the lab's tuned runs exactly.")
    exact = exact or chosen
    seeds = [lab["seed"] + i for i in range(args.repeats)]
    make_reproducible(lab["seed"])
    print(f"Lab run: {run_dir}")
    print(f"Chosen configuration: {', '.join(f'{k}={v}' for k, v in chosen.items())}")

    (x_full, y_full), test = load_cifar10()
    train, val = split_validation(x_full, y_full)

    results = {}
    for augment in (False, True):
        name = "augmented" if augment else "not_augmented"
        hp = fixed_hyperparameters({**exact, "augment": augment})
        print(f"\n==> Chosen configuration with augmentation {'on' if augment else 'off'}, seeds {seeds}")
        runs = []
        for seed in seeds:
            run = train_and_evaluate(lambda s: CNNHyperModel(seed=s).build(hp), exact["batch_size"], seed,
                                     train, val, test, args.final_max_epochs, args.patience,
                                     run_dir / "tensorboard" / "augment" / f"{name}-seed{seed}")
            print(f"  seed {seed}: best epoch {run['best_epoch']:>2} of {run['epochs_trained']:>2}, "
                  f"val_accuracy {run['val_accuracy']:.4f}, test_accuracy {run['test_accuracy']:.4f}")
            runs.append(run)
        results[name] = summarize(runs)

    same_training = (lab.get("final_max_epochs"), lab.get("patience")) == (args.final_max_epochs, args.patience)
    baseline_runs = [r for r in lab["baseline"]["runs"] if r["seed"] in seeds] if same_training else []
    output = {
        "lab_run": str(run_dir),
        "chosen": chosen,
        "seeds": seeds,
        "final_max_epochs": args.final_max_epochs,
        "not_augmented": results["not_augmented"],
        "augmented": results["augmented"],
        "augmented_minus_not_augmented": compare(results["augmented"]["runs"], results["not_augmented"]["runs"]),
    }
    if baseline_runs and len(baseline_runs) == len(seeds):
        output["augmented_minus_baseline"] = compare(results["augmented"]["runs"], baseline_runs)
    (run_dir / "augment.json").write_text(json.dumps(output, indent=2))

    print(f"\n==> Results on the test set ({len(seeds)} seed{'s' * (len(seeds) > 1)} each)")
    for name in ("not_augmented", "augmented"):
        r = results[name]
        print(f"{name.replace('_', ' ').capitalize():14} test_accuracy={r['test_accuracy_mean']:.4f} ± "
              f"{r['test_accuracy_sd']:.4f}  (best epoch {r['best_epoch_mean']:g} on average)")
    if "augmented_minus_baseline" in output:
        print(f"{'Baseline':14} test_accuracy="
              f"{statistics.mean(r['test_accuracy'] for r in baseline_runs):.4f}  (the lab's runs, same seeds)")
    print(describe_difference("Augmented - not augmented", output["augmented_minus_not_augmented"]))
    if "augmented_minus_baseline" in output:
        print(describe_difference("Augmented - baseline", output["augmented_minus_baseline"]))
    print(f"\nSaved {run_dir / 'augment.json'}")
    return output


if __name__ == "__main__":
    main()
