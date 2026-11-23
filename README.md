# Hyperparameter Tuning Lab: Keras Tuner on Fashion MNIST

This lab tunes a Fashion MNIST image classifier with [Keras Tuner](https://keras.io/keras_tuner/). It searches the network's depth, width, dropout, learning rate and batch size with **Hyperband**, re-checks the search's top configurations with fresh seeds before choosing one, then compares the tuned model with the lab's hand-picked baseline fairly: both trained the same way, with several seeds, on the test set only at the end. A second command compares Keras Tuner's three search strategies (Hyperband, random search and Bayesian optimization) at the same budget of training epochs.

It covers:
- a Keras Tuner **`HyperModel`** with **conditional hyperparameters** (each layer's settings exist only when the layer does), a **log-scale** learning rate, and the **batch size** tuned in `fit()`
- **Hyperband**, **random search** and **Bayesian optimization**, compared at the same budget of training epochs
- the **winner's curse**: the search's best score is optimistic, so its top configurations are re-checked with fresh seeds before one is chosen
- a fair comparison with the baseline (same training, 5 seeds, a paired 95% interval) and a search that is **reproducible** and **fails loudly**
- **tests** and a one-command run

It is based on the course's hyperparameter tuning lab (`Labs/Model_Development/Hyper_Parameters_Tuning/Keras_Tuner.ipynb`), kept unchanged in [original/Keras_Tuner.ipynb](original/Keras_Tuner.ipynb). With current packages that notebook doesn't work: every training run crashes, yet it still reports a "tuned" model far better than the baseline. [What changed](#what-changed-from-the-original-lab) lists every change and why.

## Contents

- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Run the whole lab in one command](#run-the-whole-lab-in-one-command)
- [Run the steps yourself](#run-the-steps-yourself)
- [Compare search strategies](#compare-search-strategies)
- [Results and findings](#results-and-findings)
- [What changed from the original lab](#what-changed-from-the-original-lab)
- [Project layout](#project-layout)
- [Troubleshooting](#troubleshooting)

## Prerequisites

| Requirement | Version tested | Install |
|---|---|---|
| Python | 3.12 (TensorFlow 2.21 supports 3.10 to 3.13, not 3.14) | macOS: `brew install python@3.12`; Ubuntu: `sudo apt install python3.12 python3.12-venv`; Windows: [python.org](https://www.python.org/downloads/) |
| Git | any | macOS: `xcode-select --install`; Ubuntu: `sudo apt install git`; Windows: `winget install Git.Git` |
| bash | any, for `run_lab.sh` | built into macOS and Linux. On Windows, use WSL 2 or Git Bash, or run the steps yourself |

Check Python:

```bash
python3.12 --version
```

This should print `Python 3.12.x`. Python 3.10, 3.11 or 3.13 also work; 3.14 does not, because TensorFlow 2.21 has no packages for it.

Packages, installed from `requirements.txt` by the setup below (or by `run_lab.sh`):

| Package | Version | Used for |
|---|---|---|
| tensorflow | 2.21.0 | training the models |
| keras | 3.15.1 | the model API |
| keras-tuner | 1.4.8 | the hyperparameter search |
| tensorboard | 2.21.0 | training curves. TensorFlow 2.21 no longer installs it, and Keras's TensorBoard callback fails without it |
| scipy / scikit-learn | 1.18.1 / 1.9.1 | Keras Tuner's `BayesianOptimization` (its optional `bayesian` extra), used by the strategy comparison |
| matplotlib | 3.11.2 | the strategy comparison's plot |
| pytest | 9.1.1 | the tests in `tests/` |

Training runs on the CPU; no GPU is needed. You need about 1.7 GB of disk for the packages, plus about 130 MB per lab run and 600 MB per full strategy comparison (the tuner keeps every trial's weights).

## Setup

`run_lab.sh` (next section) does all of this for you. To set up by hand:

```bash
git clone https://github.com/Srikarmk/MLOps-Lab5.git
cd MLOps-Lab5
python3.12 -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` pins every package's version.

The install downloads about 300 MB, 220 MB of it TensorFlow, and took 11 minutes here. Check it:

```bash
python -c "import tensorflow as tf, keras, keras_tuner as kt; print(tf.__version__, keras.__version__, kt.__version__)"
```

This should print `2.21.0 3.15.1 1.4.8`.

## Run the whole lab in one command

```bash
git clone https://github.com/Srikarmk/MLOps-Lab5.git
cd MLOps-Lab5
./run_lab.sh
```

`run_lab.sh` runs four stages and stops at the first failure:
1. **Python:** finds Python 3.10–3.13 (preferring 3.12) and creates `.venv`, unless it already exists. To pick one, set `PYTHON=/path/to/python3.12`.
2. **Packages:** installs the pinned packages from `requirements.txt`. The first time takes about 10 minutes.
3. **Tests:** runs the 16 tests in `tests/`, about 15 seconds.
4. **The lab:** runs `python -m tuning`, about 10 minutes.

Options:
- `./run_lab.sh --quick` runs a short version of everything in about 1 minute after the install.
- `./run_lab.sh --compare` also runs the strategy comparison, about 35 more minutes.
- The two combine: `./run_lab.sh --quick --compare` takes about 1 minute after the install.

## Run the steps yourself

With the virtual environment active (`source .venv/bin/activate`):

```bash
python -m tuning
```

It takes about 10 minutes on a laptop CPU: about 2 minutes of search, 2 minutes of re-checks, then 10 final models (longer if other programs are busy). The first run also downloads Fashion MNIST (30 MB) to `~/.keras/datasets/`. `python -m tuning --quick` checks everything runs in about 25 seconds.

What it does:
1. **Splits the data.** It sets aside 12,000 of the 60,000 training images as a validation set: 20% of each class, always the same images (`tuning/data.py`). The 10,000 test images are only used in step 5.
2. **Searches** with Hyperband (30 trials, at most 10 epochs each), scored by validation accuracy. The search space is `FashionHyperModel` in `tuning/models.py`:

   | Hyperparameter | Values | Notes |
   |---|---|---|
   | `layers` | 1, 2 or 3 hidden layers | |
   | `units_1` … `units_3` | 32 to 512, in steps of 32 | conditional: only exists for a layer the network has |
   | `dropout_1` … `dropout_3` | 0 to 0.5, in steps of 0.1 | conditional, like the units |
   | `learning_rate` | 0.0001 to 0.01 | sampled on a log scale, so each power of ten is equally likely |
   | `batch_size` | 32, 64, 128 or 256 | a training setting, tuned in the hypermodel's `fit()` |
3. **Re-checks the top 5.** It takes the search's 5 best distinct configurations among trials that trained for the full 10 epochs, and retrains each with 3 fresh seeds under exactly the search's conditions. The best trial's score is the luckiest of 30 single runs, so it is optimistic; the re-checks' mean is not. The configuration with the best re-checked mean is chosen.
4. **Trains both models the same way**, with 5 seeds each: up to 30 epochs, stopping when validation accuracy hasn't improved for 3 epochs, and keeping the best epoch's weights.
5. **Compares them on the test set**: mean ± standard deviation over the seeds, and the tuned-minus-baseline difference seed by seed, with a 95% interval.

It ends with:

```text
30 trials in 126s. Best (val_accuracy 0.8942): layers=1, learning_rate=0.00121, batch_size=32, units_1=288, dropout_1=0.0

==> Re-check the top 5 full-length configurations with seeds [142, 143, 144]
  #1: search 0.8942, re-checked 0.8864 [0.8913, 0.8845, 0.8835]  layers=1, learning_rate=0.00121, batch_size=32, units_1=288, dropout_1=0.0
  #2: search 0.8940, re-checked 0.8918 [0.8891, 0.8925, 0.8938]  layers=1, learning_rate=0.001095, batch_size=64, units_1=512, dropout_1=0.2
  #3: search 0.8938, re-checked 0.8910 [0.8922, 0.8924, 0.8883]  layers=3, learning_rate=0.000349, batch_size=32, units_1=448, dropout_1=0.0, units_2=288, dropout_2=0.0, units_3=256, dropout_3=0.3
  #4: search 0.8919, re-checked 0.8880 [0.8827, 0.892, 0.8893]  layers=1, learning_rate=0.00145, batch_size=32, units_1=512, dropout_1=0.0
  #5: search 0.8905, re-checked 0.8859 [0.8867, 0.8867, 0.8843]  layers=3, learning_rate=0.000445, batch_size=256, units_1=416, dropout_1=0.1, units_2=416, dropout_2=0.2, units_3=32, dropout_3=0.4
Chosen: search's #2, the best re-checked mean (0.8918). The search's scores were +0.0043 above the re-checks on average; its #1 by +0.0078.
...
==> Results on the test set (5 seeds each)
Baseline  test_accuracy=0.8798 ± 0.0054  (best epoch 11.6 on average)  layers=1, learning_rate=0.001, batch_size=32, units_1=512, dropout_1=0.2
Tuned     test_accuracy=0.8767 ± 0.0036  (best epoch 9 on average)  layers=1, learning_rate=0.001095, batch_size=64, units_1=512, dropout_1=0.2
Tuned - baseline: -0.0031, 95% interval [-0.0071, +0.0009]; tuned better with 1 of 5 seeds

Saved runs/<date-time>/results.json
View the training curves: tensorboard --logdir runs/<date-time>/tensorboard
```

Every run is seeded (`--seed`, default 42) and TensorFlow's ops are made deterministic, so the same seed gives exactly these numbers on the same setup. Every search trial also starts from the same random state, so a trial's score depends only on its hyperparameters. Each run writes to its own folder, `runs/<date-time>/`: the tuner's trials, TensorBoard logs and `results.json`. `python -m tuning --help` lists the options (seed, number of seeds, epoch limits).

Look at the training curves of the baseline, every search trial and the final models in TensorBoard, then open http://localhost:6006:

```bash
tensorboard --logdir runs/<date-time>/tensorboard
```

Run the tests on their own:

```bash
python -m pytest -v
```

They check the validation split, the search space (conditional layers, log-scale learning rate, batch size, per-trial seeding), the safeguards (a failed trial or a missing TensorBoard stops the run), the re-check's choice of candidates, the paired interval, and that the strategy comparison gives every strategy the same budget. Two of them run the lab and the comparison end to end on a 2,000-image slice of the data.

TensorFlow prints a line starting `E0000 ... NodeDef mentions attribute use_unbounded_threadpool` after training. It is a harmless message from TensorFlow itself; `run_lab.sh` sets `TF_CPP_MIN_LOG_LEVEL=2` to hide TensorFlow's other informational lines.

## Compare search strategies

```bash
python -m tuning.compare
```

This compares Keras Tuner's three search strategies on the same search space, with the same budget of **training epochs**. Trials are not a fair unit, because Hyperband deliberately cuts most of its trials short. It takes about 35 minutes; `python -m tuning.compare --quick` checks it runs in about 40 seconds.

For each of 3 search seeds (42, 10042, 20042):
1. **Hyperband** runs as in the lab, and its hypermodel counts the epochs each trial actually trains: 120 per search.
2. **Random search** and **Bayesian optimization** get the same budget: 12 trials of up to 10 epochs. Bayesian optimization starts with 4 random trials, then lets its model pick the rest.
3. Each search's chosen configuration is **retrained with 3 seeds** the same way as the lab's final models, and scored on validation and test. The baseline is retrained the same way for reference.

The results of the full run are saved in [results/strategy_comparison.json](results/strategy_comparison.json) and [results/strategy_comparison.png](results/strategy_comparison.png). A new run writes its own copy to `runs/<date-time>-compare/`.

![Best validation accuracy so far against epochs spent, for each strategy and search seed](results/strategy_comparison.png)

Averages over the 3 search seeds:

| Strategy | Trials | Epochs | Search's best validation accuracy | Chosen configuration retrained: validation | Test |
|---|---|---|---|---|---|
| Hyperband | 30 | 120 | 89.27% | 89.77% | 88.72% |
| Random search | 12 | 120 | 89.29% | 90.01% | 88.91% |
| Bayesian optimization | 12 | 119 | 89.48% | 89.85% | 88.63% |
| *Baseline, retrained the same way* | | | | *89.46%* | *88.19%* |

(The retrained scores are higher than the search's because retraining allows up to 30 epochs, against at most 10 in a trial.)

## Results and findings

**From the lab** (`python -m tuning`, one Hyperband search):

The search's top 5, re-checked with 3 fresh seeds each under the search's own conditions:

| Search rank | Configuration | Search's score | Re-checked mean | Re-checked rank |
|---|---|---|---|---|
| 1 | 1 layer of 288, no dropout, learning rate 0.0012, batch 32 | 89.42% | 88.64% | 4 |
| 2 | 1 layer of 512, dropout 0.2, learning rate 0.0011, batch 64 | 89.40% | **89.18%** | **1** |
| 3 | 3 layers (448, 288, 256), learning rate 0.00035, batch 32 | 89.38% | 89.10% | 2 |
| 4 | 1 layer of 512, no dropout, learning rate 0.0015, batch 32 | 89.19% | 88.80% | 3 |
| 5 | 3 layers (416, 416, 32), learning rate 0.00045, batch 256 | 89.05% | 88.59% | 5 |

- **The search's scores are optimistic, and its #1 most of all.** All 5 scored lower when re-checked, by 0.43 points on average. The #1 dropped 0.78 points and ranked only 4th. A search score is the luckiest of many single runs, and the top one is the luckiest of all. Re-checking with the exact search conditions and the search's own seed reproduces its score exactly, so the drop is seed luck, not a different setup.
- **The re-checked choice is almost the baseline.** It is the baseline's network (512 units, dropout 0.2) with a learning rate of 0.0011 instead of 0.001 and batches of 64 instead of 32. Trained like the baseline, it scores 0.31 points below it on test (95% interval −0.71 to +0.09). This search found nothing better than the hand-picked model.
- **The re-check fixes luck, not the short horizon.** It uses the search's 10-epoch trials, so it still favours configurations that learn fast. The strategy comparison below shows that slower, deeper configurations do better when trained to the end.

How the lab's result changed through the steps (the baseline is the same in every row: 87.98% ± 0.54 on test):

| Version | Chosen configuration | Test, 5 seeds | Tuned − baseline (95% interval) |
|---|---|---|---|
| Step 2: the original's search space (width, 3 learning rates) | 1 layer of 448, learning rate 0.001 | 88.38% ± 0.44 | +0.40 points (−0.15 to +0.95) |
| Step 3: the larger search space, search's #1 | 1 layer of 288, learning rate 0.0012 | 87.87% ± 0.46 | −0.11 points (−0.92 to +0.70) |
| Step 5: the larger search space, best of the top 5 re-checked | 1 layer of 512, learning rate 0.0011, batch 64 | 87.67% ± 0.36 | −0.31 points (−0.71 to +0.09) |

(Step 2's search ran before every trial started from the same random state, so it is not exactly reproducible with the current code.)

- **One run of each proves nothing.** Across seeds, a single model's test accuracy varies by about ±0.5 points. That's more than the difference the original notebook reported between its baseline and tuned models (0.36 points in one run).
- **The width the small search picks is luck.** In step 2, at learning rate 0.001, trials from 96 to 448 units all scored 88.7–89.6% on validation, a range about the size of seed-to-seed variation.
- **10 epochs was too few for the baseline.** Training until validation accuracy stops improving, its best epoch averaged 11.6. The original's fixed 10 epochs cut it short.

**From the strategy comparison** (`python -m tuning.compare`, 9 searches):

- **At this budget, no strategy wins.** The three strategies' average search scores are within 0.21 points of each other, and their retrained test scores within 0.28 points (88.63% to 88.91%). The spread between search seeds of one strategy is larger: Hyperband's three picks score 87.97% to 89.14% on test.
- **Tuning does help a little, and the lab's single search was unlucky.** 8 of the 9 searches picked a configuration that beats the baseline on test, by 0.56 points on average. The lab's own search is Hyperband with seed 42, whose pick was the worst of the nine and the only one below the baseline. One search is as noisy as one training run.
- **The good configurations are deeper, with a lower learning rate.** 8 of the 9 picks have 2 or 3 hidden layers, and random search and Bayesian optimization chose learning rates of 0.0001 to 0.0003, below the baseline's 0.001.
- **Hyperband's short first rounds favour fast learners.** Its picks have learning rates of 0.0004 to 0.0012, higher than every pick of the other two strategies (0.0001 to 0.0003). Hyperband judges most configurations after 2 epochs, where a high learning rate looks best. With seed 42, random search's winner (learning rate 0.000149) was among Hyperband's trials too, scored 86.54% after 2 epochs, and was dropped.
- **Keras Tuner's default Bayesian optimization would have been random search.** By default it starts with 3 random trials per hyperparameter, 27 for this space, more than its whole budget of 12. So it would never use its model, and its trials would be exactly random search's. The comparison sets `num_initial_points=4`.
- **Consecutive search seeds aren't independent.** Keras Tuner draws hyperparameter values from a counter that starts at the seed and goes up by one per draw, so seed 44's draws are seed 42's shifted by two. In a first run with seeds 42, 43 and 44, the 42 and 44 random searches shared 4 of their 12 configurations and picked the same one. The seeds are now 10,000 apart.

## What changed from the original lab

| Problem in the original | Change |
|---|---|
| No package list or versions. TensorFlow 2.21 no longer installs TensorBoard, so **every training run in the notebook crashes**. The notebook still carries on: the baseline is scored untrained (26.4% test accuracy), every search trial fails, `get_best_hyperparameters()` still returns the first trial's random settings as "optimal", and the "tuned" model (86.5%) appears far better than the baseline | Pinned `requirements.txt` that includes TensorBoard. The run checks for it before training, and **a failed trial stops the search** (`max_consecutive_failed_trials=1`). It then refuses to report a best unless every trial finished |
| Rerunning reuses the search saved in `kt_dir/` and silently returns the old result | Each run has its own folder, and the tuner starts fresh (`overwrite=True`) |
| No seeds: results change from run to run | Seeded and deterministic |
| The last cell starts TensorBoard on `/tb_logs`, a folder that doesn't exist | Logs for the baseline, every trial and the tuned model in one folder, with the exact `tensorboard` command printed |
| A notebook run cell by cell | A `tuning` package run with one command, with options for the seed and epochs |
| `validation_split=0.2` inside every `fit`, and a final retrain for a fixed 10 epochs | One fixed, stratified validation set for the search and the final models. The final models train until validation accuracy stops improving and keep their best epoch |
| Baseline and tuned model compared on one unseeded run each | Both trained the same way with 5 seeds, and compared by mean ± spread and a paired 95% interval. The test set is used only for this comparison |
| A model-builder function tuning two hyperparameters: one layer's units and three fixed learning rates | A `HyperModel` subclass tuning the depth (1–3 layers), each layer's units and dropout as **conditional** hyperparameters, the learning rate on a **log scale**, and the **batch size** in its `fit()` |
| Only Hyperband, with no way to tell whether another strategy would do better | `python -m tuning.compare`: Hyperband, random search and Bayesian optimization at the same budget of training epochs, over 3 independent search seeds, with every chosen configuration retrained over 3 seeds |
| The tuned model is the search's single best trial | The search's top 5 configurations are re-checked with 3 fresh seeds under the search's conditions, and the best re-checked mean is chosen. `results.json` records every candidate's search score and re-checked scores |
| Search trials start from whatever random state the previous one left. With the TensorBoard callback, a repeated search sometimes scored trials differently | Each trial reseeds before building its model, so repeated searches match exactly and a configuration gets the same score whichever strategy tries it |
| No tests | 16 pytest tests, including end-to-end runs of the lab and the comparison on a slice of the data |
| Manual cell-by-cell steps | `run_lab.sh`: Python check, pinned install, tests and the lab in one command |

## Project layout

```text
MLOps-Lab5/
├── run_lab.sh                  the whole lab in one command
├── requirements.txt            pinned packages
├── pytest.ini                  test settings
├── tuning/
│   ├── data.py                 Fashion MNIST and the fixed, stratified validation split
│   ├── models.py               the baseline and FashionHyperModel, the search space
│   ├── lab.py                  python -m tuning: search, re-check the top 5, fair comparison with the baseline
│   └── compare.py              python -m tuning.compare: the three search strategies at the same epoch budget
├── tests/                      pytest tests
├── results/                    the full strategy comparison's results and plot
├── original/Keras_Tuner.ipynb  the course's original notebook (unchanged)
└── runs/                       created by each run: trials, TensorBoard logs, results.json (git-ignored)
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `Python 3.10 to 3.13 is needed` from `run_lab.sh`, or `No matching distribution found for tensorflow==2.21.0` | Your Python is too new (3.14) or too old. Install Python 3.12 (see [Prerequisites](#prerequisites)), delete `.venv`, and run again, or point at it: `PYTHON=/path/to/python3.12 ./run_lab.sh` |
| `TensorBoard is not installed` | The packages were installed without `requirements.txt`. Run `pip install -r requirements.txt` |
| `Please install scipy before using the BayesianOptimization` | Same cause: run `pip install -r requirements.txt` |
| `The search did not finish cleanly (N of M trials failed)` | A trial crashed, and the lab stopped rather than report a result from an incomplete search. The message shows the first errors, and `runs/<date-time>/hyperband/trial_*/` has each trial's details |
| The run takes much longer than the times above | Training runs on the CPU and shares it with everything else; a busy browser tab can double the time. Close heavy programs, or use `--quick` to check that everything works |
| A long run stops when the laptop sleeps | Keep the machine awake: on macOS, prefix the command with `caffeinate -i` |
| `./run_lab.sh: Permission denied` | `chmod +x run_lab.sh`, or run `bash run_lab.sh` |
| Downloading Fashion MNIST fails | The first run downloads it from Google's storage. Check your internet connection; later runs use the copy in `~/.keras/datasets/` |
