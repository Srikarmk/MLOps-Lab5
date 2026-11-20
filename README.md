# Hyperparameter Tuning Lab: Keras Tuner on Fashion MNIST

This lab tunes a Fashion MNIST image classifier with [Keras Tuner](https://keras.io/keras_tuner/). It trains a hand-picked baseline, searches two hyperparameters with **Hyperband**, retrains the best combination and compares the two on the test set.

It is based on the course's hyperparameter tuning lab (`Labs/Model_Development/Hyper_Parameters_Tuning/Keras_Tuner.ipynb`), kept unchanged in [original/Keras_Tuner.ipynb](original/Keras_Tuner.ipynb). This version is a work in progress; the sections below describe it as it stands.

## Prerequisites

| Requirement | Version tested | Install |
|---|---|---|
| Python | 3.12 (TensorFlow 2.21 supports 3.10 to 3.13, not 3.14) | macOS: `brew install python@3.12`; Ubuntu: `sudo apt install python3.12 python3.12-venv`; Windows: [python.org](https://www.python.org/downloads/) |
| Git | any | macOS: `xcode-select --install`; Ubuntu: `sudo apt install git`; Windows: `winget install Git.Git` |

Packages, installed by the setup below from `requirements.txt`:

| Package | Version | Used for |
|---|---|---|
| tensorflow | 2.21.0 | training the models |
| keras | 3.15.1 | the model API |
| keras-tuner | 1.4.8 | the hyperparameter search |
| tensorboard | 2.21.0 | training curves. TensorFlow 2.21 no longer installs it, and Keras's TensorBoard callback fails without it |

Training runs on the CPU; no GPU is needed. You need about 1.5 GB of disk for the packages, plus 100 MB per run.

## Setup

```bash
git clone https://github.com/Srikarmk/MLOps-Lab5.git
cd MLOps-Lab5
python3.12 -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The install downloads about 280 MB, 220 MB of it TensorFlow, and took 11 minutes here. Check it:

```bash
python -c "import tensorflow as tf, keras, keras_tuner as kt; print(tf.__version__, keras.__version__, kt.__version__)"
```

This should print `2.21.0 3.15.1 1.4.8`.

## Run

```bash
python -m tuning
```

It takes about 4.5 minutes on a laptop CPU, almost all of it the search. The first run also downloads Fashion MNIST (30 MB) to `~/.keras/datasets/`. `python -m tuning --max-epochs 2 --epochs 2` checks everything runs in about 25 seconds.

It ends with:

```text
30 trials in 222s. Best: units=256, learning_rate=0.001, val_accuracy=0.8874
...
==> Results on the test set
Baseline  units=512  learning_rate=0.001   test_accuracy=0.8736 test_loss=0.3612
Tuned     units=256  learning_rate=0.001   test_accuracy=0.8818 test_loss=0.3373

Saved runs/<date-time>/results.json
View the training curves: tensorboard --logdir runs/<date-time>/tensorboard
```

Every run is seeded (`--seed`, default 42) and TensorFlow's ops are made deterministic, so the same seed gives exactly these numbers on the same setup. Each run writes to its own folder, `runs/<date-time>/`: the tuner's trials, TensorBoard logs and `results.json`.

TensorFlow prints a line starting `E0000 ... NodeDef mentions attribute use_unbounded_threadpool` after training. It is a harmless message from TensorFlow itself.

## What changed from the original lab

| Problem in the original | Change |
|---|---|
| No package list or versions. TensorFlow 2.21 no longer installs TensorBoard, so **every training run in the notebook crashes**. The notebook still carries on: the baseline is scored untrained (26.4% test accuracy), every search trial fails, `get_best_hyperparameters()` still returns the first trial's random settings as "optimal", and the "tuned" model (86.5%) appears far better than the baseline | Pinned `requirements.txt` that includes TensorBoard. The run checks for it before training, and **a failed trial stops the search** (`max_consecutive_failed_trials=1`). It then refuses to report a best unless every trial finished |
| Rerunning reuses the search saved in `kt_dir/` and silently returns the old result | Each run has its own folder, and the tuner starts fresh (`overwrite=True`) |
| No seeds: results change from run to run | Seeded and deterministic |
| The last cell starts TensorBoard on `/tb_logs`, a folder that doesn't exist | Logs for the baseline, every trial and the tuned model in one folder, with the exact `tensorboard` command printed |
| A notebook run cell by cell | A `tuning` package run with one command, with options for the seed and epochs |
