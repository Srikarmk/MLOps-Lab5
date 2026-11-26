#!/usr/bin/env bash
# Runs the whole lab end to end:
#   find Python 3.10-3.13 -> .venv + pinned packages -> tests -> the lab (search, re-check, final comparison)
#   -> the augmentation check
#
#   ./run_lab.sh              the full lab (about 55 minutes after the install)
#   ./run_lab.sh --quick      a short version of everything (about 3 minutes after the install)
#   ./run_lab.sh --compare    also compare the three search strategies (about 3 more hours; with --quick, 2)
#
# Override the Python used to create .venv with PYTHON=/path/to/python3.12
set -euo pipefail
cd "$(dirname "$0")"

QUICK=()
COMPARE=false
for arg in "$@"; do
    case "$arg" in
        --quick) QUICK=(--quick) ;;
        --compare) COMPARE=true ;;
        *) echo "Unknown option: $arg (use --quick and/or --compare)" >&2; exit 1 ;;
    esac
done

step() { printf '\n==> %s\n' "$*"; }

supported() {  # TensorFlow 2.21 has packages for Python 3.10 to 3.13
    "$1" -c 'import sys; sys.exit(not ((3, 10) <= sys.version_info[:2] <= (3, 13)))' 2>/dev/null
}

if [ ! -x .venv/bin/python ]; then
    step "Creating .venv"
    PY="${PYTHON:-}"
    if [ -z "$PY" ]; then
        for candidate in python3.12 python3.13 python3.11 python3.10 python3; do
            if command -v "$candidate" >/dev/null && supported "$(command -v "$candidate")"; then
                PY="$(command -v "$candidate")"
                break
            fi
        done
    fi
    if [ -z "$PY" ] || ! supported "$PY"; then
        echo "Python 3.10 to 3.13 is needed (TensorFlow 2.21 has no packages for other versions)." >&2
        echo "Install Python 3.12 (see the README), or point PYTHON at one: PYTHON=/path/to/python3.12 ./run_lab.sh" >&2
        exit 1
    fi
    echo "Using $PY ($("$PY" --version))"
    "$PY" -m venv .venv
fi

step "Installing the pinned packages (the first time downloads about 300 MB)"
.venv/bin/python -m pip install --quiet --upgrade pip
.venv/bin/python -m pip install --quiet -r requirements.txt
.venv/bin/python -c "import sys, tensorflow as tf, keras, keras_tuner as kt; print(f'Python {sys.version.split()[0]}, TensorFlow {tf.__version__}, Keras {keras.__version__}, Keras Tuner {kt.__version__}')"

export TF_CPP_MIN_LOG_LEVEL=2  # hide TensorFlow's informational and warning lines

step "Tests"
.venv/bin/python -m pytest -q

step "The lab: search, re-check the top 5, compare with the baseline"
.venv/bin/python -m tuning ${QUICK[@]+"${QUICK[@]}"}

step "The augmentation check: the chosen configuration with augmentation off and on"
.venv/bin/python -m tuning.augment ${QUICK[@]+"${QUICK[@]}"}

if $COMPARE; then
    step "Search strategies at the same epoch budget"
    .venv/bin/python -m tuning.compare ${QUICK[@]+"${QUICK[@]}"}
fi

step "Done"
echo "Each run's results, trials and TensorBoard logs are in runs/. View the training curves with:"
echo "  .venv/bin/tensorboard --logdir runs/<date-time>/tensorboard"
