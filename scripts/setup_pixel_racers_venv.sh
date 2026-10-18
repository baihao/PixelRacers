#!/usr/bin/env bash
# Create and populate the pixel-racers virtual environment at PixelRacers/pixel-racers.
#
# Usage (from anywhere):
#   bash scripts/setup_pixel_racers_venv.sh
#   bash scripts/setup_pixel_racers_venv.sh --recreate   # delete existing venv first
#
# After setup:
#   source scripts/env.sh
#   jupyter notebook submission-demo/run_submission.ipynb
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV_DIR="${ROOT}/pixel-racers"
RECREATE=false
INSTALL_KERNEL=true
SKIP_SUBMODULE=false

for arg in "$@"; do
  case "$arg" in
    --recreate) RECREATE=true ;;
    --no-kernel) INSTALL_KERNEL=false ;;
    --skip-submodule) SKIP_SUBMODULE=true ;;
    -h|--help)
      echo "Usage: $0 [--recreate] [--no-kernel] [--skip-submodule]"
      exit 0
      ;;
    *)
      echo "Unknown option: $arg" >&2
      exit 1
      ;;
  esac
done

if $RECREATE && [[ -d "$VENV_DIR" ]]; then
  echo "[setup] Removing existing $VENV_DIR"
  rm -rf "$VENV_DIR"
fi

PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  for candidate in python3.12 python3.11 python3; do
    if command -v "$candidate" &>/dev/null; then
      PYTHON="$candidate"
      break
    fi
  done
fi
if [[ -z "$PYTHON" ]]; then
  echo "[setup] ERROR: no python3 found. Install Python 3.11+ (3.12 recommended)." >&2
  exit 1
fi

echo "[setup] Using interpreter: $($PYTHON --version 2>&1) ($("$PYTHON" -c 'import sys; print(sys.executable)'))"
echo "[setup] Creating venv: $VENV_DIR"
"$PYTHON" -m venv "$VENV_DIR"

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m pip install -U pip wheel setuptools

if ! $SKIP_SUBMODULE; then
  echo "[setup] Initializing git submodules (Hierarchical-Localization / hloc)..."
  git submodule update --init --recursive
fi

echo "[setup] Installing Python dependencies from requirements.txt..."
python -m pip install -r requirements.txt

if $INSTALL_KERNEL; then
  echo "[setup] Registering Jupyter kernel: pixel-racers"
  python -m ipykernel install --user --name pixel-racers --display-name "pixel-racers"
fi

cat <<EOF

[setup] Done.

  source ${ROOT}/scripts/env.sh
  # optional: git lfs install && git lfs pull   (submission SfM maps)

  cd ${ROOT}
  jupyter notebook submission-demo/run_submission.ipynb

See README.md and THIRD_PARTY_NOTICES.md for licenses and downloaded model weights.
EOF
