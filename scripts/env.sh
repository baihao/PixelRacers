#!/usr/bin/env bash
# source scripts/env.sh
if [ -n "${BASH_SOURCE[0]:-}" ]; then
  _ENV_FILE="${BASH_SOURCE[0]}"
elif [ -n "${ZSH_VERSION:-}" ]; then
  _ENV_FILE="${(%):-%x}"
else
  _ENV_FILE="$0"
fi
ROOT="$(cd "$(dirname "${_ENV_FILE}")/.." && pwd)"
unset PYTHONPATH
export HLOC_ROOT="${ROOT}/third_party/Hierarchical-Localization"
export KMP_DUPLICATE_LIB_OK=TRUE
if [ -f "${ROOT}/pixel-racers/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${ROOT}/pixel-racers/bin/activate"
fi
unset _ENV_FILE
