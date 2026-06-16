#!/usr/bin/env bash
set -euo pipefail

ENV_DIR="${1:-.venv}"
PYTHON_BIN="${PYTHON:-}"

python_supported() {
  "$1" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if (3, 11) <= sys.version_info < (3, 13) else 1)
PY
}

if [[ -z "$PYTHON_BIN" ]]; then
  for candidate in python3.11 python3.12 python3; do
    if command -v "$candidate" >/dev/null 2>&1 && python_supported "$candidate"; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi

if [[ -z "$PYTHON_BIN" ]] || ! python_supported "$PYTHON_BIN"; then
  echo "llm_coherence requires Python 3.11 or 3.12 for the pinned analysis stack." >&2
  echo "Install Python 3.11/3.12, or rerun with PYTHON=/path/to/python3.11 $0 ${ENV_DIR}" >&2
  exit 1
fi

"$PYTHON_BIN" -m venv "$ENV_DIR"

# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"

python -m pip install --upgrade pip
python -m pip install -e .

python - <<'PY'
import importlib

for module_name in (
    "llm_coherence",
    "numpy",
    "pandas",
    "sklearn",
    "scipy",
    "statsmodels",
):
    importlib.import_module(module_name)

print("llm_coherence environment ready")
PY

echo
echo "Activate with:"
echo "  source ${ENV_DIR}/bin/activate"
