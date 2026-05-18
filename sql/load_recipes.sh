#!/usr/bin/env bash
# Load recipe schema + data (see scripts/load_recipes.py).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
pip install -q -r requirements.txt
python3 scripts/load_recipes.py "$@"
