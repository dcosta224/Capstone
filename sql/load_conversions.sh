#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
pip install -q -r requirements.txt
python3 scripts/load_food_density.py "$@"
