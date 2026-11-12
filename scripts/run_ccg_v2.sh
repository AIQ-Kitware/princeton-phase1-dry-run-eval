#!/usr/bin/env bash
#
# Run the ccg_v2 card, cards/ccg_v2.yaml.
#
# The card generates nothing: it scores the bundled ladders in data/ccg_v2/ladders (or, with
# ccg_v2.ladders_dpath overridden in the card, the ladders of a version-1 run).
# No GPU and no model endpoint are needed.
#
#   bash scripts/run_ccg_v2.sh                     # full run
#   bash scripts/run_ccg_v2.sh --dry_run True      # compile the pipeline only
#
# Environment variables:
#   OUTPUT_PATH   where results go (default: evaluation_runs)
#   BACKEND       kwdagger backend: serial (default, runs in the foreground) or tmux
#
# Any further arguments are passed to `magnet evaluate_new`.

set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

# The pipeline's stages import `cards` from the repository root and
# `contextual_drag` from src/.
export PYTHONPATH="$repo_root:$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"

if ! python -c "import magnet, sys; sys.exit(0 if magnet.__version__.startswith('0.1') else 1)" 2>/dev/null; then
    echo "[run] MAGNET 0.1.x is not installed in this environment." >&2
    echo "[run] Install it with:  bash scripts/install.sh   (or pip install -e \".[magnet,analysis,eval,inference]\")" >&2
    exit 1
fi

exec python -m magnet evaluate_new cards/ccg_v2.yaml \
    --output_path "${OUTPUT_PATH:-evaluation_runs}" \
    --backend "${BACKEND:-serial}" \
    "$@"
