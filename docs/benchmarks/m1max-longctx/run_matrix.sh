#!/bin/zsh
# Run configs/plan.json cell by cell, then regenerate results/summary.* and charts/.
#   MTPLX_REPO  local MTPLX clone that has both arm commits
#               (git remote add fork https://github.com/shunya1810/MTPLX && git fetch fork)
#   MODEL_PATH  Qwen3.8-27B-MTPLX-Optimized-Speed-FP16 directory
#   MTPLX_PY    python with mlx / mlx-lm / transformers / tokenizers (e.g. the MTPLX runtime venv)
#   WORK_DIR    scratch dir for worktrees and raw per-cell logs (not committed)
set -eu
: "${MTPLX_REPO:?}" "${MODEL_PATH:?}" "${MTPLX_PY:?}" "${WORK_DIR:?}"
cd "${0:A:h}"
caffeinate -dimsu "$MTPLX_PY" scripts/bench_longctx.py --plan configs/plan.json \
  --mtplx-repo "$MTPLX_REPO" --model "$MODEL_PATH" --python "$MTPLX_PY" \
  --work-dir "$WORK_DIR" --out results/raw-cells.jsonl --skip-done "$@"
python3 scripts/summarize.py
