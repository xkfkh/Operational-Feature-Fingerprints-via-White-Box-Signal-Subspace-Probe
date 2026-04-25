#!/usr/bin/env bash
set -euo pipefail

# 用法示例：
# bash run_one_dataset_reproduce.sh \
#   /path/to/src_v16c_paper_experiments \
#   /path/to/planetoid/data \
#   texas \
#   default \
#   10 \
#   20260419 \
#   /path/to/output_dir

CODE_DIR="${1:-}"
DATA_ROOT="${2:-}"
DATASET="${3:-}"
GRID="${4:-default}"
REPEATS="${5:-10}"
SEED0="${6:-20260419}"
OUT_DIR="${7:-./reproduce_out}"

if [ -z "$CODE_DIR" ] || [ -z "$DATA_ROOT" ] || [ -z "$DATASET" ]; then
  echo "Usage:"
  echo "  bash run_one_dataset_reproduce.sh CODE_DIR DATA_ROOT DATASET [GRID] [REPEATS] [SEED0] [OUT_DIR]"
  exit 1
fi

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

cd "$CODE_DIR"

python scripts/run_E03_random_split_stability.py \
  --data-root "$DATA_ROOT" \
  --datasets "$DATASET" \
  --repeats "$REPEATS" \
  --seed0 "$SEED0" \
  --grid "$GRID" \
  --out-dir "$OUT_DIR"

