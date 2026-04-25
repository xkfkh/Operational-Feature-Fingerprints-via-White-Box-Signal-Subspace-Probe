#!/usr/bin/env bash
set -u

CODE_DIR="/root/autodl-tmp/srcv16c/src_v16c_paper_experiments"
DATA_ROOT="/root/autodl-tmp/planetoid/data"
OUT_ROOT="/root/autodl-tmp/texas_seed_sweep_parallel2_safe"
LOG_ROOT="/root/autodl-tmp/logs_src_v16c"

mkdir -p "$OUT_ROOT"
mkdir -p "$LOG_ROOT"

export CUDA_VISIBLE_DEVICES=""

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

cd "$CODE_DIR" || exit 1

run_one_seed0() {
    seed0="$1"
    out_dir="${OUT_ROOT}/seed0_${seed0}"
    log_file="${LOG_ROOT}/E03_texas_seed0_${seed0}_10repeats_safe.log"

    mkdir -p "$out_dir"

    echo "========== START texas seed0=${seed0} $(date '+%F %T') ==========" | tee -a "$log_file"

    python scripts/run_E03_random_split_stability.py \
        --data-root "$DATA_ROOT" \
        --datasets texas \
        --repeats 10 \
        --seed0 "$seed0" \
        --grid default \
        --out-dir "$out_dir" \
        >> "$log_file" 2>&1

    status=$?

    echo "========== END texas seed0=${seed0} $(date '+%F %T') exit_code=${status} ==========" | tee -a "$log_file"

    echo "$status" > "${out_dir}/EXIT_CODE.txt"
    date '+%F %T' > "${out_dir}/FINISH_TIME.txt"

    if [ "$status" -eq 0 ]; then
        touch "${out_dir}/DONE_SUCCESS"
    else
        touch "${out_dir}/DONE_FAILED"
    fi
}

run_one_seed0 20260439 &
run_one_seed0 20260449 &

wait

echo "========== SAFE TEXAS SEED SWEEP BATCH1 FINISHED $(date '+%F %T') =========="

