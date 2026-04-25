#!/usr/bin/env bash
set -u

MASTER="/root/autodl-tmp/srcv16c/src_v16c_paper_experiments"
DATA_ROOT="/root/autodl-tmp/planetoid/data"

JOB_ROOT="/root/autodl-tmp/srcv16c_jobs"
PACKET_ROOT="/root/autodl-tmp/srcv16c_finished_packets"
LOG_ROOT="/root/autodl-tmp/logs_src_v16c"

mkdir -p "$JOB_ROOT"
mkdir -p "$PACKET_ROOT"
mkdir -p "$LOG_ROOT"

export CUDA_VISIBLE_DEVICES=""

# 25 vCPU：6个数据集同时跑，每个进程给2线程，合计约12线程
# 这样不会把CPU和内存压爆
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

if [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate base || true
fi

run_one_dataset() {
    ds="$1"

    JOB_DIR="${JOB_ROOT}/${ds}"
    PACKET_DIR="${PACKET_ROOT}/${ds}"
    LOG_FILE="${LOG_ROOT}/E03_${ds}_10repeats.log"

    echo "========== PREPARE ${ds} =========="

    rm -rf "$JOB_DIR"
    mkdir -p "$JOB_DIR"
    mkdir -p "$PACKET_DIR"

    rsync -a \
        --exclude "__pycache__" \
        --exclude ".ipynb_checkpoints" \
        --exclude "results" \
        --exclude "test_runs_backup" \
        "$MASTER/" "$JOB_DIR/"

    cd "$JOB_DIR" || exit 1

    echo "========== START ${ds} $(date '+%F %T') ==========" | tee -a "$LOG_FILE"

    if [ -f fix_src_v16c_experiment_package_v4.py ]; then
        python fix_src_v16c_experiment_package_v4.py >> "$LOG_FILE" 2>&1 || true
    fi

    python scripts/run_E03_random_split_stability.py \
        --data-root "$DATA_ROOT" \
        --datasets "$ds" \
        --repeats 10 \
        --grid default \
        >> "$LOG_FILE" 2>&1

    status=$?

    echo "========== END ${ds} $(date '+%F %T') exit_code=${status} ==========" | tee -a "$LOG_FILE"

    mkdir -p "$PACKET_DIR"

    if [ -d "$JOB_DIR/results" ]; then
        rsync -a "$JOB_DIR/results/" "$PACKET_DIR/results/"
    fi

    cp "$LOG_FILE" "$PACKET_DIR/" 2>/dev/null || true
    echo "$status" > "$PACKET_DIR/EXIT_CODE.txt"
    date '+%F %T' > "$PACKET_DIR/FINISH_TIME.txt"

    cd "$PACKET_ROOT" || exit 1
    tar -czf "${ds}_E03_10repeats_packet.tar.gz" "$ds"

    if [ "$status" -eq 0 ]; then
        touch "$PACKET_DIR/DONE_SUCCESS"
        echo "SUCCESS ${ds}"
    else
        touch "$PACKET_DIR/DONE_FAILED"
        echo "FAILED ${ds}, check log: ${LOG_FILE}"
    fi
}

# 六个数据集一起启动，不串联
run_one_dataset amazon-computers &
run_one_dataset amazon-photo &
run_one_dataset cornell &
run_one_dataset squirrel &
run_one_dataset texas &
run_one_dataset wisconsin &

wait

echo "========== ALL 6 DATASETS FINISHED $(date '+%F %T') =========="

