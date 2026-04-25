#!/usr/bin/env bash
set -u

CODE_DIR="/root/autodl-tmp/srcv16c/src_v16c_paper_experiments"
DATA_ROOT="/root/autodl-tmp/planetoid/data"
OUT_ROOT="/root/autodl-tmp/whitebox_atlas_6datasets"
FINAL_ROOT="/root/autodl-tmp/whitebox_atlas_6datasets_final"
LOG_ROOT="/root/autodl-tmp/logs_src_v16c"

mkdir -p "$OUT_ROOT"
mkdir -p "$FINAL_ROOT"
mkdir -p "$LOG_ROOT"

export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="${CODE_DIR}:${PYTHONPATH:-}"

# 稳一点：6个数据集并行，每个任务最多2线程
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

cd "$CODE_DIR" || exit 1

run_one() {
    ds="$1"
    grid="$2"

    out_dir="${OUT_ROOT}/${ds}"
    log_file="${LOG_ROOT}/ATLAS_${ds}_${grid}_split1.log"

    rm -rf "$out_dir"
    mkdir -p "$out_dir"

    echo "========== START ${ds}, grid=${grid}, $(date '+%F %T') ==========" | tee "$log_file"

    python scripts/run_whitebox_graph_mechanism_atlas.py \
        --data-root "$DATA_ROOT" \
        --dataset "$ds" \
        --grid "$grid" \
        --splits 1 \
        --seed0 20260419 \
        --out-dir "$out_dir" \
        >> "$log_file" 2>&1

    status=$?

    echo "========== END ${ds}, grid=${grid}, $(date '+%F %T'), exit_code=${status} ==========" | tee -a "$log_file"

    echo "$status" > "${out_dir}/EXIT_CODE.txt"
    date '+%F %T' > "${out_dir}/FINISH_TIME.txt"

    if [ "$status" -eq 0 ]; then
        touch "${out_dir}/DONE_SUCCESS"
    else
        touch "${out_dir}/DONE_FAILED"
    fi
}

# 大图用 fast，防止再次炸时间；其他用 default
run_one amazon-computers fast &
run_one amazon-photo default &
run_one cornell default &
run_one squirrel fast &
run_one texas default &
run_one wisconsin default &

wait

echo "========== ALL DATASETS FINISHED $(date '+%F %T') =========="

python - <<'PY'
import pandas as pd
from pathlib import Path

root = Path("/root/autodl-tmp/whitebox_atlas_6datasets")
final = Path("/root/autodl-tmp/whitebox_atlas_6datasets_final")
final.mkdir(parents=True, exist_ok=True)

order = ["amazon-computers", "amazon-photo", "cornell", "squirrel", "texas", "wisconsin"]

summary_frames = []
stability_frames = []
block_frames = []
class_frames = []
pair_frames = []
shift_frames = []
type_frames = []

for ds in order:
    d = root / ds

    f = d / "whitebox_dataset_summary_by_split.csv"
    if f.exists():
        summary_frames.append(pd.read_csv(f))

    f = d / "F_split_mechanism_stability_strip.csv"
    if f.exists():
        stability_frames.append(pd.read_csv(f))

    f = d / "A_graph_signal_block_fingerprint.csv"
    if f.exists():
        block_frames.append(pd.read_csv(f))

    f = d / "B_class_signal_glyph.csv"
    if f.exists():
        class_frames.append(pd.read_csv(f))

    f = d / "D_subspace_overlap_constellation.csv"
    if f.exists():
        pair_frames.append(pd.read_csv(f))

    f = d / "E_error_signal_shift.csv"
    if f.exists():
        shift_frames.append(pd.read_csv(f))

    f = d / "node_mechanism_type_accuracy.csv"
    if f.exists():
        type_frames.append(pd.read_csv(f))

if summary_frames:
    summary = pd.concat(summary_frames, ignore_index=True)
    summary["dataset"] = pd.Categorical(summary["dataset"], categories=order, ordered=True)
    summary = summary.sort_values(["dataset", "split"])
    summary.to_csv(final / "whitebox_atlas_6datasets_summary.csv", index=False)

    readable = summary.copy()

    pct_cols = [
        "train_acc", "val_acc", "test_acc",
        "R_raw_dominance", "L_lowpass_dominance", "H_highpass_dominance",
        "B_boundary_dominance",
        "A_PR_pca_ridge_agreement",
        "D_geometry_discriminant_conflict",
        "Q_easy", "Q_geo_only", "Q_boundary_only", "Q_hard",
    ]

    for c in pct_cols:
        if c in readable.columns:
            readable[c + "_pct"] = readable[c] * 100

    show_cols = [
        "dataset", "grid", "test_acc_pct",
        "R_raw_dominance_pct",
        "L_lowpass_dominance_pct",
        "H_highpass_dominance_pct",
        "HHI_highpass_heterophily_index",
        "T_twohop_return_index",
        "O_deephop_utility",
        "S_hub_sensitivity",
        "C_mean_class_geometry_complexity",
        "C_std_class_geometry_imbalance",
        "A_PR_pca_ridge_agreement_pct",
        "D_geometry_discriminant_conflict_pct",
        "Q_easy_pct",
        "Q_geo_only_pct",
        "Q_boundary_only_pct",
        "Q_hard_pct",
        "B_boundary_dominance_pct",
        "graph_signal_identity",
        "decision_mechanism_identity",
    ]

    show_cols = [c for c in show_cols if c in readable.columns]
    readable[show_cols].to_csv(final / "whitebox_atlas_6datasets_summary_readable.csv", index=False)

    print("===== WHITEBOX ATLAS 6 DATASETS SUMMARY =====")
    print(readable[show_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
else:
    print("[ERROR] no summary files found")

if stability_frames:
    pd.concat(stability_frames, ignore_index=True).to_csv(final / "F_all_datasets_stability_strip_values.csv", index=False)
if block_frames:
    pd.concat(block_frames, ignore_index=True).to_csv(final / "A_all_datasets_graph_signal_fingerprint.csv", index=False)
if class_frames:
    pd.concat(class_frames, ignore_index=True).to_csv(final / "B_all_datasets_class_signal_glyph.csv", index=False)
if pair_frames:
    pd.concat(pair_frames, ignore_index=True).to_csv(final / "D_all_datasets_subspace_overlap.csv", index=False)
if shift_frames:
    pd.concat(shift_frames, ignore_index=True).to_csv(final / "E_all_datasets_error_signal_shift.csv", index=False)
if type_frames:
    pd.concat(type_frames, ignore_index=True).to_csv(final / "node_mechanism_type_accuracy_all_datasets.csv", index=False)

readme = """
White-box Graph Mechanism Atlas

This run does not depend on E08.
Each dataset is rerun once.
The model selects best config by validation only.
All mechanism statistics are side-channel statistics after best config selection.

A. Graph Signal Barcode
B. Class Signal Glyph
C. Geometry-Boundary Mechanism Plane
D. Subspace Constellation
E. Error Signal Shift
F. Split Mechanism Stability Strip

Note:
Since splits=1, F is a single-split mechanism strip.
True split stability requires splits > 1.
"""
(final / "README_WHITEBOX_ATLAS.txt").write_text(readme, encoding="utf-8")
PY

cd /root/autodl-tmp

tar -czf whitebox_atlas_6datasets_final_$(date +%Y%m%d_%H%M%S).tar.gz \
  whitebox_atlas_6datasets \
  whitebox_atlas_6datasets_final \
  logs_src_v16c/ATLAS_*.log \
  srcv16c/src_v16c_paper_experiments/scripts/run_whitebox_graph_mechanism_atlas.py \
  srcv16c/src_v16c_paper_experiments/run_whitebox_atlas_6datasets_parallel_final.sh 2>/dev/null || true

echo
echo "===== FINAL PACKAGE ====="
ls -lh /root/autodl-tmp/whitebox_atlas_6datasets_final_*.tar.gz

