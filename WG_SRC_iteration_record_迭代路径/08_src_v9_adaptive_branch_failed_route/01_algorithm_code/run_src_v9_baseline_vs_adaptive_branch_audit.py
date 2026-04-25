#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audit script: baseline single-subspace PCA residual vs adaptive-branch PCA residual
===================================================================================

Purpose
-------
Run both models on the exact same Chameleon splits, then write aligned node-level
comparison tables so we can inspect:

1. Which test nodes changed prediction.
2. Which nodes were fixed by adaptive branching.
3. Which previously-correct nodes were broken.
4. Whether the adaptive decision was made by a root branch or an extra branch.
5. Which class-pair transitions dominate the rewrite.

Directory assumptions
---------------------
Project root: .../experiments_g1/wgsrc_development_workspace/
Adaptive source: src_v9/algo1_adaptive_branch_pca_src_v9.py
Data root auto-discovered from drive root containing planetoid/data/chameleon

Outputs
-------
results_src_v9_baseline_vs_adaptive_branch_chameleon_audit/
  - node_prediction_comparison.csv
  - changed_prediction_nodes.csv
  - fixed_nodes.csv
  - broken_nodes.csv
  - changed_wrong_target_nodes.csv
  - decision_rewrite_summary_by_split.csv
  - decision_rewrite_summary_by_true_class.csv
  - decision_rewrite_summary_by_pair.csv
  - adaptive_branch_usage_on_changed_nodes.csv
  - run_summary.json
"""

from __future__ import annotations

import csv
import json
import sys
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# ============================================================
# Generic IO helpers
# ============================================================

def write_csv(path: Path, rows: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, 'w', newline='', encoding='utf-8-sig') as f:
            pass
        return
    keys = list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def discover_project_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / 'src_v9').exists() and (p / 'scripts').exists():
            return p
    raise FileNotFoundError('Cannot locate project root containing src_v9 and scripts.')


# ============================================================
# Import adaptive algorithm source dynamically
# ============================================================

def load_adaptive_module(project_root: Path):
    module_path = project_root / 'src_v9' / 'algo1_adaptive_branch_pca_src_v9.py'
    if not module_path.exists():
        raise FileNotFoundError(f'Cannot find adaptive source file: {module_path}')
    spec = importlib.util.spec_from_file_location('algo1_adaptive_branch_pca_src_v9', module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# ============================================================
# Baseline = one root PCA subspace per class, no branching
# ============================================================

def baseline_scores_from_root_subspaces(adapt, F, y, train_idx, classes, max_dim):
    root_sub = adapt.fit_root_subspaces(F, y, train_idx, classes, max_dim=max_dim)
    R = adapt.residuals_to_subspaces(F, root_sub, len(classes))
    return root_sub, R


# ============================================================
# Score utilities
# ============================================================

def top2_from_score_row(classes: np.ndarray, score_row: np.ndarray) -> Tuple[int, float, int, float, float]:
    order = np.argsort(score_row)
    best_pos = int(order[0])
    second_pos = int(order[1]) if score_row.shape[0] >= 2 else int(order[0])
    best_class = int(classes[best_pos])
    second_class = int(classes[second_pos])
    best_score = float(score_row[best_pos])
    second_score = float(score_row[second_pos])
    gap = float(second_score - best_score)
    return best_class, best_score, second_class, second_score, gap


# ============================================================
# Main audit routine
# ============================================================

def run_audit(dataset='chameleon', data_base=None, out_dir=None, dim_candidates=None, num_splits=10):
    if dim_candidates is None:
        dim_candidates = [16, 24, 32, 48, 64]

    this_file = Path(__file__).resolve()
    project_root = discover_project_root(this_file.parent)
    adapt = load_adaptive_module(project_root)

    # Auto-discover data/output using the same logic as the adaptive source.
    _, _, default_data_base, _ = adapt.resolve_default_paths(dataset)
    data_base = Path(data_base) if data_base is not None else default_data_base
    out_dir = Path(out_dir) if out_dir is not None else (project_root / 'scripts' / f'results_src_v9_baseline_vs_adaptive_branch_{dataset}_audit')
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_root = adapt.find_raw_root(data_base)
    X_raw, y, A_sym = adapt.load_chameleon_raw(raw_root)

    # Build the exact same multihop feature as both algorithms use.
    P = adapt.row_normalize(A_sym)
    PX = np.asarray(P @ X_raw)
    P2X = np.asarray(P @ PX)
    F = np.hstack([
        adapt.row_l2_normalize(X_raw),
        adapt.row_l2_normalize(PX),
        adapt.row_l2_normalize(P2X),
        adapt.row_l2_normalize(X_raw - PX),
        adapt.row_l2_normalize(PX - P2X),
    ])

    print(f'Project root: {project_root}')
    print(f'Data base:    {data_base}')
    print(f'Raw root:     {raw_root}')
    print(f'Output dir:   {out_dir}')
    print(f'Feature dim:  {F.shape[1]}')

    # Keep adaptive hyperparameters exactly aligned with the current src_v9 source.
    branch_kwargs = dict(
        internal_n_folds=4,
        internal_seed=0,
        hard_residual_quantile=0.75,
        hard_margin_quantile=0.25,
        min_branch_size=10,
        expansion_ratio=2.0,
        max_extra_branches=2,
        repel_strength=0.08,
        extra_branch_bias=0.05,
    )

    node_rows = []
    changed_rows = []
    fixed_rows = []
    broken_rows = []
    changed_wrong_rows = []
    split_rows = []

    for split in range(int(num_splits)):
        train_idx, val_idx, test_idx = adapt.load_split(raw_root, split)
        classes = np.unique(y[train_idx])
        class_to_pos = {int(c): pos for pos, c in enumerate(classes)}

        # ------------------------------
        # Baseline: select best dim by val
        # ------------------------------
        best_base_val, best_base_dim, best_base_test = -1.0, -1, -1.0
        best_base_R = None
        best_base_pred = None
        best_base_root = None
        for dim in dim_candidates:
            root_sub, R = baseline_scores_from_root_subspaces(adapt, F, y, train_idx, classes, max_dim=dim)
            pred = classes[np.argmin(R, axis=1)]
            v = float(np.mean(pred[val_idx] == y[val_idx]))
            t = float(np.mean(pred[test_idx] == y[test_idx]))
            if v > best_base_val:
                best_base_val = v
                best_base_dim = int(dim)
                best_base_test = t
                best_base_R = R.copy()
                best_base_pred = pred.copy()
                best_base_root = root_sub

        # ------------------------------
        # Adaptive: select best dim by val
        # ------------------------------
        best_ad_val, best_ad_dim, best_ad_test = -1.0, -1, -1.0
        best_ad_scores = None
        best_ad_pred = None
        best_ad_branch_kind = None
        best_ad_branch_models = None
        for dim in dim_candidates:
            branch_models, meta = adapt.fit_adaptive_branch_subspaces(
                F, y, train_idx, classes, max_dim=dim, **branch_kwargs
            )
            scores, best_branch_kind = adapt.class_scores_from_branch_models(F, branch_models, classes)
            pred = classes[np.argmin(scores, axis=1)]
            v = float(np.mean(pred[val_idx] == y[val_idx]))
            t = float(np.mean(pred[test_idx] == y[test_idx]))
            if v > best_ad_val:
                best_ad_val = v
                best_ad_dim = int(dim)
                best_ad_test = t
                best_ad_scores = scores.copy()
                best_ad_pred = pred.copy()
                best_ad_branch_kind = best_branch_kind.copy()
                best_ad_branch_models = branch_models

        # Build per-class mapping from best score back to exact branch local id / confuser / size.
        # This is needed because class_scores_from_branch_models returns only the best branch kind.
        n_all = F.shape[0]
        best_branch_local_id = np.full((n_all, len(classes)), -1, dtype=np.int64)
        best_branch_confuser = np.full((n_all, len(classes)), -1, dtype=np.int64)
        best_branch_member_size = np.full((n_all, len(classes)), -1, dtype=np.int64)
        best_branch_score = np.full((n_all, len(classes)), np.inf, dtype=np.float64)
        best_branch_kind_exact = np.empty((n_all, len(classes)), dtype=object)

        for pos, c in enumerate(classes):
            for local_id, branch in enumerate(best_ad_branch_models[int(c)]):
                _, _, residual = adapt.subspace_stats(F, branch['mu'], branch['basis'])
                score = residual / max(branch['score_scale'], 1e-8) + float(branch['score_bias'])
                better = score < best_branch_score[:, pos]
                best_branch_score[better, pos] = score[better]
                best_branch_local_id[better, pos] = int(local_id)
                best_branch_confuser[better, pos] = int(branch.get('confuser_class', -1)) if branch.get('confuser_class', None) is not None else -1
                best_branch_member_size[better, pos] = int(len(branch['member_idx']))
                best_branch_kind_exact[better, pos] = branch['kind']

        # Split summary.
        split_rows.append({
            'split': split,
            'baseline_best_dim': best_base_dim,
            'baseline_val_acc': best_base_val,
            'baseline_test_acc': best_base_test,
            'adaptive_best_dim': best_ad_dim,
            'adaptive_val_acc': best_ad_val,
            'adaptive_test_acc': best_ad_test,
            'adaptive_minus_baseline_test': float(best_ad_test - best_base_test),
            'num_test_nodes': int(len(test_idx)),
            'num_changed_predictions': int(np.sum(best_base_pred[test_idx] != best_ad_pred[test_idx])),
            'num_fixed': int(np.sum((best_base_pred[test_idx] != y[test_idx]) & (best_ad_pred[test_idx] == y[test_idx]))),
            'num_broken': int(np.sum((best_base_pred[test_idx] == y[test_idx]) & (best_ad_pred[test_idx] != y[test_idx]))),
            'num_changed_wrong_target': int(np.sum((best_base_pred[test_idx] != y[test_idx]) & (best_ad_pred[test_idx] != y[test_idx]) & (best_base_pred[test_idx] != best_ad_pred[test_idx]))),
        })

        # Node-level aligned comparison.
        for node_idx in test_idx:
            true_label = int(y[node_idx])
            true_pos = class_to_pos[true_label]

            base_pred = int(best_base_pred[node_idx])
            base_pred_pos = class_to_pos[base_pred]
            _, base_best_score, base_second_class, base_second_score, base_gap = top2_from_score_row(classes, best_base_R[node_idx])
            base_true_score = float(best_base_R[node_idx, true_pos])
            base_correct = int(base_pred == true_label)

            ad_pred = int(best_ad_pred[node_idx])
            ad_pred_pos = class_to_pos[ad_pred]
            _, ad_best_score, ad_second_class, ad_second_score, ad_gap = top2_from_score_row(classes, best_ad_scores[node_idx])
            ad_true_score = float(best_ad_scores[node_idx, true_pos])
            ad_correct = int(ad_pred == true_label)

            changed = int(base_pred != ad_pred)
            fixed = int((base_pred != true_label) and (ad_pred == true_label))
            broken = int((base_pred == true_label) and (ad_pred != true_label))
            changed_wrong_target = int((base_pred != true_label) and (ad_pred != true_label) and (base_pred != ad_pred))

            if base_correct and ad_correct:
                transition = 'same_correct'
            elif (not base_correct) and (not ad_correct) and base_pred == ad_pred:
                transition = 'same_wrong_same_target'
            elif fixed:
                transition = 'fixed'
            elif broken:
                transition = 'broken'
            elif changed_wrong_target:
                transition = 'changed_wrong_target'
            else:
                transition = 'other_changed'

            row = {
                'split': split,
                'node_idx_sorted': int(node_idx),
                'true_label': true_label,
                'baseline_best_dim': best_base_dim,
                'adaptive_best_dim': best_ad_dim,
                'baseline_pred_label': base_pred,
                'baseline_true_score': base_true_score,
                'baseline_pred_score': base_best_score,
                'baseline_pred_minus_true': float(base_best_score - base_true_score),
                'baseline_second_class': base_second_class,
                'baseline_second_score': base_second_score,
                'baseline_top2_gap': base_gap,
                'baseline_correct': base_correct,
                'adaptive_pred_label': ad_pred,
                'adaptive_true_score': ad_true_score,
                'adaptive_pred_score': ad_best_score,
                'adaptive_pred_minus_true': float(ad_best_score - ad_true_score),
                'adaptive_second_class': ad_second_class,
                'adaptive_second_score': ad_second_score,
                'adaptive_top2_gap': ad_gap,
                'adaptive_correct': ad_correct,
                'adaptive_pred_branch_local_id': int(best_branch_local_id[node_idx, ad_pred_pos]),
                'adaptive_pred_branch_kind': str(best_branch_kind_exact[node_idx, ad_pred_pos]),
                'adaptive_pred_branch_confuser_class': int(best_branch_confuser[node_idx, ad_pred_pos]),
                'adaptive_pred_branch_member_size': int(best_branch_member_size[node_idx, ad_pred_pos]),
                'adaptive_true_best_branch_local_id': int(best_branch_local_id[node_idx, true_pos]),
                'adaptive_true_best_branch_kind': str(best_branch_kind_exact[node_idx, true_pos]),
                'adaptive_true_best_branch_confuser_class': int(best_branch_confuser[node_idx, true_pos]),
                'adaptive_true_best_branch_member_size': int(best_branch_member_size[node_idx, true_pos]),
                'prediction_changed': changed,
                'fixed': fixed,
                'broken': broken,
                'changed_wrong_target': changed_wrong_target,
                'transition_type': transition,
            }
            node_rows.append(row)
            if changed:
                changed_rows.append(row)
            if fixed:
                fixed_rows.append(row)
            if broken:
                broken_rows.append(row)
            if changed_wrong_target:
                changed_wrong_rows.append(row)

        print(
            f'split={split:2d}  '
            f'baseline(dim={best_base_dim:2d}) val={best_base_val:.4f} test={best_base_test:.4f} | '
            f'adaptive(dim={best_ad_dim:2d}) val={best_ad_val:.4f} test={best_ad_test:.4f}'
        )

    # --------------------------------------------------------
    # Aggregated summaries from node_rows
    # --------------------------------------------------------
    by_true_class = {}
    by_pair = {}
    by_branch_usage = {}
    for r in node_rows:
        t = int(r['true_label'])
        by_true_class.setdefault(t, {
            'true_label': t,
            'num_test_nodes': 0,
            'baseline_correct': 0,
            'adaptive_correct': 0,
            'prediction_changed': 0,
            'fixed': 0,
            'broken': 0,
            'changed_wrong_target': 0,
        })
        agg = by_true_class[t]
        agg['num_test_nodes'] += 1
        agg['baseline_correct'] += int(r['baseline_correct'])
        agg['adaptive_correct'] += int(r['adaptive_correct'])
        agg['prediction_changed'] += int(r['prediction_changed'])
        agg['fixed'] += int(r['fixed'])
        agg['broken'] += int(r['broken'])
        agg['changed_wrong_target'] += int(r['changed_wrong_target'])

        pair_key = (int(r['true_label']), int(r['baseline_pred_label']), int(r['adaptive_pred_label']))
        by_pair.setdefault(pair_key, {
            'true_label': int(r['true_label']),
            'baseline_pred_label': int(r['baseline_pred_label']),
            'adaptive_pred_label': int(r['adaptive_pred_label']),
            'count': 0,
        })
        by_pair[pair_key]['count'] += 1

        if int(r['prediction_changed']) == 1:
            bu_key = (
                int(r['adaptive_pred_label']),
                str(r['adaptive_pred_branch_kind']),
                int(r['adaptive_pred_branch_confuser_class']),
            )
            by_branch_usage.setdefault(bu_key, {
                'adaptive_pred_label': int(r['adaptive_pred_label']),
                'adaptive_pred_branch_kind': str(r['adaptive_pred_branch_kind']),
                'adaptive_pred_branch_confuser_class': int(r['adaptive_pred_branch_confuser_class']),
                'count': 0,
            })
            by_branch_usage[bu_key]['count'] += 1

    true_class_rows = []
    for _, agg in sorted(by_true_class.items()):
        n = max(1, agg['num_test_nodes'])
        true_class_rows.append({
            **agg,
            'baseline_acc_within_true_class': agg['baseline_correct'] / n,
            'adaptive_acc_within_true_class': agg['adaptive_correct'] / n,
            'prediction_changed_rate': agg['prediction_changed'] / n,
            'fixed_rate': agg['fixed'] / n,
            'broken_rate': agg['broken'] / n,
        })

    pair_rows = sorted(by_pair.values(), key=lambda z: z['count'], reverse=True)
    branch_usage_rows = sorted(by_branch_usage.values(), key=lambda z: z['count'], reverse=True)

    # --------------------------------------------------------
    # Write outputs
    # --------------------------------------------------------
    write_csv(out_dir / 'node_prediction_comparison.csv', node_rows)
    write_csv(out_dir / 'changed_prediction_nodes.csv', changed_rows)
    write_csv(out_dir / 'fixed_nodes.csv', fixed_rows)
    write_csv(out_dir / 'broken_nodes.csv', broken_rows)
    write_csv(out_dir / 'changed_wrong_target_nodes.csv', changed_wrong_rows)
    write_csv(out_dir / 'decision_rewrite_summary_by_split.csv', split_rows)
    write_csv(out_dir / 'decision_rewrite_summary_by_true_class.csv', true_class_rows)
    write_csv(out_dir / 'decision_rewrite_summary_by_pair.csv', pair_rows)
    write_csv(out_dir / 'adaptive_branch_usage_on_changed_nodes.csv', branch_usage_rows)

    baseline_test_mean = float(np.mean([r['baseline_test_acc'] for r in split_rows])) if split_rows else float('nan')
    adaptive_test_mean = float(np.mean([r['adaptive_test_acc'] for r in split_rows])) if split_rows else float('nan')
    summary = {
        'dataset': dataset,
        'data_base': str(data_base),
        'raw_root': str(raw_root),
        'out_dir': str(out_dir),
        'dim_candidates': list(dim_candidates),
        'num_splits': int(num_splits),
        'baseline_test_mean': baseline_test_mean,
        'adaptive_test_mean': adaptive_test_mean,
        'adaptive_minus_baseline_test_mean': float(adaptive_test_mean - baseline_test_mean),
        'num_node_rows': int(len(node_rows)),
        'num_changed_predictions': int(len(changed_rows)),
        'num_fixed': int(len(fixed_rows)),
        'num_broken': int(len(broken_rows)),
        'num_changed_wrong_target': int(len(changed_wrong_rows)),
    }
    (out_dir / 'run_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    print('\nDone.')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Audit baseline vs adaptive-branch decisions on Chameleon.')
    parser.add_argument('--dataset', type=str, default='chameleon')
    parser.add_argument('--data-base', type=str, default=None)
    parser.add_argument('--out-dir', type=str, default=None)
    parser.add_argument('--num-splits', type=int, default=10)
    parser.add_argument('--dims', type=int, nargs='*', default=[16, 24, 32, 48, 64])
    args = parser.parse_args()

    run_audit(
        dataset=args.dataset,
        data_base=args.data_base,
        out_dir=args.out_dir,
        dim_candidates=args.dims,
        num_splits=args.num_splits,
    )


