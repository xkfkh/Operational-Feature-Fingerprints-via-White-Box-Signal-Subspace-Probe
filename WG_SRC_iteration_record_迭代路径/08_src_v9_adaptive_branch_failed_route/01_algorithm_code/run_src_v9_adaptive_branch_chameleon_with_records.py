#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adaptive-Branch PCA runner with explicit diagnostics/records.

What this script records
------------------------
1) split_summary_adaptive_branch_records.csv
   Split-level accuracy and branch counts.

2) misclassified_test_nodes.csv
   Every misclassified test node for every split, including:
   - true/pred labels
   - true/pred scores
   - true/pred winning branch ids and branch kinds
   - top-2 class gap

3) branch_subspace_summary.csv
   Every branch actually used in the selected best model for each split, including:
   - class label
   - branch id
   - branch kind (root/extra)
   - basis dimension
   - member size
   - score scale / bias
   - confuser class
   - seed size (for extra branches)
   - seed / member node lists (sorted-index and original-id JSON)

4) train_internal_misclassified_nodes.csv
   Out-of-fold train-time internal errors used to create branches.

5) branch_member_rows.csv
   One row per branch-member node, so the exact split of each branch is inspectable.

This script is intentionally record-oriented: it does not interpret the results.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import sparse


# ============================================================
# Utilities
# ============================================================

def row_l2_normalize(X, eps=1e-12):
    X = np.asarray(X, dtype=np.float64)
    norm = np.sqrt(np.sum(X * X, axis=1, keepdims=True))
    return X / np.maximum(norm, eps)


def row_normalize(A):
    A = A.tocsr().astype(np.float64)
    deg = np.asarray(A.sum(axis=1)).ravel()
    inv = np.zeros_like(deg)
    inv[deg > 0] = 1.0 / deg[deg > 0]
    return sparse.diags(inv) @ A


def find_raw_root(base):
    base = Path(base)
    hits = list(base.rglob('out1_node_feature_label.txt'))
    if not hits:
        raise FileNotFoundError('Cannot find out1_node_feature_label.txt')
    return hits[0].parent


def load_chameleon_raw(raw_root):
    raw_root = Path(raw_root)
    node_file = raw_root / 'out1_node_feature_label.txt'
    edge_file = raw_root / 'out1_graph_edges.txt'
    ids, feats, labels = [], [], []
    with open(node_file, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            node_id, feat, label = line.rstrip('\n').split('\t')
            ids.append(int(node_id))
            feats.append(np.array(feat.split(','), dtype=np.float64))
            labels.append(int(label))
    order = np.argsort(ids)
    X = np.vstack(feats)[order]
    y = np.asarray(labels, dtype=np.int64)[order]
    original_node_ids = np.asarray(ids, dtype=np.int64)[order]

    src, dst = [], []
    with open(edge_file, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            a, b = line.strip().split('\t')
            src.append(int(a))
            dst.append(int(b))
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    n = X.shape[0]
    A_out = sparse.csr_matrix((np.ones(len(src)), (src, dst)), shape=(n, n), dtype=np.float64)
    A_out.setdiag(0)
    A_out.eliminate_zeros()
    A_sym = ((A_out + A_out.T) > 0).astype(np.float64).tocsr()
    A_sym.setdiag(0)
    A_sym.eliminate_zeros()
    return X, y, A_sym, original_node_ids


def load_split(raw_root, repeat):
    path = Path(raw_root) / f'chameleon_split_0.6_0.2_{repeat}.npz'
    z = np.load(path)
    return (
        np.where(z['train_mask'].astype(bool))[0],
        np.where(z['val_mask'].astype(bool))[0],
        np.where(z['test_mask'].astype(bool))[0],
    )


def robust_median(arr, default=1.0):
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(default)
    return float(np.median(arr))


def write_csv(path, rows):
    path = Path(path)
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


def discover_drive_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / 'planetoid' / 'data').exists():
            return p
    raise FileNotFoundError(
        'Cannot locate drive/project root containing planetoid/data. Please pass --data-base explicitly.'
    )


def discover_project_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / 'src_v9').exists() and (p / 'scripts').exists():
            return p
    raise FileNotFoundError(
        'Cannot locate project root containing src_v9 and scripts. '
        'Please place this file inside scripts or edit the path logic.'
    )


def resolve_default_paths(dataset: str):
    this_file = Path(__file__).resolve()
    project_root = discover_project_root(this_file.parent)
    drive_root = discover_drive_root(this_file.parent)
    data_base = drive_root / 'planetoid' / 'data' / dataset
    out_dir = project_root / 'scripts' / f'results_src_v9_adaptive_branch_{dataset}_records'
    return project_root, drive_root, data_base, out_dir


# ============================================================
# PCA / subspace ops
# ============================================================

def orthonormalize_columns(B):
    B = np.asarray(B, dtype=np.float64)
    if B.size == 0:
        return B.reshape(B.shape[0], 0)
    Q, _ = np.linalg.qr(B)
    keep = min(B.shape[1], Q.shape[1])
    return np.asarray(Q[:, :keep], dtype=np.float64)


def fit_pca_subspace(X, max_dim):
    X = np.asarray(X, dtype=np.float64)
    mu = np.mean(X, axis=0)
    C = X - mu
    d_eff = min(int(max_dim), max(0, X.shape[0] - 1), X.shape[1])
    if d_eff == 0:
        B = np.empty((X.shape[1], 0), dtype=np.float64)
        kept_evals = np.empty((0,), dtype=np.float64)
    else:
        Gram = C @ C.T
        evals, evecs = np.linalg.eigh(Gram)
        order = np.argsort(evals)[::-1]
        keep = order[evals[order] > 1e-12][:d_eff]
        if keep.size == 0:
            B = np.empty((X.shape[1], 0), dtype=np.float64)
            kept_evals = np.empty((0,), dtype=np.float64)
        else:
            kept_evals = evals[keep].copy()
            B = C.T @ (evecs[:, keep] / np.sqrt(evals[keep])[None, :])
            B = np.asarray(B, dtype=np.float64)
            B = orthonormalize_columns(B)
    return mu, B, kept_evals


def purify_basis_against_confusers(B, confuser_bases, repel_strength=0.08):
    B = np.asarray(B, dtype=np.float64)
    if B.size == 0 or repel_strength <= 0:
        return B
    B_new = B.copy()
    for weight, Bref in confuser_bases:
        if Bref is None or Bref.size == 0 or weight <= 0:
            continue
        proj = Bref @ (Bref.T @ B_new)
        B_new = B_new - (repel_strength * float(weight)) * proj
    return orthonormalize_columns(B_new)


def subspace_stats(F, mu, B):
    C = F - mu
    total = np.sum(C * C, axis=1)
    if B.shape[1] == 0:
        evidence = np.zeros(F.shape[0], dtype=np.float64)
    else:
        coord = C @ B
        evidence = np.sum(coord * coord, axis=1)
    residual = np.maximum(total - evidence, 0.0)
    return total, evidence, residual


# ============================================================
# Root model
# ============================================================

def fit_root_subspaces(F, y, train_idx, classes, max_dim):
    subspaces = {}
    for pos, c in enumerate(classes):
        idx = train_idx[y[train_idx] == c]
        mu, B, kept_evals = fit_pca_subspace(F[idx], max_dim=max_dim)
        subspaces[pos] = {
            'class_label': int(c),
            'mu': mu,
            'basis': B,
            'member_idx': np.asarray(idx, dtype=np.int64),
            'kept_evals': kept_evals,
        }
    return subspaces


def residuals_to_subspaces(F, subspaces, n_classes):
    R = np.zeros((F.shape[0], n_classes), dtype=np.float64)
    for pos in range(n_classes):
        mu = subspaces[pos]['mu']
        B = subspaces[pos]['basis']
        _, _, residual = subspace_stats(F, mu, B)
        R[:, pos] = residual
    return R


# ============================================================
# Training-time internal diagnostics
# ============================================================

def make_stratified_folds(train_idx, y, n_splits=4, seed=0):
    rng = np.random.RandomState(seed)
    train_idx = np.asarray(train_idx, dtype=np.int64)
    fold_id = np.zeros(train_idx.shape[0], dtype=np.int64)
    unique_classes = np.unique(y[train_idx])
    for c in unique_classes:
        cls_local = np.where(y[train_idx] == c)[0].copy()
        rng.shuffle(cls_local)
        for k, local_pos in enumerate(cls_local):
            fold_id[local_pos] = k % n_splits
    return fold_id


def collect_internal_diagnostics(F, y, train_idx, classes, max_dim, n_folds=4, seed=0):
    train_idx = np.asarray(train_idx, dtype=np.int64)
    n_all = F.shape[0]
    cnum = len(classes)
    class_to_pos = {int(c): pos for pos, c in enumerate(classes)}
    folds = make_stratified_folds(train_idx, y, n_splits=n_folds, seed=seed)

    diag = {
        'pred_label': np.full(n_all, -1, dtype=np.int64),
        'true_residual': np.full(n_all, np.nan, dtype=np.float64),
        'best_other_label': np.full(n_all, -1, dtype=np.int64),
        'best_other_residual': np.full(n_all, np.nan, dtype=np.float64),
        'margin_to_other': np.full(n_all, np.nan, dtype=np.float64),
        'true_total': np.full(n_all, np.nan, dtype=np.float64),
        'true_evidence': np.full(n_all, np.nan, dtype=np.float64),
        'fold_id': np.full(n_all, -1, dtype=np.int64),
    }
    confusion_counts = {int(c): {} for c in classes}
    incoming_counts = {int(c): {} for c in classes}

    for fold in range(n_folds):
        inner_train = train_idx[folds != fold]
        inner_eval = train_idx[folds == fold]
        diag['fold_id'][inner_eval] = fold
        sub = fit_root_subspaces(F, y, inner_train, classes, max_dim=max_dim)
        R = residuals_to_subspaces(F[inner_eval], sub, cnum)
        pred = classes[np.argmin(R, axis=1)]

        for row_pos, node_idx in enumerate(inner_eval):
            true_label = int(y[node_idx])
            true_pos = class_to_pos[true_label]
            pred_label = int(pred[row_pos])
            diag['pred_label'][node_idx] = pred_label
            diag['true_residual'][node_idx] = float(R[row_pos, true_pos])

            other_res = R[row_pos].copy()
            other_res[true_pos] = np.inf
            best_other_pos = int(np.argmin(other_res))
            best_other_label = int(classes[best_other_pos])
            diag['best_other_label'][node_idx] = best_other_label
            diag['best_other_residual'][node_idx] = float(other_res[best_other_pos])
            diag['margin_to_other'][node_idx] = float(other_res[best_other_pos] - R[row_pos, true_pos])

            mu_t = sub[true_pos]['mu']
            B_t = sub[true_pos]['basis']
            total_t, evidence_t, _ = subspace_stats(F[[node_idx]], mu_t, B_t)
            diag['true_total'][node_idx] = float(total_t[0])
            diag['true_evidence'][node_idx] = float(evidence_t[0])

            if pred_label != true_label:
                confusion_counts[true_label][pred_label] = confusion_counts[true_label].get(pred_label, 0) + 1
                incoming_counts[pred_label][true_label] = incoming_counts[pred_label].get(true_label, 0) + 1

    return diag, confusion_counts, incoming_counts


# ============================================================
# Adaptive branching
# ============================================================

def select_extra_branch_budget(hard_rate, class_size, max_extra_branches=2):
    if class_size < 24:
        return 0
    if hard_rate < 0.14:
        return 0
    if hard_rate < 0.28:
        return min(1, max_extra_branches)
    if hard_rate < 0.40:
        return min(2, max_extra_branches)
    return min(3, max_extra_branches)


def build_candidate_group(F, pool_idx, seed_idx, min_branch_size=10, expansion_ratio=2.0):
    pool_idx = np.asarray(pool_idx, dtype=np.int64)
    seed_idx = np.asarray(seed_idx, dtype=np.int64)
    if seed_idx.size == 0:
        return np.empty((0,), dtype=np.int64)
    centroid = np.mean(F[seed_idx], axis=0, keepdims=True)
    sqdist = np.sum((F[pool_idx] - centroid) ** 2, axis=1)
    target_size = max(int(min_branch_size), int(math.ceil(seed_idx.size * expansion_ratio)))
    target_size = min(target_size, pool_idx.size)
    keep_local = np.argsort(sqdist)[:target_size]
    chosen = np.unique(np.concatenate([seed_idx, pool_idx[keep_local]])).astype(np.int64)
    return chosen


def fit_branch_model(F, member_idx, max_dim, class_scale, confuser_bases=None,
                     repel_strength=0.0, branch_bias=0.0, branch_kind='root',
                     seed_idx=None, confuser_class=None):
    X = F[member_idx]
    mu, B, kept_evals = fit_pca_subspace(X, max_dim=max_dim)
    if branch_kind != 'root' and confuser_bases:
        B = purify_basis_against_confusers(B, confuser_bases, repel_strength=repel_strength)

    _, _, train_residual = subspace_stats(X, mu, B)
    local_scale = robust_median(train_residual, default=class_scale)
    score_scale = max(1e-8, 0.7 * float(class_scale) + 0.3 * local_scale)
    return {
        'mu': mu,
        'basis': B,
        'member_idx': np.asarray(member_idx, dtype=np.int64),
        'seed_idx': np.asarray(seed_idx if seed_idx is not None else [], dtype=np.int64),
        'kind': branch_kind,
        'score_scale': float(score_scale),
        'score_bias': float(branch_bias),
        'confuser_class': None if confuser_class is None else int(confuser_class),
        'basis_dim': int(B.shape[1]),
        'kept_evals': kept_evals,
    }


def fit_adaptive_branch_subspaces(
    F,
    y,
    train_idx,
    classes,
    max_dim,
    internal_n_folds=4,
    internal_seed=0,
    hard_residual_quantile=0.75,
    hard_margin_quantile=0.25,
    min_branch_size=10,
    expansion_ratio=2.0,
    max_extra_branches=2,
    repel_strength=0.08,
    extra_branch_bias=0.05,
):
    classes = np.asarray(classes, dtype=np.int64)
    class_to_pos = {int(c): pos for pos, c in enumerate(classes)}

    root_sub = fit_root_subspaces(F, y, train_idx, classes, max_dim=max_dim)
    diag, confusion_counts, incoming_counts = collect_internal_diagnostics(
        F, y, train_idx, classes, max_dim=max_dim, n_folds=internal_n_folds, seed=internal_seed
    )

    class_scales = {}
    for c in classes:
        idx_c = train_idx[y[train_idx] == c]
        class_scales[int(c)] = max(1e-8, robust_median(diag['true_residual'][idx_c], default=1.0))

    branch_models = {int(c): [] for c in classes}
    branch_build_log = []

    for pos, c in enumerate(classes):
        idx_c = root_sub[pos]['member_idx']
        root_model = fit_branch_model(
            F,
            idx_c,
            max_dim=max_dim,
            class_scale=class_scales[int(c)],
            confuser_bases=None,
            repel_strength=0.0,
            branch_bias=0.0,
            branch_kind='root',
            seed_idx=idx_c,
            confuser_class=None,
        )
        root_model['branch_local_id'] = 0
        branch_models[int(c)].append(root_model)
        branch_build_log.append({
            'class_label': int(c),
            'branch_local_id': 0,
            'kind': 'root',
            'confuser_class': '',
            'seed_size': int(idx_c.size),
            'member_size': int(idx_c.size),
            'basis_dim': int(root_model['basis_dim']),
            'score_scale': float(root_model['score_scale']),
            'score_bias': float(root_model['score_bias']),
        })

    for c in classes:
        c = int(c)
        idx_c = np.asarray(train_idx[y[train_idx] == c], dtype=np.int64)
        n_c = idx_c.size
        if n_c == 0:
            continue

        true_res_c = diag['true_residual'][idx_c]
        margin_c = diag['margin_to_other'][idx_c]
        pred_c = diag['pred_label'][idx_c]
        confuser_c = diag['best_other_label'][idx_c]

        res_thr = np.quantile(true_res_c[np.isfinite(true_res_c)], hard_residual_quantile) if np.any(np.isfinite(true_res_c)) else np.inf
        mar_thr = np.quantile(margin_c[np.isfinite(margin_c)], hard_margin_quantile) if np.any(np.isfinite(margin_c)) else -np.inf
        misclf_mask = (pred_c != c)
        high_res_mask = np.isfinite(true_res_c) & (true_res_c >= res_thr)
        low_margin_mask = np.isfinite(margin_c) & (margin_c <= mar_thr)
        hard_mask = misclf_mask | high_res_mask | low_margin_mask
        hard_idx = idx_c[hard_mask]

        hard_rate = float(hard_idx.size / max(1, n_c))
        budget = select_extra_branch_budget(hard_rate, n_c, max_extra_branches=max_extra_branches)
        if budget == 0:
            continue

        min_seed = max(6, min_branch_size // 2)
        class_res_med = robust_median(true_res_c, default=class_scales[c])

        candidate_groups = []
        unique_confusers = sorted(set(int(z) for z in confuser_c[np.isfinite(confuser_c)] if int(z) != c and int(z) >= 0))
        for conf in unique_confusers:
            seed = idx_c[hard_mask & (confuser_c == conf)]
            if seed.size < min_seed:
                continue
            group = build_candidate_group(
                F,
                pool_idx=idx_c,
                seed_idx=seed,
                min_branch_size=min_branch_size,
                expansion_ratio=expansion_ratio,
            )
            if group.size < min_branch_size:
                continue
            score = float(seed.size) * max(0.0, robust_median(diag['true_residual'][seed], default=class_res_med) - class_res_med)
            score += 0.25 * float(seed.size)
            candidate_groups.append((score, conf, seed, group))

        if not candidate_groups:
            continue

        candidate_groups.sort(key=lambda z: (z[0], z[2].size, z[3].size), reverse=True)
        selected_groups = []
        used_members = []
        for _, conf, seed, group in candidate_groups:
            if len(selected_groups) >= budget:
                break
            duplicate = False
            group_set = set(int(x) for x in group)
            for prev in used_members:
                inter = len(group_set & prev)
                union = max(1, len(group_set | prev))
                if inter / union >= 0.65:
                    duplicate = True
                    break
            if duplicate:
                continue
            selected_groups.append((conf, seed, group))
            used_members.append(group_set)

        for conf, seed, group in selected_groups:
            conf_pos = class_to_pos.get(int(conf), None)
            conf_basis = None
            if conf_pos is not None:
                conf_basis = root_sub[conf_pos]['basis']
            confuser_bases = []
            if conf_basis is not None and conf_basis.size > 0:
                confuser_bases.append((1.0, conf_basis))

            extra_incoming = incoming_counts.get(c, {})
            if extra_incoming:
                ranked_extra = sorted(extra_incoming.items(), key=lambda kv: kv[1], reverse=True)
                for conf2, _ in ranked_extra[:2]:
                    if int(conf2) == int(conf):
                        continue
                    conf2_pos = class_to_pos.get(int(conf2), None)
                    if conf2_pos is None:
                        continue
                    basis2 = root_sub[conf2_pos]['basis']
                    if basis2.size > 0:
                        confuser_bases.append((0.5, basis2))
                        break

            extra_model = fit_branch_model(
                F,
                member_idx=group,
                max_dim=max(4, min(max_dim, group.size - 1)),
                class_scale=class_scales[c],
                confuser_bases=confuser_bases,
                repel_strength=repel_strength,
                branch_bias=extra_branch_bias,
                branch_kind='extra',
                seed_idx=seed,
                confuser_class=int(conf),
            )
            extra_model['branch_local_id'] = len(branch_models[c])
            branch_models[c].append(extra_model)
            branch_build_log.append({
                'class_label': int(c),
                'branch_local_id': int(extra_model['branch_local_id']),
                'kind': 'extra',
                'confuser_class': int(conf),
                'seed_size': int(seed.size),
                'member_size': int(group.size),
                'basis_dim': int(extra_model['basis_dim']),
                'score_scale': float(extra_model['score_scale']),
                'score_bias': float(extra_model['score_bias']),
            })

    meta = {
        'class_scales': class_scales,
        'root_confusion_counts': confusion_counts,
        'incoming_counts': incoming_counts,
        'internal_diag': diag,
        'branch_build_log': branch_build_log,
    }
    return branch_models, meta


# ============================================================
# Inference with explicit branch tracing
# ============================================================

def class_scores_from_branch_models(F, branch_models, classes):
    classes = np.asarray(classes, dtype=np.int64)
    n = F.shape[0]
    cnum = len(classes)
    class_scores = np.full((n, cnum), np.inf, dtype=np.float64)
    best_branch_idx = np.full((n, cnum), -1, dtype=np.int64)
    best_branch_kind = np.empty((n, cnum), dtype=object)

    for pos, c in enumerate(classes):
        for local_id, branch in enumerate(branch_models[int(c)]):
            _, _, residual = subspace_stats(F, branch['mu'], branch['basis'])
            score = residual / max(branch['score_scale'], 1e-8) + float(branch['score_bias'])
            better = score < class_scores[:, pos]
            class_scores[better, pos] = score[better]
            best_branch_idx[better, pos] = local_id
            best_branch_kind[better, pos] = branch['kind']
    return class_scores, best_branch_idx, best_branch_kind


# ============================================================
# Main runner
# ============================================================

def run_experiment(dataset='chameleon', data_base=None, out_dir=None, dim_candidates=None, num_splits=10):
    if dim_candidates is None:
        dim_candidates = [16, 24, 32, 48, 64]

    project_root, drive_root, default_data_base, default_out_dir = resolve_default_paths(dataset)
    data_base = Path(data_base) if data_base is not None else default_data_base
    out_dir = Path(out_dir) if out_dir is not None else default_out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_root = find_raw_root(data_base)
    X_raw, y, A_sym, original_node_ids = load_chameleon_raw(raw_root)

    P = row_normalize(A_sym)
    PX = np.asarray(P @ X_raw)
    P2X = np.asarray(P @ PX)
    F = np.hstack([
        row_l2_normalize(X_raw),
        row_l2_normalize(PX),
        row_l2_normalize(P2X),
        row_l2_normalize(X_raw - PX),
        row_l2_normalize(PX - P2X),
    ])

    print(f'Project root: {project_root}')
    print(f'Drive root:   {drive_root}')
    print(f'Data base:    {data_base}')
    print(f'Raw root:     {raw_root}')
    print(f'Output dir:   {out_dir}')
    print(f'Feature dim:  {F.shape[1]}')

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

    split_rows = []
    misclf_rows = []
    branch_summary_rows = []
    branch_member_rows = []
    train_internal_misclf_rows = []

    for split in range(int(num_splits)):
        train_idx, val_idx, test_idx = load_split(raw_root, split)
        classes = np.unique(y[train_idx])
        class_to_pos = {int(c): pos for pos, c in enumerate(classes)}

        best_val, best_dim, best_test = -1.0, -1, -1.0
        best_meta = None

        for dim in dim_candidates:
            branch_models, meta = fit_adaptive_branch_subspaces(
                F, y, train_idx, classes, max_dim=dim, **branch_kwargs
            )
            scores, best_branch_idx, best_branch_kind = class_scores_from_branch_models(F, branch_models, classes)
            pred = classes[np.argmin(scores, axis=1)]
            v = float(np.mean(pred[val_idx] == y[val_idx]))
            t = float(np.mean(pred[test_idx] == y[test_idx]))
            if v > best_val:
                best_val, best_dim, best_test = v, dim, t
                best_meta = {
                    'branch_models': branch_models,
                    'meta': meta,
                    'pred': pred,
                    'scores': scores,
                    'best_branch_idx': best_branch_idx,
                    'best_branch_kind': best_branch_kind,
                }

        total_branch_count = 0
        extra_branch_count = 0
        for c in classes:
            total_branch_count += len(best_meta['branch_models'][int(c)])
            extra_branch_count += sum(1 for b in best_meta['branch_models'][int(c)] if b['kind'] == 'extra')

        split_row = {
            'split': split,
            'best_dim': int(best_dim),
            'val_acc': float(best_val),
            'test_acc': float(best_test),
            'total_branches': int(total_branch_count),
            'extra_branches': int(extra_branch_count),
        }
        for c in classes:
            split_row[f'class_{int(c)}_branch_count'] = int(len(best_meta['branch_models'][int(c)]))
        split_rows.append(split_row)
        print(
            f'  split={split:2d}  best_dim={best_dim:3d}  '
            f'val={best_val:.4f}  test={best_test:.4f}  '
            f'branches={total_branch_count} (extra={extra_branch_count})'
        )

        pred = best_meta['pred']
        scores = best_meta['scores']
        best_branch_idx = best_meta['best_branch_idx']
        diag = best_meta['meta']['internal_diag']

        # Record train internal OOF misclassified nodes
        for node_idx in train_idx:
            true_label = int(y[node_idx])
            pred_label = int(diag['pred_label'][node_idx])
            if pred_label < 0 or pred_label == true_label:
                continue
            train_internal_misclf_rows.append({
                'split': split,
                'best_dim': int(best_dim),
                'node_idx_sorted': int(node_idx),
                'original_node_id': int(original_node_ids[node_idx]),
                'fold_id': int(diag['fold_id'][node_idx]),
                'true_label': true_label,
                'pred_label': pred_label,
                'true_residual': float(diag['true_residual'][node_idx]),
                'best_other_label': int(diag['best_other_label'][node_idx]),
                'best_other_residual': float(diag['best_other_residual'][node_idx]),
                'margin_to_other': float(diag['margin_to_other'][node_idx]),
                'true_total': float(diag['true_total'][node_idx]),
                'true_evidence': float(diag['true_evidence'][node_idx]),
            })

        # Record branch structure and exact members
        for c in classes:
            c = int(c)
            for local_id, branch in enumerate(best_meta['branch_models'][c]):
                member_idx = np.asarray(branch['member_idx'], dtype=np.int64)
                seed_idx = np.asarray(branch.get('seed_idx', []), dtype=np.int64)
                branch_id = f'split{split}_class{c}_branch{local_id}'
                branch_summary_rows.append({
                    'split': split,
                    'best_dim': int(best_dim),
                    'class_label': c,
                    'branch_local_id': int(local_id),
                    'branch_id': branch_id,
                    'kind': str(branch['kind']),
                    'confuser_class': '' if branch.get('confuser_class', None) is None else int(branch['confuser_class']),
                    'basis_dim': int(branch['basis'].shape[1]),
                    'member_size': int(member_idx.size),
                    'seed_size': int(seed_idx.size),
                    'score_scale': float(branch['score_scale']),
                    'score_bias': float(branch['score_bias']),
                    'member_idx_json': json.dumps(member_idx.tolist(), ensure_ascii=False),
                    'member_original_id_json': json.dumps(original_node_ids[member_idx].astype(int).tolist(), ensure_ascii=False),
                    'seed_idx_json': json.dumps(seed_idx.tolist(), ensure_ascii=False),
                    'seed_original_id_json': json.dumps(original_node_ids[seed_idx].astype(int).tolist(), ensure_ascii=False),
                })
                seed_set = set(int(x) for x in seed_idx.tolist())
                for node_idx in member_idx.tolist():
                    branch_member_rows.append({
                        'split': split,
                        'best_dim': int(best_dim),
                        'class_label': c,
                        'branch_local_id': int(local_id),
                        'branch_id': branch_id,
                        'kind': str(branch['kind']),
                        'node_idx_sorted': int(node_idx),
                        'original_node_id': int(original_node_ids[node_idx]),
                        'is_seed': int(int(node_idx) in seed_set),
                    })

        # Record misclassified test nodes
        for node_idx in test_idx:
            true_label = int(y[node_idx])
            pred_label = int(pred[node_idx])
            if pred_label == true_label:
                continue
            pred_pos = class_to_pos[pred_label]
            true_pos = class_to_pos[true_label]

            class_scores_row = scores[node_idx]
            sorted_pos = np.argsort(class_scores_row)
            best_pos = int(sorted_pos[0])
            second_pos = int(sorted_pos[1]) if sorted_pos.size > 1 else int(sorted_pos[0])
            best_branch_local = int(best_branch_idx[node_idx, pred_pos])
            true_branch_local = int(best_branch_idx[node_idx, true_pos])
            pred_branch = best_meta['branch_models'][pred_label][best_branch_local]
            true_branch = best_meta['branch_models'][true_label][true_branch_local]

            misclf_rows.append({
                'split': split,
                'best_dim': int(best_dim),
                'node_idx_sorted': int(node_idx),
                'original_node_id': int(original_node_ids[node_idx]),
                'true_label': true_label,
                'pred_label': pred_label,
                'pred_score': float(scores[node_idx, pred_pos]),
                'true_score': float(scores[node_idx, true_pos]),
                'pred_minus_true_score': float(scores[node_idx, pred_pos] - scores[node_idx, true_pos]),
                'top2_second_class': int(classes[second_pos]),
                'top2_gap': float(class_scores_row[second_pos] - class_scores_row[best_pos]),
                'pred_branch_local_id': best_branch_local,
                'pred_branch_kind': str(pred_branch['kind']),
                'pred_branch_confuser_class': '' if pred_branch.get('confuser_class', None) is None else int(pred_branch['confuser_class']),
                'pred_branch_member_size': int(np.asarray(pred_branch['member_idx']).size),
                'true_branch_local_id': true_branch_local,
                'true_branch_kind': str(true_branch['kind']),
                'true_branch_confuser_class': '' if true_branch.get('confuser_class', None) is None else int(true_branch['confuser_class']),
                'true_branch_member_size': int(np.asarray(true_branch['member_idx']).size),
            })

    vals = [r['val_acc'] for r in split_rows]
    tests = [r['test_acc'] for r in split_rows]
    print()
    print(f'val_mean  = {float(np.mean(vals)):.4f} +- {float(np.std(vals)):.4f}')
    print(f'test_mean = {float(np.mean(tests)):.4f} +- {float(np.std(tests)):.4f}')

    write_csv(out_dir / 'split_summary_adaptive_branch_records.csv', split_rows)
    write_csv(out_dir / 'misclassified_test_nodes.csv', misclf_rows)
    write_csv(out_dir / 'branch_subspace_summary.csv', branch_summary_rows)
    write_csv(out_dir / 'branch_member_rows.csv', branch_member_rows)
    write_csv(out_dir / 'train_internal_misclassified_nodes.csv', train_internal_misclf_rows)

    run_summary = {
        'dataset': dataset,
        'data_base': str(data_base),
        'raw_root': str(raw_root),
        'out_dir': str(out_dir),
        'dim_candidates': dim_candidates,
        'num_splits': int(num_splits),
        'val_mean': float(np.mean(vals)),
        'val_std': float(np.std(vals)),
        'test_mean': float(np.mean(tests)),
        'test_std': float(np.std(tests)),
    }
    (out_dir / 'run_summary.json').write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return run_summary


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Run adaptive-branch PCA on Chameleon and record misclassified nodes + branch splits.'
    )
    parser.add_argument('--dataset', type=str, default='chameleon')
    parser.add_argument('--data-base', type=str, default=None)
    parser.add_argument('--out-dir', type=str, default=None)
    parser.add_argument('--num-splits', type=int, default=10)
    parser.add_argument('--dims', type=int, nargs='*', default=[16, 24, 32, 48, 64])
    args = parser.parse_args()

    run_experiment(
        dataset=args.dataset,
        data_base=args.data_base,
        out_dir=args.out_dir,
        dim_candidates=args.dims,
        num_splits=args.num_splits,
    )


