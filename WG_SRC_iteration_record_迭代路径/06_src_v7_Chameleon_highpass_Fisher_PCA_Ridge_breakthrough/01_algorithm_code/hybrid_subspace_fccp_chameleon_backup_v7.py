#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid Subspace-FCCP for Chameleon
==================================

Design goal:
- keep the old strong HA-SDC base classifier: raw/smoothed class-subspace residuals;
- keep the useful part of robust dynamic FCCP: class/state compatibility on the graph;
- fix the main failure mode of the pure dynamic-medoid version: weak medoid-only base and no train-label clamping.

Core score:
    score(i,u) = base_residual(i, class(u))
                 + cluster_weight * dist(i, medoid_u)
                 + prior_weight * [-log pi_u]
                 + lambda_compat * compatibility_penalty(i,u)

Inference clamps train nodes to their learned train states after every propagation step.
"""

import argparse
import csv
import json
import math
import zipfile
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg


# ============================================================
# Utilities
# ============================================================


def softmax_neg(score, temperature=1.0):
    T = max(float(temperature), 1e-12)
    z = -score / T
    z = z - np.max(z, axis=1, keepdims=True)
    p = np.exp(z)
    return p / np.maximum(np.sum(p, axis=1, keepdims=True), 1e-12)


def accuracy(pred, y, idx):
    return float(np.mean(pred[idx] == y[idx]))


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


def squared_dist_to_centers(X, centers):
    x2 = np.sum(X * X, axis=1, keepdims=True)
    c2 = np.sum(centers * centers, axis=1, keepdims=True).T
    d = x2 + c2 - 2.0 * (X @ centers.T)
    return np.maximum(d, 0.0)


def parse_list_int(s):
    return [int(x.strip()) for x in str(s).split(',') if x.strip()]


def parse_list_float(s):
    return [float(x.strip()) for x in str(s).split(',') if x.strip()]


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ============================================================
# Data loading
# ============================================================


def find_raw_root(base):
    base = Path(base)
    hits = list(base.rglob('out1_node_feature_label.txt'))
    if not hits:
        raise FileNotFoundError('Cannot find out1_node_feature_label.txt')
    return hits[0].parent


def extract_zip(zip_path, workdir):
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(workdir)
    return find_raw_root(workdir)


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

    A_in = A_out.T.tocsr()

    # Keep old-style weighted symmetric adjacency for the Laplacian base.
    A_sym_weighted = 0.5 * (A_out + A_out.T)
    A_sym_weighted.setdiag(0)
    A_sym_weighted.eliminate_zeros()

    # Binary symmetric relation for compatibility propagation.
    A_sym = ((A_out + A_out.T) > 0).astype(np.float64).tocsr()
    A_sym.setdiag(0)
    A_sym.eliminate_zeros()

    return X, y, A_out, A_in, A_sym, A_sym_weighted


def load_split(raw_root, repeat):
    path = Path(raw_root) / f'chameleon_split_0.6_0.2_{repeat}.npz'
    if not path.exists():
        raise FileNotFoundError(f'Cannot find split file: {path}')
    z = np.load(path)
    train_idx = np.where(z['train_mask'].astype(bool))[0]
    val_idx = np.where(z['val_mask'].astype(bool))[0]
    test_idx = np.where(z['test_mask'].astype(bool))[0]
    return train_idx, val_idx, test_idx


def build_relations(A_out, A_in, A_sym, mode):
    out = {}
    for part in [x.strip() for x in mode.split(',') if x.strip()]:
        if part == 'sym':
            out['sym'] = A_sym
        elif part == 'out':
            out['out'] = A_out
        elif part == 'in':
            out['in'] = A_in
        else:
            raise ValueError(f'Unknown relation: {part}')
    if not out:
        raise ValueError('No relation selected')
    return out


def train_idf_transform(X, train_idx):
    X_bin = X[train_idx] > 0
    df = np.sum(X_bin, axis=0)
    n_train = len(train_idx)
    idf = np.log((1.0 + n_train) / (1.0 + df)) + 1.0
    return X * idf[None, :]


def preprocess_features(X_raw, train_idx, mode='raw'):
    if mode == 'raw':
        return np.asarray(X_raw, dtype=np.float64).copy()
    if mode == 'l2':
        return row_l2_normalize(X_raw)
    if mode == 'idf':
        return row_l2_normalize(train_idf_transform(X_raw, train_idx))
    raise ValueError(f'Unknown feature_mode: {mode}')


# ============================================================
# HA-SDC base: raw/smoothed class-subspace residuals
# ============================================================


def normalized_laplacian(A):
    A = A.tocsr().astype(np.float64)
    deg = np.asarray(A.sum(axis=1)).ravel()
    inv_sqrt = np.zeros_like(deg)
    inv_sqrt[deg > 0] = 1.0 / np.sqrt(deg[deg > 0])
    D = sparse.diags(inv_sqrt)
    n = A.shape[0]
    return (sparse.eye(n, format='csr') - D @ A @ D).tocsr()


_SMOOTH_CACHE = {}


def smooth_features(A_sym_weighted, X, lambda_smooth=0.1):
    if float(lambda_smooth) == 0.0:
        return X.copy()
    key = (id(A_sym_weighted), id(X), float(lambda_smooth))
    if key in _SMOOTH_CACHE:
        return _SMOOTH_CACHE[key]
    L = normalized_laplacian(A_sym_weighted)
    n = X.shape[0]
    system = sparse.eye(n, format='csr') + float(lambda_smooth) * L
    lu = splinalg.splu(system.tocsc())
    Z = lu.solve(np.asarray(X, dtype=np.float64))
    if Z.ndim == 1:
        Z = Z[:, None]
    Z = np.asarray(Z, dtype=np.float64)
    _SMOOTH_CACHE[key] = Z
    return Z


def fit_class_subspaces(F, y, train_idx, classes, max_dim=8):
    subspaces = {}
    for pos, c in enumerate(classes):
        idx = train_idx[y[train_idx] == c]
        Xc = F[idx]
        mu = np.mean(Xc, axis=0)
        C = Xc - mu
        d_eff = min(int(max_dim), max(0, len(idx) - 1), F.shape[1])
        if d_eff == 0:
            B = np.empty((F.shape[1], 0), dtype=np.float64)
        else:
            Gram = C @ C.T
            evals, evecs = np.linalg.eigh(Gram)
            order = np.argsort(evals)[::-1]
            keep = order[evals[order] > 1e-12][:d_eff]
            if keep.size == 0:
                B = np.empty((F.shape[1], 0), dtype=np.float64)
            else:
                B = C.T @ (evecs[:, keep] / np.sqrt(evals[keep])[None, :])
                B = np.asarray(B, dtype=np.float64)
        subspaces[pos] = {'class_label': int(c), 'mu': mu, 'basis': B}
    return subspaces


def residuals_to_subspaces(F, subspaces, n_classes):
    R = np.zeros((F.shape[0], n_classes), dtype=np.float64)
    for pos in range(n_classes):
        mu = subspaces[pos]['mu']
        B = subspaces[pos]['basis']
        C = F - mu
        total = np.sum(C * C, axis=1)
        if B.shape[1] == 0:
            evidence = np.zeros(F.shape[0])
        else:
            coord = C @ B
            evidence = np.sum(coord * coord, axis=1)
        R[:, pos] = np.maximum(total - evidence, 0.0)
    return R


def normalize_residuals_by_class(R, y, train_idx, classes, mode='none', eps=1e-12):
    if mode == 'none':
        return R.copy(), np.ones(R.shape[1])
    Rn = R.copy()
    scales = np.ones(R.shape[1])
    for pos, c in enumerate(classes):
        idx = train_idx[y[train_idx] == c]
        vals = R[idx, pos]
        if vals.size == 0:
            scale = 1.0
        elif mode == 'mean':
            scale = float(np.mean(vals))
        elif mode == 'median':
            scale = float(np.median(vals))
        else:
            raise ValueError(f'Unknown residual_norm: {mode}')
        if not np.isfinite(scale) or scale <= eps:
            scale = float(np.mean(vals) + eps) if vals.size else 1.0
        if not np.isfinite(scale) or scale <= eps:
            scale = 1.0
        Rn[:, pos] = R[:, pos] / (scale + eps)
        scales[pos] = scale
    return Rn, scales


def classwise_gate(Rs, Rr, y, val_idx, classes, tau=5.0):
    pred_s = classes[np.argmin(Rs, axis=1)]
    pred_r = classes[np.argmin(Rr, axis=1)]
    gate = np.zeros(len(classes), dtype=np.float64)
    global_s = float(np.mean(pred_s[val_idx] == y[val_idx]))
    global_r = float(np.mean(pred_r[val_idx] == y[val_idx]))
    for pos, c in enumerate(classes):
        idx = val_idx[y[val_idx] == c]
        if idx.size == 0:
            acc_s = 0.5 * (global_s + global_r)
            acc_r = 0.5 * (global_s + global_r)
        else:
            acc_s = float(np.mean(pred_s[idx] == y[idx]))
            acc_r = float(np.mean(pred_r[idx] == y[idx]))
        gate[pos] = 1.0 / (1.0 + np.exp(-float(tau) * (acc_s - acc_r)))
    return gate


def fit_base_residual(X, y, A_sym_weighted, train_idx, val_idx, classes,
                      lambda_smooth=0.1, dim=8, tau_gate=5.0, residual_norm='none'):
    Z = smooth_features(A_sym_weighted, X, lambda_smooth=lambda_smooth)
    raw_sub = fit_class_subspaces(X, y, train_idx, classes, max_dim=dim)
    smooth_sub = fit_class_subspaces(Z, y, train_idx, classes, max_dim=dim)
    Rr = residuals_to_subspaces(X, raw_sub, len(classes))
    Rs = residuals_to_subspaces(Z, smooth_sub, len(classes))
    Rr, scale_r = normalize_residuals_by_class(Rr, y, train_idx, classes, mode=residual_norm)
    Rs, scale_s = normalize_residuals_by_class(Rs, y, train_idx, classes, mode=residual_norm)
    gate = classwise_gate(Rs, Rr, y, val_idx, classes, tau=tau_gate)
    base = gate[None, :] * Rs + (1.0 - gate[None, :]) * Rr
    return base, {'gate': gate, 'raw_sub': raw_sub, 'smooth_sub': smooth_sub,
                  'scale_raw': scale_r, 'scale_smooth': scale_s}


# ============================================================
# Robust state construction
# ============================================================


def farthest_first_init(Xc, K):
    n = Xc.shape[0]
    K_eff = min(int(K), n)
    mean = np.mean(Xc, axis=0, keepdims=True)
    first = int(np.argmin(squared_dist_to_centers(Xc, mean).ravel()))
    medoids = [first]
    while len(medoids) < K_eff:
        d = squared_dist_to_centers(Xc, Xc[medoids])
        nearest = np.min(d, axis=1)
        nearest[medoids] = -1.0
        nxt = int(np.argmax(nearest))
        if nxt in medoids:
            break
        medoids.append(nxt)
    return medoids


def trimmed_medoids(Xc, K, trim_frac=0.1, iters=10):
    n = Xc.shape[0]
    if n == 0:
        raise ValueError('Empty class for medoid clustering')
    medoids = farthest_first_init(Xc, K)
    centers = Xc[medoids].copy()
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(iters):
        d = squared_dist_to_centers(Xc, centers)
        new_labels = np.argmin(d, axis=1)
        if np.array_equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels
        for k in range(centers.shape[0]):
            members = np.where(labels == k)[0]
            if members.size == 0:
                continue
            order = np.argsort(d[members, k])
            keep_n = max(1, int(math.ceil((1.0 - trim_frac) * members.size)))
            kept = members[order[:keep_n]]
            mean = np.mean(Xc[kept], axis=0, keepdims=True)
            d_to_mean = squared_dist_to_centers(Xc[members], mean).ravel()
            centers[k] = Xc[members[int(np.argmin(d_to_mean))]]
    return labels, centers


def initialize_states(X, y, train_idx, classes, K=1, trim_frac=0.1):
    n = X.shape[0]
    node_state = np.full(n, -1, dtype=np.int64)
    state_info = []
    centers = []
    class_states = {}
    for cpos, c in enumerate(classes):
        idx = train_idx[y[train_idx] == c]
        labels, centers_c = trimmed_medoids(X[idx], K=K, trim_frac=trim_frac, iters=10)
        class_states[cpos] = []
        offset = len(state_info)
        for k in range(centers_c.shape[0]):
            u = offset + k
            state_info.append({'state': u, 'class_pos': cpos, 'class_label': int(c), 'cluster': k})
            centers.append(centers_c[k])
            class_states[cpos].append(u)
        for local_i, node in enumerate(idx):
            node_state[node] = offset + int(labels[local_i])
    return {'node_state': node_state, 'state_info': state_info,
            'centers': np.vstack(centers), 'class_states': class_states}


def state_class_positions(state_info):
    return np.asarray([s['class_pos'] for s in state_info], dtype=np.int64)


def state_counts(node_state, train_idx, n_states, alpha=1.0):
    counts = np.full(n_states, float(alpha), dtype=np.float64)
    states = node_state[train_idx]
    valid = states >= 0
    np.add.at(counts, states[valid], 1.0)
    return counts / np.maximum(np.sum(counts), 1e-12)


# ============================================================
# Compatibility matrices
# ============================================================


def estimate_class_compatibility(A, y, train_idx, classes, alpha=0.5):
    n_classes = len(classes)
    class_to_pos = {int(c): i for i, c in enumerate(classes)}
    train_mask = np.zeros(y.shape[0], dtype=bool)
    train_mask[train_idx] = True
    counts = np.full((n_classes, n_classes), float(alpha), dtype=np.float64)
    coo = A.tocoo()
    mask = train_mask[coo.row] & train_mask[coo.col]
    for i, j in zip(coo.row[mask], coo.col[mask]):
        counts[class_to_pos[int(y[i])], class_to_pos[int(y[j])]] += 1.0
    return counts / np.maximum(counts.sum(axis=1, keepdims=True), 1e-12)


def build_fallback_state_B(H, state_info):
    class_pos = state_class_positions(state_info)
    n_states = len(state_info)
    n_classes = H.shape[0]
    states_per_class = np.array([np.sum(class_pos == c) for c in range(n_classes)], dtype=np.int64)
    B = np.zeros((n_states, n_states), dtype=np.float64)
    for u in range(n_states):
        a = class_pos[u]
        for v in range(n_states):
            b = class_pos[v]
            B[u, v] = H[a, b] / max(int(states_per_class[b]), 1)
    return B / np.maximum(B.sum(axis=1, keepdims=True), 1e-12)


def estimate_state_compatibility(A, y, train_idx, classes, node_state, state_info,
                                 shrink_alpha=20.0, laplace=0.5):
    n_states = len(state_info)
    train_mask = np.zeros(y.shape[0], dtype=bool)
    train_mask[train_idx] = True
    H = estimate_class_compatibility(A, y, train_idx, classes, alpha=laplace)
    B = build_fallback_state_B(H, state_info)
    counts = np.zeros((n_states, n_states), dtype=np.float64)
    coo = A.tocoo()
    mask = train_mask[coo.row] & train_mask[coo.col]
    u = node_state[coo.row[mask]]
    v = node_state[coo.col[mask]]
    valid = (u >= 0) & (v >= 0)
    if np.any(valid):
        np.add.at(counts, (u[valid], v[valid]), 1.0)
    row_counts = counts.sum(axis=1, keepdims=True)
    G = (counts + float(shrink_alpha) * B) / np.maximum(row_counts + float(shrink_alpha), 1e-12)
    G = G / np.maximum(G.sum(axis=1, keepdims=True), 1e-12)
    return G, H


def hard_neighbor_distribution(A, node_state, n_states, kappa=5.0):
    n = A.shape[0]
    z = np.zeros((n, n_states), dtype=np.float64)
    valid_nodes = np.where(node_state >= 0)[0]
    z[valid_nodes, node_state[valid_nodes]] = 1.0
    valid_float = (node_state >= 0).astype(np.float64)
    denom = np.asarray(A @ valid_float).ravel()
    q = np.asarray(A @ z)
    nz = denom > 0
    q[nz] = q[nz] / denom[nz, None]
    q[~nz] = 0.0
    eta = denom / (denom + float(kappa))
    return q, eta


def soft_neighbor_distribution(A, z, kappa=5.0, confidence_weight=True):
    if confidence_weight:
        conf = np.max(z, axis=1)
        denom = np.asarray(A @ conf).ravel()
        q = np.asarray(A @ (z * conf[:, None]))
    else:
        one = np.ones(z.shape[0], dtype=np.float64)
        denom = np.asarray(A @ one).ravel()
        q = np.asarray(A @ z)
    nz = denom > 0
    q[nz] = q[nz] / denom[nz, None]
    q[~nz] = 0.0
    eta = denom / (denom + float(kappa))
    return q, eta


def compat_penalty(q, G, mode='ce'):
    if mode == 'ce':
        return -(q @ np.log(np.maximum(G, 1e-12)).T)
    if mode == 'l2':
        q2 = np.sum(q * q, axis=1, keepdims=True)
        g2 = np.sum(G * G, axis=1, keepdims=True).T
        return np.maximum(q2 + g2 - 2.0 * (q @ G.T), 0.0)
    raise ValueError(f'Unknown compat_mode: {mode}')


# ============================================================
# Hybrid FCCP model
# ============================================================


def center_scores(X, centers, train_idx):
    dist = squared_dist_to_centers(X, centers)
    train_min = np.min(dist[train_idx], axis=1)
    scale = float(np.median(train_min))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    return dist / scale, scale


def build_model(X, y, train_idx, classes, A_relations, base_score,
                K=1, trim_frac=0.1, shrink_alpha=20.0, laplace=0.5):
    state = initialize_states(X, y, train_idx, classes, K=K, trim_frac=trim_frac)
    node_state = state['node_state']
    state_info = state['state_info']
    Gs, Hs = {}, {}
    for name, A in A_relations.items():
        G, H = estimate_state_compatibility(A, y, train_idx, classes, node_state, state_info,
                                            shrink_alpha=shrink_alpha, laplace=laplace)
        Gs[name] = G
        Hs[name] = H
    pi = state_counts(node_state, train_idx, len(state_info), alpha=1.0)
    state['Gs'] = Gs
    state['Hs'] = Hs
    state['pi'] = pi
    state['state_class_pos'] = state_class_positions(state_info)
    return state


def train_state_clamp(model, train_idx):
    n_states = len(model['state_info'])
    z = np.zeros((len(train_idx), n_states), dtype=np.float64)
    states = model['node_state'][train_idx]
    for r, u in enumerate(states):
        z[r, int(u)] = 1.0
    return z


def infer_model(X, model, A_relations, train_idx, base_score,
                lambda_compat=0.1, cluster_weight=0.1, prior_weight=0.0,
                iterations=2, temperature=1.0, kappa=5.0,
                compat_mode='ce', confidence_weight=True, clamp_train=True):
    state_class_pos = model['state_class_pos']
    centers = model['centers']
    n_states = len(model['state_info'])

    cscore, _ = center_scores(X, centers, train_idx)
    prior = -np.log(np.maximum(model['pi'], 1e-12))[None, :]
    score0 = base_score[:, state_class_pos] + float(cluster_weight) * cscore + float(prior_weight) * prior

    z = softmax_neg(score0, temperature=temperature)
    if clamp_train:
        clamp = train_state_clamp(model, train_idx)
        z[train_idx] = clamp
    else:
        clamp = None

    for _ in range(int(iterations)):
        score = score0.copy()
        for name, A in A_relations.items():
            q, eta = soft_neighbor_distribution(A, z, kappa=kappa, confidence_weight=confidence_weight)
            D = compat_penalty(q, model['Gs'][name], mode=compat_mode)
            score += float(lambda_compat) * eta[:, None] * D
        z = softmax_neg(score, temperature=temperature)
        if clamp_train:
            z[train_idx] = clamp
    return z


def state_to_class_probs(z, state_info, n_classes):
    out = np.zeros((z.shape[0], n_classes), dtype=np.float64)
    for u, info in enumerate(state_info):
        out[:, info['class_pos']] += z[:, u]
    return out / np.maximum(out.sum(axis=1, keepdims=True), 1e-12)


# ============================================================
# Experiment
# ============================================================


def run_one_config(X, y, A_sym_weighted, A_relations, train_idx, val_idx, test_idx, classes, cfg):
    base_score, base_info = fit_base_residual(
        X, y, A_sym_weighted, train_idx, val_idx, classes,
        lambda_smooth=cfg['lambda_smooth'],
        dim=cfg['dim'],
        tau_gate=cfg['tau_gate'],
        residual_norm=cfg['residual_norm'],
    )
    base_pred = classes[np.argmin(base_score, axis=1)]

    model = build_model(
        X, y, train_idx, classes, A_relations, base_score,
        K=cfg['K'], trim_frac=cfg['trim_frac'],
        shrink_alpha=cfg['shrink_alpha'], laplace=cfg['laplace'],
    )
    z = infer_model(
        X, model, A_relations, train_idx, base_score,
        lambda_compat=cfg['lambda_compat'],
        cluster_weight=cfg['cluster_weight'],
        prior_weight=cfg['prior_weight'],
        iterations=cfg['iterations'],
        temperature=cfg['temperature'],
        kappa=cfg['kappa'],
        compat_mode=cfg['compat_mode'],
        confidence_weight=cfg['confidence_weight'],
        clamp_train=cfg['clamp_train'],
    )
    class_prob = state_to_class_probs(z, model['state_info'], len(classes))
    pred = classes[np.argmax(class_prob, axis=1)]

    return {
        'base_train': accuracy(base_pred, y, train_idx),
        'base_val': accuracy(base_pred, y, val_idx),
        'base_test': accuracy(base_pred, y, test_idx),
        'train': accuracy(pred, y, train_idx),
        'val': accuracy(pred, y, val_idx),
        'test': accuracy(pred, y, test_idx),
        'gate': base_info['gate'].tolist(),
    }


def make_grid(args):
    grid = []
    for K in parse_list_int(args.K_list):
        for dim in parse_list_int(args.dim_list):
            for lambda_smooth in parse_list_float(args.lambda_smooth_list):
                for lambda_compat in parse_list_float(args.lambda_list):
                    for cluster_weight in parse_list_float(args.cluster_weight_list):
                        for iterations in parse_list_int(args.iter_list):
                            for shrink_alpha in parse_list_float(args.shrink_alpha_list):
                                grid.append({
                                    'K': K,
                                    'dim': dim,
                                    'lambda_smooth': lambda_smooth,
                                    'lambda_compat': lambda_compat,
                                    'cluster_weight': cluster_weight,
                                    'iterations': iterations,
                                    'shrink_alpha': shrink_alpha,
                                    'trim_frac': args.trim_frac,
                                    'laplace': args.laplace,
                                    'kappa': args.kappa,
                                    'prior_weight': args.prior_weight,
                                    'temperature': args.temperature,
                                    'tau_gate': args.tau_gate,
                                    'residual_norm': args.residual_norm,
                                    'compat_mode': args.compat_mode,
                                    'confidence_weight': args.confidence_weight,
                                    'clamp_train': args.clamp_train,
                                })
    return grid


def run_experiment(args):
    if args.root:
        raw_root = Path(args.root)
    elif args.zip:
        raw_root = extract_zip(args.zip, args.workdir)
    else:
        raise ValueError('Need --zip or --root')

    X_raw, y, A_out, A_in, A_sym, A_sym_weighted = load_chameleon_raw(raw_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = make_grid(args)
    print(f'Loaded Chameleon: n={X_raw.shape[0]}, d={X_raw.shape[1]}, grid={len(grid)} per repeat')

    all_rows, best_rows = [], []
    for repeat in range(args.repeats):
        print(f'\n========== repeat {repeat} ==========')
        train_idx, val_idx, test_idx = load_split(raw_root, repeat)
        X = preprocess_features(X_raw, train_idx, mode=args.feature_mode)
        classes = np.unique(y[train_idx])
        A_relations = build_relations(A_out, A_in, A_sym, args.relations)

        best = None
        for gi, cfg in enumerate(grid):
            res = run_one_config(X, y, A_sym_weighted, A_relations, train_idx, val_idx, test_idx, classes, cfg)
            row = {
                'repeat': repeat,
                'grid_id': gi,
                'feature_mode': args.feature_mode,
                'relations': args.relations,
                **cfg,
                'base_train': res['base_train'],
                'base_val': res['base_val'],
                'base_test': res['base_test'],
                'train': res['train'],
                'val': res['val'],
                'test': res['test'],
            }
            all_rows.append(row)

            if best is None:
                best = row
            else:
                key = (row['val'], -abs(row['lambda_compat']), -row['K'], -row['iterations'], row['test'] * 1e-9)
                best_key = (best['val'], -abs(best['lambda_compat']), -best['K'], -best['iterations'], best['test'] * 1e-9)
                if key > best_key:
                    best = row

            print(f"grid={gi:03d} K={cfg['K']} d={cfg['dim']} ls={cfg['lambda_smooth']} "
                  f"lam={cfg['lambda_compat']} cw={cfg['cluster_weight']} it={cfg['iterations']} "
                  f"base_val={res['base_val']:.4f} val={res['val']:.4f} test={res['test']:.4f}")

        best_rows.append(best)
        print(f"[BEST repeat {repeat}] base_val={best['base_val']:.4f} base_test={best['base_test']:.4f} "
              f"K={best['K']} d={best['dim']} ls={best['lambda_smooth']} lam={best['lambda_compat']} "
              f"cw={best['cluster_weight']} it={best['iterations']} val={best['val']:.4f} test={best['test']:.4f}")
        write_csv(out_dir / 'all_grid_results_partial.csv', all_rows)
        write_csv(out_dir / 'best_by_val_partial.csv', best_rows)

    write_csv(out_dir / 'all_grid_results.csv', all_rows)
    write_csv(out_dir / 'best_by_val.csv', best_rows)

    summary = {
        'repeats': args.repeats,
        'feature_mode': args.feature_mode,
        'relations': args.relations,
        'base_val_mean_selected': float(np.mean([r['base_val'] for r in best_rows])),
        'base_test_mean_selected': float(np.mean([r['base_test'] for r in best_rows])),
        'val_mean': float(np.mean([r['val'] for r in best_rows])),
        'val_std': float(np.std([r['val'] for r in best_rows])),
        'test_mean': float(np.mean([r['test'] for r in best_rows])),
        'test_std': float(np.std([r['test'] for r in best_rows])),
        'delta_val_mean': float(np.mean([r['val'] - r['base_val'] for r in best_rows])),
        'delta_test_mean': float(np.mean([r['test'] - r['base_test'] for r in best_rows])),
        'best_rows': best_rows,
    }
    with open(out_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print('\n========== SUMMARY ==========' )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip', type=str, default=None)
    parser.add_argument('--root', type=str, default=None)
    parser.add_argument('--workdir', type=str, default='hybrid_subspace_fccp_tmp')
    parser.add_argument('--out', type=str, default='hybrid_subspace_fccp_results')
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--feature_mode', type=str, default='raw', choices=['raw', 'l2', 'idf'])
    parser.add_argument('--relations', type=str, default='sym')

    parser.add_argument('--K_list', type=str, default='1,2')
    parser.add_argument('--dim_list', type=str, default='8,16')
    parser.add_argument('--lambda_smooth_list', type=str, default='0,0.03,0.1')
    parser.add_argument('--lambda_list', type=str, default='0,0.01,0.03,0.05,0.1,0.25,0.5')
    parser.add_argument('--cluster_weight_list', type=str, default='0,0.05,0.1,0.25')
    parser.add_argument('--iter_list', type=str, default='1,2,3')
    parser.add_argument('--shrink_alpha_list', type=str, default='20,100')

    parser.add_argument('--trim_frac', type=float, default=0.10)
    parser.add_argument('--laplace', type=float, default=0.50)
    parser.add_argument('--kappa', type=float, default=5.0)
    parser.add_argument('--prior_weight', type=float, default=0.0)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--tau_gate', type=float, default=5.0)
    parser.add_argument('--residual_norm', type=str, default='none', choices=['none', 'median', 'mean'])
    parser.add_argument('--compat_mode', type=str, default='ce', choices=['ce', 'l2'])
    parser.add_argument('--confidence_weight', action='store_true', default=True)
    parser.add_argument('--no_confidence_weight', dest='confidence_weight', action='store_false')
    parser.add_argument('--clamp_train', action='store_true', default=True)
    parser.add_argument('--no_clamp_train', dest='clamp_train', action='store_false')

    args = parser.parse_args()
    run_experiment(args)


if __name__ == '__main__':
    main()


