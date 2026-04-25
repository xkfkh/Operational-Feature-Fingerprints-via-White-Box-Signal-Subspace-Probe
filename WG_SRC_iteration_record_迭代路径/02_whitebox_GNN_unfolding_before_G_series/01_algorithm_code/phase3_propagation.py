"""
Phase 3A: Subspace Propagation Response Analysis

Goal: Quantify how different subspaces respond to GNN propagation (A^k).
This module measures energy changes, dimensionality shifts, smoothing effects,
and spectral energy distributions when features are propagated through the graph.

All metrics are WHITE-BOX with explicit formulas.

Four core response metrics:
1. Energy Ratio:  rho_c^(k) = ||P_c (A^k X)^T||_F^2 / ||P_c X^T||_F^2
2. Effective Rank Change:  erank(P_c (A^k X)^T) / erank(P_c X^T)
3. Smoothness Ratio: within_class_var / between_class_var after propagation
4. Spectral Energy Distribution: energy in low/mid/high frequency bands

Author: Phase 3A agent (p3-propagation)
"""

import numpy as np
import torch
import sys
import os

# Add parent dir so we can import utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from utils import (
    effective_rank, within_class_scatter, between_class_scatter,
    compute_hop_features
)


# ============================================================
# 1. PCA Proxy Subspace Construction
# ============================================================

def build_pca_subspaces(X, Y, num_classes, sub_dim, train_mask=None):
    """
    Build PCA-based proxy subspaces for each class.

    For each class c:
      1. Select training nodes of class c
      2. Center the features
      3. Compute top-sub_dim singular vectors as basis
      4. P_c = V_c^T  (sub_dim x feat_dim), orthonormal rows

    Parameters
    ----------
    X : Tensor[n, D]  -- node features
    Y : Tensor[n]     -- labels
    num_classes : int
    sub_dim : int      -- subspace dimension
    train_mask : Tensor[n] bool, optional  -- if given, only use train nodes

    Returns
    -------
    projections : dict {class_id: Tensor[sub_dim, D]}
        Orthonormal projection matrices P_c (rows are basis vectors)
    """
    n, D = X.shape
    projections = {}

    for c in range(num_classes):
        if train_mask is not None:
            mask = (Y == c) & train_mask
        else:
            mask = (Y == c)

        nc = mask.sum().item()
        if nc < 2:
            # Fallback: random orthonormal
            Q, _ = torch.linalg.qr(torch.randn(D, sub_dim, device=X.device))
            projections[c] = Q[:, :sub_dim].t()  # [sub_dim, D]
            continue

        Xc = X[mask]  # [nc, D]
        Xc_centered = Xc - Xc.mean(0, keepdim=True)

        # SVD to get top sub_dim directions
        # Xc_centered = U S V^T, take V[:, :sub_dim]
        actual_dim = min(sub_dim, nc - 1, D)
        try:
            U, S, Vh = torch.linalg.svd(Xc_centered, full_matrices=False)
            P_c = Vh[:actual_dim, :]  # [actual_dim, D]
        except Exception:
            Q, _ = torch.linalg.qr(torch.randn(D, sub_dim, device=X.device))
            P_c = Q[:, :sub_dim].t()
            actual_dim = sub_dim

        # Pad with zeros if actual_dim < sub_dim
        if actual_dim < sub_dim:
            pad = torch.zeros(sub_dim - actual_dim, D, device=X.device)
            P_c = torch.cat([P_c, pad], dim=0)

        projections[c] = P_c  # [sub_dim, D]

    return projections


# ============================================================
# 2. Core Response Metric 1: Energy Ratio
# ============================================================

def energy_ratio(P_c, X_k, X_0):
    """
    Energy ratio: how much energy does subspace c capture after k hops?

    Formula:
        rho_c^(k) = ||P_c @ X_k^T||_F^2 / ||P_c @ X_0^T||_F^2

    where:
        P_c:  [sub_dim, D]  -- projection matrix for class c
        X_k:  [n, D]        -- features after k hops (A^k X)
        X_0:  [n, D]        -- original features (k=0)

    Geometric meaning:
        rho > 1: propagation amplifies energy in this subspace
        rho < 1: propagation attenuates energy in this subspace
        rho = 1: energy preserved (subspace is invariant to propagation)

    Numerical example:
        If P_c captures directions aligned with low-frequency eigenvectors of A,
        then A^k amplifies those directions, so rho > 1.
        If P_c captures high-frequency noise, A^k smooths it, so rho < 1.

    Parameters
    ----------
    P_c : Tensor[sub_dim, D]
    X_k : Tensor[n, D]
    X_0 : Tensor[n, D]

    Returns
    -------
    float : energy ratio rho_c^(k)
    """
    # Project features into subspace c
    # Z_k = P_c @ X_k^T -> [sub_dim, n], then ||Z_k||_F^2
    Z_k = P_c @ X_k.t()  # [sub_dim, n]
    Z_0 = P_c @ X_0.t()  # [sub_dim, n]

    energy_k = (Z_k ** 2).sum().item()
    energy_0 = (Z_0 ** 2).sum().item()

    if energy_0 < 1e-12:
        return 0.0

    return energy_k / energy_0


# ============================================================
# 3. Core Response Metric 2: Effective Rank Change
# ============================================================

def effective_rank_change(P_c, X_k, X_0):
    """
    Effective rank change: does propagation expand or collapse dimensionality?

    Formula:
        erc_c^(k) = erank(P_c @ X_k^T) / erank(P_c @ X_0^T)

    where erank(M) = exp(H(sigma/sum(sigma))), sigma = singular values of M

    Geometric meaning:
        erc > 1: propagation increases effective dimensionality
                 (features spread more uniformly across subspace directions)
        erc < 1: propagation collapses dimensionality
                 (features concentrate on fewer subspace directions)
        erc = 1: dimensionality preserved

    Numerical example:
        If original data has erank=5.2 and after 2 hops erank=3.1,
        then erc = 3.1/5.2 = 0.60 -> propagation collapses 40% of rank.
        This indicates over-smoothing in this subspace.

    Parameters
    ----------
    P_c : Tensor[sub_dim, D]
    X_k : Tensor[n, D]
    X_0 : Tensor[n, D]

    Returns
    -------
    float : effective rank change ratio
    """
    Z_k = (P_c @ X_k.t()).t()  # [n, sub_dim]
    Z_0 = (P_c @ X_0.t()).t()  # [n, sub_dim]

    er_k = effective_rank(Z_k)
    er_0 = effective_rank(Z_0)

    if er_0 < 1e-6:
        return 1.0

    return er_k / er_0


# ============================================================
# 4. Core Response Metric 3: Smoothness Ratio
# ============================================================

def smoothness_ratio(P_c, X_k, Y, num_classes):
    """
    Smoothness ratio: does propagation help or hurt class separability?

    Formula:
        s_c^(k) = within_class_var(Z_k) / between_class_var(Z_k)

    where Z_k = (P_c @ X_k^T)^T = X_k @ P_c^T  [n, sub_dim]

    Geometric meaning:
        s < 1: classes are more separated than spread -> good separability
        s > 1: classes overlap -> poor separability
        s decreasing with k: propagation improves separability
        s increasing with k: propagation hurts (over-smoothing)

    Note: We compute within/between on the PROJECTED features Z_k,
    not on the original features. This tells us about separability
    WITHIN this specific subspace.

    Numerical example:
        If within_var=0.3, between_var=2.0 -> s = 0.15 (great separation)
        After 2 hops: within_var=0.1, between_var=0.5 -> s = 0.20 (slightly worse)
        -> propagation helps compactness but also merges class means

    Parameters
    ----------
    P_c : Tensor[sub_dim, D]
    X_k : Tensor[n, D]
    Y : Tensor[n]
    num_classes : int

    Returns
    -------
    float : smoothness ratio (lower = better separability)
    """
    Z_k = X_k @ P_c.t()  # [n, sub_dim]

    w_var = within_class_scatter(Z_k, Y, num_classes)
    b_var = between_class_scatter(Z_k, Y, num_classes)

    if b_var < 1e-12:
        return float('inf')

    return w_var / b_var


# ============================================================
# 5. Core Response Metric 4: Spectral Energy Distribution
# ============================================================

def spectral_energy_distribution(P_c, X_k, eigvecs, n_bands=3):
    """
    Spectral energy distribution: which frequency bands survive propagation?

    Formula:
        1. Decompose Z_k = P_c @ X_k^T into Laplacian eigenspaces:
           Z_k_spectral = P_c @ X_k^T @ U_band  (project onto frequency bands)
        2. Energy in band b: E_b = ||Z_k_spectral_b||_F^2 / ||Z_k||_F^2

    where U_band are columns of Laplacian eigenvectors grouped by eigenvalue:
        - Low frequency:  eigenvalues in [0, 0.67]     -> smooth signals
        - Mid frequency:  eigenvalues in (0.67, 1.33]  -> transitional
        - High frequency: eigenvalues in (1.33, 2.0]   -> noisy/local signals

    Geometric meaning:
        Low freq energy dominant:  subspace captures smooth graph signals
        High freq energy dominant: subspace captures local/noisy variations
        After propagation (k>0), low freq should dominate more (smoothing effect)

    Numerical example:
        k=0: [0.40, 0.35, 0.25] -> balanced across bands
        k=2: [0.75, 0.20, 0.05] -> heavy low-freq, high-freq filtered out
        -> propagation acts as low-pass filter on this subspace

    Parameters
    ----------
    P_c : Tensor[sub_dim, D]
    X_k : Tensor[n, D]
    eigvecs : Tensor[n, n] -- Laplacian eigenvectors (columns), sorted by eigenvalue
    n_bands : int -- number of frequency bands (default 3: low/mid/high)

    Returns
    -------
    list[float] : energy fraction in each band (sums to 1)
    """
    # Z_k = P_c @ X_k^T  -> [sub_dim, n]
    Z_k = P_c @ X_k.t()  # [sub_dim, n]

    total_energy = (Z_k ** 2).sum().item()
    if total_energy < 1e-12:
        return [1.0 / n_bands] * n_bands

    n = eigvecs.shape[0]
    band_size = n // n_bands

    band_energies = []
    for b in range(n_bands):
        start = b * band_size
        end = (b + 1) * band_size if b < n_bands - 1 else n
        U_band = eigvecs[:, start:end]  # [n, band_size]

        # Project Z_k onto this frequency band
        # Z_k_band = Z_k @ U_band  -> [sub_dim, band_size]
        Z_k_band = Z_k @ U_band
        band_energy = (Z_k_band ** 2).sum().item()
        band_energies.append(band_energy)

    # Normalize to fractions
    total_band = sum(band_energies)
    if total_band < 1e-12:
        return [1.0 / n_bands] * n_bands

    return [e / total_band for e in band_energies]


# ============================================================
# 6. Compute Laplacian Eigendecomposition (for spectral analysis)
# ============================================================

def compute_laplacian_eigen(L, n_eigen=None):
    """
    Compute eigendecomposition of Laplacian matrix.

    L = U Lambda U^T

    Parameters
    ----------
    L : Tensor[n, n] -- symmetric Laplacian matrix
    n_eigen : int or None -- number of eigenpairs (None = all)

    Returns
    -------
    eigenvalues : Tensor[k]   -- sorted ascending
    eigenvectors : Tensor[n, k] -- columns are eigenvectors
    """
    if n_eigen is not None and n_eigen < L.shape[0]:
        # For large graphs, use partial eigendecomposition via scipy
        L_np = L.numpy()
        from scipy.sparse.linalg import eigsh
        eigenvalues, eigenvectors = eigsh(L_np, k=n_eigen, which='SM')
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        return torch.tensor(eigenvalues, dtype=torch.float32), \
               torch.tensor(eigenvectors, dtype=torch.float32)
    else:
        eigenvalues, eigenvectors = torch.linalg.eigh(L)
        return eigenvalues, eigenvectors


# ============================================================
# 7. Main Analysis: Run All Metrics for All Classes and Hops
# ============================================================

def analyze_propagation_response(
    X, Y, A_norm, L,
    projections,
    num_classes,
    max_hops=3,
    eigvecs=None,
    n_bands=3,
    verbose=True
):
    """
    Run complete propagation response analysis.

    For each class c and each hop k:
      1. Compute A^k X
      2. Project into subspace c: Z_k = P_c @ (A^k X)^T
      3. Compute all 4 response metrics

    Parameters
    ----------
    X : Tensor[n, D]
    Y : Tensor[n]
    A_norm : Tensor[n, n]
    L : Tensor[n, n]
    projections : dict {class_id: Tensor[sub_dim, D]}
    num_classes : int
    max_hops : int
    eigvecs : Tensor[n, n] or None (will compute if None)
    n_bands : int
    verbose : bool

    Returns
    -------
    results : dict with full response analysis
    """
    # Compute hop features: [X, AX, A^2X, ...]
    hops = compute_hop_features(X, A_norm, max_hops)

    # Compute Laplacian eigenvectors if not provided
    if eigvecs is None:
        if verbose:
            print("Computing Laplacian eigendecomposition...")
        _, eigvecs = compute_laplacian_eigen(L)

    X_0 = hops[0]  # Original features

    response_per_hop = {}

    for c in range(num_classes):
        P_c = projections[c]
        response_per_hop[c] = {}

        for k in range(max_hops + 1):
            X_k = hops[k]

            # Metric 1: Energy ratio
            er = energy_ratio(P_c, X_k, X_0)

            # Metric 2: Effective rank change
            erc = effective_rank_change(P_c, X_k, X_0)

            # Metric 3: Smoothness ratio
            sr = smoothness_ratio(P_c, X_k, Y, num_classes)

            # Metric 4: Spectral energy distribution
            sed = spectral_energy_distribution(P_c, X_k, eigvecs, n_bands)

            response_per_hop[c][k] = {
                'energy_ratio': er,
                'effective_rank_change': erc,
                'smoothness_ratio': sr,
                'spectral_energy_low': sed[0],
                'spectral_energy_mid': sed[1] if n_bands >= 2 else 0.0,
                'spectral_energy_high': sed[2] if n_bands >= 3 else 0.0,
            }

            if verbose:
                sr_str = f"{sr:.4f}" if sr != float('inf') else "inf"
                print(f"  Class {c}, Hop {k}: "
                      f"energy_ratio={er:.4f}, "
                      f"erank_change={erc:.4f}, "
                      f"smoothness={sr_str}, "
                      f"spectral=[{sed[0]:.3f},{sed[1]:.3f},{sed[2]:.3f}]")

    # Compute summary per class: average response across hops
    response_summary = {}
    for c in range(num_classes):
        # Use metrics at last hop as summary
        last = response_per_hop[c][max_hops]
        response_summary[c] = torch.tensor([
            last['energy_ratio'],
            last['effective_rank_change'],
            last['smoothness_ratio'] if last['smoothness_ratio'] != float('inf') else 100.0,
            last['spectral_energy_low'],
        ], dtype=torch.float32)

    # Compute per-node response metrics
    # For each node, compute its energy in each class subspace at each hop
    response_per_node = {}

    # Node-level energy response at best hop
    for k in range(max_hops + 1):
        X_k = hops[k]
        node_energy = torch.zeros(X.shape[0])
        for c in range(num_classes):
            P_c = projections[c]
            # z_i_c = P_c @ x_i  -> [sub_dim]
            Z_k_c = X_k @ P_c.t()  # [n, sub_dim]
            node_energy += (Z_k_c ** 2).sum(dim=1)
        response_per_node[f'total_subspace_energy_hop{k}'] = node_energy

    # Energy change per node: hop_k vs hop_0
    for k in range(1, max_hops + 1):
        e_k = response_per_node[f'total_subspace_energy_hop{k}']
        e_0 = response_per_node['total_subspace_energy_hop0']
        ratio = torch.where(e_0 > 1e-12, e_k / e_0, torch.ones_like(e_0))
        response_per_node[f'energy_change_hop{k}'] = ratio

    results = {
        'response_per_hop': response_per_hop,
        'response_summary_per_class': response_summary,
        'response_per_node': response_per_node,
        'recommended_responses': [
            'energy_ratio',
            'effective_rank_change',
            'smoothness_ratio',
            'spectral_energy_low'
        ],
        'num_hops_analyzed': max_hops,
    }

    return results


# ============================================================
# 8. Aggregate Statistics for Report
# ============================================================

def compute_aggregate_stats(results, num_classes, max_hops):
    """
    Compute aggregate statistics across classes for reporting.

    Returns dict with:
    - mean/std of each metric across classes for each hop
    - trend analysis (increasing/decreasing with hops)
    """
    stats = {}
    metric_names = [
        'energy_ratio', 'effective_rank_change', 'smoothness_ratio',
        'spectral_energy_low', 'spectral_energy_mid', 'spectral_energy_high'
    ]

    for metric in metric_names:
        stats[metric] = {}
        for k in range(max_hops + 1):
            values = []
            for c in range(num_classes):
                v = results['response_per_hop'][c][k][metric]
                if v != float('inf'):
                    values.append(v)
            if values:
                stats[metric][k] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                }
            else:
                stats[metric][k] = {
                    'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0
                }

        # Trend: compare hop 0 to last hop
        v0 = stats[metric][0]['mean']
        vK = stats[metric][max_hops]['mean']
        if abs(v0) > 1e-10:
            stats[metric]['trend'] = (vK - v0) / abs(v0)
        else:
            stats[metric]['trend'] = 0.0

    return stats

