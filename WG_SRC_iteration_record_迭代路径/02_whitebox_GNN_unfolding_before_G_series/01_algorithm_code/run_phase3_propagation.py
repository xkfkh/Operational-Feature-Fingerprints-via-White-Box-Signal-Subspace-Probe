"""
Phase 3A Run Script: Propagation Response Analysis (FIRST HALF - PCA Proxy)

This script:
1. Loads Cora dataset
2. Builds PCA proxy subspaces for each class (dims=4,8,16)
3. Computes propagation response metrics for k=0,1,2,3
4. Saves results to results/phase3/propagation_preliminary.pt
5. Prints detailed analysis

Environment: CPU only, seed=42
"""

import os, sys, time
import numpy as np
import torch

# Setup paths
PROJECT_DIR = "D:/桌面/MSR实验复现与创新"
EXP_DIR = os.path.join(PROJECT_DIR, "experiments_g1/wgsrc_development_workspace")
sys.path.insert(0, os.path.join(EXP_DIR, "src"))

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from utils import load_cora, to_torch, make_masks, save_results
from phase3_propagation import (
    build_pca_subspaces,
    analyze_propagation_response,
    compute_aggregate_stats,
    compute_laplacian_eigen
)


def run_experiment(sub_dim, seed=42):
    """Run propagation response analysis with given subspace dimension."""
    print(f"\n{'='*70}")
    print(f"Phase 3A: Propagation Response Analysis (PCA proxy, sub_dim={sub_dim})")
    print(f"{'='*70}")

    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load data
    print("\n[1] Loading Cora dataset...")
    t0 = time.time()
    features, labels, adj, adj_norm, lap, train_idx, val_idx, test_idx = load_cora()
    X, Y, A, L = to_torch(features, labels, adj_norm, lap)
    train_mask, val_mask, test_mask = make_masks(X.shape[0], train_idx, val_idx, test_idx)
    n, D = X.shape
    num_classes = int(Y.max().item()) + 1
    print(f"  Loaded: n={n}, D={D}, classes={num_classes}")
    print(f"  Train: {train_mask.sum().item()}, Val: {val_mask.sum().item()}, Test: {test_mask.sum().item()}")
    print(f"  Time: {time.time()-t0:.2f}s")

    # Build PCA proxy subspaces
    print(f"\n[2] Building PCA proxy subspaces (sub_dim={sub_dim})...")
    t1 = time.time()
    projections = build_pca_subspaces(X, Y, num_classes, sub_dim, train_mask=train_mask)
    print(f"  Built {len(projections)} class subspaces")
    for c in range(num_classes):
        P = projections[c]
        # Verify orthonormality
        orth_err = (P @ P.t() - torch.eye(sub_dim)).norm().item()
        class_count = (Y[train_mask] == c).sum().item()
        print(f"  Class {c}: P shape={list(P.shape)}, orth_err={orth_err:.6f}, train_count={class_count}")
    print(f"  Time: {time.time()-t1:.2f}s")

    # Compute Laplacian eigendecomposition
    print(f"\n[3] Computing Laplacian eigendecomposition...")
    t2 = time.time()
    eigvals, eigvecs = compute_laplacian_eigen(L)
    print(f"  Eigenvalue range: [{eigvals.min().item():.6f}, {eigvals.max().item():.6f}]")
    print(f"  Time: {time.time()-t2:.2f}s")

    # Show eigenvalue distribution for context
    n_eig = len(eigvals)
    low_cutoff = int(n_eig * 0.33)
    mid_cutoff = int(n_eig * 0.67)
    print(f"  Low band:  eigenvalues [{eigvals[0].item():.4f}, {eigvals[low_cutoff-1].item():.4f}]")
    print(f"  Mid band:  eigenvalues [{eigvals[low_cutoff].item():.4f}, {eigvals[mid_cutoff-1].item():.4f}]")
    print(f"  High band: eigenvalues [{eigvals[mid_cutoff].item():.4f}, {eigvals[-1].item():.4f}]")

    # Run propagation response analysis
    print(f"\n[4] Running propagation response analysis (hops=0,1,2,3)...")
    t3 = time.time()
    max_hops = 3
    results = analyze_propagation_response(
        X, Y, A, L,
        projections=projections,
        num_classes=num_classes,
        max_hops=max_hops,
        eigvecs=eigvecs,
        n_bands=3,
        verbose=True
    )
    print(f"  Analysis time: {time.time()-t3:.2f}s")

    # Aggregate statistics
    print(f"\n[5] Aggregate Statistics Across Classes:")
    stats = compute_aggregate_stats(results, num_classes, max_hops)

    for metric in ['energy_ratio', 'effective_rank_change', 'smoothness_ratio',
                    'spectral_energy_low', 'spectral_energy_mid', 'spectral_energy_high']:
        print(f"\n  {metric}:")
        for k in range(max_hops + 1):
            s = stats[metric][k]
            print(f"    hop {k}: mean={s['mean']:.4f}, std={s['std']:.4f}, "
                  f"range=[{s['min']:.4f}, {s['max']:.4f}]")
        trend = stats[metric]['trend']
        direction = "INCREASING" if trend > 0.05 else ("DECREASING" if trend < -0.05 else "STABLE")
        print(f"    Trend: {trend:+.4f} ({direction})")

    # Key findings summary
    print(f"\n[6] Key Findings (sub_dim={sub_dim}):")
    print("-" * 50)

    # Energy ratio analysis
    er_mean_k3 = stats['energy_ratio'][3]['mean']
    if er_mean_k3 > 1.1:
        print(f"  * Energy AMPLIFIED by propagation (ratio={er_mean_k3:.3f})")
        print(f"    -> Subspaces are aligned with low-frequency graph signals")
    elif er_mean_k3 < 0.9:
        print(f"  * Energy ATTENUATED by propagation (ratio={er_mean_k3:.3f})")
        print(f"    -> Subspaces capture high-frequency / noisy components")
    else:
        print(f"  * Energy roughly PRESERVED (ratio={er_mean_k3:.3f})")

    # Effective rank analysis
    erc_mean_k3 = stats['effective_rank_change'][3]['mean']
    if erc_mean_k3 < 0.8:
        print(f"  * Effective rank COLLAPSED ({erc_mean_k3:.3f})")
        print(f"    -> Over-smoothing risk: propagation reduces diversity")
    elif erc_mean_k3 > 1.2:
        print(f"  * Effective rank EXPANDED ({erc_mean_k3:.3f})")
    else:
        print(f"  * Effective rank stable ({erc_mean_k3:.3f})")

    # Smoothness analysis
    sr_k0 = stats['smoothness_ratio'][0]['mean']
    sr_k3 = stats['smoothness_ratio'][3]['mean']
    if sr_k3 < sr_k0 * 0.8:
        print(f"  * Separability IMPROVED with propagation (ratio {sr_k0:.3f} -> {sr_k3:.3f})")
    elif sr_k3 > sr_k0 * 1.2:
        print(f"  * Separability DEGRADED with propagation (ratio {sr_k0:.3f} -> {sr_k3:.3f})")
    else:
        print(f"  * Separability roughly stable (ratio {sr_k0:.3f} -> {sr_k3:.3f})")

    # Spectral analysis
    low_k3 = stats['spectral_energy_low'][3]['mean']
    high_k3 = stats['spectral_energy_high'][3]['mean']
    print(f"  * Spectral distribution at hop 3: low={low_k3:.3f}, high={high_k3:.3f}")
    if low_k3 > 0.5:
        print(f"    -> Low-pass filtering effect dominates")

    return results, stats


def main():
    print("=" * 70)
    print("Phase 3A: Subspace Propagation Response Analysis")
    print("FIRST HALF: Using PCA proxy subspaces (Phase 1 not ready yet)")
    print("Dataset: Cora, Seed: 42")
    print("=" * 70)

    all_results = {}
    all_stats = {}

    for sub_dim in [4, 8, 16]:
        results, stats = run_experiment(sub_dim, seed=42)
        all_results[sub_dim] = results
        all_stats[sub_dim] = stats

    # Cross-dimension comparison
    print("\n" + "=" * 70)
    print("CROSS-DIMENSION COMPARISON")
    print("=" * 70)

    header = f"{'Metric':<30} {'dim=4':>10} {'dim=8':>10} {'dim=16':>10}"
    print(header)
    print("-" * 60)

    for metric in ['energy_ratio', 'effective_rank_change', 'smoothness_ratio',
                    'spectral_energy_low']:
        for k in [0, 1, 3]:
            values = []
            for sub_dim in [4, 8, 16]:
                v = all_stats[sub_dim][metric][k]['mean']
                values.append(v)
            label = f"{metric}(hop={k})"
            print(f"  {label:<28} {values[0]:>10.4f} {values[1]:>10.4f} {values[2]:>10.4f}")

    # Save all results
    print("\n\n[7] Saving results...")
    save_data = {
        'all_results': {},
        'all_stats': all_stats,
        'config': {
            'sub_dims': [4, 8, 16],
            'max_hops': 3,
            'seed': 42,
            'subspace_method': 'PCA_proxy',
            'dataset': 'Cora',
        }
    }

    # Convert results to serializable format
    for sub_dim in [4, 8, 16]:
        r = all_results[sub_dim]
        save_data['all_results'][sub_dim] = {
            'response_per_hop': r['response_per_hop'],
            'response_summary_per_class': {
                c: v.tolist() for c, v in r['response_summary_per_class'].items()
            },
            'response_per_node': {
                k: v.tolist() if isinstance(v, torch.Tensor) else v
                for k, v in r['response_per_node'].items()
            },
            'recommended_responses': r['recommended_responses'],
            'num_hops_analyzed': r['num_hops_analyzed'],
        }

    filepath = save_results(save_data, "phase3", "propagation_preliminary.pt")
    print(f"  Saved to: {filepath}")

    # Also save in Phase 3 output format for downstream compatibility
    # Use sub_dim=8 as primary (matches baseline's subspace_dim=8)
    primary_results = all_results[8]
    phase3_output = {
        'response_per_hop': primary_results['response_per_hop'],
        'response_summary_per_class': primary_results['response_summary_per_class'],
        'response_per_node': primary_results['response_per_node'],
        'recommended_responses': primary_results['recommended_responses'],
        'num_hops_analyzed': primary_results['num_hops_analyzed'],
    }
    filepath2 = save_results(phase3_output, "phase3", "output_preliminary.pt")
    print(f"  Phase 3 interface output: {filepath2}")

    total_time = time.time()
    print(f"\n[DONE] Phase 3A preliminary analysis complete.")


if __name__ == '__main__':
    main()

