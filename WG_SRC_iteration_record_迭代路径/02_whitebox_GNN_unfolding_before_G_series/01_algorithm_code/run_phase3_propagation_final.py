"""
Phase 3A Run Script: SECOND HALF - Real Phase 1 Subspaces

This script:
1. Loads Cora dataset
2. Loads Phase 1's real subspace projections (Coding Rate Gradient Flow, sub_dim=32)
3. Computes all 4 propagation response metrics for k=0,1,2,3
4. Compares with PCA proxy results from FIRST HALF
5. Saves final results to results/phase3/propagation_final.pt

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

from utils import load_cora, to_torch, make_masks, save_results, load_results
from phase3_propagation import (
    build_pca_subspaces,
    analyze_propagation_response,
    compute_aggregate_stats,
    compute_laplacian_eigen
)


def main():
    print("=" * 70)
    print("Phase 3A SECOND HALF: Real Phase 1 Subspaces")
    print("Subspace method: Coding Rate Gradient Flow (sub_dim=32)")
    print("Dataset: Cora, Seed: 42")
    print("=" * 70)

    torch.manual_seed(42)
    np.random.seed(42)

    # ========================================================
    # 1. Load Cora data
    # ========================================================
    print("\n[1] Loading Cora dataset...")
    t0 = time.time()
    features, labels, adj, adj_norm, lap, train_idx, val_idx, test_idx = load_cora()
    X, Y, A, L = to_torch(features, labels, adj_norm, lap)
    train_mask, val_mask, test_mask = make_masks(X.shape[0], train_idx, val_idx, test_idx)
    n, D = X.shape
    num_classes = int(Y.max().item()) + 1
    print(f"  Loaded: n={n}, D={D}, classes={num_classes}")
    print(f"  Time: {time.time()-t0:.2f}s")

    # ========================================================
    # 2. Load Phase 1 real subspaces
    # ========================================================
    print("\n[2] Loading Phase 1 subspace projections...")
    p1_data = torch.load(
        os.path.join(EXP_DIR, "results/phase1/coderate_output.pt"),
        weights_only=False
    )
    real_projections = p1_data['subspace_projections']
    sub_dim = p1_data['subspace_dims'][0]  # 32
    method = p1_data['method']
    metadata = p1_data['metadata']

    print(f"  Method: {method}")
    print(f"  Sub_dim: {sub_dim}")
    print(f"  MCR2 value: {metadata['mcr2_value']:.4f}")
    if 'test_acc' in metadata:
        print(f"  Phase 1 test_acc: {metadata['test_acc']:.4f}")
    if 'train_acc' in metadata:
        print(f"  Phase 1 train_acc: {metadata['train_acc']:.4f}")
    print(f"  Phase 1 metadata keys: {list(metadata.keys())[:10]}")

    for c in range(num_classes):
        P = real_projections[c]
        orth_err = (P @ P.t() - torch.eye(P.shape[0])).norm().item()
        print(f"  Class {c}: P shape={list(P.shape)}, orth_err={orth_err:.6f}")

    # ========================================================
    # 3. Compute Laplacian eigendecomposition
    # ========================================================
    print("\n[3] Computing Laplacian eigendecomposition...")
    t1 = time.time()
    eigvals, eigvecs = compute_laplacian_eigen(L)
    print(f"  Eigenvalue range: [{eigvals.min().item():.6f}, {eigvals.max().item():.6f}]")
    print(f"  Time: {time.time()-t1:.2f}s")

    # ========================================================
    # 4. Run propagation response with REAL subspaces
    # ========================================================
    print("\n[4] Running propagation response with REAL Phase 1 subspaces...")
    t2 = time.time()
    max_hops = 3
    real_results = analyze_propagation_response(
        X, Y, A, L,
        projections=real_projections,
        num_classes=num_classes,
        max_hops=max_hops,
        eigvecs=eigvecs,
        n_bands=3,
        verbose=True
    )
    print(f"  Analysis time: {time.time()-t2:.2f}s")

    real_stats = compute_aggregate_stats(real_results, num_classes, max_hops)

    # ========================================================
    # 5. Also run with PCA proxy (sub_dim=32) for fair comparison
    # ========================================================
    print("\n[5] Running propagation response with PCA proxy (sub_dim=32) for comparison...")
    t3 = time.time()
    pca_projections = build_pca_subspaces(X, Y, num_classes, sub_dim, train_mask=train_mask)
    pca_results = analyze_propagation_response(
        X, Y, A, L,
        projections=pca_projections,
        num_classes=num_classes,
        max_hops=max_hops,
        eigvecs=eigvecs,
        n_bands=3,
        verbose=False
    )
    pca_stats = compute_aggregate_stats(pca_results, num_classes, max_hops)
    print(f"  Time: {time.time()-t3:.2f}s")

    # ========================================================
    # 6. COMPARISON: Real vs PCA
    # ========================================================
    print("\n" + "=" * 70)
    print("COMPARISON: Phase 1 Real Subspaces vs PCA Proxy (both sub_dim=32)")
    print("=" * 70)

    metrics_to_compare = [
        'energy_ratio', 'effective_rank_change', 'smoothness_ratio',
        'spectral_energy_low', 'spectral_energy_mid', 'spectral_energy_high'
    ]

    print(f"\n{'Metric':<30} {'Hop':>4} {'Real':>10} {'PCA':>10} {'Diff':>10} {'Better':>8}")
    print("-" * 75)

    for metric in metrics_to_compare:
        for k in [0, 1, 2, 3]:
            r_val = real_stats[metric][k]['mean']
            p_val = pca_stats[metric][k]['mean']
            diff = r_val - p_val

            # Determine which is "better"
            if metric == 'energy_ratio':
                # Higher = more stable under propagation = better
                better = "Real" if r_val > p_val else "PCA"
            elif metric == 'effective_rank_change':
                # Higher = less collapse = better
                better = "Real" if r_val > p_val else "PCA"
            elif metric == 'smoothness_ratio':
                # Lower = better separability
                better = "Real" if r_val < p_val else "PCA"
            elif metric == 'spectral_energy_low':
                # Context-dependent, but generally higher low-freq = more stable
                better = "Real" if r_val > p_val else "PCA"
            else:
                better = "-"

            print(f"  {metric:<28} {k:>4} {r_val:>10.4f} {p_val:>10.4f} {diff:>+10.4f} {better:>8}")

    # ========================================================
    # 7. Per-class comparison at hop=3
    # ========================================================
    print("\n" + "=" * 70)
    print("PER-CLASS COMPARISON at hop=3: Real vs PCA")
    print("=" * 70)

    print(f"\n{'Cls':>3} | {'Real_ER':>8} {'PCA_ER':>8} | {'Real_ERC':>8} {'PCA_ERC':>8} | {'Real_SR':>8} {'PCA_SR':>8} | {'Real_LF':>8} {'PCA_LF':>8}")
    print("-" * 95)

    for c in range(num_classes):
        rr = real_results['response_per_hop'][c][3]
        pr = pca_results['response_per_hop'][c][3]
        print(f"  {c:>1} | {rr['energy_ratio']:>8.4f} {pr['energy_ratio']:>8.4f} | "
              f"{rr['effective_rank_change']:>8.4f} {pr['effective_rank_change']:>8.4f} | "
              f"{rr['smoothness_ratio']:>8.4f} {pr['smoothness_ratio']:>8.4f} | "
              f"{rr['spectral_energy_low']:>8.4f} {pr['spectral_energy_low']:>8.4f}")

    # ========================================================
    # 8. Key findings
    # ========================================================
    print("\n" + "=" * 70)
    print("KEY FINDINGS: Real vs PCA Subspaces")
    print("=" * 70)

    # Energy ratio comparison
    real_er3 = real_stats['energy_ratio'][3]['mean']
    pca_er3 = pca_stats['energy_ratio'][3]['mean']
    print(f"\n1. Energy Ratio at hop 3:")
    print(f"   Real: {real_er3:.4f}, PCA: {pca_er3:.4f}")
    if real_er3 > pca_er3:
        improvement = (real_er3 - pca_er3) / pca_er3 * 100
        print(f"   -> Real subspaces retain {improvement:.1f}% MORE energy!")
        print(f"   -> Phase 1's MCR2-optimized subspaces are better aligned with graph structure")
    else:
        print(f"   -> PCA retains more energy (unexpected)")

    # Smoothness comparison
    real_sr3 = real_stats['smoothness_ratio'][3]['mean']
    pca_sr3 = pca_stats['smoothness_ratio'][3]['mean']
    print(f"\n2. Smoothness Ratio at hop 3 (lower = better):")
    print(f"   Real: {real_sr3:.4f}, PCA: {pca_sr3:.4f}")
    if real_sr3 < pca_sr3:
        improvement = (pca_sr3 - real_sr3) / pca_sr3 * 100
        print(f"   -> Real subspaces have {improvement:.1f}% BETTER separability!")
    else:
        print(f"   -> PCA has better separability (unexpected)")

    # Effective rank comparison
    real_erc3 = real_stats['effective_rank_change'][3]['mean']
    pca_erc3 = pca_stats['effective_rank_change'][3]['mean']
    print(f"\n3. Effective Rank Change at hop 3:")
    print(f"   Real: {real_erc3:.4f}, PCA: {pca_erc3:.4f}")
    if real_erc3 > pca_erc3:
        print(f"   -> Real subspaces are more dimensionally stable under propagation")
    else:
        print(f"   -> PCA is more stable (real subspaces may use more dims actively)")

    # Spectral comparison
    real_lf3 = real_stats['spectral_energy_low'][3]['mean']
    pca_lf3 = pca_stats['spectral_energy_low'][3]['mean']
    print(f"\n4. Spectral Low-Freq Energy at hop 3:")
    print(f"   Real: {real_lf3:.4f}, PCA: {pca_lf3:.4f}")

    # ========================================================
    # 9. Save final results
    # ========================================================
    print("\n[9] Saving final results...")

    # Save comprehensive results
    final_data = {
        'real_results': {
            'response_per_hop': real_results['response_per_hop'],
            'response_summary_per_class': {
                c: v.tolist() for c, v in real_results['response_summary_per_class'].items()
            },
            'response_per_node': {
                k: v.tolist() if isinstance(v, torch.Tensor) else v
                for k, v in real_results['response_per_node'].items()
            },
            'recommended_responses': real_results['recommended_responses'],
            'num_hops_analyzed': real_results['num_hops_analyzed'],
        },
        'real_stats': real_stats,
        'pca_results': {
            'response_per_hop': pca_results['response_per_hop'],
            'response_summary_per_class': {
                c: v.tolist() for c, v in pca_results['response_summary_per_class'].items()
            },
        },
        'pca_stats': pca_stats,
        'config': {
            'sub_dim': sub_dim,
            'max_hops': max_hops,
            'seed': 42,
            'phase1_method': method,
            'phase1_mcr2_value': metadata['mcr2_value'],
            'phase1_test_acc': metadata.get('test_acc', None),
            'dataset': 'Cora',
        }
    }

    filepath1 = save_results(final_data, "phase3", "propagation_final.pt")
    print(f"  Final results: {filepath1}")

    # Save Phase 3 output in interface format
    phase3_output = {
        'response_per_hop': real_results['response_per_hop'],
        'response_summary_per_class': real_results['response_summary_per_class'],
        'response_per_node': real_results['response_per_node'],
        'recommended_responses': real_results['recommended_responses'],
        'num_hops_analyzed': real_results['num_hops_analyzed'],
    }
    filepath2 = save_results(phase3_output, "phase3", "output.pt")
    print(f"  Interface output: {filepath2}")

    print(f"\n[DONE] Phase 3A SECOND HALF complete.")


if __name__ == '__main__':
    main()

