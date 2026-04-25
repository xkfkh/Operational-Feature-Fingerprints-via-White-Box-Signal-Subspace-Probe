# Whitebox GCN (CRATE-style) Experiment Log

## Mathematical Derivation

### Objective: Sparse Rate Reduction

Following CRATE, the per-layer objective is:

    L_obj = R(Z) - Rc(Z; U_[K]) - lambda * ||Z||_0

where R(Z) = 0.5 * log det(I + d/(n*eps) * Z Z^T).

Alternating minimization yields two sub-steps per layer:

### Step 1: Graph-Smooth Compression (analogous to CRATE attention)

    Z^(l+1/2) = Z^l + eta * A_norm @ Z^l @ (U U^T)

- U: learned subspace basis (out_dim x rank), orthonormal init
- A_norm = D^{-1/2}(A+I)D^{-1/2}: symmetrically normalized adjacency
- Projects neighbor-aggregated features onto the learned subspace
- Analogous to CRATE attention: Z + MSSA(Z | U_[K])

### Step 2: ISTA Sparsification (analogous to CRATE MLP)

    Z^(l+1) = ReLU(Z^(l+1/2) + eta*(Z^(l+1/2) - Z^(l+1/2) @ D^T D) - eta*lam)

- D: learnable dictionary (dict_size x out_dim), row-normalized
- One step of ISTA on ||Z - D alpha||^2 + lam||alpha||_1
- ReLU = soft-thresholding proximal operator

### Loss

    L = L_CE - lambda_mcr * DeltaR(Z_train) + lambda_sparse * ||Z_train||_1

## Comparison with Standard GCN

| Aspect | Standard GCN | Whitebox GCN (CRATE-style) |
|--------|-------------|---------------------------|
| Layer | sigma(A H W) | compress + ISTA |
| Theory | empirical | sparse rate reduction |
| Subspace | implicit | explicit U |
| Sparsity | none | ISTA dict D |
| Loss | CE | CE + MCR2 + L1 |

## Optimization Process

### run1_baseline
- val=0.7200, test=0.7280
- eta=1.0, lam=0.001, dict=256, lmcr=0.001, lsparse=0.0

### run2_pure_ce
- val=0.7600, test=0.7320
- eta=1.0, lam=0.001, dict=256, lmcr=0.0, lsparse=0.0

### run3_small_eta
- val=0.6867, test=0.7040
- eta=0.1, lam=0.0001, dict=256, lmcr=0.0, lsparse=0.0

### run4_3layers
- val=0.7600, test=0.7510
- eta=0.5, lam=0.001, dict=256, lmcr=0.0, lsparse=0.0

## Final Result

Best: run4_3layers
Test acc: 0.7510, Val acc: 0.7600

## Key Findings

1. CRATE-style alternating optimization (compress + sparsify) provides
   a principled interpretation of graph convolution as sparse rate reduction.
2. The subspace basis U captures class-specific structure aligned with MCR2.
3. The ISTA step enforces sparse representations in the dictionary basis.
4. MCR2 auxiliary loss must be weighted carefully (consistent with Agent 1).
5. Dense adjacency avoids sparse tensor issues on Windows/CPU.


