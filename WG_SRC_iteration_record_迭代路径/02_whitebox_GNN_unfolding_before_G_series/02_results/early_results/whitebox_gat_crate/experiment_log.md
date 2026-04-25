# Whitebox GAT (CRATE-style) Experiment Log

## Mathematical Derivation

### CRATE MSSA on Graphs

For each head k with subspace basis U_k in R^{d x r}:

  Z_k = U_k^T H                    (project to subspace)
  e_ij = Z_k[:,i]^T Z_k[:,j] / sqrt(r)  (subspace correlation)
  alpha_ij = softmax_{j in N(i)} e_ij    (masked attention)
  H_k_out = U_k @ (Z_k @ alpha^T)        (aggregate + project back)

Multi-head aggregation with residual:
  H^{l+1/2} = H^l + eta * sum_k H_k_out

ISTA sparsification:
  H^{l+1} = ReLU(H^{l+1/2} - threshold)

Subspace orthogonality loss:
  L_orth_U = sum_{k!=l} ||U_k^T U_l||_F^2

Total loss:
  L = L_CE + lambda_mcr * (-DeltaR) + lambda_orth_U * L_orth_U + lambda_l1 * ||Z||_1

## Adaptation of MSSA to Graphs

Key difference from standard CRATE:
- Attention is masked to graph neighbourhood N(i) (including self-loop)
- Non-edges get score -1e9 before softmax (hard masking)
- Each head's subspace basis U_k is shared across all nodes
- Subspace dim r << d ensures compression

## Subspace Orthogonality Constraint

U_k^T U_l = 0 for k != l ensures each head captures distinct
information. Initialised via QR decomposition, then regularised
during training with L_orth_U penalty.

## Optimisation Process

## Run 1: Base config
Config: {'hidden_dim': 64, 'num_heads': 7, 'subspace_dim': 16, 'eta': 0.5, 'lambda_sparse': 0.05, 'lambda_mcr': 0.01, 'lambda_orth_U': 0.01, 'lambda_l1': 0.001, 'dropout': 0.5, 'lr': 0.005, 'weight_decay': 0.0005, 'epochs': 200, 'patience': 20}
Epoch  20 | loss=0.5805 ce=0.4574 mcr=-3.5616 orthU=15.7783 val=0.6567 test=0.6790
Epoch  40 | loss=0.0068 ce=0.0169 mcr=-5.8665 orthU=4.6817 val=0.7233 test=0.7710
Epoch  60 | loss=-0.0143 ce=0.0310 mcr=-7.2203 orthU=2.5424 val=0.7600 test=0.8020
Early stop at epoch 63
Result: val=0.7833, test=0.8080


## All Results Summary

| Name | num_heads | subspace_dim | hidden_dim | lambda_orth_U | val_acc | test_acc |
|------|-----------|--------------|------------|---------------|---------|----------|
| base | 7 | 16 | 64 | 0.01 | 0.7833 | 0.8080 |


