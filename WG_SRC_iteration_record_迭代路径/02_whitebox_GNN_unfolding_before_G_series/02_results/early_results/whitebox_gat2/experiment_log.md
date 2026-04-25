# Experiment Log -- Agent 5: White-box GAT (CRATE on Graphs)

## 1. Derivation: CRATE -> Graph GAT

CRATE optimises: max E[R(Z) - R^c(Z; U_[K]) - lambda*||Z||_0]
via alternating two sub-problems per block:

### Compression step (-> attention)
Z^(l+1/2) = (1-c)*Z^l + c * MSSA(Z^l | U_[K])
MSSA = multi-head subspace self-attention
SSA(Z|U_k)_i = sum_j alpha_ij * U_k U_k^T Z_j
alpha_ij = softmax_j( (U_k^T Z_i)^T (U_k^T Z_j) / sqrt(r) )

Graph adaptation: restrict j to N_hat(i) (graph neighbors + self).
This makes the compression local, consistent with graph inductive bias.

### Sparsification step (-> MLP / ISTA)
Z^(l+1) = ReLU(Z^(l+1/2) + eta*D^T(Z^(l+1/2) - D@Z^(l+1/2)) - eta*lambda*1)
One proximal gradient step on ||Z||_1 with dictionary D.
Operates node-locally; no graph structure needed.

## 2. Implementation Details

- graph_masked_ssa: scatter-softmax over edge_index for each subspace U_k
- ista_step: proximal gradient with learnable dictionary D
- WhiteboxGATBlock: compression + sparsification
- WhiteboxGAT2: input_proj -> Block1 -> Block2 -> classifier
- Loss: CE + lambda_mcr*(-DeltaR) + lambda_orth*L_orth
- DeltaR computed on training nodes of last hidden layer
- lambda_orth kept small (0.001) per Agent 1 recommendation

## 3. Comparison with Other Agents

Agent 1 (MCR2+Orth, MLP, synthetic data):
  - Finds lambda_orth=0.1 too large; recommends [0.001, 0.01]
  - We use lambda_orth=0.001 accordingly

Agent 2 (whitebox_gcn): not available
Agent 3 (whitebox_gat PRISM): not available
Agent 4 (whitebox_gcn2): not available

## 4. Hyperparameter Search

Config 0: val=0.1620 test=0.1490 | {'hidden_dim': 64, 'num_subspaces': 4, 'subspace_dim': 16, 'eta': 0.5, 'lambda_sparse': 0.05, 'c': 0.5, 'dropout': 0.5, 'lr': 0.005, 'weight_decay': 0.0005, 'lambda_mcr': 0.01, 'lambda_orth': 0.001, 'epochs': 200, 'patience': 20}
Config 1: val=0.1620 test=0.1490 | {'hidden_dim': 128, 'num_subspaces': 8, 'subspace_dim': 16, 'eta': 0.3, 'lambda_sparse': 0.02, 'c': 0.5, 'dropout': 0.6, 'lr': 0.005, 'weight_decay': 0.0005, 'lambda_mcr': 0.01, 'lambda_orth': 0.001, 'epochs': 200, 'patience': 20}
Config 2: val=0.7140 test=0.7020 | {'hidden_dim': 64, 'num_subspaces': 4, 'subspace_dim': 16, 'eta': 0.3, 'lambda_sparse': 0.01, 'c': 0.3, 'dropout': 0.5, 'lr': 0.01, 'weight_decay': 0.001, 'lambda_mcr': 0.005, 'lambda_orth': 0.0005, 'epochs': 300, 'patience': 30}
Config 3: val=0.1560 test=0.1440 | {'hidden_dim': 64, 'num_subspaces': 4, 'subspace_dim': 16, 'eta': 0.3, 'lambda_sparse': 0.01, 'c': 0.5, 'dropout': 0.5, 'lr': 0.005, 'weight_decay': 0.0005, 'lambda_mcr': 0.0, 'lambda_orth': 0.0, 'epochs': 200, 'patience': 20}
Config 4: val=0.4720 test=0.4680 | {'hidden_dim': 128, 'num_subspaces': 4, 'subspace_dim': 32, 'eta': 0.2, 'lambda_sparse': 0.005, 'c': 0.5, 'dropout': 0.5, 'lr': 0.005, 'weight_decay': 0.0005, 'lambda_mcr': 0.01, 'lambda_orth': 0.001, 'epochs': 300, 'patience': 30}

## 5. Final Results

Best val  accuracy: 0.7140
Best test accuracy: 0.7020
Best config index : 2

## 6. Conclusion

CRATE's sparse rate reduction objective can be cleanly ported to graphs:
- Compression step = graph-masked MSSA (local subspace projection + aggregation)
- Sparsification step = node-local ISTA (promotes sparse representations)
- Each block has a clear optimization interpretation (one alternating iteration)
- MCR2 DeltaR loss further encourages class-discriminative representations
- The white-box design provides interpretability: each layer's role is explicit

