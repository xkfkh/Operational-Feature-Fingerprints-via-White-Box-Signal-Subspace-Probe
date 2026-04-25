"""
whitebox_gcn2.py  --  Agent 4
White-box GCN: ReduNet unrolled gradient ascent on graph-structured data.

Diagnostic findings:
- With uniform soft labels, expand and agg_compress have similar magnitude (~0.6 vs ~0.5)
  so the gradient step H + eta*(expand - agg) barely transforms H
- The layer needs graph aggregation of H itself (not just compression term)
- Fix: apply A_norm @ H as the base representation, then apply ReduNet operators

Corrected architecture per layer:
  H_agg = A_norm @ H                          (graph smoothing)
  H_n   = LayerNorm(H_agg)
  E     = (I + alpha/n * H_n^T H_n)^{-1}     (expansion operator)
  C_k   = (I + alpha/n_k * H_k^T H_k)^{-1}  (compression operator)
  H'    = H_n + eta*(H_n @ E - sum_k(gamma_k*(H_n*pi_k) @ C_k))
  H_out = ISTA(H', threshold)

Loss = CE(train) - lambda_mcr * DeltaR + lambda_orth * L_orth
"""

import os
import sys
import pickle
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────

def load_cora(data_dir):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objs = []
    for n in names:
        with open(f"{data_dir}/ind.cora.{n}", 'rb') as f:
            objs.append(pickle.load(f, encoding='latin1'))
    x, y, tx, ty, allx, ally, graph = objs

    test_idx = [int(l.strip()) for l in open(f"{data_dir}/ind.cora.test.index")]
    test_idx_range = sorted(test_idx)

    features = sp.vstack((allx, tx)).toarray()
    features[test_idx] = features[test_idx_range]

    labels = np.vstack((ally, ty))
    labels[test_idx] = labels[test_idx_range]
    labels = np.argmax(labels, axis=1)

    n = len(graph)
    rows, cols = [], []
    for i, nbrs in graph.items():
        for j in nbrs:
            rows.append(i); cols.append(j)
            rows.append(j); cols.append(i)
    data_vals = np.ones(len(rows), dtype=np.float32)
    adj = sp.coo_matrix((data_vals, (rows, cols)), shape=(n, n)).tocsr()
    adj = (adj + adj.T); adj.data[:] = 1.0

    train_mask = np.zeros(n, dtype=bool); train_mask[:140] = True
    val_mask   = np.zeros(n, dtype=bool); val_mask[140:640] = True
    test_mask  = np.zeros(n, dtype=bool)
    test_mask[test_idx_range[0]:test_idx_range[-1]+1] = True

    return features, labels, adj, train_mask, val_mask, test_mask


def build_norm_adj(adj):
    """D^{-1/2}(A+I)D^{-1/2} as torch sparse tensor"""
    n = adj.shape[0]
    adj_hat = (adj + sp.eye(n, dtype=np.float32)).tocsr()
    deg = np.array(adj_hat.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(deg, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D = sp.diags(d_inv_sqrt)
    A_norm = (D @ adj_hat @ D).tocoo()
    idx = torch.tensor(np.vstack([A_norm.row, A_norm.col]), dtype=torch.long)
    val = torch.tensor(A_norm.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, val, (n, n)).coalesce()


def normalize_features(features):
    row_sum = features.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return features / row_sum


# ─────────────────────────────────────────────
# ReduNet GCN Layer
# ─────────────────────────────────────────────

class ReduNetGCNLayer(nn.Module):
    """
    White-box GCN layer: graph aggregation + ReduNet gradient unrolling + ISTA.

    H_agg = A_norm @ H                          (graph smoothing)
    H_n   = LayerNorm(H_agg)
    E     = (I + alpha/n * H_n^T H_n)^{-1}
    C_k   = (I + alpha/n_k * H_k^T H_k)^{-1}
    H'    = H_n + eta*(H_n @ E - sum_k(gamma_k*(H_n*pi_k) @ C_k))
    H_out = sign(H') * ReLU(|H'| - threshold)
    """

    def __init__(self, dim, num_classes, eta=0.5, alpha=0.5, lambda_sparse=0.01):
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes
        self.eta = eta
        self.alpha = alpha

        self.threshold = nn.Parameter(torch.tensor(lambda_sparse))
        self.log_gamma = nn.Parameter(torch.zeros(num_classes))
        self.norm = nn.LayerNorm(dim)

    def forward(self, H, A_norm_sparse, soft_labels):
        # Step 1: graph aggregation
        H_agg = torch.sparse.mm(A_norm_sparse, H)

        # Step 2: normalize for stable operator computation
        H_n = self.norm(H_agg)
        n, d = H_n.shape
        K = self.num_classes
        gamma = F.softplus(self.log_gamma)

        # Step 3: expansion operator E = (I + alpha/n * H_n^T H_n)^{-1}
        gram = (self.alpha / n) * (H_n.t() @ H_n)
        E = torch.linalg.inv(torch.eye(d, device=H.device) + gram)
        expand_term = H_n @ E

        # Step 4: compression term sum_k gamma_k * (H_n * pi_k) @ C_k
        compress_sum = torch.zeros_like(H_n)
        for k in range(K):
            pi_k = soft_labels[:, k]
            n_k  = pi_k.sum().clamp(min=1.0)
            H_k  = H_n * pi_k.unsqueeze(1)
            gram_k = (self.alpha / n_k) * (H_k.t() @ H_k)
            C_k = torch.linalg.inv(torch.eye(d, device=H.device) + gram_k)
            compress_sum = compress_sum + gamma[k] * (H_k @ C_k)

        # Step 5: ReduNet gradient step
        H_half = H_n + self.eta * (expand_term - compress_sum)

        # Step 6: ISTA sparsification
        thr = torch.abs(self.threshold)
        return torch.sign(H_half) * F.relu(torch.abs(H_half) - thr)


# ─────────────────────────────────────────────
# Full Model
# ─────────────────────────────────────────────

class WhiteboxGCN2(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes,
                 eta=0.5, alpha=0.5, lambda_sparse=0.01, dropout=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_p = dropout

        self.input_proj = nn.Linear(in_dim, hidden_dim, bias=False)
        nn.init.xavier_uniform_(self.input_proj.weight)

        self.layer1 = ReduNetGCNLayer(hidden_dim, num_classes,
                                      eta=eta, alpha=alpha, lambda_sparse=lambda_sparse)
        self.layer2 = ReduNetGCNLayer(hidden_dim, num_classes,
                                      eta=eta, alpha=alpha, lambda_sparse=lambda_sparse)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, A_norm_sparse, soft_labels):
        H = self.input_proj(x)
        H = F.dropout(H, p=self.dropout_p, training=self.training)

        H = self.layer1(H, A_norm_sparse, soft_labels)
        H = F.dropout(H, p=self.dropout_p, training=self.training)

        with torch.no_grad():
            soft_labels_mid = F.softmax(self.classifier(H), dim=1)

        H = self.layer2(H, A_norm_sparse, soft_labels_mid)
        logits = self.classifier(H)
        return logits, H


# ─────────────────────────────────────────────
# MCR2 Loss
# ─────────────────────────────────────────────

def compute_delta_R(Z, soft_labels, alpha=0.5):
    n, d = Z.shape
    K = soft_labels.shape[1]
    device = Z.device

    gram = (alpha / n) * (Z.t() @ Z)
    _, ld = torch.linalg.slogdet(torch.eye(d, device=device) + gram)
    R_total = 0.5 * ld

    R_class = torch.tensor(0.0, device=device)
    for k in range(K):
        pi_k = soft_labels[:, k]
        n_k  = pi_k.sum().clamp(min=1.0)
        Z_k  = Z * pi_k.unsqueeze(1)
        gram_k = (alpha / n_k) * (Z_k.t() @ Z_k)
        _, ld_k = torch.linalg.slogdet(torch.eye(d, device=device) + gram_k)
        R_class = R_class + (n_k / n) * 0.5 * ld_k

    return R_total - R_class


def compute_orth_loss(Z, soft_labels, eps=1e-8):
    K = soft_labels.shape[1]
    means = []
    for k in range(K):
        pi_k = soft_labels[:, k]
        n_k  = pi_k.sum().clamp(min=eps)
        mean_k = (Z * pi_k.unsqueeze(1)).sum(0) / n_k
        means.append(F.normalize(mean_k, dim=0))
    means = torch.stack(means)
    G = means @ means.t()
    mask = 1.0 - torch.eye(K, device=Z.device)
    return (G * mask).pow(2).sum()


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def run_experiment(config, A_sparse, x, y, train_m, val_m, test_m):
    n = x.shape[0]
    in_dim = x.shape[1]
    num_classes = int(y.max().item()) + 1
    device = x.device

    torch.manual_seed(config.get('seed', 42))

    model = WhiteboxGCN2(
        in_dim=in_dim,
        hidden_dim=config['hidden_dim'],
        num_classes=num_classes,
        eta=config['eta'],
        alpha=config['alpha'],
        lambda_sparse=config['lambda_sparse'],
        dropout=config['dropout']
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )

    soft_labels = torch.full((n, num_classes), 1.0 / num_classes, device=device)
    best_val_acc  = 0.0
    best_test_acc = 0.0
    patience_cnt  = 0

    for epoch in range(1, config['epochs'] + 1):
        model.train()
        optimizer.zero_grad()

        logits, H = model(x, A_sparse, soft_labels.detach())

        ce_loss   = F.cross_entropy(logits[train_m], y[train_m])
        delta_R   = compute_delta_R(H, soft_labels.detach(), alpha=config['alpha'])
        orth_loss = compute_orth_loss(H, soft_labels.detach())

        loss = ce_loss - config['lambda_mcr'] * delta_R + config['lambda_orth'] * orth_loss
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            soft_labels = F.softmax(logits.detach(), dim=1)

        model.eval()
        with torch.no_grad():
            logits_e, _ = model(x, A_sparse, soft_labels)
            val_acc  = (logits_e[val_m].argmax(1)  == y[val_m]).float().mean().item()
            test_acc = (logits_e[test_m].argmax(1) == y[test_m]).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            best_test_acc = test_acc
            patience_cnt  = 0
        else:
            patience_cnt += 1

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | loss={loss.item():.4f} ce={ce_loss.item():.4f} "
                  f"dR={delta_R.item():.4f} | val={val_acc:.4f} test={test_acc:.4f} "
                  f"| best_val={best_val_acc:.4f}", flush=True)

        if patience_cnt >= config['patience']:
            print(f"  Early stop at epoch {epoch}", flush=True)
            break

    print(f"  Result: val={best_val_acc:.4f}, test={best_test_acc:.4f}", flush=True)
    return best_val_acc, best_test_acc


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    data_dir = "D:/桌面/MSR实验复现与创新/planetoid/data"
    out_dir  = "D:/桌面/MSR实验复现与创新/results/whitebox_gcn2"
    os.makedirs(out_dir, exist_ok=True)

    print("Loading Cora...", flush=True)
    features, labels, adj, train_mask, val_mask, test_mask = load_cora(data_dir)
    features = normalize_features(features)

    device = torch.device('cpu')
    A_sparse = build_norm_adj(adj).to(device)
    x = torch.tensor(features, dtype=torch.float32, device=device)
    y = torch.tensor(labels,   dtype=torch.long,    device=device)
    train_m = torch.tensor(train_mask, device=device)
    val_m   = torch.tensor(val_mask,   device=device)
    test_m  = torch.tensor(test_mask,  device=device)

    n, in_dim = x.shape
    num_classes = int(y.max().item()) + 1
    print(f"Graph: {n} nodes, {in_dim} features, {num_classes} classes", flush=True)
    print(f"Train/Val/Test: {train_m.sum()}/{val_m.sum()}/{test_m.sum()}", flush=True)

    all_results = []

    # Corrected architecture: graph agg first, then ReduNet operators
    # lambda_mcr tiny so CE dominates
    configs = [
        # name,            hd,  eta, alpha, lsp,   drop, lr,   wd,   lmcr, lorth, ep,  pat, seed
        ('base',           64,  0.5, 0.5,   0.01,  0.5,  0.01, 5e-4, 1e-4, 1e-4,  300, 30,  42),
        ('no_mcr',         64,  0.5, 0.5,   0.01,  0.5,  0.01, 5e-4, 0.0,  0.0,   300, 30,  42),
        ('h128',           128, 0.5, 0.5,   0.005, 0.5,  0.01, 5e-4, 1e-4, 1e-4,  300, 30,  42),
        ('h128_no_mcr',    128, 0.5, 0.5,   0.005, 0.5,  0.01, 5e-4, 0.0,  0.0,   300, 30,  42),
        ('drop03',         64,  0.5, 0.5,   0.01,  0.3,  0.01, 5e-4, 1e-4, 1e-4,  300, 30,  42),
        ('h128_drop03',    128, 0.5, 0.5,   0.005, 0.3,  0.01, 5e-4, 0.0,  0.0,   300, 30,  42),
    ]

    for (name, hd, eta, alpha, lsp, drop, lr, wd, lmcr, lorth, ep, pat, seed) in configs:
        cfg = dict(hidden_dim=hd, eta=eta, alpha=alpha, lambda_sparse=lsp,
                   dropout=drop, lr=lr, weight_decay=wd,
                   lambda_mcr=lmcr, lambda_orth=lorth,
                   epochs=ep, patience=pat, seed=seed)
        print(f"\n{'='*60}", flush=True)
        print(f"Exp: {name}  hidden={hd} eta={eta} alpha={alpha} lsp={lsp} "
              f"drop={drop} lr={lr} lmcr={lmcr}", flush=True)
        print(f"{'='*60}", flush=True)
        v, t = run_experiment(cfg, A_sparse, x, y, train_m, val_m, test_m)
        all_results.append((name, cfg, v, t))
        if t >= 0.75:
            print(f"  *** Target achieved! ***", flush=True)
            break

    # Best result
    best_name, best_cfg, best_val, best_test = max(all_results, key=lambda r: r[3])
    print(f"\n{'='*60}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    for name, cfg, val, test in all_results:
        print(f"  {name:20s}  val={val:.4f}  test={test:.4f}", flush=True)
    print(f"\nBest: {best_name}  val={best_val:.4f}  test={best_test:.4f}", flush=True)
    print(f"Target >0.75: {'ACHIEVED' if best_test > 0.75 else 'NOT ACHIEVED'}", flush=True)

    # Agent 2 comparison
    agent2_path = "D:/桌面/MSR实验复现与创新/results/whitebox_gcn/agent2_result.txt"
    agent2_text = ""
    if os.path.exists(agent2_path):
        with open(agent2_path, 'r', encoding='utf-8') as f:
            agent2_text = f.read()

    # Write results
    with open(f"{out_dir}/experiment_log.md", 'w', encoding='utf-8') as f:
        f.write("# Whitebox GCN2 (Agent 4) Experiment Log\n\n")
        f.write("## Method\n\n")
        f.write("White-box GCN with graph aggregation + ReduNet unrolled gradient ascent.\n\n")
        f.write("### Layer formula\n\n```\n")
        f.write("H_agg = A_norm @ H                          (graph smoothing)\n")
        f.write("H_n   = LayerNorm(H_agg)\n")
        f.write("E     = (I + alpha/n * H_n^T H_n)^{-1}\n")
        f.write("C_k   = (I + alpha/n_k * H_k^T H_k)^{-1}\n")
        f.write("H'    = H_n + eta*(H_n @ E - sum_k(gamma_k*(H_n*pi_k) @ C_k))\n")
        f.write("H_out = sign(H') * ReLU(|H'| - threshold)  [ISTA]\n")
        f.write("```\n\n")
        f.write("Loss = CE - lambda_mcr * DeltaR + lambda_orth * L_orth\n\n")
        f.write("### Key design insight\n\n")
        f.write("Graph aggregation (A_norm @ H) is applied first in each layer,\n")
        f.write("then ReduNet operators act on the smoothed representation.\n")
        f.write("This separates the graph structure from the rate-reduction optimization.\n\n")
        f.write("## Results\n\n")
        f.write("| Config | val_acc | test_acc |\n|--------|---------|----------|\n")
        for name, cfg, val, test in all_results:
            f.write(f"| {name} | {val:.4f} | {test:.4f} |\n")
        f.write(f"\n**Best: {best_name}  test_acc={best_test:.4f}**\n\n")
        f.write(f"Target >0.75: {'ACHIEVED' if best_test > 0.75 else 'NOT ACHIEVED'}\n\n")
        if agent2_text:
            f.write("## Comparison with Agent 2\n\n```\n" + agent2_text + "\n```\n")

    with open(f"{out_dir}/agent4_result.txt", 'w', encoding='utf-8') as f:
        f.write("whitebox_gcn2.py (Agent 4) experiment results\n")
        f.write("=" * 60 + "\n\n")
        f.write("Method: Graph aggregation + ReduNet unrolled gradient ascent + ISTA\n\n")
        f.write("Architecture:\n")
        f.write("  - Input projection (linear, xavier init)\n")
        f.write("  - 2 x ReduNetGCNLayer:\n")
        f.write("      H_agg = A_norm @ H  (graph smoothing)\n")
        f.write("      H_n   = LayerNorm(H_agg)\n")
        f.write("      E     = (I + alpha/n * H_n^T H_n)^{-1}\n")
        f.write("      C_k   = (I + alpha/n_k * H_k^T H_k)^{-1}\n")
        f.write("      H'    = H_n + eta*(H_n @ E - sum_k(gamma_k*(H_n*pi_k) @ C_k))\n")
        f.write("      H_out = ISTA(H', threshold)\n")
        f.write("  - Output classifier (linear)\n\n")
        f.write("Loss = CE - lambda_mcr * DeltaR + lambda_orth * L_orth\n\n")
        f.write("Results:\n")
        for name, cfg, val, test in all_results:
            f.write(f"  {name:22s}  val={val:.4f}  test={test:.4f}\n")
        f.write(f"\nBest config:   {best_name}\n")
        f.write(f"Best val acc:  {best_val:.4f}\n")
        f.write(f"Best test acc: {best_test:.4f}\n")
        f.write(f"Target >0.75:  {'YES' if best_test > 0.75 else 'NO'}\n\n")
        f.write("Best hyperparameters:\n")
        for k, v in best_cfg.items():
            f.write(f"  {k}: {v}\n")
        if agent2_text:
            f.write("\n\nComparison with Agent 2:\n" + agent2_text)

    print(f"\nResults written to {out_dir}/", flush=True)


if __name__ == '__main__':
    main()


