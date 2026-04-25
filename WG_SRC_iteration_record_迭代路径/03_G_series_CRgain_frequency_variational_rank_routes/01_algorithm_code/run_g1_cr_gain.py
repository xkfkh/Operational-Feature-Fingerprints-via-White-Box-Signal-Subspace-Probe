"""
G1: Coding Rate Gain Driven Hop Weights (White-box GNN)

Core idea: w_k ∝ ΔR_k = R(Z_k) - R(Z_{k-1})
Hop weights are derived directly from the objective function (coding rate marginal gain),
not independent learnable parameters. Only tau (temperature) is learned.

Layer forward pass (proximal gradient unrolling):
  1. Compute per-hop features: Z_k = A^k H W  (k=0..K)
  2. Compute coding rate R_k for each hop, then marginal gain ΔR_k = R_k - R_{k-1}
  3. w_k = softmax(ΔR_k / tau)  -- tau is learnable temperature
  4. Gradient step: H_half = H + eta * sum_k w_k * grad_R_k - eta * lambda_lap * L @ H
  5. Proximal step: soft_threshold + LayerNorm
"""

import os, sys, pickle, time
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

torch.manual_seed(42)
np.random.seed(42)

DATA_DIR = "D:/桌面/MSR实验复现与创新/planetoid/data"
OUT_DIR  = "D:/桌面/MSR实验复现与创新/results/whitebox_gat_v3"
os.makedirs(OUT_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(OUT_DIR, "g1_output.txt")
RESULT_PATH = os.path.join(OUT_DIR, "g1_result.txt")

output_lines = []

def log(msg):
    print(msg, flush=True)
    output_lines.append(str(msg))

def save_output():
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))

# ── Data Loading ──────────────────────────────────────────────────────────────

def load_cora(data_dir):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objs = []
    for n in names:
        with open(f"{data_dir}/ind.cora.{n}", 'rb') as f:
            objs.append(pickle.load(f, encoding='latin1'))
    x, y, tx, ty, allx, ally, graph = objs
    test_idx_raw    = [int(l.strip()) for l in open(f"{data_dir}/ind.cora.test.index")]
    test_idx_sorted = sorted(test_idx_raw)
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_raw, :] = features[test_idx_sorted, :]
    features = np.array(features.todense(), dtype=np.float32)
    labels = np.vstack((ally, ty))
    labels[test_idx_raw, :] = labels[test_idx_sorted, :]
    labels = np.argmax(labels, axis=1)
    n = features.shape[0]
    adj = np.zeros((n, n), dtype=np.float32)
    for node, neighbors in graph.items():
        for nb in neighbors:
            adj[node, nb] = 1.0
            adj[nb, node] = 1.0
    np.fill_diagonal(adj, 1.0)
    deg = adj.sum(axis=1)
    d_inv_sqrt = np.where(deg > 0, np.power(deg, -0.5), 0.0)
    adj_norm = d_inv_sqrt[:, None] * adj * d_inv_sqrt[None, :]  # D^{-1/2} A D^{-1/2}
    lap = np.eye(n, dtype=np.float32) - adj_norm                # L = I - adj_norm
    train_idx = list(range(140))
    val_idx   = list(range(200, 500))
    test_idx  = test_idx_sorted[:1000]
    return features, labels, adj, adj_norm, lap, train_idx, val_idx, test_idx

# ── Coding Rate ───────────────────────────────────────────────────────────────

def coding_rate(Z, eps=0.5):
    """
    R(Z) = 0.5 * log det(I + d/(n*eps^2) * Z^T Z)
    Z: [n, d]
    Returns scalar tensor.
    """
    n, d = Z.shape
    coeff = d / (n * eps * eps)
    # Use d x d form: 0.5 * logdet(I_d + coeff * Z^T Z)
    M = torch.eye(d, device=Z.device) + coeff * (Z.t() @ Z)
    sign, logdet = torch.linalg.slogdet(M)
    return 0.5 * logdet

def coding_rate_gradient(Z, eps=0.5):
    """
    dR/dZ = coeff * Z * (I + coeff * Z^T Z)^{-1}
    Z: [n, d]  returns [n, d]
    """
    n, d = Z.shape
    coeff = d / (n * eps * eps)
    M = torch.eye(d, device=Z.device) + coeff * (Z.t() @ Z)
    M_inv = torch.linalg.inv(M)
    return coeff * (Z @ M_inv)

# ── CRGainHopLayer ────────────────────────────────────────────────────────────

class CRGainHopLayer(nn.Module):
    """
    One layer of CR-Gain driven hop aggregation.

    For each hop k in {0, 1, ..., K}:
      Z_k = A^k H W_k   (projected hop-k features)
      R_k = coding_rate(Z_k); ΔR_k = R_k - R_{k-1}  (ΔR_0 = R_0)
      w_k = softmax(ΔR_k / tau)

    Gradient step:
      H_half = H + eta * sum_k w_k * grad_R(Z_k) @ W_k^T
                 - eta * lambda_lap * L @ H

    Proximal step:
      H_out = LayerNorm(soft_threshold(H_half, threshold))
    """
    def __init__(self, in_dim, out_dim, subspace_dim, num_hops=2,
                 eta=0.5, eps=0.5, lambda_lap=0.3, lambda_sparse=0.05,
                 tau_init=1.0):
        super().__init__()
        self.in_dim       = in_dim
        self.out_dim      = out_dim
        self.subspace_dim = subspace_dim
        self.num_hops     = num_hops
        self.eta          = eta
        self.eps          = eps
        self.lambda_lap   = lambda_lap
        self.lambda_sparse = lambda_sparse

        # Learnable temperature (only learned parameter for hop weighting)
        self.log_tau = nn.Parameter(torch.tensor(float(tau_init)).log())

        # Per-hop projection matrices W_k: in_dim -> subspace_dim
        self.W = nn.ModuleList([
            nn.Linear(in_dim, subspace_dim, bias=False)
            for _ in range(num_hops + 1)
        ])

        # Output projection: subspace_dim -> out_dim
        # FIX: input must be in_dim (H_half lives in H-space), not subspace_dim
        self.out_proj = nn.Linear(in_dim, out_dim, bias=False)

        # Learnable soft-threshold
        self.threshold = nn.Parameter(torch.full((out_dim,), lambda_sparse))

        # LayerNorm
        self.ln = nn.LayerNorm(out_dim)

        self.dropout = nn.Dropout(p=0.0)  # set externally if needed

    @property
    def tau(self):
        return self.log_tau.exp()

    def forward(self, H, hop_feats, L):
        """
        H:         [n, in_dim]  current node features
        hop_feats: list of K+1 tensors [n, in_dim], hop_feats[k] = A^k H
        L:         [n, n] normalized Laplacian
        Returns:   H_out [n, out_dim], hop_weights [K+1], coding_rates [K+1]
        """
        tau = self.tau
        K = self.num_hops

        # Step 1: project each hop
        Z_list = [self.W[k](hop_feats[k]) for k in range(K + 1)]  # each [n, subspace_dim]

        # Step 2: compute coding rate for each hop, then marginal gain
        R_list = [coding_rate(Z_list[k], self.eps) for k in range(K + 1)]
        # delta_R_k = R_k - R_{k-1}; delta_R_0 = R_0 (baseline)
        delta_R_list = []
        for k in range(K + 1):
            if k == 0:
                delta_R_list.append(R_list[k])          # delta_R_0 = R_0
            else:
                delta_R_list.append(R_list[k] - R_list[k - 1])  # marginal gain
        delta_R_tensor = torch.stack(delta_R_list)  # [K+1]

        # Step 3: hop weights via softmax over delta_R/tau  (true ΔR weighting)
        w = F.softmax(delta_R_tensor / tau, dim=0)  # [K+1]

        # Step 4: gradient step
        # grad_R contribution: sum_k w_k * grad_R(Z_k) @ W_k^T  -> [n, in_dim]
        grad_contrib = torch.zeros_like(H)

        for k in range(K + 1):
            g_Zk = coding_rate_gradient(Z_list[k], self.eps)  # [n, subspace_dim]
            # back-project to in_dim space
            g_H = g_Zk @ self.W[k].weight                     # [n, in_dim]
            grad_contrib = grad_contrib + w[k] * g_H

        H_half = H + self.eta * grad_contrib - self.eta * self.lambda_lap * (L @ H)

        # Step 5: dimension alignment in_dim -> out_dim, then proximal
        # FIX: out_proj receives H_half [n, in_dim], the true whitebox gradient step result.
        #      Deleted dead-code block (Z_agg path) that was overwriting H_half_in.
        if self.in_dim == self.out_dim:
            H_out_pre = H_half
        else:
            H_out_pre = self.out_proj(H_half)

        # Step 6: proximal operator — soft threshold + LayerNorm
        thr = self.threshold.abs().unsqueeze(0)                # [1, out_dim]
        H_soft = H_out_pre.sign() * F.relu(H_out_pre.abs() - thr)
        H_out  = self.ln(H_soft)

        return H_out, w.detach(), delta_R_tensor.detach()


# ── Full Model ────────────────────────────────────────────────────────────────

class CRGainGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, subspace_dim,
                 num_hops=2, eta=0.5, eps=0.5,
                 lambda_lap=0.3, lambda_sparse=0.05,
                 tau_init=1.0, dropout=0.6):
        super().__init__()
        self.num_hops = num_hops
        self.dropout  = dropout

        self.layer1 = CRGainHopLayer(
            in_dim, hidden_dim, subspace_dim, num_hops,
            eta, eps, lambda_lap, lambda_sparse, tau_init
        )
        self.layer2 = CRGainHopLayer(
            hidden_dim, num_classes, subspace_dim, num_hops,
            eta, eps, lambda_lap, lambda_sparse, tau_init
        )

    def _precompute_hops(self, H, adj_norm, num_hops):
        """Precompute A^0 H, A^1 H, ..., A^K H"""
        hops = [H]
        cur = H
        for _ in range(num_hops):
            cur = adj_norm @ cur
            hops.append(cur)
        return hops

    def forward(self, H, adj_norm, L):
        # Layer 1
        hops1 = self._precompute_hops(H, adj_norm, self.num_hops)
        H1, w1, R1 = self.layer1(H, hops1, L)
        H1 = F.dropout(H1, p=self.dropout, training=self.training)
        H1 = F.elu(H1)

        # Layer 2
        hops2 = self._precompute_hops(H1, adj_norm, self.num_hops)
        H2, w2, R2 = self.layer2(H1, hops2, L)

        return H2, (w1, w2), (R1, R2)


# ── MCR2 Loss ─────────────────────────────────────────────────────────────────

def mcr2_loss(Z, y, num_classes, eps=0.5):
    """MCR2: maximize Delta_R = R(Z) - mean_k R(Z_k)"""
    R_total = coding_rate(Z, eps)
    R_class_sum = 0.0
    for c in range(num_classes):
        mask = (y == c)
        if mask.sum() < 2:
            continue
        Zc = Z[mask]
        R_class_sum = R_class_sum + coding_rate(Zc, eps)
    delta_R = R_total - R_class_sum / num_classes
    return -delta_R  # minimize negative delta_R

def orth_loss(Z, y, num_classes):
    """Encourage class subspaces to be orthogonal"""
    means = []
    for c in range(num_classes):
        mask = (y == c)
        if mask.sum() < 1:
            means.append(torch.zeros(Z.shape[1], device=Z.device))
        else:
            means.append(Z[mask].mean(0))
    M = torch.stack(means)  # [C, d]
    M = F.normalize(M, dim=1)
    gram = M @ M.t()        # [C, C]
    eye  = torch.eye(num_classes, device=Z.device)
    return (gram - eye).pow(2).sum() / (num_classes * num_classes)


# ── Training ──────────────────────────────────────────────────────────────────

def run_experiment(name, cfg):
    log(f"\n{'='*60}")
    log(f"Experiment: {name}")
    log(f"Config: {cfg}")
    log(f"{'='*60}")

    torch.manual_seed(cfg.get('seed', 42))
    np.random.seed(cfg.get('seed', 42))

    # Load data
    features, labels, adj, adj_norm, lap, train_idx, val_idx, test_idx = load_cora(DATA_DIR)
    n, in_dim = features.shape
    num_classes = int(labels.max()) + 1

    device = torch.device('cpu')
    X       = torch.tensor(features, dtype=torch.float32, device=device)
    Y       = torch.tensor(labels,   dtype=torch.long,    device=device)
    A_norm  = torch.tensor(adj_norm, dtype=torch.float32, device=device)
    L_mat   = torch.tensor(lap,      dtype=torch.float32, device=device)

    train_mask = torch.zeros(n, dtype=torch.bool, device=device)
    val_mask   = torch.zeros(n, dtype=torch.bool, device=device)
    test_mask  = torch.zeros(n, dtype=torch.bool, device=device)
    train_mask[train_idx] = True
    val_mask[val_idx]     = True
    test_mask[test_idx]   = True

    model = CRGainGNN(
        in_dim      = in_dim,
        hidden_dim  = cfg['hidden_dim'],
        num_classes = num_classes,
        subspace_dim= cfg['subspace_dim'],
        num_hops    = cfg['num_hops'],
        eta         = cfg['eta'],
        eps         = cfg['eps'],
        lambda_lap  = cfg['lambda_lap'],
        lambda_sparse= cfg['lambda_sparse'],
        tau_init    = cfg['tau_init'],
        dropout     = cfg['dropout'],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd']
    )

    best_val_acc = 0.0
    best_test_acc = 0.0
    patience_cnt  = 0
    epochs        = cfg['epochs']
    patience      = cfg['patience']

    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits, (w1, w2), (R1, R2) = model(X, A_norm, L_mat)

        # Cross-entropy loss
        ce = F.cross_entropy(logits[train_mask], Y[train_mask])

        # MCR2 loss on final layer output (train nodes only, avoid label leakage)
        mcr = mcr2_loss(logits[train_mask], Y[train_mask], num_classes, eps=cfg['eps'])

        # Orthogonality loss (train nodes only, avoid label leakage)
        orth = orth_loss(logits[train_mask], Y[train_mask], num_classes)

        loss = ce + cfg['lambda_mcr'] * mcr + cfg['lambda_orth'] * orth
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            logits_e, (w1e, w2e), (R1e, R2e) = model(X, A_norm, L_mat)
            pred = logits_e.argmax(dim=1)
            val_acc  = (pred[val_mask]  == Y[val_mask]).float().mean().item()
            test_acc = (pred[test_mask] == Y[test_mask]).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            best_test_acc = test_acc
            patience_cnt  = 0
        else:
            patience_cnt += 1

        if epoch % 50 == 0:
            w1_str = '[' + ', '.join(f'{v:.3f}' for v in w1e.tolist()) + ']'
            w2_str = '[' + ', '.join(f'{v:.3f}' for v in w2e.tolist()) + ']'
            R1_str = '[' + ', '.join(f'{v:.3f}' for v in R1e.tolist()) + ']'
            R2_str = '[' + ', '.join(f'{v:.3f}' for v in R2e.tolist()) + ']'
            tau1 = model.layer1.tau.item()
            tau2 = model.layer2.tau.item()
            log(f"Epoch {epoch:4d} | loss={loss.item():.4f} | "
                f"val={val_acc:.4f} | test={test_acc:.4f}")
            log(f"  Layer1 tau={tau1:.3f} | hop_w={w1_str} | dR={R1_str}")
            log(f"  Layer2 tau={tau2:.3f} | hop_w={w2_str} | dR={R2_str}")

        if patience_cnt >= patience:
            log(f"  Early stop at epoch {epoch}")
            break

    elapsed = time.time() - t0
    log(f"  Best val={best_val_acc:.4f} | Best test={best_test_acc:.4f} | "
        f"Time={elapsed:.1f}s")

    return best_val_acc, best_test_acc


# ── Experiment Configs ────────────────────────────────────────────────────────

BASE = dict(
    hidden_dim    = 64,
    subspace_dim  = 16,
    num_hops      = 2,
    eta           = 0.5,
    eps           = 0.5,
    lambda_lap    = 0.3,
    lambda_sparse = 0.05,
    lambda_mcr    = 0.005,
    lambda_orth   = 0.005,
    dropout       = 0.6,
    lr            = 0.005,
    wd            = 1e-3,
    epochs        = 400,
    patience      = 50,
    seed          = 42,
    tau_init      = 1.0,
)

def merge(base, **kwargs):
    d = dict(base)
    d.update(kwargs)
    return d

EXPERIMENTS = [
    ('g1_base',      merge(BASE)),
    ('g1_tau_high',  merge(BASE, tau_init=2.0)),
    ('g1_tau_low',   merge(BASE, tau_init=0.5)),
    ('g1_hidden128', merge(BASE, hidden_dim=128, subspace_dim=32)),
    ('g1_hop3',      merge(BASE, num_hops=3)),
    ('g1_best',      merge(BASE, hidden_dim=128, subspace_dim=32, num_hops=2)),
]

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    log("G1: Coding Rate Gain Driven Hop Weights")
    log(f"Date: 2026-04-08")
    log(f"Output dir: {OUT_DIR}")

    results = []
    for name, cfg in EXPERIMENTS:
        val_acc, test_acc = run_experiment(name, cfg)
        results.append((name, val_acc, test_acc))
        save_output()

    # Summary table
    log(f"\n{'='*60}")
    log("SUMMARY TABLE")
    log(f"{'='*60}")
    log(f"{'Experiment':<20} {'Val Acc':>10} {'Test Acc':>10}")
    log(f"{'-'*42}")
    best_name, best_val, best_test = '', 0.0, 0.0
    for name, val_acc, test_acc in results:
        marker = ' <-- best' if val_acc == max(r[1] for r in results) else ''
        log(f"{name:<20} {val_acc:>10.4f} {test_acc:>10.4f}{marker}")
        if val_acc > best_val:
            best_val, best_test, best_name = val_acc, test_acc, name

    log(f"\nBest config: {best_name}")
    log(f"  Val Acc:  {best_val:.4f}")
    log(f"  Test Acc: {best_test:.4f}")

    # Save result summary
    result_lines = [
        "G1 Experiment Results",
        "=" * 40,
        f"{'Experiment':<20} {'Val':>8} {'Test':>8}",
        "-" * 40,
    ]
    for name, val_acc, test_acc in results:
        result_lines.append(f"{name:<20} {val_acc:>8.4f} {test_acc:>8.4f}")
    result_lines += [
        "-" * 40,
        f"Best: {best_name}  val={best_val:.4f}  test={best_test:.4f}",
    ]
    with open(RESULT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(result_lines))
    log(f"\nResults saved to {RESULT_PATH}")

    save_output()
    log("Done.")


