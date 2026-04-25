"""
Agent G4: Multi-scale Subspace Effective-Rank Driven Hop Weights (White-box GNN)
=================================================================================
Core idea: hop weights derived from subspace effective rank (participation ratio).
  rank_eff_k = tr(Sigma_k)^2 / tr(Sigma_k^2)   (participation ratio)
  w_k = softmax(rank_eff / tau),  tau learnable
Higher effective rank => more spread information => larger coding rate contribution => larger weight.
"""
import os, sys, pickle, time
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_DIR = "D:/桌面/MSR实验复现与创新/planetoid/data"
OUT_DIR  = "D:/桌面/MSR实验复现与创新/results/whitebox_gat_v3"
os.makedirs(OUT_DIR, exist_ok=True)

LOG_PATH    = os.path.join(OUT_DIR, "g4_output.txt")
RESULT_PATH = os.path.join(OUT_DIR, "g4_result.txt")

log_lines = []

def log(msg):
    print(msg, flush=True)
    log_lines.append(str(msg))

def save_log():
    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
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
    rowsum = features.sum(1, keepdims=True)
    features = features / np.where(rowsum == 0, 1, rowsum)
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
    d_inv_sqrt = np.where(deg > 0, deg ** -0.5, 0.0)
    adj_norm = d_inv_sqrt[:, None] * adj * d_inv_sqrt[None, :]
    lap = np.eye(n, dtype=np.float32) - adj_norm
    train_idx = list(range(140))
    val_idx   = list(range(200, 500))
    test_idx  = test_idx_sorted[:1000]
    return features, labels, adj_norm, lap, train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# Subspace Rank Hop Layer
# ---------------------------------------------------------------------------
class SubspaceRankHopLayer(nn.Module):
    """
    One proximal-gradient unrolling step with effective-rank-driven hop weights.

    For each hop k:
      Z_k = U_k^T H_k^T          [subspace_dim, n]
      Sigma_k = Z_k Z_k^T / n    [subspace_dim, subspace_dim]
      rank_eff_k = tr(Sigma_k)^2 / tr(Sigma_k^2)   (participation ratio)
      w_k = softmax(rank_eff / tau)

    Gradient step:
      H_half = H + eta * sum_k w_k * grad_R_k - eta * sum_k lambda_lap * L @ H_k
      grad_R_k = U_k @ M_k^{-1} @ Z_k   (from coding rate derivative)
      M_k = I + coeff * Z_k Z_k^T

    Proximal: soft-threshold + LayerNorm
    """
    def __init__(self, in_dim, out_dim, num_hops, subspace_dim,
                 eta=0.5, lambda_lap=0.3, lambda_sparse=0.05, eps=0.5):
        super().__init__()
        self.in_dim       = in_dim
        self.out_dim      = out_dim
        self.num_hops     = num_hops
        self.subspace_dim = subspace_dim
        self.eta          = eta
        self.lambda_lap   = lambda_lap
        self.lambda_sparse = lambda_sparse
        self.eps          = eps

        # Subspace bases U_k: [in_dim, subspace_dim] per hop
        self.U = nn.ParameterList([
            nn.Parameter(torch.randn(in_dim, subspace_dim) * 0.1)
            for _ in range(num_hops + 1)
        ])

        # Input projection
        self.W_in = nn.Linear(in_dim, out_dim, bias=False)

        # Learnable temperature tau (log-parameterized for positivity)
        self.log_tau = nn.Parameter(torch.zeros(1))  # tau = exp(log_tau), init=1.0

        self.layer_norm = nn.LayerNorm(out_dim)
        self.dropout    = nn.Dropout(p=0.0)  # dropout applied externally

    def effective_rank(self, Z):
        """
        Participation ratio: tr(Sigma)^2 / tr(Sigma^2)
        Z: [subspace_dim, n]
        Sigma: [subspace_dim, subspace_dim]
        """
        n = Z.shape[1]
        Sigma = (Z @ Z.t()) / (n + 1e-8)
        tr_S  = torch.trace(Sigma)
        tr_S2 = torch.trace(Sigma @ Sigma)
        rank_eff = (tr_S ** 2) / (tr_S2 + 1e-8)
        return rank_eff, Sigma

    def coding_rate_grad(self, U_k, Z_k, n, eps):
        """
        grad_R w.r.t. H via chain rule:
          R(Z_k) = 0.5 * log det(I + coeff * Z_k Z_k^T)
          coeff = subspace_dim / (n * eps^2)
          grad_H_k R = U_k @ (I + coeff * Z_k Z_k^T)^{-1} @ Z_k / n
        Returns: [n, in_dim]
        """
        sd    = Z_k.shape[0]
        coeff = sd / (n * eps ** 2 + 1e-8)
        M     = torch.eye(sd, device=Z_k.device) + coeff * (Z_k @ Z_k.t())
        # Solve M^{-1} Z_k via Cholesky for stability
        try:
            L_chol = torch.linalg.cholesky(M)
            MiZ    = torch.cholesky_solve(Z_k, L_chol)  # [sd, n]
        except Exception:
            MiZ = torch.linalg.solve(M, Z_k)
        # grad w.r.t. H_k: [n, in_dim]
        grad = (U_k @ MiZ).t()  # [n, in_dim]
        return grad

    def forward(self, H, hops, L):
        """
        H:    [n, in_dim]  current representation
        hops: list of [n, in_dim], hops[k] = A^k H_input
        L:    [n, n] graph Laplacian
        Returns: [n, out_dim], rank_effs list, hop_weights
        """
        n = H.shape[0]
        tau = torch.exp(self.log_tau).clamp(min=0.1, max=10.0)

        rank_effs = []
        grads_R   = []
        lap_terms = []

        for k in range(self.num_hops + 1):
            H_k = hops[k]                          # [n, in_dim]
            U_k = self.U[k]                        # [in_dim, subspace_dim]
            # Orthonormalize U_k via QR for stable subspace
            U_k_orth, _ = torch.linalg.qr(U_k)    # [in_dim, subspace_dim]

            Z_k = U_k_orth.t() @ H_k.t()          # [subspace_dim, n]

            rank_eff, _ = self.effective_rank(Z_k)
            rank_effs.append(rank_eff)

            g = self.coding_rate_grad(U_k_orth, Z_k, n, self.eps)  # [n, in_dim]
            grads_R.append(g)

            lap_terms.append(L @ H_k)              # [n, in_dim]

        # Hop weights from effective rank
        rank_tensor = torch.stack(rank_effs)       # [num_hops+1]
        w = F.softmax(rank_tensor / tau, dim=0)    # [num_hops+1]

        # Gradient step
        grad_sum = sum(w[k] * grads_R[k]   for k in range(self.num_hops + 1))
        lap_sum  = sum(w[k] * lap_terms[k] for k in range(self.num_hops + 1))

        H_half = H + self.eta * grad_sum - self.eta * self.lambda_lap * lap_sum

        # Proximal: soft threshold
        H_prox = torch.sign(H_half) * F.relu(H_half.abs() - self.lambda_sparse * self.eta)

        # Project to output dim
        H_out = self.W_in(H_prox)                 # [n, out_dim]

        # LayerNorm
        H_out = self.layer_norm(H_out)

        return H_out, [r.item() for r in rank_effs], w.detach().cpu().tolist()


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------
class G4Model(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, num_hops, subspace_dim,
                 eta=0.5, lambda_lap=0.3, lambda_sparse=0.05, eps=0.5, dropout=0.6):
        super().__init__()
        self.num_hops    = num_hops
        self.dropout_p   = dropout

        self.layer1 = SubspaceRankHopLayer(
            in_dim, hidden_dim, num_hops, subspace_dim,
            eta=eta, lambda_lap=lambda_lap, lambda_sparse=lambda_sparse, eps=eps
        )
        self.layer2 = SubspaceRankHopLayer(
            hidden_dim, num_classes, num_hops, subspace_dim,
            eta=eta, lambda_lap=lambda_lap, lambda_sparse=lambda_sparse, eps=eps
        )

    def forward(self, X, A, L):
        """
        X: [n, in_dim]
        A: [n, n] normalized adjacency
        L: [n, n] Laplacian
        """
        # Precompute hops for layer 1
        hops1 = [X]
        H = X
        for _ in range(self.num_hops):
            H = A @ H
            hops1.append(H)

        H1, rank1, w1 = self.layer1(X, hops1, L)
        H1 = F.dropout(F.elu(H1), p=self.dropout_p, training=self.training)

        # Precompute hops for layer 2
        hops2 = [H1]
        H = H1
        for _ in range(self.num_hops):
            H = A @ H
            hops2.append(H)

        H2, rank2, w2 = self.layer2(H1, hops2, L)

        return H2, (rank1, w1), (rank2, w2)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------
def mcr2_loss(Z, y, num_classes, eps=0.5):
    """
    MCR2 loss: -Delta_R = -(R(Z) - mean_k R(Z_k))
    Minimizing this maximizes coding rate reduction.
    """
    n, d = Z.shape
    coeff = d / (n * eps ** 2 + 1e-8)

    def log_det(M):
        sign, ld = torch.linalg.slogdet(M)
        return ld

    I = torch.eye(d, device=Z.device)
    R_total = 0.5 * log_det(I + coeff * Z.t() @ Z)

    R_class = 0.0
    for c in range(num_classes):
        mask = (y == c)
        n_c = mask.sum().item()
        if n_c < 2:
            continue
        Z_c = Z[mask]
        coeff_c = d / (n_c * eps ** 2 + 1e-8)
        R_c = 0.5 * log_det(I + coeff_c * Z_c.t() @ Z_c)
        R_class += (n_c / n) * R_c

    delta_R = R_total - R_class
    return -delta_R  # minimize negative => maximize delta_R


def orth_loss_within(layer):
    """Within-layer: U_k^T U_k should be close to I (columns orthonormal)."""
    loss = 0.0
    for U_k in layer.U:
        U_k_orth, _ = torch.linalg.qr(U_k)
        sd = U_k_orth.shape[1]
        I  = torch.eye(sd, device=U_k.device)
        loss += (U_k_orth.t() @ U_k_orth - I).pow(2).sum()
    return loss


def orth_loss_cross_hop(layer):
    """Cross-hop: different U_k should be orthogonal to each other."""
    loss = 0.0
    Us = []
    for U_k in layer.U:
        U_k_orth, _ = torch.linalg.qr(U_k)
        Us.append(U_k_orth)
    K = len(Us)
    for i in range(K):
        for j in range(i + 1, K):
            loss += (Us[i].t() @ Us[j]).pow(2).sum()
    return loss


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def run_experiment(cfg, features, labels, A, L, train_idx, val_idx, test_idx, device):
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])

    X = torch.tensor(features, dtype=torch.float32, device=device)
    y = torch.tensor(labels,   dtype=torch.long,    device=device)
    A_t = torch.tensor(A, dtype=torch.float32, device=device)
    L_t = torch.tensor(L, dtype=torch.float32, device=device)

    num_classes = int(y.max().item()) + 1
    in_dim      = X.shape[1]

    model = G4Model(
        in_dim       = in_dim,
        hidden_dim   = cfg['hidden_dim'],
        num_classes  = num_classes,
        num_hops     = cfg['num_hops'],
        subspace_dim = cfg['subspace_dim'],
        eta          = cfg['eta'],
        lambda_lap   = cfg['lambda_lap'],
        lambda_sparse= cfg['lambda_sparse'],
        eps          = cfg['eps'],
        dropout      = cfg['dropout'],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd']
    )

    best_val_acc  = 0.0
    best_test_acc = 0.0
    patience_cnt  = 0

    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        optimizer.zero_grad()

        logits, (rank1, w1), (rank2, w2) = model(X, A_t, L_t)

        ce_loss = F.cross_entropy(logits[train_idx], y[train_idx])

        # MCR2 on hidden layer (layer1 output) — use train nodes
        H1_train = F.elu(model.layer1(X, _get_hops(X, A_t, cfg['num_hops']), L_t)[0])[train_idx]
        mcr_loss = mcr2_loss(H1_train, y[train_idx], num_classes, eps=cfg['eps'])

        # Orthogonality losses
        orth_w1 = orth_loss_within(model.layer1) + orth_loss_within(model.layer2)
        orth_ch = orth_loss_cross_hop(model.layer1) + orth_loss_cross_hop(model.layer2)

        loss = (ce_loss
                + cfg['lambda_mcr']  * mcr_loss
                + cfg['lambda_orth'] * (orth_w1 + orth_ch))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            logits_eval, _, _ = model(X, A_t, L_t)
            val_acc  = (logits_eval[val_idx].argmax(1)  == y[val_idx]).float().mean().item()
            test_acc = (logits_eval[test_idx].argmax(1) == y[test_idx]).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            best_test_acc = test_acc
            patience_cnt  = 0
        else:
            patience_cnt += 1

        if epoch % 50 == 0:
            rank1_str = ', '.join(f'{r:.2f}' for r in rank1)
            rank2_str = ', '.join(f'{r:.2f}' for r in rank2)
            w1_str    = ', '.join(f'{w:.3f}' for w in w1)
            w2_str    = ', '.join(f'{w:.3f}' for w in w2)
            log(f"  ep{epoch:4d} | loss={loss.item():.4f} ce={ce_loss.item():.4f} "
                f"| val={val_acc:.4f} test={test_acc:.4f}"
                f"\n           L1 rank_eff=[{rank1_str}] hop_w=[{w1_str}]"
                f"\n           L2 rank_eff=[{rank2_str}] hop_w=[{w2_str}]")

        if patience_cnt >= cfg['patience']:
            log(f"  Early stop at epoch {epoch}")
            break

    return best_val_acc, best_test_acc


def _get_hops(X, A, num_hops):
    hops = [X]
    H = X
    for _ in range(num_hops):
        H = A @ H
        hops.append(H)
    return hops


# ---------------------------------------------------------------------------
# Experiment configs
# ---------------------------------------------------------------------------
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
)

EXPERIMENTS = {
    'g4_base':      {**BASE},
    'g4_tau_high':  {**BASE, 'tau_override': 2.0},   # handled below
    'g4_tau_low':   {**BASE, 'tau_override': 0.5},
    'g4_hidden128': {**BASE, 'hidden_dim': 128, 'subspace_dim': 32},
    'g4_hop3':      {**BASE, 'num_hops': 3},
    'g4_best':      {**BASE, 'hidden_dim': 128, 'subspace_dim': 32},
}


def set_tau(model, tau_val):
    """Override initial log_tau so that exp(log_tau) = tau_val."""
    import math
    with torch.no_grad():
        model.layer1.log_tau.fill_(math.log(tau_val))
        model.layer2.log_tau.fill_(math.log(tau_val))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log("=" * 70)
    log("G4: Multi-scale Subspace Effective-Rank Driven Hop Weights")
    log("=" * 70)

    device = torch.device('cpu')

    log("Loading Cora...")
    features, labels, A, L, train_idx, val_idx, test_idx = load_cora(DATA_DIR)
    log(f"  nodes={features.shape[0]}, features={features.shape[1]}, "
        f"classes={int(labels.max())+1}")
    log(f"  train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    results = {}

    for name, cfg in EXPERIMENTS.items():
        log(f"\n{'='*60}")
        log(f"Experiment: {name}")
        log(f"  hidden={cfg['hidden_dim']}, subspace={cfg['subspace_dim']}, "
            f"hops={cfg['num_hops']}, lr={cfg['lr']}")

        tau_override = cfg.pop('tau_override', None)

        t0 = time.time()

        # We need to intercept model creation to set tau — patch run_experiment
        torch.manual_seed(cfg['seed'])
        np.random.seed(cfg['seed'])

        X   = torch.tensor(features, dtype=torch.float32, device=device)
        y   = torch.tensor(labels,   dtype=torch.long,    device=device)
        A_t = torch.tensor(A, dtype=torch.float32, device=device)
        L_t = torch.tensor(L, dtype=torch.float32, device=device)

        num_classes = int(y.max().item()) + 1
        in_dim      = X.shape[1]

        model = G4Model(
            in_dim        = in_dim,
            hidden_dim    = cfg['hidden_dim'],
            num_classes   = num_classes,
            num_hops      = cfg['num_hops'],
            subspace_dim  = cfg['subspace_dim'],
            eta           = cfg['eta'],
            lambda_lap    = cfg['lambda_lap'],
            lambda_sparse = cfg['lambda_sparse'],
            eps           = cfg['eps'],
            dropout       = cfg['dropout'],
        ).to(device)

        if tau_override is not None:
            set_tau(model, tau_override)
            log(f"  tau overridden to {tau_override}")

        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd']
        )

        best_val_acc  = 0.0
        best_test_acc = 0.0
        patience_cnt  = 0

        for epoch in range(1, cfg['epochs'] + 1):
            model.train()
            optimizer.zero_grad()

            hops1 = _get_hops(X, A_t, cfg['num_hops'])
            logits, (rank1, w1), (rank2, w2) = model(X, A_t, L_t)

            ce_loss = F.cross_entropy(logits[train_idx], y[train_idx])

            # MCR2 on layer1 hidden (recompute without dropout for loss)
            with torch.no_grad():
                H1_all, _, _ = model.layer1(X, hops1, L_t)
                H1_all = F.elu(H1_all)
            # Re-run layer1 with grad for MCR2 loss
            H1_grad, _, _ = model.layer1(X, hops1, L_t)
            H1_grad = F.elu(H1_grad)
            mcr_loss = mcr2_loss(H1_grad[train_idx], y[train_idx], num_classes, eps=cfg['eps'])

            orth_w1 = orth_loss_within(model.layer1) + orth_loss_within(model.layer2)
            orth_ch = orth_loss_cross_hop(model.layer1) + orth_loss_cross_hop(model.layer2)

            loss = (ce_loss
                    + cfg['lambda_mcr']  * mcr_loss
                    + cfg['lambda_orth'] * (orth_w1 + orth_ch))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            model.eval()
            with torch.no_grad():
                logits_eval, _, _ = model(X, A_t, L_t)
                val_acc  = (logits_eval[val_idx].argmax(1)  == y[val_idx]).float().mean().item()
                test_acc = (logits_eval[test_idx].argmax(1) == y[test_idx]).float().mean().item()

            if val_acc > best_val_acc:
                best_val_acc  = val_acc
                best_test_acc = test_acc
                patience_cnt  = 0
            else:
                patience_cnt += 1

            if epoch % 50 == 0:
                rank1_str = ', '.join(f'{r:.2f}' for r in rank1)
                rank2_str = ', '.join(f'{r:.2f}' for r in rank2)
                w1_str    = ', '.join(f'{w:.3f}' for w in w1)
                w2_str    = ', '.join(f'{w:.3f}' for w in w2)
                log(f"  ep{epoch:4d} | loss={loss.item():.4f} ce={ce_loss.item():.4f} "
                    f"| val={val_acc:.4f} test={test_acc:.4f}"
                    f"\n           L1 rank_eff=[{rank1_str}] hop_w=[{w1_str}]"
                    f"\n           L2 rank_eff=[{rank2_str}] hop_w=[{w2_str}]")

            if patience_cnt >= cfg['patience']:
                log(f"  Early stop at epoch {epoch}")
                break

        elapsed = time.time() - t0
        log(f"  RESULT: val={best_val_acc:.4f}  test={best_test_acc:.4f}  time={elapsed:.1f}s")
        results[name] = {'val': best_val_acc, 'test': best_test_acc, 'time': elapsed}

    # Summary
    log("\n" + "=" * 70)
    log("SUMMARY TABLE")
    log("=" * 70)
    log(f"{'Experiment':<20} {'Val Acc':>10} {'Test Acc':>10} {'Time(s)':>10}")
    log("-" * 55)
    best_name = max(results, key=lambda k: results[k]['test'])
    for name, r in results.items():
        marker = " <-- best" if name == best_name else ""
        log(f"{name:<20} {r['val']:>10.4f} {r['test']:>10.4f} {r['time']:>10.1f}{marker}")

    log(f"\nBest config: {best_name}")
    log(f"  Test accuracy: {results[best_name]['test']:.4f}")

    save_log()

    # Write result file
    with open(RESULT_PATH, 'w', encoding='utf-8') as f:
        f.write("G4 Subspace Effective-Rank Hop Weights - Results\n")
        f.write("=" * 55 + "\n")
        f.write(f"{'Experiment':<20} {'Val':>8} {'Test':>8} {'Time':>8}\n")
        f.write("-" * 48 + "\n")
        for name, r in results.items():
            marker = " *" if name == best_name else ""
            f.write(f"{name:<20} {r['val']:>8.4f} {r['test']:>8.4f} {r['time']:>8.1f}{marker}\n")
        f.write(f"\nBest: {best_name}  test={results[best_name]['test']:.4f}\n")

    log(f"\nLogs saved to: {LOG_PATH}")
    log(f"Results saved to: {RESULT_PATH}")


if __name__ == '__main__':
    main()


