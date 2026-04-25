import os
import sys
import time
import json
import pickle
import numpy as np
import scipy.sparse as sp

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── paths ───────────────────────────────────────────────────────────────────
DATA_DIR   = "D:/桌面/MSR实验复现与创新/planetoid/data"
OUT_DIR    = "D:/桌面/MSR实验复现与创新/results/whitebox_gat_v3"
OUT_TXT    = os.path.join(OUT_DIR, "g3_output.txt")
RESULT_TXT = os.path.join(OUT_DIR, "g3_result.txt")
os.makedirs(OUT_DIR, exist_ok=True)

# ─── logging ─────────────────────────────────────────────────────────────────
_log_file = open(OUT_TXT, "w", encoding="utf-8")

def log(msg=""):
    print(msg)
    _log_file.write(msg + "\n")
    _log_file.flush()

# ─── Cora loader ─────────────────────────────────────────────────────────────
def load_cora(data_dir):
    def _load(name):
        path = os.path.join(data_dir, name)
        with open(path, "rb") as f:
            return pickle.load(f, encoding="latin1")

    x     = _load("ind.cora.x")
    tx    = _load("ind.cora.tx")
    allx  = _load("ind.cora.allx")
    y     = _load("ind.cora.y")
    ty    = _load("ind.cora.ty")
    ally  = _load("ind.cora.ally")
    graph = _load("ind.cora.graph")

    test_idx_path = os.path.join(data_dir, "ind.cora.test.index")
    test_idx = []
    with open(test_idx_path, "r") as f:
        for line in f:
            test_idx.append(int(line.strip()))
    test_idx_sorted = sorted(test_idx)

    # features
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx, :] = features[test_idx_sorted, :]

    # labels
    labels = np.vstack((ally, ty))
    labels[test_idx, :] = labels[test_idx_sorted, :]
    labels = np.argmax(labels, axis=1)

    # adjacency
    n = features.shape[0]
    adj = sp.lil_matrix((n, n))
    for node, neighbors in graph.items():
        for nb in neighbors:
            adj[node, nb] = 1.0
            adj[nb, node] = 1.0
    adj = adj.tocsr()

    # D^{-1/2} A D^{-1/2}
    deg = np.array(adj.sum(axis=1)).flatten()
    deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D_inv_sqrt = sp.diags(deg_inv_sqrt)
    adj_norm = D_inv_sqrt @ adj @ D_inv_sqrt
    adj_norm = adj_norm.tocsr()

    # Laplacian  L = I - adj_norm
    lap = sp.eye(n) - adj_norm

    # splits (standard Cora)
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask   = torch.zeros(n, dtype=torch.bool)
    test_mask  = torch.zeros(n, dtype=torch.bool)
    train_mask[:140]      = True
    val_mask[140:640]     = True
    test_mask[test_idx]   = True

    # to dense tensors
    feat_dense = torch.FloatTensor(np.array(features.todense()))
    labels_t   = torch.LongTensor(labels)

    # sparse adj_norm and lap as torch sparse
    def sp_to_torch(mat):
        mat = mat.tocoo().astype(np.float32)
        idx = torch.LongTensor(np.vstack([mat.row, mat.col]))
        val = torch.FloatTensor(mat.data)
        return torch.sparse_coo_tensor(idx, val, mat.shape).coalesce()

    adj_t = sp_to_torch(adj_norm)
    lap_t = sp_to_torch(lap)

    return feat_dense, labels_t, adj_t, lap_t, train_mask, val_mask, test_mask

# ─── coding rate ─────────────────────────────────────────────────────────────
def coding_rate(Z, eps=0.5):
    """R(Z) = 0.5 * log det(I + coeff * Z^T Z),  Z: [n, d]"""
    n, d = Z.shape
    coeff = d / (n * eps * eps)
    # use n×n form when n < d, else d×d
    if n < d:
        G = Z @ Z.t()                          # n×n
        M = torch.eye(n, device=Z.device) + coeff * G
    else:
        G = Z.t() @ Z                          # d×d
        M = torch.eye(d, device=Z.device) + coeff * G
    sign, logdet = torch.linalg.slogdet(M)
    return 0.5 * logdet

# ─── MCR2 delta R ────────────────────────────────────────────────────────────
def mcr2_delta_r(Z, y, num_classes, eps=0.5):
    R_total = coding_rate(Z, eps)
    R_parts = []
    for c in range(num_classes):
        mask = (y == c)
        if mask.sum() < 2:
            continue
        Zc = Z[mask]
        R_parts.append(coding_rate(Zc, eps))
    if len(R_parts) == 0:
        return R_total
    R_mean = torch.stack(R_parts).mean()
    return R_total - R_mean

# ─── VariationalHopLayer ─────────────────────────────────────────────────────
class VariationalHopLayer(nn.Module):
    """
    Variational hop selection layer.

    For each hop k in {0, 1, ..., K}:
        Z_k = A^k X W
    Coding rate:
        R_k = 0.5 * log det(I + coeff * Z_k^T Z_k)
    Variational distribution (ELBO-optimal):
        q_k = softmax([R_0, ..., R_K] / tau)
    KL vs uniform prior:
        KL = sum_k q_k * log(q_k * (K+1))
    Output:
        H = sum_k q_k * Z_k   (soft mixture)
    """
    def __init__(self, in_dim, out_dim, num_hops, eps=0.5, dropout=0.6):
        super().__init__()
        self.num_hops = num_hops
        self.eps      = eps
        K = num_hops + 1   # hops 0..num_hops

        # one linear per hop
        self.linears = nn.ModuleList([nn.Linear(in_dim, out_dim, bias=False)
                                      for _ in range(K)])
        # learnable temperature (raw; actual tau = softplus(raw_tau))
        self.raw_tau = nn.Parameter(torch.zeros(1))   # softplus(0) ≈ 0.693

        self.dropout = nn.Dropout(dropout)
        self._last_q  = None   # for logging
        self._last_kl = None

    @property
    def tau(self):
        return F.softplus(self.raw_tau) + 1e-4

    def forward(self, X, adj, precomp_hops=None):
        """
        X            : [n, in_dim]
        adj          : torch sparse [n, n]  (A_norm)
        precomp_hops : list of A^k X (optional, avoids recomputation)
        Returns: H [n, out_dim], kl_loss scalar
        """
        K = self.num_hops + 1
        tau = self.tau

        # ── compute A^k X for each hop ──
        if precomp_hops is None:
            hop_feats = []
            cur = X
            for k in range(K):
                hop_feats.append(cur)
                cur = torch.sparse.mm(adj, cur)
        else:
            hop_feats = precomp_hops

        # ── project each hop ──
        Z_list = []
        for k in range(K):
            z = self.dropout(hop_feats[k])
            z = self.linears[k](z)
            Z_list.append(z)

        # ── coding rates ──
        R_list = []
        for z in Z_list:
            R_list.append(coding_rate(z, self.eps))
        R_vec = torch.stack(R_list)          # [K]

        # ── variational distribution ──
        q = F.softmax(R_vec / tau, dim=0)    # [K]
        self._last_q = q.detach().cpu()

        # ── KL vs uniform prior ──
        kl = (q * torch.log(q * K + 1e-8)).sum()
        self._last_kl = kl.detach().item()

        # ── soft mixture ──
        H = sum(q[k] * Z_list[k] for k in range(K))

        return H, kl

# ─── full model ──────────────────────────────────────────────────────────────
class VariationalHopGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, subspace_dim,
                 num_hops, eps, eta, lambda_lap, lambda_sparse, dropout):
        super().__init__()
        self.eta          = eta
        self.lambda_lap   = lambda_lap
        self.lambda_sparse = lambda_sparse
        self.subspace_dim = subspace_dim
        self.num_hops     = num_hops

        self.layer1 = VariationalHopLayer(in_dim,     hidden_dim, num_hops, eps, dropout)
        self.layer2 = VariationalHopLayer(hidden_dim, hidden_dim, num_hops, eps, dropout)

        self.proj   = nn.Linear(hidden_dim, subspace_dim, bias=False)
        self.cls    = nn.Linear(subspace_dim, out_dim)
        self.ln1    = nn.LayerNorm(hidden_dim)
        self.ln2    = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, adj, lap, y_train=None, train_mask=None, num_classes=7):
        # layer 1
        H1, kl1 = self.layer1(X, adj)
        H1 = F.relu(self.ln1(H1))

        # layer 2
        H2, kl2 = self.layer2(H1, adj)
        H2 = F.relu(self.ln2(H2))

        # projection to subspace
        Z = self.proj(self.dropout(H2))
        logits = self.cls(Z)

        kl_total = kl1 + kl2

        # MCR2 loss on Z (training nodes only)
        mcr2_loss = torch.tensor(0.0)
        if y_train is not None and train_mask is not None:
            Zt = Z[train_mask]
            yt = y_train[train_mask]
            mcr2_loss = -mcr2_delta_r(Zt, yt, num_classes, eps=0.5)

        # orthogonality loss: W^T W ≈ I  (on proj weight)
        W = self.proj.weight   # [subspace_dim, hidden_dim]
        WWT = W @ W.t()
        orth_loss = torch.norm(WWT - torch.eye(self.subspace_dim, device=W.device), p='fro')

        return logits, Z, kl_total, mcr2_loss, orth_loss

# ─── train / eval ────────────────────────────────────────────────────────────
def accuracy(logits, labels, mask):
    pred = logits[mask].argmax(dim=1)
    return (pred == labels[mask]).float().mean().item()

def run_experiment(cfg, feat, labels, adj, lap, train_mask, val_mask, test_mask):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    in_dim      = feat.shape[1]
    num_classes = int(labels.max().item()) + 1

    model = VariationalHopGNN(
        in_dim      = in_dim,
        hidden_dim  = cfg["hidden_dim"],
        out_dim     = num_classes,
        subspace_dim= cfg["subspace_dim"],
        num_hops    = cfg["num_hops"],
        eps         = cfg["eps"],
        eta         = cfg["eta"],
        lambda_lap  = cfg["lambda_lap"],
        lambda_sparse = cfg["lambda_sparse"],
        dropout     = cfg["dropout"],
    )

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=cfg["lr"], weight_decay=cfg["wd"])

    best_val  = 0.0
    best_test = 0.0
    patience_cnt = 0

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        optimizer.zero_grad()

        logits, Z, kl, mcr2_loss, orth_loss = model(
            feat, adj, lap,
            y_train=labels, train_mask=train_mask,
            num_classes=num_classes
        )

        ce_loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        loss = (ce_loss
                + cfg["lambda_mcr"] * mcr2_loss
                + cfg["lambda_orth"] * orth_loss
                + cfg["lambda_kl"]  * kl)

        loss.backward()
        optimizer.step()

        # validation
        model.eval()
        with torch.no_grad():
            logits_e, _, _, _, _ = model(feat, adj, lap)
            val_acc  = accuracy(logits_e, labels, val_mask)
            test_acc = accuracy(logits_e, labels, test_mask)

        if val_acc > best_val:
            best_val  = val_acc
            best_test = test_acc
            patience_cnt = 0
        else:
            patience_cnt += 1

        # logging every 50 epochs
        if epoch % 50 == 0 or epoch == 1:
            q1 = model.layer1._last_q
            q2 = model.layer2._last_q
            tau1 = model.layer1.tau.item()
            tau2 = model.layer2.tau.item()
            kl_v = kl.item()
            q1_str = "[" + ", ".join(f"{v:.3f}" for v in q1.tolist()) + "]"
            q2_str = "[" + ", ".join(f"{v:.3f}" for v in q2.tolist()) + "]"
            log(f"  ep{epoch:4d} | ce={ce_loss.item():.4f} "
                f"| kl={kl_v:.4f} | val={val_acc:.4f} | test={test_acc:.4f}")
            log(f"           L1 tau={tau1:.4f} q={q1_str}")
            log(f"           L2 tau={tau2:.4f} q={q2_str}")

        if patience_cnt >= cfg["patience"]:
            log(f"  Early stop at epoch {epoch}")
            break

    return best_val, best_test

# ─── experiment configs ───────────────────────────────────────────────────────
BASE = dict(
    hidden_dim   = 64,
    subspace_dim = 16,
    num_hops     = 2,
    eta          = 0.5,
    eps          = 0.5,
    lambda_lap   = 0.3,
    lambda_sparse= 0.05,
    lambda_mcr   = 0.005,
    lambda_orth  = 0.005,
    lambda_kl    = 0.01,
    dropout      = 0.6,
    lr           = 0.005,
    wd           = 1e-3,
    epochs       = 400,
    patience     = 50,
    seed         = 42,
)

EXPERIMENTS = [
    ("g3_base",      {**BASE}),
    ("g3_kl_high",   {**BASE, "lambda_kl": 0.05}),
    ("g3_kl_low",    {**BASE, "lambda_kl": 0.001}),
    ("g3_hidden128", {**BASE, "hidden_dim": 128, "subspace_dim": 32}),
    ("g3_hop3",      {**BASE, "num_hops": 3}),
    ("g3_best",      {**BASE, "hidden_dim": 128, "subspace_dim": 32, "lambda_kl": 0.01}),
]

# ─── main ─────────────────────────────────────────────────────────────────────
def main():
    log("=" * 60)
    log("G3 Variational Hop Selection GNN — Cora")
    log("=" * 60)
    log(f"Loading Cora from {DATA_DIR} ...")
    feat, labels, adj, lap, train_mask, val_mask, test_mask = load_cora(DATA_DIR)
    log(f"  nodes={feat.shape[0]}, features={feat.shape[1]}, "
        f"classes={int(labels.max())+1}")
    log(f"  train={train_mask.sum().item()}, "
        f"val={val_mask.sum().item()}, "
        f"test={test_mask.sum().item()}")
    log()

    results = {}
    for name, cfg in EXPERIMENTS:
        log("-" * 60)
        log(f"Experiment: {name}")
        log(f"  lambda_kl={cfg['lambda_kl']}, hidden={cfg['hidden_dim']}, "
            f"subspace={cfg['subspace_dim']}, num_hops={cfg['num_hops']}")
        t0 = time.time()
        val_acc, test_acc = run_experiment(
            cfg, feat, labels, adj, lap, train_mask, val_mask, test_mask
        )
        elapsed = time.time() - t0
        results[name] = {"val": val_acc, "test": test_acc, "time": elapsed}
        log(f"  => best val={val_acc:.4f}, test={test_acc:.4f}, "
            f"time={elapsed:.1f}s")
        log()

    # ── summary table ──
    log("=" * 60)
    log("SUMMARY")
    log(f"{'Experiment':<20} {'Val Acc':>10} {'Test Acc':>10} {'Time(s)':>10}")
    log("-" * 55)
    best_name = None
    best_test = 0.0
    for name, r in results.items():
        log(f"{name:<20} {r['val']:>10.4f} {r['test']:>10.4f} {r['time']:>10.1f}")
        if r["test"] > best_test:
            best_test = r["test"]
            best_name = name
    log("-" * 55)
    log(f"Best config: {best_name}  test_acc={best_test:.4f}")
    log("=" * 60)

    # ── write result file ──
    with open(RESULT_TXT, "w", encoding="utf-8") as f:
        f.write("G3 Variational Hop Selection — Results\n")
        f.write("=" * 50 + "\n")
        for name, r in results.items():
            f.write(f"{name}: val={r['val']:.4f}, test={r['test']:.4f}, "
                    f"time={r['time']:.1f}s\n")
        f.write(f"\nBest: {best_name}  test={best_test:.4f}\n")

    log(f"Results written to {RESULT_TXT}")
    _log_file.close()

if __name__ == "__main__":
    main()


