import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import math
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import scipy.sparse as sp

# ── paths ──────────────────────────────────────────────────────────────────────
DATA_DIR   = "D:/桌面/MSR实验复现与创新/planetoid/data"
OUT_DIR    = "D:/桌面/MSR实验复现与创新/results/whitebox_gat_v3"
OUT_TXT    = os.path.join(OUT_DIR, "g2_output.txt")
RES_TXT    = os.path.join(OUT_DIR, "g2_result.txt")
os.makedirs(OUT_DIR, exist_ok=True)

# ── logging ────────────────────────────────────────────────────────────────────
_log_fh = open(OUT_TXT, "w", encoding="utf-8")

def log(msg=""):
    print(msg)
    _log_fh.write(msg + "\n")
    _log_fh.flush()

# ── Cora loader ────────────────────────────────────────────────────────────────
def load_cora(data_dir):
    import pickle, itertools
    names = ["x", "y", "tx", "ty", "allx", "ally", "graph"]
    objects = []
    for n in names:
        fpath = os.path.join(data_dir, "ind.cora.{}".format(n))
        with open(fpath, "rb") as f:
            objects.append(pickle.load(f, encoding="latin1"))
    x, y, tx, ty, allx, ally, graph = objects

    # test indices
    test_idx_file = os.path.join(data_dir, "ind.cora.test.index")
    test_idx = []
    with open(test_idx_file, "r") as f:
        for line in f:
            test_idx.append(int(line.strip()))
    test_idx_sorted = sorted(test_idx)

    # features
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx, :] = features[test_idx_sorted, :]
    features = features.toarray().astype(np.float32)

    # labels
    labels_onehot = np.vstack((ally, ty))
    labels_onehot[test_idx, :] = labels_onehot[test_idx_sorted, :]
    labels = labels_onehot.argmax(axis=1)

    # adjacency
    n = features.shape[0]
    adj = sp.lil_matrix((n, n), dtype=np.float32)
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
    adj_norm = adj_norm.toarray().astype(np.float32)

    # Laplacian L = I - adj_norm
    lap = np.eye(n, dtype=np.float32) - adj_norm

    # splits (standard Cora)
    train_mask = np.zeros(n, dtype=bool)
    val_mask   = np.zeros(n, dtype=bool)
    test_mask  = np.zeros(n, dtype=bool)
    train_mask[:140]       = True
    val_mask[140:640]      = True
    test_mask[test_idx]    = True

    return features, labels, adj_norm, lap, train_mask, val_mask, test_mask

# ── MCR2 helpers ───────────────────────────────────────────────────────────────
def coding_rate(Z, eps):
    """R(Z) = 0.5 * log det(I + d/(n*eps) * Z Z^T), Z: n x d"""
    n, d = Z.shape
    coeff = d / (n * eps)
    S = torch.eye(n, device=Z.device) + coeff * Z @ Z.t()
    sign, logdet = torch.linalg.slogdet(S)
    return 0.5 * logdet

def mcr2_loss(Z, y, num_classes, eps):
    Rz = coding_rate(Z, eps)
    Rc = 0.0
    for c in range(num_classes):
        mask = (y == c)
        if mask.sum() < 2:
            continue
        Zc = Z[mask]
        nc = Zc.shape[0]
        Rc = Rc + (nc / Z.shape[0]) * coding_rate(Zc, eps)
    return -(Rz - Rc)   # minimise negative delta-R

def orth_loss(W):
    """Encourage W^T W = I"""
    d = W.shape[1]
    G = W.t() @ W
    return ((G - torch.eye(d, device=W.device)) ** 2).mean()

# ── FreqLapHopLayer ────────────────────────────────────────────────────────────
class FreqLapHopLayer(nn.Module):
    def __init__(self, in_dim, out_dim, subspace_dim, num_hops,
                 eta, eps, lambda_lap, lambda_sparse):
        super().__init__()
        self.in_dim       = in_dim
        self.out_dim      = out_dim
        self.subspace_dim = subspace_dim
        self.num_hops     = num_hops
        self.eta          = eta
        self.eps          = eps
        self.lambda_lap   = lambda_lap
        self.lambda_sparse = lambda_sparse

        # learnable temperature for softmax over hop weights
        self.log_tau = nn.Parameter(torch.zeros(1))   # tau = exp(log_tau), init 1.0

        # projection matrices U_k  (in_dim x subspace_dim) per hop
        self.U = nn.ParameterList([
            nn.Parameter(torch.randn(in_dim, subspace_dim) * 0.1)
            for _ in range(num_hops)
        ])

        # linear transform W: in_dim -> out_dim
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.xavier_uniform_(self.W.weight)

        self.ln = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(p=0.0)   # dropout set externally via model

    def forward(self, H, hops, L):
        """
        H    : n x in_dim  (input features)
        hops : list of K tensors, each n x in_dim  (pre-computed hop features)
        L    : n x n Laplacian
        Returns: n x out_dim
        """
        n = H.shape[0]
        K = self.num_hops
        tau = torch.exp(self.log_tau).clamp(min=1e-3)

        # ── 1. Laplacian energies E_k ──────────────────────────────────────
        E = []
        for k in range(K):
            Hk = hops[k]                          # n x in_dim
            # E_k = tr(H_k^T L H_k) / (n * d)
            LHk = L @ Hk                          # n x in_dim
            ek  = (Hk * LHk).sum() / (n * Hk.shape[1])
            E.append(ek)
        E_stack = torch.stack(E)                  # K

        # ── 2. hop weights w_k = softmax(-E / tau) ────────────────────────
        w = F.softmax(-E_stack / tau, dim=0)      # K

        # ── 3. grad_R_k and Laplacian gradient ────────────────────────────
        grad_sum = torch.zeros(n, self.in_dim, device=H.device)
        lap_sum  = torch.zeros(n, self.in_dim, device=H.device)

        for k in range(K):
            Hk = hops[k]                          # n x in_dim
            Uk = self.U[k]                        # in_dim x subspace_dim

            # Z_k = U_k^T H_k^T  [subspace_dim, n]
            Zk = Uk.t() @ Hk.t()                  # subspace_dim x n
            s  = Zk.shape[0]
            coeff = s / (n * self.eps)
            # M_k = I_s + coeff * Z_k Z_k^T  [s, s]
            Mk = torch.eye(s, device=H.device) + coeff * (Zk @ Zk.t())
            try:
                Mk_inv = torch.cholesky_inverse(torch.linalg.cholesky(Mk))
            except Exception:
                Mk_inv = torch.linalg.pinv(Mk)
            # grad_R_k = U_k @ M_inv @ Z_k  [in_dim, n] -> transpose -> [n, in_dim]
            gk = (Uk @ (Mk_inv @ Zk)).t()         # n x in_dim
            grad_sum = grad_sum + w[k] * gk

            # Laplacian gradient: L @ H_k
            lap_sum = lap_sum + self.lambda_lap * (L @ Hk)

        # ── 4. gradient step ──────────────────────────────────────────────
        H_half = H + self.eta * grad_sum - self.eta * lap_sum   # n x in_dim

        # ── 5. linear transform ───────────────────────────────────────────
        out = self.W(H_half)                      # n x out_dim

        # ── 6. proximal: soft threshold + LayerNorm ───────────────────────
        thresh = self.lambda_sparse * self.eta
        out = F.relu(out.abs() - thresh) * out.sign()
        out = self.ln(out)

        return out, w.detach(), E_stack.detach()


# ── Full model ─────────────────────────────────────────────────────────────────
class FreqLapGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes,
                 subspace_dim, num_hops, eta, eps,
                 lambda_lap, lambda_sparse, dropout):
        super().__init__()
        self.num_hops = num_hops
        self.dropout  = dropout

        self.layer1 = FreqLapHopLayer(in_dim,     hidden_dim, subspace_dim,
                                      num_hops, eta, eps, lambda_lap, lambda_sparse)
        self.layer2 = FreqLapHopLayer(hidden_dim, num_classes, subspace_dim,
                                      num_hops, eta, eps, lambda_lap, lambda_sparse)

    def _build_hops(self, H, adj_norm):
        """Build hop features: H_k = (adj_norm)^k @ H, k=1..K"""
        hops = []
        Hk = H
        for _ in range(self.num_hops):
            Hk = adj_norm @ Hk
            hops.append(Hk)
        return hops

    def forward(self, x, adj_norm, L):
        x = F.dropout(x, p=self.dropout, training=self.training)

        hops1 = self._build_hops(x, adj_norm)
        h, w1, e1 = self.layer1(x, hops1, L)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        hops2 = self._build_hops(h, adj_norm)
        out, w2, e2 = self.layer2(h, hops2, L)

        return out, (w1, w2), (e1, e2)


# ── training / eval ────────────────────────────────────────────────────────────
def accuracy(logits, labels, mask):
    pred = logits[mask].argmax(dim=1)
    return (pred == labels[mask]).float().mean().item()

def run_experiment(name, cfg, data):
    features, labels, adj_norm, lap, train_mask, val_mask, test_mask = data
    device = torch.device("cpu")

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    X   = torch.tensor(features, dtype=torch.float32, device=device)
    Y   = torch.tensor(labels,   dtype=torch.long,    device=device)
    A   = torch.tensor(adj_norm, dtype=torch.float32, device=device)
    L   = torch.tensor(lap,      dtype=torch.float32, device=device)
    tm  = torch.tensor(train_mask, dtype=torch.bool,  device=device)
    vm  = torch.tensor(val_mask,   dtype=torch.bool,  device=device)
    tsm = torch.tensor(test_mask,  dtype=torch.bool,  device=device)

    num_classes = int(Y.max().item()) + 1
    in_dim      = X.shape[1]

    model = FreqLapGNN(
        in_dim      = in_dim,
        hidden_dim  = cfg["hidden_dim"],
        num_classes = num_classes,
        subspace_dim= cfg["subspace_dim"],
        num_hops    = cfg["num_hops"],
        eta         = cfg["eta"],
        eps         = cfg["eps"],
        lambda_lap  = cfg["lambda_lap"],
        lambda_sparse= cfg["lambda_sparse"],
        dropout     = cfg["dropout"],
    ).to(device)

    # override tau if specified
    if "tau" in cfg:
        with torch.no_grad():
            model.layer1.log_tau.fill_(math.log(cfg["tau"]))
            model.layer2.log_tau.fill_(math.log(cfg["tau"]))

    optimizer = Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])

    best_val  = 0.0
    best_test = 0.0
    patience_cnt = 0

    log("=" * 60)
    log("Experiment: {}".format(name))
    log("Config: {}".format({k: v for k, v in cfg.items()}))
    log("=" * 60)

    t0 = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        optimizer.zero_grad()

        logits, (w1, w2), (e1, e2) = model(X, A, L)

        # cross-entropy on train nodes
        loss_ce = F.cross_entropy(logits[tm], Y[tm])

        # MCR2 loss on train nodes
        Z_train = logits[tm]
        loss_mcr = mcr2_loss(Z_train, Y[tm], num_classes, cfg["eps"])

        # orthogonality loss on U matrices
        loss_orth = 0.0
        for layer in [model.layer1, model.layer2]:
            for Uk in layer.U:
                loss_orth = loss_orth + orth_loss(Uk)
        loss_orth = loss_orth / (2 * cfg["num_hops"])

        loss = loss_ce + cfg["lambda_mcr"] * loss_mcr + cfg["lambda_orth"] * loss_orth
        loss.backward()
        optimizer.step()

        # eval
        model.eval()
        with torch.no_grad():
            logits_e, (w1e, w2e), (e1e, e2e) = model(X, A, L)
            val_acc  = accuracy(logits_e, Y, vm)
            test_acc = accuracy(logits_e, Y, tsm)

        if val_acc > best_val:
            best_val  = val_acc
            best_test = test_acc
            patience_cnt = 0
        else:
            patience_cnt += 1

        if epoch % 50 == 0:
            w1_np = w1e.cpu().numpy()
            w2_np = w2e.cpu().numpy()
            e1_np = e1e.cpu().numpy()
            e2_np = e2e.cpu().numpy()
            log("Epoch {:4d} | loss={:.4f} | val={:.4f} | test={:.4f}".format(
                epoch, loss.item(), val_acc, test_acc))
            log("  L1 hop_w=[{}]  E=[{}]".format(
                ", ".join("{:.3f}".format(v) for v in w1_np),
                ", ".join("{:.4f}".format(v) for v in e1_np)))
            log("  L2 hop_w=[{}]  E=[{}]".format(
                ", ".join("{:.3f}".format(v) for v in w2_np),
                ", ".join("{:.4f}".format(v) for v in e2_np)))

        if patience_cnt >= cfg["patience"]:
            log("Early stop at epoch {}".format(epoch))
            break

    elapsed = time.time() - t0
    log("Done: best_val={:.4f}  best_test={:.4f}  time={:.1f}s".format(
        best_val, best_test, elapsed))
    return best_val, best_test


# ── experiment configs ─────────────────────────────────────────────────────────
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
    dropout      = 0.6,
    lr           = 0.005,
    wd           = 1e-3,
    epochs       = 400,
    patience     = 50,
    seed         = 42,
)

def merge(base, **kwargs):
    d = dict(base)
    d.update(kwargs)
    return d

EXPERIMENTS = [
    ("g2_base",      merge(BASE)),
    ("g2_tau_high",  merge(BASE, tau=2.0)),
    ("g2_tau_low",   merge(BASE, tau=0.5)),
    ("g2_hidden128", merge(BASE, hidden_dim=128, subspace_dim=32)),
    ("g2_hop3",      merge(BASE, num_hops=3)),
    ("g2_best",      merge(BASE, hidden_dim=128, subspace_dim=32)),
]

# ── main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log("Loading Cora from {}".format(DATA_DIR))
    data = load_cora(DATA_DIR)
    features, labels, adj_norm, lap, train_mask, val_mask, test_mask = data
    log("Nodes={}, Features={}, Classes={}".format(
        features.shape[0], features.shape[1], int(labels.max()) + 1))
    log("")

    results = {}
    for name, cfg in EXPERIMENTS:
        val_acc, test_acc = run_experiment(name, cfg, data)
        results[name] = {"val": round(val_acc, 4), "test": round(test_acc, 4)}
        log("")

    # ── summary ───────────────────────────────────────────────────────────────
    log("=" * 60)
    log("SUMMARY")
    log("=" * 60)
    log("{:<20s}  {:>8s}  {:>8s}".format("Experiment", "Val Acc", "Test Acc"))
    log("-" * 42)
    best_name = None
    best_test = 0.0
    for name, res in results.items():
        log("{:<20s}  {:>8.4f}  {:>8.4f}".format(name, res["val"], res["test"]))
        if res["test"] > best_test:
            best_test = res["test"]
            best_name = name
    log("-" * 42)
    log("Best config: {}  test_acc={:.4f}".format(best_name, best_test))

    # ── save result file ──────────────────────────────────────────────────────
    with open(RES_TXT, "w", encoding="utf-8") as f:
        f.write("G2 FreqLap Experiment Results\n")
        f.write("=" * 60 + "\n")
        f.write("{:<20s}  {:>8s}  {:>8s}\n".format("Experiment", "Val Acc", "Test Acc"))
        f.write("-" * 42 + "\n")
        for name, res in results.items():
            f.write("{:<20s}  {:>8.4f}  {:>8.4f}\n".format(name, res["val"], res["test"]))
        f.write("-" * 42 + "\n")
        f.write("Best config: {}  test_acc={:.4f}\n".format(best_name, best_test))
        f.write("\nFull results JSON:\n")
        f.write(json.dumps(results, indent=2) + "\n")

    log("")
    log("Results saved to {}".format(RES_TXT))
    _log_fh.close()


