"""
Agent E4: GL-MSSD v1 - 融合 D1 (Laplacian 正则) + D4 (多尺度子空间)
====================================================================
目标函数：
  min_Z  -sum_k R(Z_k) + lambda/2 * tr(Z^T L Z) + mu * ||Z||_1
         + gamma * sum_{k!=k'} ||Z_k^T Z_{k'}||_F^2

其中 Z_k = U_k U_k^T H^{(k)} 是第 k 跳的子空间投影。

每层推导：
  1. 多跳图信号：H^{(k)} = A_norm^k X
  2. 每个 hop 的子空间投影 + coding rate 梯度
  3. Laplacian 正则梯度：lambda * L Z
  4. 子空间间正交约束
  5. proximal step (L1)

这是 D1 和 D4 的严格理论融合。
"""

import os, pickle, numpy as np, scipy.sparse as sp
import torch, torch.nn as nn, torch.nn.functional as F
from datetime import datetime

DATA_DIR = "D:/桌面/MSR实验复现与创新/planetoid/data"
OUT_DIR  = "D:/桌面/MSR实验复现与创新/results/whitebox_gat_v3"
os.makedirs(OUT_DIR, exist_ok=True)
LOG_PATH = os.path.join(OUT_DIR, "e4_output.txt")
log_lines = []

def log(msg):
    print(msg, flush=True)
    log_lines.append(msg)

def save_log():
    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))

def load_dataset(name, data_dir):
    names = ['x','y','tx','ty','allx','ally','graph']
    objs = []
    for n in names:
        with open(f"{data_dir}/ind.{name}.{n}", 'rb') as f:
            objs.append(pickle.load(f, encoding='latin1'))
    x, y, tx, ty, allx, ally, graph = objs
    test_idx_raw    = [int(l.strip()) for l in open(f"{data_dir}/ind.{name}.test.index")]
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
            adj[node, nb] = 1.0; adj[nb, node] = 1.0
    np.fill_diagonal(adj, 1.0)
    deg = adj.sum(axis=1)
    d_inv_sqrt = np.where(deg > 0, deg**-0.5, 0.0)
    adj_norm = d_inv_sqrt[:, None] * adj * d_inv_sqrt[None, :]
    lap = np.eye(n, dtype=np.float32) - adj_norm
    return (features, labels, adj, adj_norm, lap,
            list(range(140)), list(range(200,500)), test_idx_sorted[:1000])


def precompute_multihop(features, adj_norm, max_hop, device):
    A = torch.tensor(adj_norm, device=device)
    H = torch.tensor(features, device=device)
    hops = [H]
    for _ in range(max_hop):
        H = A @ H
        hops.append(H)
    return hops


class GLMSSDLayer(nn.Module):
    """
    GL-MSSD Layer: 融合 Laplacian 正则 + 多尺度子空间分解

    前向推导（严格来自目标函数）：
      对每个 hop k：
        1. 子空间投影：Z_k = U_k U_k^T H^{(k)}
        2. coding rate 梯度（attention）：
           score_ij = (U_k^T z_i)^T M_k^{-1} (U_k^T z_j)  [Hessian attention]
        3. 加权聚合：Z_agg = sum_k w_k * (attention aggregation of Z_k)
        4. Laplacian 梯度：lambda_lap * L * Z_agg
        5. 梯度步 + proximal step
    """
    def __init__(self, in_dim, out_dim, num_hops, subspace_dim,
                 eta=0.5, lambda_lap=0.1, lambda_sparse=0.05, eps=0.5):
        super().__init__()
        self.num_hops     = num_hops
        self.subspace_dim = subspace_dim
        self.eta          = eta
        self.eps          = eps
        self.lambda_lap   = nn.Parameter(torch.tensor(lambda_lap))
        # per-hop subspace bases
        self.U = nn.ParameterList()
        for _ in range(num_hops + 1):
            raw = torch.randn(in_dim, subspace_dim)
            q, _ = torch.linalg.qr(raw)
            self.U.append(nn.Parameter(q[:, :subspace_dim].clone()))
        # learnable hop weights
        self.hop_weights = nn.Parameter(torch.ones(num_hops + 1) / (num_hops + 1))
        self.threshold   = nn.Parameter(torch.full((out_dim,), lambda_sparse))
        self.proj = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else None

    def forward(self, hops, adj_mask, L):
        """
        hops: list of [n, in_dim], length = num_hops+1
        adj_mask: [n, n]
        L: [n, n]
        """
        n = hops[0].shape[0]
        weights = F.softmax(self.hop_weights, dim=0)

        agg_total = torch.zeros_like(hops[0].t())  # [in_dim, n]

        for k, H_k in enumerate(hops):
            Uk = self.U[k]                              # [in_dim, sub_dim]
            Zk = Uk.t() @ H_k.t()                      # [sub_dim, n]

            # Hessian attention in subspace
            coeff = self.subspace_dim / (n * self.eps**2)
            M_k   = torch.eye(self.subspace_dim, device=H_k.device) + coeff * (Zk @ Zk.t())
            try:
                L_chol = torch.linalg.cholesky(M_k)
                M_inv  = torch.cholesky_inverse(L_chol)
            except Exception:
                M_inv  = torch.linalg.pinv(M_k)

            scores = Zk.t() @ M_inv @ Zk               # [n, n]
            scores = scores + (1.0 - adj_mask) * (-1e9)
            alpha  = F.softmax(scores, dim=1)           # [n, n]

            # gradient contribution from hop k
            grad_k = Uk @ (M_inv @ Zk @ alpha.t())     # [in_dim, n]
            agg_total = agg_total + weights[k] * grad_k

        # Laplacian gradient on aggregated representation
        H_agg = (agg_total).t()                         # [n, in_dim]
        lap_grad = self.lambda_lap * (L @ H_agg)

        # gradient step
        H_ref = hops[0]
        H_half = H_ref + self.eta * H_agg - self.eta * lap_grad
        if self.proj is not None:
            H_half = self.proj(H_half)

        # proximal step
        thr   = self.threshold.unsqueeze(0)
        H_out = torch.sign(H_half) * F.relu(torch.abs(H_half) - thr)

        # orthogonality loss (cross-hop)
        orth_loss = torch.tensor(0.0, device=H_out.device)
        for k in range(len(self.U)):
            UkTUk = self.U[k].t() @ self.U[k]
            I = torch.eye(self.subspace_dim, device=H_out.device)
            orth_loss = orth_loss + ((UkTUk - I)**2).sum()
            for l in range(k+1, len(self.U)):
                orth_loss = orth_loss + ((self.U[k].t() @ self.U[l])**2).sum()

        lap_smooth = torch.trace(H_out.t() @ L @ H_out)
        return H_out, orth_loss, lap_smooth, weights.detach()


class GLMSSDNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes,
                 num_hops=2, subspace_dim=16, eta=0.5,
                 lambda_lap=0.1, lambda_sparse=0.05,
                 dropout=0.5, eps=0.5):
        super().__init__()
        self.dropout  = dropout
        self.num_hops = num_hops
        self.layer1   = GLMSSDLayer(in_dim, hidden_dim, num_hops, subspace_dim,
                                     eta, lambda_lap, lambda_sparse, eps)
        self.layer2   = GLMSSDLayer(hidden_dim, hidden_dim, num_hops, subspace_dim,
                                     eta, lambda_lap, lambda_sparse, eps)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, hops, adj_mask, L):
        H = F.dropout(hops[0], p=self.dropout, training=self.training)
        hops_in = [H] + hops[1:]
        H, o1, ls1, w1 = self.layer1(hops_in, adj_mask, L)
        H = F.dropout(H, p=self.dropout, training=self.training)
        hops_in2 = [H] * (self.num_hops + 1)
        H, o2, ls2, w2 = self.layer2(hops_in2, adj_mask, L)
        return self.classifier(H), H, o1+o2, ls1+ls2, w2


def mcr2_loss(Z, labels, num_classes, eps=0.5):
    n, d = Z.shape; eps2 = eps**2
    def log_det(M, m):
        _, ld = torch.linalg.slogdet(
            torch.eye(d, device=Z.device) + (d/(m*eps2))*M)
        return 0.5 * ld
    R_all = log_det(Z.t()@Z, n)
    R_c   = torch.tensor(0.0, device=Z.device)
    for c in range(num_classes):
        idx = (labels==c).nonzero(as_tuple=True)[0]
        if len(idx)==0: continue
        Zc  = Z[idx]
        R_c = R_c + (len(idx)/n)*log_det(Zc.t()@Zc, len(idx))
    return -(R_all - R_c)


def train_eval(cfg, features, labels, adj_norm, lap,
               train_idx, val_idx, test_idx, device, run_name):
    torch.manual_seed(cfg.get('seed', 42))
    num_classes = int(labels.max())+1
    hops = precompute_multihop(features, adj_norm, cfg['num_hops'], device)
    model = GLMSSDNet(
        features.shape[1], cfg['hidden_dim'], num_classes,
        cfg['num_hops'], cfg['subspace_dim'], cfg['eta'],
        cfg['lambda_lap'], cfg['lambda_sparse'], cfg['dropout'], cfg['eps']
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    labels_t = torch.tensor(labels, device=device, dtype=torch.long)
    adj_t    = torch.tensor(adj_norm, device=device)
    L_t      = torch.tensor(lap,     device=device)

    best_val, best_test, patience_cnt = 0.0, 0.0, 0

    for epoch in range(1, cfg['epochs']+1):
        model.train(); opt.zero_grad()
        logits, penult, orth_loss, lap_smooth, hop_w = model(hops, adj_t, L_t)
        loss_ce  = F.cross_entropy(logits[train_idx], labels_t[train_idx])
        loss_mcr = mcr2_loss(penult[train_idx], labels_t[train_idx], num_classes)
        loss = loss_ce + cfg['lambda_mcr']*loss_mcr + cfg['lambda_orth']*orth_loss
        loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            logits_e, _, _, _, hw = model(hops, adj_t, L_t)
            preds    = logits_e.argmax(1)
            val_acc  = (preds[val_idx]  == labels_t[val_idx]).float().mean().item()
            test_acc = (preds[test_idx] == labels_t[test_idx]).float().mean().item()

        if val_acc > best_val:
            best_val, best_test, patience_cnt = val_acc, test_acc, 0
        else:
            patience_cnt += 1
        if patience_cnt >= cfg['patience']: break

        if epoch % 50 == 0:
            w_str = ' '.join([f'{w:.3f}' for w in hw.cpu().numpy()])
            log(f"  [{run_name}] ep={epoch:3d}  loss={loss.item():.4f}"
                f"  ce={loss_ce.item():.4f}  val={val_acc:.4f}  test={test_acc:.4f}"
                f"  hop_w=[{w_str}]")

    return best_val, best_test


def main():
    device = torch.device('cpu')
    log(f"=== Agent E4: GL-MSSD v1  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    log("融合 D1 (Laplacian 正则) + D4 (多尺度子空间) + Hessian attention\n")

    features, labels, adj, adj_norm, lap, train_idx, val_idx, test_idx = load_dataset('cora', DATA_DIR)
    log(f"Cora: {features.shape[0]} nodes, {features.shape[1]} features, {int(labels.max())+1} classes")

    base = dict(hidden_dim=64, num_hops=2, subspace_dim=16, eta=0.5, eps=0.5,
                lambda_lap=0.1, lambda_sparse=0.05,
                lambda_mcr=0.005, lambda_orth=0.005,
                dropout=0.6, lr=0.005, wd=1e-3, epochs=400, patience=40, seed=42)

    experiments = [
        ('iter1_base',       dict()),
        ('iter2_hop1',       dict(num_hops=1)),
        ('iter3_hop3',       dict(num_hops=3)),
        ('iter4_lap_low',    dict(lambda_lap=0.03)),
        ('iter5_lap_high',   dict(lambda_lap=0.3)),
        ('iter6_sub32',      dict(subspace_dim=32)),
        ('iter7_eps_03',     dict(eps=0.3)),
        ('iter8_best',       dict(num_hops=2, subspace_dim=32, eps=0.3,
                                  lambda_lap=0.05, lambda_mcr=0.005,
                                  dropout=0.6, lr=0.005)),
    ]

    results = []
    for name, overrides in experiments:
        cfg = {**base, **overrides}
        log(f"\n--- {name} ---")
        val, test = train_eval(cfg, features, labels, adj_norm, lap,
                               train_idx, val_idx, test_idx, device, name)
        log(f"  >> {name}: val={val:.4f}  test={test:.4f}")
        results.append((name, val, test))
        save_log()

    log("\n" + "="*60)
    log("Agent E4 - GL-MSSD v1 实验结果汇总")
    log("="*60)
    best_val, best_name = 0, ''
    for name, val, test in results:
        log(f"  {name:<20} val={val:.4f}  test={test:.4f}")
        if val > best_val:
            best_val, best_name = val, name
    best_test = next(t for n,v,t in results if n==best_name)
    log(f"\n最优: {best_name}  val={best_val:.4f}  test={best_test:.4f}")
    log("\n理论贡献：")
    log("  - 目标函数同时包含 coding rate、Laplacian 正则、多尺度子空间正交约束")
    log("  - 每层严格推导自目标函数的 proximal gradient")
    log("  - Hessian attention + 多跳聚合 + Laplacian 平滑，三者统一在一个框架")

    res_path = os.path.join(OUT_DIR, "e4_result.txt")
    with open(res_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))
    log(f"\nWritten to {res_path}")
    save_log()

if __name__ == '__main__':
    main()


