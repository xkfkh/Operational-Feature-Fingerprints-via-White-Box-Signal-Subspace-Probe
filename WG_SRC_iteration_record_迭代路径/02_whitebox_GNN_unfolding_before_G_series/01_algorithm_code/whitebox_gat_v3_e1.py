"""
Agent E1: GL-SRR with True Hessian Attention
=============================================
改进点：attention 权重真正来自 coding rate 的 Hessian，而不是普通内积。

coding rate R(Z) = 0.5 * log det(I + d/(n*eps^2) * Z^T Z)
Hessian w.r.t. Z:  H_R = (d/(n*eps^2)) * (I + d/(n*eps^2) * Z^T Z)^{-1}
                       = M^{-1}  where M = I + d/(n*eps^2) * Z^T Z

attention score:
  a_ij = z_i^T M^{-1} z_j  (if (i,j) in E)

这是真正的 natural gradient / Fisher information metric 下的相似度。
"""

import os, pickle, numpy as np, scipy.sparse as sp
import torch, torch.nn as nn, torch.nn.functional as F
from datetime import datetime

OUT_DIR  = "D:/桌面/MSR实验复现与创新/results/whitebox_gat_v3"
DATA_DIR = "D:/桌面/MSR实验复现与创新/planetoid/data"
os.makedirs(OUT_DIR, exist_ok=True)
LOG_PATH = os.path.join(OUT_DIR, "e1_output.txt")
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
    # row-normalize
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


class HessianAttentionLayer(nn.Module):
    """
    GL-SRR layer with true Hessian-based attention.

    attention score a_ij = z_i^T M^{-1} z_j
    where M = I + (d / (n * eps^2)) * Z^T Z  (per-head, low-rank approx)

    To keep it tractable:
      - project to subspace: z_k = U^T z  (subspace_dim << in_dim)
      - M_k = I + (subspace_dim/(n*eps^2)) * Z_k^T Z_k  in R^{subspace_dim x subspace_dim}
      - a_ij = (U^T z_i)^T M_k^{-1} (U^T z_j)
    """
    def __init__(self, in_dim, out_dim, num_heads, subspace_dim,
                 eta=0.5, lambda_lap=0.1, lambda_sparse=0.05, eps=0.5):
        super().__init__()
        self.num_heads    = num_heads
        self.subspace_dim = subspace_dim
        self.eta          = eta
        self.eps          = eps
        self.lambda_lap   = nn.Parameter(torch.tensor(lambda_lap))
        # subspace bases
        self.U = nn.ParameterList()
        for _ in range(num_heads):
            raw = torch.randn(in_dim, subspace_dim)
            q, _ = torch.linalg.qr(raw)
            self.U.append(nn.Parameter(q[:, :subspace_dim].clone()))
        self.threshold = nn.Parameter(torch.full((out_dim,), lambda_sparse))
        self.proj = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else None

    def forward(self, H, adj_mask, L):
        n, d = H.shape
        Ht = H.t()  # [in_dim, n]
        agg_sum = torch.zeros_like(Ht)

        for k in range(self.num_heads):
            Uk = self.U[k]                              # [in_dim, sub_dim]
            Zk = Uk.t() @ Ht                            # [sub_dim, n]

            # True Hessian: M_k = I + (sub_dim/(n*eps^2)) * Zk Zk^T
            coeff = self.subspace_dim / (n * self.eps**2)
            M_k   = torch.eye(self.subspace_dim, device=H.device) + coeff * (Zk @ Zk.t())
            # M_k^{-1} via Cholesky (stable)
            try:
                L_chol = torch.linalg.cholesky(M_k)
                M_inv  = torch.cholesky_inverse(L_chol)
            except Exception:
                M_inv  = torch.linalg.pinv(M_k)

            # attention: a_ij = (U^T z_i)^T M^{-1} (U^T z_j)
            # = Zk^T M_inv Zk  -> [n, n]
            scores = (Zk.t() @ M_inv @ Zk)             # [n, n]
            scores = scores + (1.0 - adj_mask) * (-1e9)
            alpha  = F.softmax(scores, dim=1)           # [n, n]
            # aggregate: U_k M_inv U_k^T H alpha^T  (gradient step)
            agg_sum = agg_sum + Uk @ (M_inv @ Zk @ alpha.t())

        lap_grad = self.lambda_lap * (L @ H)
        H_half   = (Ht + self.eta * agg_sum).t() - self.eta * lap_grad
        if self.proj is not None:
            H_half = self.proj(H_half)

        thr   = self.threshold.unsqueeze(0)
        H_out = torch.sign(H_half) * F.relu(torch.abs(H_half) - thr)

        orth_loss = torch.tensor(0.0, device=H.device)
        for k in range(self.num_heads):
            UkTUk = self.U[k].t() @ self.U[k]
            I     = torch.eye(self.subspace_dim, device=H.device)
            orth_loss = orth_loss + ((UkTUk - I)**2).sum()
            for l in range(k+1, self.num_heads):
                orth_loss = orth_loss + ((self.U[k].t() @ self.U[l])**2).sum()

        lap_smooth = torch.trace(H_out.t() @ L @ H_out)
        return H_out, orth_loss, lap_smooth


class HessianGLSRRNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes,
                 num_heads=7, subspace_dim=16, eta=0.5,
                 lambda_lap=0.1, lambda_sparse=0.05, dropout=0.5, eps=0.5):
        super().__init__()
        self.dropout = dropout
        self.layer1  = HessianAttentionLayer(in_dim, hidden_dim, num_heads,
                                              subspace_dim, eta, lambda_lap, lambda_sparse, eps)
        self.layer2  = HessianAttentionLayer(hidden_dim, hidden_dim, num_heads,
                                              subspace_dim, eta, lambda_lap, lambda_sparse, eps)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, H, adj_mask, L):
        H = F.dropout(H, p=self.dropout, training=self.training)
        H, o1, ls1 = self.layer1(H, adj_mask, L)
        H = F.dropout(H, p=self.dropout, training=self.training)
        H, o2, ls2 = self.layer2(H, adj_mask, L)
        return self.classifier(H), H, o1+o2, ls1+ls2


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


def train_eval(cfg, features, labels, adj, L, train_idx, val_idx, test_idx, device, run_name):
    num_classes = int(labels.max())+1
    torch.manual_seed(cfg.get('seed', 42))
    model = HessianGLSRRNet(
        features.shape[1], cfg['hidden_dim'], num_classes,
        cfg['num_heads'], cfg['subspace_dim'], cfg['eta'],
        cfg['lambda_lap'], cfg['lambda_sparse'], cfg['dropout'], cfg['eps']
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    feat_t   = torch.tensor(features, device=device)
    labels_t = torch.tensor(labels,   device=device, dtype=torch.long)
    adj_t    = torch.tensor(adj,       device=device)
    L_t      = torch.tensor(L,         device=device)

    best_val, best_test, patience_cnt = 0.0, 0.0, 0

    for epoch in range(1, cfg['epochs']+1):
        model.train(); opt.zero_grad()
        logits, penult, orth_loss, lap_smooth = model(feat_t, adj_t, L_t)
        loss_ce  = F.cross_entropy(logits[train_idx], labels_t[train_idx])
        loss_mcr = mcr2_loss(penult[train_idx], labels_t[train_idx], num_classes)
        loss = loss_ce + cfg['lambda_mcr']*loss_mcr + cfg['lambda_orth']*orth_loss
        loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            logits_e, _, _, _ = model(feat_t, adj_t, L_t)
            preds    = logits_e.argmax(1)
            val_acc  = (preds[val_idx]  == labels_t[val_idx]).float().mean().item()
            test_acc = (preds[test_idx] == labels_t[test_idx]).float().mean().item()

        if val_acc > best_val:
            best_val, best_test, patience_cnt = val_acc, test_acc, 0
        else:
            patience_cnt += 1
        if patience_cnt >= cfg['patience']: break

        if epoch % 50 == 0:
            log(f"  [{run_name}] ep={epoch:3d}  loss={loss.item():.4f}"
                f"  ce={loss_ce.item():.4f}  val={val_acc:.4f}  test={test_acc:.4f}")

    return best_val, best_test


def main():
    device = torch.device('cpu')
    log(f"=== Agent E1: Hessian Attention GL-SRR  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    log("改进：attention = z_i^T M^{-1} z_j，M = I + coeff*Z^T Z（真正的 coding rate Hessian）\n")

    features, labels, adj, adj_norm, lap, train_idx, val_idx, test_idx = load_dataset('cora', DATA_DIR)
    log(f"Cora: {features.shape[0]} nodes, {features.shape[1]} features, {int(labels.max())+1} classes")

    base = dict(hidden_dim=64, num_heads=7, subspace_dim=16, eta=0.5, eps=0.5,
                lambda_lap=0.1, lambda_sparse=0.05,
                lambda_mcr=0.005, lambda_orth=0.005,
                dropout=0.6, lr=0.005, wd=1e-3, epochs=400, patience=40, seed=42)

    experiments = [
        ('iter1_hessian_base',   dict()),
        ('iter2_eps_01',         dict(eps=0.1)),
        ('iter3_eps_10',         dict(eps=1.0)),
        ('iter4_lap_005',        dict(lambda_lap=0.05)),
        ('iter5_lap_02',         dict(lambda_lap=0.2)),
        ('iter6_sub32',          dict(subspace_dim=32)),
        ('iter7_heads4',         dict(num_heads=4, subspace_dim=16)),
        ('iter8_best',           dict(eps=0.3, lambda_lap=0.05, subspace_dim=32,
                                      num_heads=7, dropout=0.6, lr=0.005)),
    ]

    results = []
    for name, overrides in experiments:
        cfg = {**base, **overrides}
        log(f"\n--- {name} ---")
        val, test = train_eval(cfg, features, labels, adj, lap,
                               train_idx, val_idx, test_idx, device, name)
        log(f"  >> {name}: val={val:.4f}  test={test:.4f}")
        results.append((name, val, test))
        save_log()

    log("\n" + "="*60)
    log("Agent E1 - Hessian Attention 实验结果汇总")
    log("="*60)
    best_val, best_name = 0, ''
    for name, val, test in results:
        log(f"  {name:<25} val={val:.4f}  test={test:.4f}")
        if val > best_val:
            best_val, best_name = val, name
    best_test = next(t for n,v,t in results if n==best_name)
    log(f"\n最优: {best_name}  val={best_val:.4f}  test={best_test:.4f}")
    log("\n理论贡献：")
    log("  - attention 权重 = z_i^T M^{-1} z_j，M 为 coding rate 的 Hessian")
    log("  - 这是 natural gradient 意义下的节点相似度，有严格信息几何依据")
    log("  - 相比普通内积 attention，更接近目标函数的曲率结构")

    res_path = os.path.join(OUT_DIR, "e1_result.txt")
    with open(res_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))
    log(f"\nWritten to {res_path}")
    save_log()

if __name__ == '__main__':
    main()


