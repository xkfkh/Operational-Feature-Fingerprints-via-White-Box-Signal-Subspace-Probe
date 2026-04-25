"""
Agent F2: GL-MSSD v2 + 谱域 Proximal Gradient (白盒)
=====================================================
大方向：目标函数和层结构绑定，从 graph-aware objective 直接推出 layer。

目标函数（谱域版本）：
  min_Z  -R_spectral(Z) + lambda/2 * ||Z||_{L,spec}^2 + mu * ||Z||_1

其中：
  R_spectral(Z) = sum_k w_k * log det(I + d/(n*eps^2) * Phi_k^T Phi_k)
  Phi_k = V_k^T Z  （Z 在第 k 个频率子空间的投影）
  V_k 是 Laplacian 第 k 个特征向量
  ||Z||_{L,spec}^2 = sum_k lambda_k * ||V_k^T Z||_F^2  （谱域 Laplacian 正则）

对 Z 求梯度：
  grad_R = sum_k w_k * V_k * (d/(n*eps^2)) * (I + d/(n*eps^2) * Phi_k Phi_k^T)^{-1} * Phi_k^T
  grad_L = sum_k lambda_k * lambda_k_eig * V_k * V_k^T * Z  （谱域平滑梯度）

Proximal gradient step:
  Z_{t+1} = prox_{mu}( Z_t + eta * grad_R - eta * grad_L )
  prox_{mu}(x) = sign(x) * relu(|x| - mu)  （soft threshold）

这样 layer 完全由目标函数推导，无任何黑盒组件。
"""
import os, pickle, numpy as np, scipy.sparse as sp, scipy.linalg as sla
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch, torch.nn as nn, torch.nn.functional as F
from datetime import datetime

DATA_DIR = "D:/桌面/MSR实验复现与创新/planetoid/data"
OUT_DIR  = "D:/桌面/MSR实验复现与创新/results/whitebox_gat_v3"
os.makedirs(OUT_DIR, exist_ok=True)
LOG_PATH = os.path.join(OUT_DIR, "f2_output.txt")
log_lines = []

def log(msg):
    print(msg, flush=True)
    log_lines.append(msg)

def save_log():
    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))

def load_cora(data_dir):
    names = ['x','y','tx','ty','allx','ally','graph']
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
            adj[node, nb] = 1.0; adj[nb, node] = 1.0
    np.fill_diagonal(adj, 1.0)
    deg = adj.sum(axis=1)
    d_inv_sqrt = np.where(deg > 0, deg**-0.5, 0.0)
    adj_norm = d_inv_sqrt[:, None] * adj * d_inv_sqrt[None, :]
    lap = np.eye(n, dtype=np.float32) - adj_norm
    return (features, labels, adj_norm, lap,
            list(range(140)), list(range(200,500)), test_idx_sorted[:1000])


def compute_spectral_basis(lap, k):
    """计算 Laplacian 最大 k 个特征向量和特征值（float64 精度）
    注意：取最大 k 个（高频），而非最小 k 个（全是0附近）
    """
    log(f"  Computing top-{k} eigenvectors (largest eigenvalues)...")
    lap64 = lap.astype(np.float64)
    eigvals, eigvecs = np.linalg.eigh(lap64)
    # 取最大 k 个特征值/向量（高频分量，有区分度）
    eigvals = eigvals[-k:]
    eigvecs = eigvecs[:, -k:]
    eigvals = np.clip(eigvals, 0.0, None)
    log(f"  Eigenvalue range: [{eigvals[0]:.6f}, {eigvals[-1]:.6f}]")
    return eigvals.astype(np.float32), eigvecs.astype(np.float32)


def propagate_hops(H, A, num_hops):
    hops = [H]
    for _ in range(num_hops):
        H = A @ H
        hops.append(H)
    return hops


class SpectralProxLayer(nn.Module):
    """
    谱域 Proximal Gradient Layer（完全白盒）

    目标函数：
      min_Z  -sum_k w_k * R_k(V_k^T Z) + sum_k lambda_k/2 * eig_k * ||V_k^T Z||_F^2
             + mu * ||Z||_1

    其中 R_k(Phi) = log det(I + coeff * Phi^T Phi) 是第 k 个频率子空间的 coding rate

    梯度推导：
      grad_{Z} R_k = V_k * coeff * (I + coeff * Phi_k Phi_k^T)^{-1} * Phi_k^T
      grad_{Z} L_k = eig_k * V_k * V_k^T * Z

    Proximal step：
      Z_{t+1} = prox_mu( Z + eta * sum_k w_k * grad_R_k - eta * sum_k lambda_k * grad_L_k )
    """
    def __init__(self, in_dim, out_dim, num_hops, num_spec,
                 eta=0.5, lambda_lap=0.1, lambda_sparse=0.05, eps=0.5):
        super().__init__()
        self.num_hops    = num_hops
        self.num_spec    = num_spec
        self.eta         = eta
        self.eps         = eps
        # 可学习的频率子空间权重（对应目标函数中的 w_k）
        self.spec_logits = nn.Parameter(torch.zeros(num_spec))
        # 可学习的 per-spec Laplacian 正则强度（对应 lambda_k）
        self.lap_logits  = nn.Parameter(torch.full((num_spec,), lambda_lap))
        # 可学习的 hop 权重
        self.hop_weights = nn.Parameter(torch.ones(num_hops + 1) / (num_hops + 1))
        # proximal threshold（对应 mu）
        self.threshold   = nn.Parameter(torch.full((out_dim,), lambda_sparse))
        self.proj = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else None
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, H, A, V, eigvals):
        """
        H:       [n, in_dim]
        A:       [n, n] normalized adjacency
        V:       [n, num_spec] Laplacian eigenvectors (fixed)
        eigvals: [num_spec] Laplacian eigenvalues (fixed)
        """
        n, d = H.shape
        hops     = propagate_hops(H, A, self.num_hops)
        hop_w    = F.softmax(self.hop_weights, dim=0)
        spec_w   = F.softmax(self.spec_logits, dim=0)          # [num_spec]
        lap_lams = F.softplus(self.lap_logits)                  # [num_spec], positive

        coeff = d / (n * self.eps**2)

        grad_R_total = torch.zeros_like(H)   # accumulate coding rate gradient
        grad_L_total = torch.zeros_like(H)   # accumulate Laplacian gradient

        for h_idx, H_k in enumerate(hops):
            for s_idx in range(self.num_spec):
                v_s   = V[:, s_idx:s_idx+1]          # [n, 1]
                eig_s = eigvals[s_idx]                # scalar

                # Phi_s = V_s^T H_k: [1, d]  (projection onto s-th spectral component)
                Phi_s = v_s.t() @ H_k                # [1, d]

                # grad of R_s w.r.t. H_k:
                # M_s = I_d + coeff * Phi_s^T Phi_s: [d, d]
                M_s   = torch.eye(d, device=H.device) + coeff * (Phi_s.t() @ Phi_s)
                try:
                    L_chol = torch.linalg.cholesky(M_s)
                    M_inv  = torch.cholesky_inverse(L_chol)
                except Exception:
                    M_inv  = torch.linalg.pinv(M_s)

                # grad_R_s = V_s * coeff * M_inv * Phi_s^T: [n, d]
                grad_R_s = v_s @ (coeff * Phi_s @ M_inv)   # [n, d]
                grad_R_total = grad_R_total + hop_w[h_idx] * spec_w[s_idx] * grad_R_s

                # grad of L_s w.r.t. H_k:
                # grad_L_s = eig_s * V_s * V_s^T * H_k: [n, d]
                grad_L_s = eig_s * (v_s @ (v_s.t() @ H_k))
                grad_L_total = grad_L_total + hop_w[h_idx] * lap_lams[s_idx] * grad_L_s

        # proximal gradient step
        H_half = H + self.eta * grad_R_total - self.eta * grad_L_total
        if self.proj is not None:
            H_half = self.proj(H_half)

        # proximal operator (soft threshold)
        thr   = self.threshold.unsqueeze(0)
        H_out = torch.sign(H_half) * F.relu(torch.abs(H_half) - thr)
        H_out = self.norm(H_out)

        return H_out, hop_w.detach(), spec_w.detach()


class F2Net(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes,
                 num_hops=2, num_spec=32,
                 eta=0.5, lambda_lap=0.1, lambda_sparse=0.05,
                 dropout=0.5, eps=0.5):
        super().__init__()
        self.dropout = dropout
        self.layer1  = SpectralProxLayer(in_dim,     hidden_dim, num_hops, num_spec,
                                         eta, lambda_lap, lambda_sparse, eps)
        self.layer2  = SpectralProxLayer(hidden_dim, hidden_dim, num_hops, num_spec,
                                         eta, lambda_lap, lambda_sparse, eps)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, H0, A, V, eigvals):
        H = F.dropout(H0, p=self.dropout, training=self.training)
        H, hw1, sw1 = self.layer1(H, A, V, eigvals)
        H = F.dropout(H, p=self.dropout, training=self.training)
        H, hw2, sw2 = self.layer2(H, A, V, eigvals)
        return self.classifier(H), H, hw2, sw2


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


def train_eval(cfg, features, labels, adj_norm, lap, V_t, eig_t,
               train_idx, val_idx, test_idx, device, run_name):
    torch.manual_seed(cfg.get('seed', 42))
    num_classes = int(labels.max())+1
    model = F2Net(
        features.shape[1], cfg['hidden_dim'], num_classes,
        cfg['num_hops'], cfg['num_spec'],
        cfg['eta'], cfg['lambda_lap'], cfg['lambda_sparse'],
        cfg['dropout'], cfg['eps']
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    feat_t   = torch.tensor(features, device=device)
    labels_t = torch.tensor(labels,   device=device, dtype=torch.long)
    adj_t    = torch.tensor(adj_norm,  device=device)

    best_val, best_test, patience_cnt = 0.0, 0.0, 0

    for epoch in range(1, cfg['epochs']+1):
        model.train(); opt.zero_grad()
        logits, penult, hop_w, spec_w = model(feat_t, adj_t, V_t, eig_t)
        loss_ce  = F.cross_entropy(logits[train_idx], labels_t[train_idx])
        loss_mcr = mcr2_loss(penult[train_idx], labels_t[train_idx], num_classes)
        loss = loss_ce + cfg['lambda_mcr'] * loss_mcr
        loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            logits_e, _, hw, sw = model(feat_t, adj_t, V_t, eig_t)
            preds    = logits_e.argmax(1)
            val_acc  = (preds[val_idx]  == labels_t[val_idx]).float().mean().item()
            test_acc = (preds[test_idx] == labels_t[test_idx]).float().mean().item()

        if val_acc > best_val:
            best_val, best_test, patience_cnt = val_acc, test_acc, 0
        else:
            patience_cnt += 1
        if patience_cnt >= cfg['patience']: break

        if epoch % 50 == 0:
            w_str  = ' '.join([f'{w:.3f}' for w in hw.cpu().numpy()])
            sw_top = sw.cpu().numpy()
            top3   = np.argsort(sw_top)[-3:][::-1]
            log(f"  [{run_name}] ep={epoch:3d}  loss={loss.item():.4f}"
                f"  ce={loss_ce.item():.4f}  val={val_acc:.4f}  test={test_acc:.4f}"
                f"  hop_w=[{w_str}]  top_spec={top3.tolist()}")

    return best_val, best_test


def main():
    device = torch.device('cpu')
    log(f"=== Agent F2: 谱域 Proximal Gradient (白盒)  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    log("从 graph-aware spectral coding rate 目标函数直接推导 layer 结构\n")

    features, labels, adj_norm, lap, train_idx, val_idx, test_idx = load_cora(DATA_DIR)
    log(f"Cora: {features.shape[0]} nodes, {features.shape[1]} features, {int(labels.max())+1} classes")

    k_spec = 32  # 用前32个特征向量，避免计算太慢
    eigvals_np, eigvecs_np = compute_spectral_basis(lap, k_spec)
    V_t   = torch.tensor(eigvecs_np, dtype=torch.float32)
    eig_t = torch.tensor(eigvals_np, dtype=torch.float32)
    log("")

    base = dict(hidden_dim=64, num_hops=2, num_spec=k_spec,
                eta=0.5, eps=0.5, lambda_lap=0.1, lambda_sparse=0.05,
                lambda_mcr=0.005,
                dropout=0.6, lr=0.005, wd=1e-3, epochs=300, patience=40, seed=42)

    experiments = [
        ('f2_base',       dict()),
        ('f2_spec16',     dict(num_spec=16)),
        ('f2_lap_high',   dict(lambda_lap=0.3)),
        ('f2_lap_low',    dict(lambda_lap=0.03)),
        ('f2_hop1',       dict(num_hops=1)),
        ('f2_hop3',       dict(num_hops=3)),
        ('f2_mcr_high',   dict(lambda_mcr=0.01)),
        ('f2_hidden128',  dict(hidden_dim=128)),
        ('f2_best',       dict(hidden_dim=128, num_spec=k_spec, lambda_lap=0.1,
                                lambda_mcr=0.005, dropout=0.6, num_hops=2)),
        ('f2_seed1',      dict(seed=10)),
    ]

    results = []
    for name, overrides in experiments:
        cfg = {**base, **overrides}
        cfg['num_spec'] = min(cfg['num_spec'], k_spec)
        V_use   = V_t[:, :cfg['num_spec']]
        eig_use = eig_t[:cfg['num_spec']]
        log(f"\n--- {name} ---")
        val, test = train_eval(cfg, features, labels, adj_norm, lap,
                               V_use, eig_use,
                               train_idx, val_idx, test_idx, device, name)
        log(f"  >> {name}: val={val:.4f}  test={test:.4f}")
        results.append((name, val, test))
        save_log()

    log("\n" + "="*60)
    log("Agent F2 - 谱域 Proximal Gradient 结果汇总")
    log("="*60)
    best_val, best_name = 0, ''
    for name, val, test in results:
        log(f"  {name:<20} val={val:.4f}  test={test:.4f}")
        if val > best_val:
            best_val, best_name = val, name
    best_test = next(t for n,v,t in results if n==best_name)
    log(f"\n最优: {best_name}  val={best_val:.4f}  test={best_test:.4f}")
    log("\n理论贡献（白盒）：")
    log("  - 目标函数：谱域 coding rate + 谱域 Laplacian 正则 + L1 稀疏")
    log("  - Layer = 对目标函数的 proximal gradient step，无任何黑盒组件")
    log("  - 可学习频率权重 spec_w 对应目标函数中的 w_k，有明确数学含义")
    log("  - 与 E5（空域 Hessian）对比，验证谱域 vs 空域 proximal gradient 的差异")

    res_path = os.path.join(OUT_DIR, "f2_result.txt")
    with open(res_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))
    log(f"\nWritten to {res_path}")
    save_log()

if __name__ == '__main__':
    main()


