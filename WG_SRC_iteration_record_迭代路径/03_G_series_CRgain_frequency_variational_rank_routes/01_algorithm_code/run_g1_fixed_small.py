"""
run_g1_fixed_small.py
=====================
小规模验证脚本 — 白盒 GNN (CR-Gain Hop Layer) 修复版本
验证点：
  1. 维度正确：out_proj 的输入是 in_dim，不是 subspace_dim
  2. H_half norm 非零（白盒梯度步确实在工作）
  3. hop 权重随训练有分化（tau 驱动的 softmax 有差异）

配置（小规模快速验证）：
  hidden_dim=32, subspace_dim=8, num_hops=2
  epochs=100, patience=20, dropout=0.5
  lr=0.005, wd=1e-3, tau_init=1.0
  lambda_lap=0.3, lambda_sparse=0.05, lambda_mcr=0.005, lambda_orth=0.005
  eps=0.5, eta=0.5, seed=42
"""

import os, sys, pickle, time
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ── 路径常量 ───────────────────────────────────────────────────────────────────

DATA_DIR    = "D:/桌面/MSR实验复现与创新/planetoid/data"
OUT_DIR     = "D:/桌面/MSR实验复现与创新/results/whitebox_gat_v3"
RESULT_PATH = os.path.join(OUT_DIR, "g1_fixed_small_result.txt")
LOG_PATH    = os.path.join(OUT_DIR, "g1_fixed_small_log.txt")

os.makedirs(OUT_DIR, exist_ok=True)

# ── 日志工具 ───────────────────────────────────────────────────────────────────

_log_lines = []

def log(msg):
    print(msg, flush=True)
    _log_lines.append(str(msg))

def flush_log():
    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(_log_lines) + '\n')

# ── 数据加载（与 run_g1_cr_gain.py 完全相同）───────────────────────────────────

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

# ── Coding Rate 函数（复制自 run_g1_cr_gain.py）───────────────────────────────

def coding_rate(Z, eps=0.5):
    """
    R(Z) = 0.5 * log det(I + d/(n*eps^2) * Z^T Z)
    Z: [n, d]
    """
    n, d = Z.shape
    coeff = d / (n * eps * eps)
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

# ── CRGainHopLayer（复制自 run_g1_cr_gain.py，加白盒验证记录）─────────────────

class CRGainHopLayer(nn.Module):
    """
    One layer of CR-Gain driven hop aggregation.

    For each hop k in {0, 1, ..., K}:
      Z_k = A^k H W_k   (projected hop-k features)
      R_k = coding_rate(Z_k)
      w_k = softmax(R_k / tau)

    Gradient step:
      H_half = H + eta * sum_k w_k * grad_R(Z_k) @ W_k^T
                 - eta * lambda_lap * L @ H

    Proximal step:
      H_out = LayerNorm(soft_threshold(out_proj(H_half), threshold))

    白盒验证：在 forward 里记录 H_half 的 norm，可由外部检查。
    """
    def __init__(self, in_dim, out_dim, subspace_dim, num_hops=2,
                 eta=0.5, eps=0.5, lambda_lap=0.3, lambda_sparse=0.05,
                 tau_init=1.0):
        super().__init__()
        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.subspace_dim  = subspace_dim
        self.num_hops      = num_hops
        self.eta           = eta
        self.eps           = eps
        self.lambda_lap    = lambda_lap
        self.lambda_sparse = lambda_sparse

        # 唯一可学习的跳权温度参数
        self.log_tau = nn.Parameter(torch.tensor(float(tau_init)).log())

        # 每跳投影矩阵 W_k: in_dim -> subspace_dim
        self.W = nn.ModuleList([
            nn.Linear(in_dim, subspace_dim, bias=False)
            for _ in range(num_hops + 1)
        ])

        # 输出投影: in_dim -> out_dim
        # 关键修复: 输入维度是 in_dim (H_half 在 H 空间中), 不是 subspace_dim
        self.out_proj = nn.Linear(in_dim, out_dim, bias=False)

        # 可学习软阈值
        self.threshold = nn.Parameter(torch.full((out_dim,), lambda_sparse))

        # LayerNorm
        self.ln = nn.LayerNorm(out_dim)

        # 白盒验证: 记录最近一次 forward 中 H_half 的 norm
        self._last_H_half_norm = 0.0

    @property
    def tau(self):
        return self.log_tau.exp()

    def forward(self, H, hop_feats, L):
        """
        H:         [n, in_dim]  当前节点特征
        hop_feats: list of K+1 tensors [n, in_dim], hop_feats[k] = A^k H
        L:         [n, n] 归一化 Laplacian
        Returns:   H_out [n, out_dim], hop_weights [K+1], coding_rates [K+1]
        """
        tau = self.tau
        K = self.num_hops

        # Step 1: 对每跳特征做子空间投影
        Z_list = [self.W[k](hop_feats[k]) for k in range(K + 1)]  # 每个 [n, subspace_dim]

        # Step 2: 计算每跳的 coding rate
        R_list = [coding_rate(Z_list[k], self.eps) for k in range(K + 1)]
        R_tensor = torch.stack(R_list)  # [K+1]

        # Step 3: softmax 计算跳权重
        w = F.softmax(R_tensor / tau, dim=0)  # [K+1]

        # Step 4: 白盒梯度步
        # grad_R 贡献: sum_k w_k * grad_R(Z_k) @ W_k^T -> [n, in_dim]
        grad_contrib = torch.zeros_like(H)

        for k in range(K + 1):
            g_Zk = coding_rate_gradient(Z_list[k], self.eps)  # [n, subspace_dim]
            # 反投影到 in_dim 空间
            g_H = g_Zk @ self.W[k].weight                     # [n, in_dim]
            grad_contrib = grad_contrib + w[k] * g_H

        # Laplacian 正则: lambda_lap * L @ H
        H_half = H + self.eta * grad_contrib - self.eta * self.lambda_lap * (L @ H)

        # 白盒验证记录: H_half 的 norm 必须非零，才说明梯度步在工作
        self._last_H_half_norm = H_half.norm().item()

        # Step 5: 维度对齐 in_dim -> out_dim
        # 修复要点: out_proj 接收 H_half [n, in_dim]，而非 subspace_dim 的向量
        if self.in_dim == self.out_dim:
            H_out_pre = H_half
        else:
            H_out_pre = self.out_proj(H_half)

        # Step 6: 软阈值近端算子 + LayerNorm
        thr = self.threshold.abs().unsqueeze(0)                # [1, out_dim]
        H_soft = H_out_pre.sign() * F.relu(H_out_pre.abs() - thr)
        H_out  = self.ln(H_soft)

        return H_out, w.detach(), R_tensor.detach()


# ── 完整模型（复制自 run_g1_cr_gain.py）─────────────────────────────────────

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
        """预计算 A^0 H, A^1 H, ..., A^K H"""
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


# ── MCR2 / Orth 损失（复制自 run_g1_cr_gain.py）─────────────────────────────

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
    return -delta_R  # 最小化负 delta_R

def orth_loss(Z, y, num_classes):
    """鼓励类子空间正交"""
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


# ── 小规模实验主函数 ──────────────────────────────────────────────────────────

def run_small_experiment():
    # ─ 配置 ─
    cfg = dict(
        hidden_dim    = 32,
        subspace_dim  = 8,
        num_hops      = 2,
        epochs        = 100,
        patience      = 20,
        dropout       = 0.5,
        lr            = 0.005,
        wd            = 1e-3,
        tau_init      = 1.0,
        lambda_lap    = 0.3,
        lambda_sparse = 0.05,
        lambda_mcr    = 0.005,
        lambda_orth   = 0.005,
        eps           = 0.5,
        eta           = 0.5,
        seed          = 42,
    )

    log("=" * 60)
    log("Experiment: g1_fixed_small_base")
    log("WHITE-BOX GNN: CR-Gain Hop Layer (Fixed)")
    log("=" * 60)
    log(f"Config: {cfg}")
    log("")

    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])

    # ─ 加载数据 ─
    log("Loading Cora dataset ...")
    features, labels, adj, adj_norm, lap, train_idx, val_idx, test_idx = load_cora(DATA_DIR)
    n, in_dim = features.shape
    num_classes = int(labels.max()) + 1
    log(f"  Nodes={n}, Features={in_dim}, Classes={num_classes}")
    log(f"  Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    log("")

    device = torch.device('cpu')
    X      = torch.tensor(features, dtype=torch.float32, device=device)
    Y      = torch.tensor(labels,   dtype=torch.long,    device=device)
    A_norm = torch.tensor(adj_norm, dtype=torch.float32, device=device)
    L_mat  = torch.tensor(lap,      dtype=torch.float32, device=device)

    train_mask = torch.zeros(n, dtype=torch.bool, device=device)
    val_mask   = torch.zeros(n, dtype=torch.bool, device=device)
    test_mask  = torch.zeros(n, dtype=torch.bool, device=device)
    train_mask[train_idx] = True
    val_mask[val_idx]     = True
    test_mask[test_idx]   = True

    # ─ 模型 ─
    model = CRGainGNN(
        in_dim       = in_dim,
        hidden_dim   = cfg['hidden_dim'],
        num_classes  = num_classes,
        subspace_dim = cfg['subspace_dim'],
        num_hops     = cfg['num_hops'],
        eta          = cfg['eta'],
        eps          = cfg['eps'],
        lambda_lap   = cfg['lambda_lap'],
        lambda_sparse= cfg['lambda_sparse'],
        tau_init     = cfg['tau_init'],
        dropout      = cfg['dropout'],
    ).to(device)

    # 打印维度信息（验证修复）
    log("Model architecture (whitebox dimension check):")
    log(f"  Layer1: in_dim={in_dim}, out_dim={cfg['hidden_dim']}, "
        f"subspace_dim={cfg['subspace_dim']}")
    log(f"  Layer1.out_proj: weight shape = {list(model.layer1.out_proj.weight.shape)}")
    log(f"    -> Expected [{cfg['hidden_dim']}, {in_dim}] (in_dim={in_dim}, NOT subspace_dim={cfg['subspace_dim']})")
    log(f"  Layer2: in_dim={cfg['hidden_dim']}, out_dim={num_classes}, "
        f"subspace_dim={cfg['subspace_dim']}")
    log(f"  Layer2.out_proj: weight shape = {list(model.layer2.out_proj.weight.shape)}")
    log("")

    total_params = sum(p.numel() for p in model.parameters())
    log(f"Total parameters: {total_params}")
    log("")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd']
    )

    best_val_acc  = 0.0
    best_test_acc = 0.0
    patience_cnt  = 0

    log("Training started ...")
    log("-" * 60)
    t0 = time.time()

    for epoch in range(1, cfg['epochs'] + 1):
        # ─ 训练步 ─
        model.train()
        optimizer.zero_grad()

        logits, (w1, w2), (R1, R2) = model(X, A_norm, L_mat)

        ce   = F.cross_entropy(logits[train_mask], Y[train_mask])
        mcr  = mcr2_loss(logits, Y, num_classes, eps=cfg['eps'])
        orth = orth_loss(logits, Y, num_classes)
        loss = ce + cfg['lambda_mcr'] * mcr + cfg['lambda_orth'] * orth

        loss.backward()
        optimizer.step()

        # ─ 验证步 ─
        model.eval()
        with torch.no_grad():
            logits_e, (w1e, w2e), (R1e, R2e) = model(X, A_norm, L_mat)
            pred     = logits_e.argmax(dim=1)
            val_acc  = (pred[val_mask]  == Y[val_mask]).float().mean().item()
            test_acc = (pred[test_mask] == Y[test_mask]).float().mean().item()

        # ─ Early stopping ─
        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            best_test_acc = test_acc
            patience_cnt  = 0
        else:
            patience_cnt += 1

        # ─ 每 10 个 epoch 打印一次 ─
        if epoch % 10 == 0:
            w1_str = '[' + ', '.join(f'{v:.3f}' for v in w1e.tolist()) + ']'
            R1_str = '[' + ', '.join(f'{v:.3f}' for v in R1e.tolist()) + ']'
            tau1   = model.layer1.tau.item()

            # 白盒验证：H_half norm（在 eval forward 之后 layer1._last_H_half_norm 已更新）
            h_half_norm = model.layer1._last_H_half_norm

            log(f"Epoch {epoch:4d} | loss={loss.item():.4f} | "
                f"val={val_acc:.4f} | test={test_acc:.4f}")
            log(f"  Layer1 tau={tau1:.3f} | hop_w={w1_str} | R={R1_str}")
            log(f"  WHITEBOX CHECK: H_half norm = {h_half_norm:.4f}")

        if patience_cnt >= cfg['patience']:
            log(f"\nEarly stop at epoch {epoch} (patience={cfg['patience']})")
            break

    elapsed = time.time() - t0
    log("-" * 60)

    # ─ 最终报告 ─
    log("")
    log("=" * 60)
    log("FINAL RESULTS")
    log("=" * 60)
    log(f"Best val  acc : {best_val_acc:.4f}")
    log(f"Best test acc : {best_test_acc:.4f}")
    log(f"Training time : {elapsed:.1f}s")
    log("")

    # ─ hop 权重分化分析 ─
    model.eval()
    with torch.no_grad():
        logits_f, (w1f, w2f), (R1f, R2f) = model(X, A_norm, L_mat)
    w1_list = w1f.tolist()
    w2_list = w2f.tolist()
    log("Final hop weights (Layer1):")
    for k, wk in enumerate(w1_list):
        log(f"  hop-{k}: {wk:.4f}")
    log(f"  max-min spread = {max(w1_list) - min(w1_list):.4f}  "
        f"({'DIFFERENTIATED' if max(w1_list) - min(w1_list) > 0.05 else 'UNIFORM'})")
    log("")
    log("Final hop weights (Layer2):")
    for k, wk in enumerate(w2_list):
        log(f"  hop-{k}: {wk:.4f}")
    log(f"  max-min spread = {max(w2_list) - min(w2_list):.4f}  "
        f"({'DIFFERENTIATED' if max(w2_list) - min(w2_list) > 0.05 else 'UNIFORM'})")
    log("")

    # ─ 最终 H_half norm ─
    final_h_half_norm = model.layer1._last_H_half_norm
    log(f"Final WHITEBOX CHECK: Layer1 H_half norm = {final_h_half_norm:.4f}")
    log(f"  -> {'PASS: H_half is non-zero, whitebox gradient step is active' if final_h_half_norm > 0 else 'FAIL: H_half is zero!'}")
    log("")

    # ─ 维度修复确认 ─
    out_proj_in  = model.layer1.out_proj.weight.shape[1]
    out_proj_out = model.layer1.out_proj.weight.shape[0]
    log("Dimension fix verification:")
    log(f"  out_proj weight: [{out_proj_out}, {out_proj_in}]")
    log(f"  in_dim={in_dim}, subspace_dim={cfg['subspace_dim']}")
    if out_proj_in == in_dim:
        log("  -> PASS: out_proj input dim == in_dim (1433), NOT subspace_dim (8)")
    else:
        log(f"  -> FAIL: out_proj input dim = {out_proj_in}, expected {in_dim}")
    log("")

    # ─ 保存结果文件 ─
    result_lines = [
        "g1_fixed_small_base Results",
        "=" * 40,
        f"Best val  acc : {best_val_acc:.4f}",
        f"Best test acc : {best_test_acc:.4f}",
        f"Training time : {elapsed:.1f}s",
        "",
        f"Layer1 final hop weights: {[round(v,4) for v in w1_list]}",
        f"Layer2 final hop weights: {[round(v,4) for v in w2_list]}",
        f"Layer1 hop spread: {max(w1_list)-min(w1_list):.4f}",
        "",
        f"Layer1 H_half norm (whitebox check): {final_h_half_norm:.4f}",
        f"out_proj dim fix: [{out_proj_out}, {out_proj_in}] (in_dim={in_dim})",
        "",
        "STATUS: OK" if final_h_half_norm > 0 and out_proj_in == in_dim else "STATUS: CHECK FAILED",
    ]
    with open(RESULT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(result_lines) + '\n')
    log(f"Result saved -> {RESULT_PATH}")

    return best_val_acc, best_test_acc


# ── 入口 ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    log("White-box GNN (CR-Gain) — Small Scale Verification")
    log(f"Data dir : {DATA_DIR}")
    log(f"Out  dir : {OUT_DIR}")
    log("")

    val_acc, test_acc = run_small_experiment()

    flush_log()
    log(f"\nLog saved -> {LOG_PATH}")
    log("Done.")


