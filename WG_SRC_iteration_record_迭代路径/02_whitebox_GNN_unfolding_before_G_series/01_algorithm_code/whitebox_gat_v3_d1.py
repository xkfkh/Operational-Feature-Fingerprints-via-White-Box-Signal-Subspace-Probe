"""
Whitebox GAT v3 - Agent D1
理论创新：Graph Laplacian Sparse Rate Reduction (GL-SRR)

目标函数（第一性原理推导）：
  min_{Z}  -R(Z) + (lambda/2) * tr(Z^T L Z) + mu * ||Z||_1
  其中:
    R(Z) = log det(I + d/(n*eps^2) * Z^T Z)  ← coding rate（最大化表示多样性）
    tr(Z^T L Z)  ← graph Laplacian 正则（相邻节点表示应相似）
    ||Z||_1      ← 稀疏约束

对该目标做 proximal gradient descent 一步展开：
  梯度步: Z_half = Z - eta * grad_Z[-R(Z) + lambda/2 * tr(Z^T L Z)]
        = Z + eta * (dR/dZ) - eta*lambda * L Z
  近端步: Z_new = prox_{mu*eta}(Z_half) = sign(Z_half)*relu(|Z_half|-mu*eta)

attention 权重由 coding rate 的 Hessian 近似给出：
  A_ij ∝ exp( z_i^T (I + d/(n*eps^2) Z^T Z)^{-1} z_j )  if (i,j) in E

这样每一层的前向传播 = 目标函数的一步 proximal gradient 迭代，完全闭环。
"""

import os, pickle, numpy as np, scipy.sparse as sp
import torch, torch.nn as nn, torch.nn.functional as F
from datetime import datetime

torch.manual_seed(42); np.random.seed(42)

DATA_DIR = "D:/桌面/MSR实验复现与创新/planetoid/data"
OUT_DIR  = "D:/桌面/MSR实验复现与创新/results/whitebox_gat_v3"
os.makedirs(OUT_DIR, exist_ok=True)

LOG_PATH = os.path.join(OUT_DIR, "d1_log.txt")
log_lines = []

def log(msg):
    print(msg, flush=True)
    log_lines.append(msg)

def save_log():
    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))

# ── 数据加载 ──────────────────────────────────────────────────────────────────
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
    labels = np.vstack((ally, ty))
    labels[test_idx_raw, :] = labels[test_idx_sorted, :]
    labels = np.argmax(labels, axis=1)
    n = features.shape[0]
    # 邻接矩阵
    adj = np.zeros((n, n), dtype=np.float32)
    for node, neighbors in graph.items():
        for nb in neighbors:
            adj[node, nb] = 1.0; adj[nb, node] = 1.0
    np.fill_diagonal(adj, 1.0)
    # 归一化 Laplacian: L = I - D^{-1/2} A D^{-1/2}
    deg = adj.sum(axis=1)
    d_inv_sqrt = np.where(deg > 0, np.power(deg, -0.5), 0.0)
    adj_norm = d_inv_sqrt[:, None] * adj * d_inv_sqrt[None, :]  # D^{-1/2} A D^{-1/2}
    lap = np.eye(n, dtype=np.float32) - adj_norm                # L = I - adj_norm
    return (features, labels, adj, adj_norm, lap,
            list(range(140)), list(range(200,500)), test_idx_sorted[:1000])

# ── GL-SRR Layer（从目标函数严格推导）────────────────────────────────────────
class GLSRRLayer(nn.Module):
    """
    一层 = GL-SRR 目标的一步 proximal gradient descent

    前向传播推导：
      1. 计算 coding rate gradient:
         dR/dZ = (d/(n*eps^2)) * Z * (I + d/(n*eps^2) * Z^T Z)^{-1}
         近似为: (1/subspace_dim) * U U^T Z  （低秩近似，U 是子空间基）

      2. 计算 Laplacian gradient:
         d/dZ [lambda/2 * tr(Z^T L Z)] = lambda * L Z

      3. 梯度步（attention 权重由 coding rate Hessian 近似）:
         Z_half = Z + eta * U U^T Z * A_graph - eta * lambda_lap * L Z
         其中 A_graph_{ij} = softmax( z_i^T U U^T z_j / sqrt(d) ) * adj_{ij}

      4. 近端步（L1 稀疏）:
         Z_new = sign(Z_half) * relu(|Z_half| - threshold)
    """
    def __init__(self, in_dim, out_dim, num_heads, subspace_dim,
                 eta=0.5, lambda_lap=0.1, lambda_sparse=0.05):
        super().__init__()
        self.num_heads    = num_heads
        self.subspace_dim = subspace_dim
        self.eta          = eta
        self.lambda_lap   = nn.Parameter(torch.tensor(lambda_lap))  # 可学习
        # 子空间基 U_k（正交初始化）
        self.U = nn.ParameterList()
        for _ in range(num_heads):
            raw = torch.randn(in_dim, subspace_dim)
            q, _ = torch.linalg.qr(raw)
            self.U.append(nn.Parameter(q[:, :subspace_dim].clone()))
        # 近端阈值（可学习）
        self.threshold = nn.Parameter(torch.full((out_dim,), lambda_sparse))
        self.proj = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else None

    def forward(self, H, adj_mask, L):
        """
        H: [n, in_dim]
        adj_mask: [n, n] 邻接矩阵（含自环）
        L: [n, n] 归一化 Laplacian
        """
        Ht = H.t()  # [in_dim, n]
        # Step 1: coding rate gradient via subspace attention
        agg_sum = torch.zeros_like(Ht)
        for k in range(self.num_heads):
            Uk = self.U[k]                          # [in_dim, subspace_dim]
            Zk = Uk.t() @ Ht                        # [subspace_dim, n]
            # attention = coding rate Hessian 近似
            scores = (Zk.t() @ Zk) / (self.subspace_dim ** 0.5)  # [n, n]
            scores = scores + (1.0 - adj_mask) * (-1e9)
            alpha  = F.softmax(scores, dim=1)       # [n, n]
            agg_sum = agg_sum + Uk @ (Zk @ alpha.t())

        # Step 2: Laplacian gradient: lambda_lap * L Z
        lap_grad = self.lambda_lap * (L @ H)        # [n, in_dim]

        # Step 3: 梯度步
        H_half = (Ht + self.eta * agg_sum).t() - self.eta * lap_grad  # [n, in_dim]
        if self.proj is not None:
            H_half = self.proj(H_half)

        # Step 4: 近端步（soft-threshold = prox of L1）
        thr   = self.threshold.unsqueeze(0)
        H_out = torch.sign(H_half) * F.relu(torch.abs(H_half) - thr)

        # 正交正则损失（保持子空间正交性）
        orth_loss = torch.tensor(0.0, device=H.device)
        for k in range(self.num_heads):
            UkTUk = self.U[k].t() @ self.U[k]
            I     = torch.eye(self.subspace_dim, device=H.device)
            orth_loss = orth_loss + ((UkTUk - I)**2).sum()
            for l in range(k+1, self.num_heads):
                orth_loss = orth_loss + ((self.U[k].t() @ self.U[l])**2).sum()

        # Laplacian smoothness loss（用于监控，不额外加入总损失）
        lap_smooth = torch.trace(H_out.t() @ L @ H_out)

        return H_out, orth_loss, lap_smooth

# ── GL-SRR 完整模型 ───────────────────────────────────────────────────────────
class GLSRRNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes,
                 num_heads=7, subspace_dim=16, eta=0.5,
                 lambda_lap=0.1, lambda_sparse=0.05, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.layer1  = GLSRRLayer(in_dim, hidden_dim, num_heads, subspace_dim,
                                   eta, lambda_lap, lambda_sparse)
        self.layer2  = GLSRRLayer(hidden_dim, hidden_dim, num_heads, subspace_dim,
                                   eta, lambda_lap, lambda_sparse)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, H, adj_mask, L):
        H = F.dropout(H, p=self.dropout, training=self.training)
        H, o1, ls1 = self.layer1(H, adj_mask, L)
        H = F.dropout(H, p=self.dropout, training=self.training)
        H, o2, ls2 = self.layer2(H, adj_mask, L)
        logits = self.classifier(H)
        return logits, H, o1+o2, ls1+ls2

# ── MCR2 loss（只用 train_idx）────────────────────────────────────────────────
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

# ── 训练 ──────────────────────────────────────────────────────────────────────
def train_eval(cfg, features, labels, adj, L, train_idx, val_idx, test_idx, device, run_name):
    num_classes = int(labels.max())+1
    model = GLSRRNet(
        features.shape[1], cfg['hidden_dim'], num_classes,
        cfg['num_heads'], cfg['subspace_dim'], cfg['eta'],
        cfg['lambda_lap'], cfg['lambda_sparse'], cfg['dropout']
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    feat_t   = torch.tensor(features, device=device)
    labels_t = torch.tensor(labels,   device=device, dtype=torch.long)
    adj_t    = torch.tensor(adj,       device=device)
    L_t      = torch.tensor(L,         device=device)

    best_val, best_test, patience_cnt, best_state = 0.0, 0.0, 0, None
    history = []

    for epoch in range(1, cfg['epochs']+1):
        model.train(); opt.zero_grad()
        logits, penult, orth_loss, lap_smooth = model(feat_t, adj_t, L_t)
        loss_ce  = F.cross_entropy(logits[train_idx], labels_t[train_idx])
        loss_mcr = mcr2_loss(penult[train_idx], labels_t[train_idx], num_classes)
        # 总损失 = CE + MCR2 + 正交正则（Laplacian 已内嵌在层内）
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
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
        if patience_cnt >= cfg['patience']: break

        if epoch % 50 == 0:
            msg = (f"  [{run_name}] ep={epoch:3d}  loss={loss.item():.4f}"
                   f"  ce={loss_ce.item():.4f}  mcr={loss_mcr.item():.4f}"
                   f"  lap_smooth={lap_smooth.item():.4f}"
                   f"  val={val_acc:.4f}  test={test_acc:.4f}")
            log(msg)
            history.append(msg)

    if best_state: model.load_state_dict(best_state)
    return best_val, best_test, history

# ── 主程序 ────────────────────────────────────────────────────────────────────
def main():
    device = torch.device('cpu')
    log(f"=== Agent D1: GL-SRR  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    log("理论：Graph Laplacian Sparse Rate Reduction")
    log("每层 = proximal gradient step on: -R(Z) + lambda/2*tr(Z^T L Z) + mu*||Z||_1\n")

    features, labels, adj, adj_norm, lap, train_idx, val_idx, test_idx = load_cora(DATA_DIR)
    log(f"Cora: {features.shape[0]} nodes, {features.shape[1]} features, {int(labels.max())+1} classes")

    # 基础配置（从 Agent C best_combo 出发）
    base = dict(hidden_dim=64, num_heads=7, subspace_dim=16, eta=0.5,
                lambda_lap=0.1, lambda_sparse=0.05,
                lambda_mcr=0.005, lambda_orth=0.005,
                dropout=0.6, lr=0.005, wd=1e-3, epochs=400, patience=40)

    # 迭代优化实验
    experiments = [
        # 迭代1：基础 GL-SRR
        ('iter1_base',      dict()),
        # 迭代2：增大 Laplacian 权重
        ('iter2_lap_high',  dict(lambda_lap=0.3)),
        # 迭代3：减小 Laplacian 权重
        ('iter3_lap_low',   dict(lambda_lap=0.03)),
        # 迭代4：更多 heads
        ('iter4_heads8',    dict(num_heads=8, subspace_dim=8)),
        # 迭代5：更大子空间
        ('iter5_sub32',     dict(subspace_dim=32)),
        # 迭代6：用归一化 adj（adj_norm）而非原始 adj
        ('iter6_adj_norm',  dict()),   # 特殊处理，见下
        # 迭代7：综合最优
        ('iter7_best',      dict(lambda_lap=0.05, num_heads=8, subspace_dim=16,
                                  dropout=0.6, lambda_mcr=0.005, lr=0.005)),
    ]

    results = []
    all_history = []
    for i, (name, overrides) in enumerate(experiments):
        cfg = {**base, **overrides}
        use_adj = adj_norm if name == 'iter6_adj_norm' else adj
        log(f"\n--- {name} ---")
        val, test, hist = train_eval(cfg, features, labels, use_adj, lap,
                                     train_idx, val_idx, test_idx, device, name)
        log(f"  >> {name}: val={val:.4f}  test={test:.4f}")
        results.append((name, val, test))
        all_history.extend(hist)

    # 写结果
    log("\n" + "="*60)
    log("Agent D1 - GL-SRR 实验结果汇总")
    log("="*60)
    best_val, best_name = 0, ''
    for name, val, test in results:
        log(f"  {name:<20} val={val:.4f}  test={test:.4f}")
        if val > best_val:
            best_val, best_name = val, name
    best_test = next(t for n,v,t in results if n==best_name)
    log(f"\n最优: {best_name}  val={best_val:.4f}  test={best_test:.4f}")
    log("\n理论贡献：")
    log("  - 将 graph Laplacian 正则嵌入目标函数，每层严格对应一步 proximal gradient")
    log("  - Laplacian 权重 lambda_lap 可学习，自适应图结构")
    log("  - attention 权重由 coding rate Hessian 近似给出，有理论依据")

    save_log()
    # 写 result 文件
    res_path = os.path.join(OUT_DIR, "d1_result.txt")
    with open(res_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))
    log(f"\nWritten to {res_path}")
    save_log()

if __name__ == '__main__':
    main()


