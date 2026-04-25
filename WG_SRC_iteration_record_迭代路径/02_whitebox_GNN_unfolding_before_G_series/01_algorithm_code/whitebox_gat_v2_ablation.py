"""
Whitebox GAT CRATE v2 - Agent A: 修复标签泄漏 + 消融实验
修复：
1. MCR2 只用 train_idx（修复标签泄漏）
2. 标准 soft-threshold 替代单边 ReLU
3. 训练结束后 load best_state
4. 加入 head 内部正交约束
5. MCR2 对 penultimate embedding 算，不对 logits 算
"""
import os, sys, pickle
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
np.random.seed(42)

DATA_DIR = "D:/桌面/MSR实验复现与创新/planetoid/data"
OUT_DIR  = "D:/桌面/MSR实验复现与创新/results/whitebox_gat_crate_v2"
os.makedirs(OUT_DIR, exist_ok=True)

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
    adj = np.zeros((n, n), dtype=np.float32)
    for node, neighbors in graph.items():
        for nb in neighbors:
            adj[node, nb] = 1.0
            adj[nb, node] = 1.0
    np.fill_diagonal(adj, 1.0)
    train_idx = list(range(140))
    val_idx   = list(range(200, 500))
    test_idx  = test_idx_sorted[:1000]
    return features, labels, adj, train_idx, val_idx, test_idx

# ── 模型层 ────────────────────────────────────────────────────────────────────
class WhiteboxGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, subspace_dim,
                 eta=0.5, lambda_sparse=0.05, use_soft_thresh=True):
        super().__init__()
        self.num_heads    = num_heads
        self.subspace_dim = subspace_dim
        self.eta          = eta
        self.use_soft_thresh = use_soft_thresh
        self.U = nn.ParameterList()
        for _ in range(num_heads):
            raw = torch.randn(in_dim, subspace_dim)
            q, _ = torch.linalg.qr(raw)
            self.U.append(nn.Parameter(q[:, :subspace_dim].clone()))
        self.threshold = nn.Parameter(torch.full((out_dim,), lambda_sparse))
        self.proj = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else None

    def forward(self, H, adj_mask):
        n = H.shape[0]
        Ht = H.t()
        agg_sum = torch.zeros_like(Ht)
        for k in range(self.num_heads):
            Uk = self.U[k]
            Zk = Uk.t() @ Ht
            scale = self.subspace_dim ** 0.5
            scores = (Zk.t() @ Zk) / scale
            scores = scores + (1.0 - adj_mask) * (-1e9)
            alpha  = F.softmax(scores, dim=1)
            Zk_agg = Zk @ alpha.t()
            agg_sum = agg_sum + Uk @ Zk_agg
        H_half = (Ht + self.eta * agg_sum).t()
        if self.proj is not None:
            H_half = self.proj(H_half)
        thr = self.threshold.unsqueeze(0)
        if self.use_soft_thresh:
            # 标准 soft-threshold（修复 ISTA）
            H_out = torch.sign(H_half) * F.relu(torch.abs(H_half) - thr)
        else:
            # 原始单边 ReLU（消融对比用）
            H_out = F.relu(H_half - thr)
        # 正交损失：head 间 + head 内部
        orth_loss = torch.tensor(0.0, device=H.device)
        for k in range(self.num_heads):
            # head 内部正交：||U_k^T U_k - I||_F^2
            UkTUk = self.U[k].t() @ self.U[k]
            I = torch.eye(self.subspace_dim, device=H.device)
            orth_loss = orth_loss + ((UkTUk - I) ** 2).sum()
            # head 间正交
            for l in range(k + 1, self.num_heads):
                cross = self.U[k].t() @ self.U[l]
                orth_loss = orth_loss + (cross ** 2).sum()
        return H_out, orth_loss

class WhiteboxGATCRATE(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes,
                 num_heads=7, subspace_dim=16, eta=0.5,
                 lambda_sparse=0.05, dropout=0.5, use_soft_thresh=True):
        super().__init__()
        self.dropout = dropout
        self.layer1 = WhiteboxGATLayer(in_dim, hidden_dim, num_heads,
                                        subspace_dim, eta, lambda_sparse, use_soft_thresh)
        self.norm1  = nn.LayerNorm(hidden_dim)
        self.layer2 = WhiteboxGATLayer(hidden_dim, hidden_dim, num_heads,
                                        subspace_dim, eta, lambda_sparse, use_soft_thresh)
        self.norm2  = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, H, adj_mask):
        H = F.dropout(H, p=self.dropout, training=self.training)
        H, o1 = self.layer1(H, adj_mask)
        H = self.norm1(H)
        H = F.dropout(H, p=self.dropout, training=self.training)
        H, o2 = self.layer2(H, adj_mask)
        H = self.norm2(H)
        penultimate = H  # 用于 MCR2
        logits = self.classifier(H)
        return logits, penultimate, o1 + o2

# ── MCR2 ──────────────────────────────────────────────────────────────────────
def mcr2_loss(Z, labels, num_classes, eps=0.5):
    n, d = Z.shape
    eps2 = eps ** 2
    def log_det(M, m):
        A = torch.eye(d, device=Z.device) + (d / (m * eps2)) * M
        _, logdet = torch.linalg.slogdet(A)
        return 0.5 * logdet
    R_all = log_det(Z.t() @ Z, n)
    R_parts = torch.tensor(0.0, device=Z.device)
    for c in range(num_classes):
        idx = (labels == c).nonzero(as_tuple=True)[0]
        if len(idx) == 0: continue
        Zc = Z[idx]
        R_parts = R_parts + (len(idx) / n) * log_det(Zc.t() @ Zc, len(idx))
    return -(R_all - R_parts)

# ── 训练 ──────────────────────────────────────────────────────────────────────
def train_and_eval(cfg, features, labels, adj_mask, train_idx, val_idx, test_idx, device):
    num_classes = int(labels.max()) + 1
    model = WhiteboxGATCRATE(
        in_dim=features.shape[1], hidden_dim=cfg['hidden_dim'],
        num_classes=num_classes, num_heads=cfg['num_heads'],
        subspace_dim=cfg['subspace_dim'], eta=cfg['eta'],
        lambda_sparse=cfg['lambda_sparse'], dropout=cfg['dropout'],
        use_soft_thresh=cfg['use_soft_thresh']
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    feat_t   = torch.tensor(features, device=device)
    labels_t = torch.tensor(labels, device=device, dtype=torch.long)
    adj_t    = torch.tensor(adj_mask, device=device)
    best_val, best_test, patience_cnt, best_state = 0.0, 0.0, 0, None
    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        opt.zero_grad()
        logits, penult, orth_loss = model(feat_t, adj_t)
        loss_ce     = F.cross_entropy(logits[train_idx], labels_t[train_idx])
        # 修复：MCR2 只用 train_idx，对 penultimate embedding 算
        loss_mcr    = mcr2_loss(penult[train_idx], labels_t[train_idx], num_classes) if cfg['lambda_mcr'] > 0 else torch.tensor(0.0)
        loss_orth   = orth_loss
        loss_sparse = logits[train_idx].abs().mean()
        loss = (loss_ce
                + cfg['lambda_mcr']  * loss_mcr
                + cfg['lambda_orth'] * loss_orth
                + cfg['lambda_l1']   * loss_sparse)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            logits_e, _, _ = model(feat_t, adj_t)
            preds = logits_e.argmax(dim=1)
            val_acc  = (preds[val_idx]  == labels_t[val_idx]).float().mean().item()
            test_acc = (preds[test_idx] == labels_t[test_idx]).float().mean().item()
        if val_acc > best_val:
            best_val, best_test, patience_cnt = val_acc, test_acc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
        if patience_cnt >= cfg['patience']:
            break
    # 修复：加载最佳状态
    if best_state:
        model.load_state_dict(best_state)
    return best_val, best_test

# ── 主程序 ────────────────────────────────────────────────────────────────────
def main():
    device = torch.device('cpu')
    print("Loading Cora...")
    features, labels, adj_mask, train_idx, val_idx, test_idx = load_cora(DATA_DIR)

    base = dict(hidden_dim=64, num_heads=7, subspace_dim=16, eta=0.5,
                lambda_sparse=0.05, lambda_mcr=0.01, lambda_orth=0.01,
                lambda_l1=0.001, dropout=0.5, lr=0.005, weight_decay=5e-4,
                epochs=200, patience=20, use_soft_thresh=True)

    # 消融实验：必须全部跑完
    ablations = [
        ('full_model',   dict()),
        ('no_mcr',       dict(lambda_mcr=0.0)),
        ('no_orth',      dict(lambda_orth=0.0)),
        ('no_l1',        dict(lambda_l1=0.0)),
        ('relu_thresh',  dict(use_soft_thresh=False)),
        ('ce_only',      dict(lambda_mcr=0.0, lambda_orth=0.0, lambda_l1=0.0)),
    ]

    results = []
    for name, overrides in ablations:
        cfg = {**base, **overrides}
        print(f"Running {name}...", flush=True)
        val, test = train_and_eval(cfg, features, labels, adj_mask,
                                   train_idx, val_idx, test_idx, device)
        print(f"  {name}: val={val:.4f}  test={test:.4f}", flush=True)
        results.append((name, val, test))

    # 写结果
    out_path = os.path.join(OUT_DIR, "agent_A_result.txt")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("Whitebox GAT CRATE v2 - Agent A: Ablation Study\n")
        f.write("=" * 60 + "\n\n")
        f.write("修复内容：\n")
        f.write("1. MCR2 只用 train_idx（修复标签泄漏）\n")
        f.write("2. 标准 soft-threshold: sign(x)*relu(|x|-thr)\n")
        f.write("3. 训练结束后 load best_state\n")
        f.write("4. head 内部正交约束 ||U_k^T U_k - I||_F^2\n")
        f.write("5. MCR2 对 penultimate embedding 算\n")
        f.write("6. 加入 LayerNorm + 独立 classifier head\n\n")
        f.write("消融实验结果：\n")
        best_val, best_name = 0, ''
        for name, val, test in results:
            f.write(f"  {name:<15} val={val:.4f}  test={test:.4f}\n")
            if val > best_val:
                best_val, best_name = val, name
        best_test = next(t for n, v, t in results if n == best_name)
        f.write(f"\n最佳 config: {best_name}  val={best_val:.4f}  test={best_test:.4f}\n")
        f.write("\n结论：\n")
        mcr_r  = next((t for n,v,t in results if n=='no_mcr'), 0)
        full_r = next((t for n,v,t in results if n=='full_model'), 0)
        orth_r = next((t for n,v,t in results if n=='no_orth'), 0)
        ce_r   = next((t for n,v,t in results if n=='ce_only'), 0)
        f.write(f"  MCR2 贡献: full={full_r:.4f} vs no_mcr={mcr_r:.4f} (diff={full_r-mcr_r:+.4f})\n")
        f.write(f"  Orth 贡献: full={full_r:.4f} vs no_orth={orth_r:.4f} (diff={full_r-orth_r:+.4f})\n")
        f.write(f"  vs CE only: full={full_r:.4f} vs ce_only={ce_r:.4f} (diff={full_r-ce_r:+.4f})\n")
    print(f"\nResults written to {out_path}")

if __name__ == '__main__':
    main()


