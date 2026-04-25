"""
Agent E2: GL-SRR 多随机种子 + 多数据集 (Cora / Citeseer / Pubmed)
================================================================
目的：验证 D1 最优配置的稳定性，消除单点结果的偶然性。
种子：[0, 1, 2, 3, 4]，数据集：Cora, Citeseer, Pubmed
报告：mean ± std
"""

import os, pickle, numpy as np, scipy.sparse as sp
import torch, torch.nn as nn, torch.nn.functional as F
from datetime import datetime

OUT_DIR  = "D:/桌面/MSR实验复现与创新/results/whitebox_gat_v3"
DATA_DIR = "D:/桌面/MSR实验复现与创新/planetoid/data"
os.makedirs(OUT_DIR, exist_ok=True)
LOG_PATH = os.path.join(OUT_DIR, "e2_output.txt")
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


class GLSRRLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, subspace_dim,
                 eta=0.5, lambda_lap=0.1, lambda_sparse=0.05):
        super().__init__()
        self.num_heads    = num_heads
        self.subspace_dim = subspace_dim
        self.eta          = eta
        self.lambda_lap   = nn.Parameter(torch.tensor(lambda_lap))
        self.U = nn.ParameterList()
        for _ in range(num_heads):
            raw = torch.randn(in_dim, subspace_dim)
            q, _ = torch.linalg.qr(raw)
            self.U.append(nn.Parameter(q[:, :subspace_dim].clone()))
        self.threshold = nn.Parameter(torch.full((out_dim,), lambda_sparse))
        self.proj = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else None

    def forward(self, H, adj_mask, L):
        Ht = H.t()
        agg_sum = torch.zeros_like(Ht)
        for k in range(self.num_heads):
            Uk = self.U[k]
            Zk = Uk.t() @ Ht
            scores = (Zk.t() @ Zk) / (self.subspace_dim ** 0.5)
            scores = scores + (1.0 - adj_mask) * (-1e9)
            alpha  = F.softmax(scores, dim=1)
            agg_sum = agg_sum + Uk @ (Zk @ alpha.t())
        lap_grad = self.lambda_lap * (L @ H)
        H_half = (Ht + self.eta * agg_sum).t() - self.eta * lap_grad
        if self.proj is not None:
            H_half = self.proj(H_half)
        thr   = self.threshold.unsqueeze(0)
        H_out = torch.sign(H_half) * F.relu(torch.abs(H_half) - thr)
        orth_loss = torch.tensor(0.0, device=H.device)
        for k in range(self.num_heads):
            UkTUk = self.U[k].t() @ self.U[k]
            I = torch.eye(self.subspace_dim, device=H.device)
            orth_loss = orth_loss + ((UkTUk - I)**2).sum()
            for l in range(k+1, self.num_heads):
                orth_loss = orth_loss + ((self.U[k].t() @ self.U[l])**2).sum()
        lap_smooth = torch.trace(H_out.t() @ L @ H_out)
        return H_out, orth_loss, lap_smooth


class GLSRRNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes,
                 num_heads=7, subspace_dim=16, eta=0.5,
                 lambda_lap=0.1, lambda_sparse=0.05, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.layer1  = GLSRRLayer(in_dim, hidden_dim, num_heads, subspace_dim, eta, lambda_lap, lambda_sparse)
        self.layer2  = GLSRRLayer(hidden_dim, hidden_dim, num_heads, subspace_dim, eta, lambda_lap, lambda_sparse)
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


def train_eval_seed(cfg, features, labels, adj, L, train_idx, val_idx, test_idx, device, seed):
    torch.manual_seed(seed); np.random.seed(seed)
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

    best_val, best_test, patience_cnt = 0.0, 0.0, 0
    for epoch in range(1, cfg['epochs']+1):
        model.train(); opt.zero_grad()
        logits, penult, orth_loss, _ = model(feat_t, adj_t, L_t)
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
    return best_val, best_test


def run_dataset(ds_name, cfg, device, seeds):
    log(f"\n{'='*50}")
    log(f"Dataset: {ds_name.upper()}")
    log(f"{'='*50}")
    features, labels, adj, adj_norm, lap, train_idx, val_idx, test_idx = load_dataset(ds_name, DATA_DIR)
    log(f"  {features.shape[0]} nodes, {features.shape[1]} features, {int(labels.max())+1} classes")
    val_list, test_list = [], []
    for seed in seeds:
        val, test = train_eval_seed(cfg, features, labels, adj, lap,
                                    train_idx, val_idx, test_idx, device, seed)
        log(f"  seed={seed}  val={val:.4f}  test={test:.4f}")
        val_list.append(val); test_list.append(test)
        save_log()
    vm, vs = np.mean(val_list), np.std(val_list)
    tm, ts = np.mean(test_list), np.std(test_list)
    log(f"  >> {ds_name}: val={vm:.4f}±{vs:.4f}  test={tm:.4f}±{ts:.4f}")
    return vm, vs, tm, ts


def main():
    device = torch.device('cpu')
    log(f"=== Agent E2: GL-SRR 多种子多数据集  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    log("验证 D1 最优配置在多种子、多数据集上的稳定性\n")

    # D1 最优配置
    cfg = dict(hidden_dim=64, num_heads=7, subspace_dim=16, eta=0.5,
               lambda_lap=0.1, lambda_sparse=0.05,
               lambda_mcr=0.005, lambda_orth=0.005,
               dropout=0.6, lr=0.005, wd=1e-3, epochs=400, patience=40)

    seeds = [0, 1, 2, 3, 4]
    datasets = ['cora', 'citeseer', 'pubmed']

    summary = {}
    for ds in datasets:
        try:
            vm, vs, tm, ts = run_dataset(ds, cfg, device, seeds)
            summary[ds] = (vm, vs, tm, ts)
        except Exception as e:
            log(f"  ERROR on {ds}: {e}")
            summary[ds] = (0, 0, 0, 0)

    log("\n" + "="*60)
    log("Agent E2 - 多种子多数据集结果汇总")
    log("="*60)
    log(f"  {'Dataset':<12} {'Val mean±std':<20} {'Test mean±std':<20}")
    log(f"  {'-'*52}")
    for ds, (vm, vs, tm, ts) in summary.items():
        log(f"  {ds:<12} {vm:.4f}±{vs:.4f}        {tm:.4f}±{ts:.4f}")

    res_path = os.path.join(OUT_DIR, "e2_result.txt")
    with open(res_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))
    log(f"\nWritten to {res_path}")
    save_log()

if __name__ == '__main__':
    main()


