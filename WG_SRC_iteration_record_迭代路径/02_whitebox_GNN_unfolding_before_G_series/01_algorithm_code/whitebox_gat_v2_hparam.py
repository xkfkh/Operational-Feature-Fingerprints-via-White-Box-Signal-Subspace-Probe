"""
Whitebox GAT CRATE v2 - Agent C: 超参数搜索
修复了标签泄漏 + 标准 soft-threshold
"""
import os, pickle, numpy as np, scipy.sparse as sp
import torch, torch.nn as nn, torch.nn.functional as F

torch.manual_seed(42); np.random.seed(42)

DATA_DIR = "D:/桌面/MSR实验复现与创新/planetoid/data"
OUT_DIR  = "D:/桌面/MSR实验复现与创新/results/whitebox_gat_crate_v2"
os.makedirs(OUT_DIR, exist_ok=True)

def load_cora(data_dir):
    names = ['x','y','tx','ty','allx','ally','graph']
    objs = []
    for n in names:
        with open(f"{data_dir}/ind.cora.{n}", 'rb') as f:
            objs.append(pickle.load(f, encoding='latin1'))
    x, y, tx, ty, allx, ally, graph = objs
    test_idx_raw = [int(l.strip()) for l in open(f"{data_dir}/ind.cora.test.index")]
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
            adj[node, nb] = 1.0; adj[nb, node] = 1.0
    np.fill_diagonal(adj, 1.0)
    return features, labels, adj, list(range(140)), list(range(200,500)), test_idx_sorted[:1000]

class WhiteboxGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, subspace_dim, eta=0.5, lambda_sparse=0.05):
        super().__init__()
        self.num_heads = num_heads; self.subspace_dim = subspace_dim; self.eta = eta
        self.U = nn.ParameterList()
        for _ in range(num_heads):
            raw = torch.randn(in_dim, subspace_dim)
            q, _ = torch.linalg.qr(raw)
            self.U.append(nn.Parameter(q[:, :subspace_dim].clone()))
        self.threshold = nn.Parameter(torch.full((out_dim,), lambda_sparse))
        self.proj = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else None

    def forward(self, H, adj_mask):
        Ht = H.t(); agg_sum = torch.zeros_like(Ht)
        for k in range(self.num_heads):
            Uk = self.U[k]; Zk = Uk.t() @ Ht
            scores = (Zk.t() @ Zk) / (self.subspace_dim ** 0.5)
            scores = scores + (1.0 - adj_mask) * (-1e9)
            alpha = F.softmax(scores, dim=1)
            agg_sum = agg_sum + Uk @ (Zk @ alpha.t())
        H_half = (Ht + self.eta * agg_sum).t()
        if self.proj is not None: H_half = self.proj(H_half)
        thr = self.threshold.unsqueeze(0)
        H_out = torch.sign(H_half) * F.relu(torch.abs(H_half) - thr)
        orth_loss = torch.tensor(0.0, device=H.device)
        for k in range(self.num_heads):
            si = self.U[k].t() @ self.U[k] - torch.eye(self.subspace_dim, device=H.device)
            orth_loss = orth_loss + (si**2).sum()
            for l in range(k+1, self.num_heads):
                orth_loss = orth_loss + ((self.U[k].t() @ self.U[l])**2).sum()
        return H_out, orth_loss

class WhiteboxGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, num_heads=7,
                 subspace_dim=16, eta=0.5, lambda_sparse=0.05, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.layer1 = WhiteboxGATLayer(in_dim, hidden_dim, num_heads, subspace_dim, eta, lambda_sparse)
        self.layer2 = WhiteboxGATLayer(hidden_dim, hidden_dim, num_heads, subspace_dim, eta, lambda_sparse)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, H, adj_mask):
        H = F.dropout(H, p=self.dropout, training=self.training)
        H, o1 = self.layer1(H, adj_mask)
        H = F.dropout(H, p=self.dropout, training=self.training)
        H, o2 = self.layer2(H, adj_mask)
        return self.classifier(H), H, o1+o2

def mcr2_loss(Z, labels, num_classes, eps=0.5):
    n, d = Z.shape; eps2 = eps**2
    def log_det(M, m):
        _, ld = torch.linalg.slogdet(torch.eye(d, device=Z.device) + (d/(m*eps2))*M)
        return 0.5 * ld
    R_all = log_det(Z.t()@Z, n)
    R_parts = torch.tensor(0.0, device=Z.device)
    for c in range(num_classes):
        idx = (labels==c).nonzero(as_tuple=True)[0]
        if len(idx)==0: continue
        Zc = Z[idx]; R_parts = R_parts + (len(idx)/n)*log_det(Zc.t()@Zc, len(idx))
    return -(R_all - R_parts)

def train_eval(cfg, features, labels, adj, train_idx, val_idx, test_idx, device):
    num_classes = int(labels.max())+1
    model = WhiteboxGAT(features.shape[1], cfg['hidden_dim'], num_classes,
                        cfg['num_heads'], cfg['subspace_dim'], cfg['eta'],
                        cfg['lambda_sparse'], cfg['dropout']).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    feat_t = torch.tensor(features, device=device)
    labels_t = torch.tensor(labels, device=device, dtype=torch.long)
    adj_t = torch.tensor(adj, device=device)
    best_val, best_test, patience_cnt, best_state = 0.0, 0.0, 0, None
    for epoch in range(1, cfg['epochs']+1):
        model.train(); opt.zero_grad()
        logits, penult, orth_loss = model(feat_t, adj_t)
        loss_ce = F.cross_entropy(logits[train_idx], labels_t[train_idx])
        loss_mcr = mcr2_loss(penult[train_idx], labels_t[train_idx], num_classes) if cfg['lambda_mcr']>0 else torch.tensor(0.0)
        loss = loss_ce + cfg['lambda_mcr']*loss_mcr + cfg['lambda_orth']*orth_loss + cfg['lambda_l1']*penult.abs().mean()
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            logits_e, _, _ = model(feat_t, adj_t)
            preds = logits_e.argmax(1)
            val_acc  = (preds[val_idx]  == labels_t[val_idx]).float().mean().item()
            test_acc = (preds[test_idx] == labels_t[test_idx]).float().mean().item()
        if val_acc > best_val:
            best_val, best_test, patience_cnt = val_acc, test_acc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
        if patience_cnt >= cfg['patience']: break
    if best_state: model.load_state_dict(best_state)
    return best_val, best_test

def main():
    device = torch.device('cpu')
    print("Loading Cora...")
    features, labels, adj, train_idx, val_idx, test_idx = load_cora(DATA_DIR)

    base = dict(hidden_dim=64, num_heads=7, subspace_dim=16, eta=0.5,
                lambda_sparse=0.05, lambda_mcr=0.01, lambda_orth=0.01,
                lambda_l1=0.001, dropout=0.5, lr=0.005, weight_decay=5e-4,
                epochs=300, patience=30)

    configs = [
        ('base_fixed',   dict()),
        ('lr_high',      dict(lr=0.01)),
        ('lr_low',       dict(lr=0.002)),
        ('drop06',       dict(dropout=0.6)),
        ('mcr_small',    dict(lambda_mcr=0.001)),
        ('mcr_large',    dict(lambda_mcr=0.05)),
        ('sparse_small', dict(lambda_sparse=0.01)),
        ('sparse_large', dict(lambda_sparse=0.1)),
        ('orth_small',   dict(lambda_orth=0.001)),
        ('best_combo',   dict(lr=0.005, weight_decay=1e-3, dropout=0.6,
                              lambda_mcr=0.005, lambda_orth=0.005, lambda_sparse=0.03)),
    ]

    results = []
    for name, overrides in configs:
        cfg = {**base, **overrides}
        print(f"Running {name}...", flush=True)
        val, test = train_eval(cfg, features, labels, adj, train_idx, val_idx, test_idx, device)
        print(f"  {name}: val={val:.4f}  test={test:.4f}", flush=True)
        results.append((name, val, test))

    results_sorted = sorted(results, key=lambda x: x[1], reverse=True)
    best = results_sorted[0]

    out = ["Whitebox GAT CRATE - Hyperparameter Search Results", "="*60, ""]
    for name, val, test in results:
        out.append(f"  {name:<15} val={val:.4f}  test={test:.4f}")
    out += ["", f"Best: {best[0]}  val={best[1]:.4f}  test={best[2]:.4f}"]

    path = os.path.join(OUT_DIR, "agent_C_result.txt")
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out))
    print('\n'.join(out))
    print(f"\nWritten to {path}")

if __name__ == '__main__':
    main()


