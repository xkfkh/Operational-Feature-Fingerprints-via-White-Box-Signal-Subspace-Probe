"""
Agent F1: E5 + MCR2 best_combo + Residual + LayerNorm
=======================================================
方向：在 E5 动态多跳基础上，融合 v2 的最优超参（best_combo），
加入 residual connection 和 LayerNorm，目标突破 0.80。
"""
import os, pickle, numpy as np, scipy.sparse as sp
import torch, torch.nn as nn, torch.nn.functional as F
from datetime import datetime

DATA_DIR = "D:/桌面/MSR实验复现与创新/planetoid/data"
OUT_DIR  = "D:/桌面/MSR实验复现与创新/results/whitebox_gat_v3"
os.makedirs(OUT_DIR, exist_ok=True)
LOG_PATH = os.path.join(OUT_DIR, "f1_output.txt")
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


def propagate_hops(H, A, num_hops):
    hops = [H]
    for _ in range(num_hops):
        H = A @ H
        hops.append(H)
    return hops


class F1Layer(nn.Module):
    """E5 layer + residual + LayerNorm"""
    def __init__(self, in_dim, out_dim, num_hops, subspace_dim,
                 eta=0.5, lambda_lap=0.1, lambda_sparse=0.05, eps=0.5):
        super().__init__()
        self.num_hops     = num_hops
        self.subspace_dim = subspace_dim
        self.eta          = eta
        self.eps          = eps
        self.lambda_laps  = nn.Parameter(torch.full((num_hops+1,), lambda_lap))
        self.U = nn.ParameterList()
        for _ in range(num_hops + 1):
            raw = torch.randn(in_dim, subspace_dim)
            q, _ = torch.linalg.qr(raw)
            self.U.append(nn.Parameter(q[:, :subspace_dim].clone()))
        self.hop_weights = nn.Parameter(torch.ones(num_hops + 1) / (num_hops + 1))
        self.threshold   = nn.Parameter(torch.full((out_dim,), lambda_sparse))
        self.proj = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else None
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, H, A, adj_mask, L):
        n = H.shape[0]
        hops    = propagate_hops(H, A, self.num_hops)
        weights = F.softmax(self.hop_weights, dim=0)
        laps    = F.softplus(self.lambda_laps)

        agg_total = torch.zeros_like(H.t())
        lap_total = torch.zeros_like(H)

        for k, H_k in enumerate(hops):
            Uk = self.U[k]
            Zk = Uk.t() @ H_k.t()
            coeff = self.subspace_dim / (n * self.eps**2)
            M_k   = torch.eye(self.subspace_dim, device=H.device) + coeff * (Zk @ Zk.t())
            try:
                L_chol = torch.linalg.cholesky(M_k)
                M_inv  = torch.cholesky_inverse(L_chol)
            except Exception:
                M_inv  = torch.linalg.pinv(M_k)
            scores = Zk.t() @ M_inv @ Zk
            scores = scores + (1.0 - adj_mask) * (-1e9)
            alpha  = F.softmax(scores, dim=1)
            grad_k = Uk @ (M_inv @ Zk @ alpha.t())
            agg_total = agg_total + weights[k] * grad_k
            lap_total = lap_total + laps[k] * (L @ H_k)

        H_half = H + self.eta * agg_total.t() - self.eta * lap_total
        if self.proj is not None:
            H_half = self.proj(H_half)

        thr   = self.threshold.unsqueeze(0)
        H_out = torch.sign(H_half) * F.relu(torch.abs(H_half) - thr)

        # residual
        if H_out.shape == H.shape:
            H_out = H_out + H
        H_out = self.norm(H_out)

        orth_loss = torch.tensor(0.0, device=H.device)
        for k in range(len(self.U)):
            UkTUk = self.U[k].t() @ self.U[k]
            I = torch.eye(self.subspace_dim, device=H.device)
            orth_loss = orth_loss + ((UkTUk - I)**2).sum()
            for l in range(k+1, len(self.U)):
                orth_loss = orth_loss + ((self.U[k].t() @ self.U[l])**2).sum()

        lap_smooth = torch.trace(H_out.t() @ L @ H_out)
        return H_out, orth_loss, lap_smooth, weights.detach()


class F1Net(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes,
                 num_hops=2, subspace_dim=16, eta=0.5,
                 lambda_lap=0.1, lambda_sparse=0.05, dropout=0.5, eps=0.5):
        super().__init__()
        self.dropout = dropout
        self.layer1  = F1Layer(in_dim, hidden_dim, num_hops, subspace_dim, eta, lambda_lap, lambda_sparse, eps)
        self.layer2  = F1Layer(hidden_dim, hidden_dim, num_hops, subspace_dim, eta, lambda_lap, lambda_sparse, eps)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, H0, A, adj_mask, L):
        H = F.dropout(H0, p=self.dropout, training=self.training)
        H, o1, ls1, w1 = self.layer1(H, A, adj_mask, L)
        H = F.dropout(H, p=self.dropout, training=self.training)
        H, o2, ls2, w2 = self.layer2(H, A, adj_mask, L)
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
    model = F1Net(
        features.shape[1], cfg['hidden_dim'], num_classes,
        cfg['num_hops'], cfg['subspace_dim'], cfg['eta'],
        cfg['lambda_lap'], cfg['lambda_sparse'], cfg['dropout'], cfg['eps']
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    feat_t   = torch.tensor(features, device=device)
    labels_t = torch.tensor(labels,   device=device, dtype=torch.long)
    adj_t    = torch.tensor(adj_norm,  device=device)
    L_t      = torch.tensor(lap,       device=device)

    best_val, best_test, patience_cnt, best_state = 0.0, 0.0, 0, None

    for epoch in range(1, cfg['epochs']+1):
        model.train(); opt.zero_grad()
        logits, penult, orth_loss, lap_smooth, hop_w = model(feat_t, adj_t, adj_t, L_t)
        loss_ce  = F.cross_entropy(logits[train_idx], labels_t[train_idx])
        loss_mcr = mcr2_loss(penult[train_idx], labels_t[train_idx], num_classes)
        loss = loss_ce + cfg['lambda_mcr']*loss_mcr + cfg['lambda_orth']*orth_loss
        loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            logits_e, _, _, _, hw = model(feat_t, adj_t, adj_t, L_t)
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
            w_str = ' '.join([f'{w:.3f}' for w in hw.cpu().numpy()])
            log(f"  [{run_name}] ep={epoch:3d}  loss={loss.item():.4f}"
                f"  ce={loss_ce.item():.4f}  val={val_acc:.4f}  test={test_acc:.4f}"
                f"  hop_w=[{w_str}]")

    return best_val, best_test


def main():
    device = torch.device('cpu')
    log(f"=== Agent F1: E5 + Residual + LayerNorm + MCR2 best_combo  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    log("方向：E5动态多跳 + residual connection + LayerNorm + v2 best_combo 超参\n")

    features, labels, adj_norm, lap, train_idx, val_idx, test_idx = load_cora(DATA_DIR)
    log(f"Cora: {features.shape[0]} nodes, {features.shape[1]} features, {int(labels.max())+1} classes\n")

    # E5 best_combo 超参基础
    base = dict(hidden_dim=64, num_hops=2, subspace_dim=16, eta=0.5, eps=0.5,
                lambda_lap=0.3, lambda_sparse=0.05,
                lambda_mcr=0.005, lambda_orth=0.005,
                dropout=0.6, lr=0.005, wd=1e-3, epochs=400, patience=50, seed=42)

    experiments = [
        ('f1_base',          dict()),
        ('f1_mcr_up',        dict(lambda_mcr=0.01)),
        ('f1_mcr_down',      dict(lambda_mcr=0.001)),
        ('f1_lap_low',       dict(lambda_lap=0.1)),
        ('f1_sub32',         dict(subspace_dim=32)),
        ('f1_hop3',          dict(num_hops=3)),
        ('f1_hidden128',     dict(hidden_dim=128, subspace_dim=32)),
        ('f1_best_combo',    dict(hidden_dim=128, subspace_dim=32, num_hops=2,
                                  lambda_lap=0.3, lambda_mcr=0.005, lambda_orth=0.005,
                                  dropout=0.6, lr=0.005, wd=1e-3)),
        ('f1_multiseed_s1',  dict(seed=10)),
        ('f1_multiseed_s2',  dict(seed=20)),
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
    log("Agent F1 - 结果汇总")
    log("="*60)
    best_val, best_name = 0, ''
    for name, val, test in results:
        log(f"  {name:<20} val={val:.4f}  test={test:.4f}")
        if val > best_val:
            best_val, best_name = val, name
    best_test = next(t for n,v,t in results if n==best_name)
    log(f"\n最优: {best_name}  val={best_val:.4f}  test={best_test:.4f}")

    res_path = os.path.join(OUT_DIR, "f1_result.txt")
    with open(res_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))
    log(f"\nWritten to {res_path}")
    save_log()

if __name__ == '__main__':
    main()


