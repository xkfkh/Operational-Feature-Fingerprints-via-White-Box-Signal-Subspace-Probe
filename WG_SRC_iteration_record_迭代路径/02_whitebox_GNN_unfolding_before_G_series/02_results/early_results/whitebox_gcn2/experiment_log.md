# Whitebox GCN2 (Agent 4) Experiment Log

## Method

White-box GCN with full ReduNet unrolled gradient ascent.

### Layer formula

```
E^l   = (I + alpha/n * H^T H)^{-1}
C_k^l = (I + alpha/n_k * H_k^T H_k)^{-1}
H'    = H + eta*(H @ E - A_norm @ sum_k(gamma_k*(H*pi_k) @ C_k))
H_out = sign(H') * ReLU(|H'| - threshold)  [ISTA]
```

Loss = CE - lambda_mcr * DeltaR + lambda_orth * L_orth

### Key insight

DeltaR ~ 10-30 >> CE ~ 2.0, so lambda_mcr must be < 1e-4 for CE to dominate.

## Results

| Config | val_acc | test_acc |
|--------|---------|----------|
| CE_dominant | 0.3320 | 0.2830 |
| tiny_mcr | 0.3320 | 0.2840 |
| h128_tiny_mcr | 0.4520 | 0.4470 |
| no_mcr | 0.3320 | 0.2840 |
| no_mcr_h128 | 0.4680 | 0.4400 |
| no_mcr_drop03 | 0.4360 | 0.4060 |

**Best: h128_tiny_mcr  test_acc=0.4470**

Target >0.75: NOT ACHIEVED



