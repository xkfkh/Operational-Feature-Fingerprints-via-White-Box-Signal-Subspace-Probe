# 11 src_v13 / src_v14 Per-class Dimension Coverage and Floor

Goal: allow different classes to use different PCA dimensions.

src_v13: per-class dimension search.
src_v14: dimension floor and global fallback.

Key audits:
- dim_search_trace;
- true-class residual;
- true-class explained ratio;
- false-positive rate;
- dimension fraction.

