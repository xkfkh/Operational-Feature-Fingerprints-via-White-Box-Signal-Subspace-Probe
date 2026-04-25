# 06 src_v7 Chameleon High-pass / Fisher / PCA / Ridge Breakthrough

Goal: handle Chameleon heterophily.

Main discoveries:
- low-pass smoothing alone is insufficient;
- high-pass differences such as X-PX and PX-P2X are useful;
- PCA residual and Ridge boundary are complementary;
- Fisher coordinate selection gives a major improvement.

Main lesson: explicit graph-signal dictionary + Fisher + PCA + Ridge became the backbone.

