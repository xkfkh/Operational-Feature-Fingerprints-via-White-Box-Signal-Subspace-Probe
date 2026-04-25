# 13 src_v16 / src_v16b Reliability Map and Anti-hub

Goal: replace pairwise specialists with reliability-guided PCA/Ridge fusion and anti-hub correction.

Key audits:
- reliability_map;
- penalty_search_trace;
- antihub_class_stats;
- analysis_report.

Main lesson: reliability and anti-hub statistics were interpretable, but correction mechanisms did not reliably beat the simpler base.

