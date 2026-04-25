# Operational Feature Fingerprints of Graph Datasets via a White-Box Signal-Subspace Probe

**A white-box graph signal–subspace probe for revealing how graph datasets rely on raw features, low-pass propagation, high-pass differences, class-subspace geometry, and ridge-boundary decisions.**

This repository contains the implementation, experiments, and full research-iteration records for **WG-SRC**, the white-box signal–subspace probe used in the paper.

WG-SRC is not only a node classifier. It is also an intrinsic diagnostic instrument: every signal block and decision module is named, measurable, and auditable. Therefore, the fitted model produces not only predictions, but also **node-level mechanism atlases** and **dataset-level operational feature fingerprints**.

---

## Start Here

For a quick understanding of the research process:

- [Research Process Quick Overview 中文](./WG_SRC_iteration_record_迭代路径文件夹/01_科研过程速读_中文.md)
- [Full Research Narrative English](./WG_SRC_iteration_record_迭代路径文件夹/02_Full_Research_Narrative_English_strict_long.md)
- [Complete Iteration Logic 中文](./WG_SRC_iteration_record_迭代路径文件夹/02_WG-SRC_src_v16c完整科研迭代逻辑.md)

For the final algorithm and experiments:

- `src_v16c_paper_experiments/`
- `paperexp/core.py`
- `run_all_experiments.py`

---

## What Makes This Repository Different

Most graph-learning repositories only provide final code and accuracy tables.

This repository also records **how the method was discovered**:

1. Starting from MCR² / ReduNet / CRATE representation geometry.
2. Attempting white-box GNN unfolding.
3. Discovering that interpretable formulas inside trainable GNN layers were not sufficient for a truly white-box mechanism.
4. Moving to closed-form graph subspace classifiers.
5. Finding that Chameleon-like heterophilic graphs require explicit high-pass graph differences.
6. Replacing accuracy-only search with node-level and mechanism-level statistical audits.
7. Rejecting more complex variants when they improved validation but failed to transfer to test or harmed interpretability.
8. Selecting `src_v16c` because it achieved the best balance among performance, simplicity, runtime, white-box clarity, and dataset-fingerprint analysis.

---

## Core Idea

WG-SRC uses:

1. an explicit multi-block graph-signal dictionary;
2. Fisher coordinate selection;
3. class-wise PCA subspace residuals;
4. closed-form multi-alpha Ridge classification;
5. validation-selected PCA/Ridge fusion;
6. node-level signal atlas and dataset-level operational fingerprint.

The final goal is not merely to classify nodes, but to expose **which mechanisms a dataset uses during classification**.
