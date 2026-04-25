# Operational Feature Fingerprints of Graph Datasets via a White-Box Signal-Subspace Probe

**A first-author-led white-box graph learning project that turns node prediction into dataset diagnosis.**

This repository contains the implementation, experiment package, atlas/fingerprint analysis, and full research-iteration record for **WG-SRC**, a white-box signal–subspace probe developed for the paper:

**Operational Feature Fingerprints of Graph Datasets via a White-Box Signal-Subspace Probe**

The project was led by **Xiong Yuchen** as first author, with **Yeap Swee Keong** and **Ban Zhen Hong** as corresponding supervisors.

Rather than learning an opaque message-passing representation, WG-SRC constructs a named graph-signal dictionary, selects Fisher coordinates, fits class-wise PCA subspaces, solves closed-form multi-alpha Ridge classifiers, and fuses PCA/Ridge scores by validation.

The key point is dual use: the same fitted white-box scaffold both predicts node labels and produces a node-level mechanism atlas. By aggregating this atlas, WG-SRC yields operational feature fingerprints of graph datasets, describing raw-feature reliance, low-pass propagation reliance, high-pass sensitivity, class-subspace complexity, and ridge-boundary dependence.

> The main contribution is not only a accurate white-box node classifier, but a fitted predictive scaffold that converts graph datasets into measurable operational fingerprints.

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
## Why the Research Iteration Record Is Included

This repository is intended to show not only the final WG-SRC scaffold, but also the first-author-led research process that produced it.

The iteration record is included because the main value of this project is not a one-shot accuracy improvement. It shows how I formulated a white-box graph-learning problem, tested several mechanistic routes, identified why some apparently promising designs failed, and used statistical audits beyond accuracy to decide which mechanisms should be kept or rejected.

The research trajectory documents:

1. how the project started from MCR² / ReduNet / CRATE representation geometry;
2. why early white-box GNN unfolding was not clean enough as a final mechanism;
3. how the work moved toward closed-form graph signal–subspace classification;
4. how Chameleon experiments exposed the need for explicit high-pass graph differences;
5. how later versions were judged by hard-node statistics, parameter effects, fixed / broken / redirect audits, per-class PCA dimension traces, pairwise-specialist overfitting checks, reliability maps, anti-hub analysis, and OOF candidate traces;
6. why more complex variants were rejected when they improved validation but failed to transfer, increased complexity, or weakened the diagnostic role of the scaffold;
7. why `src_v16c` was selected as the cleanest balance between prediction, interpretability, runtime, and operational dataset fingerprinting.

In short, the record is included to make the research path auditable: problem reframing, negative results, mechanism diagnosis, statistical decision-making, and final convergence to a white-box signal–subspace probe.

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
