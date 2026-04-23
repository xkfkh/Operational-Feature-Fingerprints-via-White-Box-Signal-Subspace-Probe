# WG-SRC: White-Box Graph Subspace–Ridge Classifier with Node-Level Signal Atlases

Official repository for **WG-SRC**.

This repository provides:

1. the full implementation of the WG-SRC algorithm,
2. all scripts for experiments, evaluation, ablations, and figure generation,
3. research notes documenting the iteration process, design decisions, failed variants, and the reasoning behind the final method.

WG-SRC is a white-box graph classifier built on an explicit multi-hop graph-signal dictionary, Fisher coordinate selection, class-wise PCA subspaces, and closed-form ridge classification. Beyond predictive performance, it also serves as an intrinsic audit and diagnostic framework by producing node-level mechanism atlases and dataset-level operational fingerprints.
