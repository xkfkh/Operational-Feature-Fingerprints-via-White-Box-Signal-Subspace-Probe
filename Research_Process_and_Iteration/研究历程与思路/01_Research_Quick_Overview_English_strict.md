# WG-SRC / src_v16c One-Page Iteration Overview

This file is for quick reading. The complete code, results, and statistical audit materials are stored chronologically in stage folders `01_` to `16_`.

## One-sentence summary

This project originally started from the white-box representation learning ideas of MCR² / ReduNet / CRATE and attempted to explain the internal mechanisms of graph neural networks. Later, it was found that trained white-box GNNs were still not clean enough, so the project turned to explicit graph signals, class-wise PCA subspaces, Fisher coordinate selection, and closed-form Ridge classification, finally forming WG-SRC: a white-box graph classifier that can both predict and generate dataset and feature characteristics.

## How the project direction changed by my own decisions
The project went through several stages:

1. At first, I wanted to build a truly interpretable white-box GNN and reduce the difficult-to-explain end-to-end training process in traditional black-box GNNs.

2. Later, I began to build a white-box algorithmic framework and decomposed the graph classification process into testable modules such as graph-signal construction, subspace representation, feature selection, and closed-form classifiers.

3. During the experiments, I realized that comparing accuracy alone was not enough to explain model behavior, so I further introduced finer-grained statistical analysis to observe what was happening across different classes, features, and datasets.

4. Furthermore, I found that this white-box framework could be used not only for classification, but also to characterize the structural and feature patterns of the dataset itself, forming a kind of dataset-specific fingerprint analysis.

5. Next, I plan to use these white-box statistical results to dissect the performance differences of traditional black-box GNNs on different datasets, infer the relationships between different black-box components and dataset fingerprints, and continue improving the model and experimental analysis methods on that basis.


## Final method

The final selected version is:

`14_src_v16c_FINAL_enhanced_multihop_Fisher_PCA_multi_alpha_Ridge`

Core components:

1. a nine-block explicit graph-signal dictionary;
2. Fisher coordinate selection;
3. class-wise PCA residual;
4. multi-alpha closed-form Ridge;
5. validation-selected PCA/Ridge fusion;
6. node-level signal atlas.

## Key research transitions

### Transition 1: From explaining black boxes to building white boxes

At the beginning, the goal was to use MCR², coding rate, effective rank, principal angle, and related metrics to explain internal model representations. Later, I wanted to try making the algorithm itself white-box as well.

Corresponding folders:

- `01_MCR2_layerwise_geometric_tracing`
- `02_whitebox_GNN_unfolding_before_G_series`

### Transition 2: From white-box GNN layers to closed-form subspace classifiers

White-box GNN routes such as G1 / CRGain were once relatively strong, but later analysis showed that simply adding an interpretable formula to a neural layer was still not clean enough. So the project moved from “white-box GNNs with trainable layers” to explicit matrix operations and closed-form classifiers.

Corresponding folders:

- `03_G_series_CRgain_frequency_variational_rank_routes`
- `04_L_series_repairs_of_whitebox_GNN_route`
- `05_src_v1_layered_closed_form_whitebox_pipeline`

### Transition 3: Chameleon exposed the importance of high-pass graph signals, so the white-box framework was decomposed into interpretably separable components, and component validity was tested by replacing components

On heterophilic graphs such as Chameleon, pure low-pass smoothing was not enough. Experiments showed that high-pass graph differences such as `X-PX` and `PX-P²X` were very important for classification. This step directly led to the final nine-block graph-signal dictionary.

Corresponding folders:
- `05b_src_v3_to_src_v6_WhiteBox_GSD_pipeline_Layer3_coupling_and_packaging`
- `06_src_v7_Chameleon_highpass_Fisher_PCA_Ridge_breakthrough`

### Transition 4: From accuracy-driven to more detailed statistics, such as PCA error-node-set characteristics

After several more iterations with only small improvements, starting from v8, iteration no longer only looked at accuracy, but instead looked at self-designed statistics such as:

- hard nodes;
- parameter effects on misclassified nodes;
- branch fixed / broken / wrong redirect;
- per-class PCA dimension trace;
- whether pairwise specialists were only validation overfit;
- whether the reliability map truly improved test transfer;
- whether OOF label beams repaired more than they damaged.

Corresponding folders:

- `07_v8_error_node_and_parameter_effect_audit`
- `09_src_v10_safe_adaptive_branch_fixed_broken_redirect_audit`
- `11_src_v13_src_v14_per_class_dimension_coverage_and_floor`
- `12_src_v15_score_calibration_pairwise_specialist_validation_overfit`
- `13_src_v16_src_v16b_reliability_map_and_antihub`
- `15_rejected_after_v16c_v16d_v16e_v16f_v16g_v17`

### Transition 5: src_v16c was finally selected because it was the cleanest, performed well in both predictive ability and interpretability, and at the end it also began to produce white-box-style results such as dataset fingerprints

Versions such as v16e and v17 tried more complex mechanisms such as iterative labels and OOF label beams, but they were slower, more complex, and did not stably outperform src_v16c. src_v16c was finally selected because it had the best balance among accuracy, simplicity, white-box clarity, runtime cost, and atlas interpretability. It also produced dataset fingerprints across different datasets, and later attempts were made to connect fingerprint characteristics with components of traditional black-box models.

Corresponding folders:

- `14_src_v16c_FINAL_enhanced_multihop_Fisher_PCA_multi_alpha_Ridge`
- `16_FINAL_paper_experiment_package_and_six_dataset_results`

## Summary

This file shows the process from project initiation to completion of the research:

1. starting from theoretical questions: MCR² / subspace representations / white-box learning;
2. discovering internal methodological problems: trained white-box GNNs were still not clean enough;
3. pivoting after failure: from white-box GNNs to closed-form classifiers;
4. changing the method according to data characteristics: heterophily requires high-pass graph signals;
5. making decisions through statistical audits: fixed / broken / redirect, dimension trace, reliability map;
6. ruling out complex but unstable options: pairwise specialists, iterative labels, OOF beam;
7. finally forming the reproducible, interpretable, and diagnostic WG-SRC framework.
