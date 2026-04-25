# WG-SRC / src_v16c Complete Research Iteration Logic and the Reasons for Statistical-Driven Decisions

> This document is intended to systematically organize the full iteration path of WG-SRC / src_v16c from the initial idea to the final paper experiment package.  
> The focus is not to simply list “which version had the highest accuracy,” but to organize why each generation appeared, what problem it solved, what the statistical results showed, why it was continued or abandoned, and how these iterations eventually converged to `src_v16c` and to WG-SRC in the paper.

---

## 0. Position of this file

This document organizes four types of materials:

1. **Early Claude JSONL iteration records**  
   These record the model-development process from MCR², ReduNet, CRATE, white-box GNN, G1/G2/G3/G4, and the L-series to `src_v16c`.

2. **Detailed path records of the main folders**  
   These record the real version directories, script directories, result directories, and the final paper-experiment package path in the project.

3. **The `version_statistical_summary_pack` statistical package**  
   This package contains a large number of statistical audit results from v8 to v17, including for example:
   - error-node statistics;
   - parameter-effect statistics;
   - branch fixed / broken / redirect statistics;
   - class-pair geometry;
   - per-class dimension search traces;
   - class-bias traces;
   - pairwise-specialist traces;
   - reliability maps;
   - anti-hub penalty search;
   - OOF label-beam candidate traces.

4. **The final paper content**  
   In the final paper, WG-SRC is defined as a white-box graph classifier and graph-dataset diagnostic probe:  
   explicit graph-signal dictionary + Fisher coordinate selection + class-wise PCA subspaces + closed-form Ridge classification + validation fusion + node-level signal atlas.

The purpose of this file is not to rewrite the paper, but to organize the **real research process** into a readable, auditable Markdown document that can be placed into GitHub `research_notes/`.

---

## 1. Changes in the overall research goal

### 1.1 Initial goal

At the beginning, the project was not aimed at simply improving graph-classification accuracy. Instead, it started from a more fundamental question:

> Do machine-learning models internally form interpretable low-dimensional structures?  
> Can black-box neural networks be replaced by mechanisms that are more white-box and more linear-algebraic?

The initial theoretical background came from:

- MCR²;
- ReduNet;
- CRATE;
- subspace learning;
- coding rate;
- within-class compression and between-class separation.

The core intuition was:

> A good representation space should place samples from the same class into relatively low-dimensional class subspaces, while making subspaces from different classes as separable as possible. If this structure can be constructed explicitly, then it may not be necessary to rely completely on black-box neural networks.

---

### 1.2 Later goal

As the experiments progressed, the research goal gradually changed from:

```text
explaining the internal representations of black-box models
```

to:

```text
constructing a graph classifier that is itself white-box
```

which finally became:

```text
WG-SRC = White-Box Graph Subspace–Ridge Classifier
```

The final method is not a GNN. Instead, it is a white-box graph-classification framework composed of linear-algebraic modules:

```text
explicit graph-signal dictionary
→ Fisher coordinate selection
→ class-wise PCA subspace residual
→ closed-form multi-alpha Ridge
→ validation-selected PCA/Ridge score fusion
→ node-level atlas / dataset fingerprint
```

---

## 2. Overall version lineage

The main evolution path of the whole project can be summarized as:

```text
MCR² / ReduNet / CRATE representation-geometry diagnosis
        ↓
layerwise geometric tracing
        ↓
whitebox_gat / whitebox_gcn / CRATE-like GNN unfolding
        ↓
G-series: G1 / G2 / G3 / G4 hop-selection routes
        ↓
G1 CRGain temporarily wins
        ↓
discovery that the theoretical white-box gradient path of G1 was covered by a trainable projection
        ↓
L-series: repairing the white-box GNN route
        ↓
shift toward layered closed-form white-box algorithms
        ↓
src_v1: Tikhonov smoothing + class subspace + response/impact + closed-form predictor
        ↓
src_v3-v6: WhiteBox-GSD pipelining, HASDC, raw/smooth coupling, packaging
        ↓
src_v7: explicit high-pass / multihop + Fisher + PCA + Ridge on Chameleon
        ↓
v8: error-node and parameter-effect statistical audit
        ↓
src_v9-v15: adaptive branch, safe branch, per-class dimension, pairwise specialist
        ↓
src_v16: reliability map + anti-hub
        ↓
src_v16b: per-class reliability + greedy dimension
        ↓
src_v16c: 9-block enhanced multihop + Fisher + PCA + multi-alpha Ridge
        ↓
v16d/e/f/g/v17: complex variants continued to be explored but were excluded
        ↓
src_v16c_paper_experiments: formal paper experiment package
        ↓
WG-SRC paper version
```

This path shows that:

> The final WG-SRC was not directly chosen from a single version with high accuracy. Instead, it was obtained after repeated mechanism failures, statistical audits, and structural simplifications.

---

# Part I: From black-box explanation to white-box GNN unfolding

---

## 3. MCR² and layerwise geometric tracing

### 3.1 Corresponding stage

Corresponding organized directory:

```text
01_MCR2_layerwise_geometric_tracing
```

Representative files include:

```text
mcr2_orth.py
mcr2_trace_fashionmnist.py
```

---

### 3.2 The problem being addressed at that time

The earliest goal was to test whether:

1. neural-network hidden layers undergo within-class compression;
2. different classes gradually separate;
3. each class forms a low-dimensional subspace;
4. the representation dimension becomes more structured layer by layer;
5. MCR² can explain representation geometry inside neural networks.

Therefore, this stage mainly built **diagnostic tools**, not the final classifier.

---

### 3.3 Core metrics

The main quantities examined at that time included:

- coding rate;
- MCR² delta;
- effective rank;
- principal angle;
- subspace coherence;
- within-class scatter;
- between-class scatter;
- layerwise representation trajectory.

The basic idea of MCR² can be written as:

\[
\Delta R(Z)=R(Z)-R_c(Z)
\]

where:

- \(Z\): the sample representation matrix at a certain layer of the model;
- \(R(Z)\): the coding rate of the whole set of samples;
- \(R_c(Z)\): the within-class coding rate computed by class;
- \(\Delta R(Z)\): the difference between overall representation complexity and within-class complexity.

Intuitively:

- the larger \(\Delta R(Z)\), the more expanded the classes are in the overall space;
- the tighter the within-class structure, the more separated the classes are;
- but this does not automatically imply orthogonal subspaces, nor does it automatically imply more stable classification.

---

### 3.4 The MCR² + orthogonality-penalty counterexample experiment

At this stage, a synthetic 10-class low-rank Gaussian-subspace experiment was also conducted:

- 100 train samples per class;
- 100 test samples per class;
- MLP hidden dims `[128, 64, 32]`;
- training for 15 epochs.

Three training modes were compared:

| Training mode | Meaning |
|---|---|
| CE only | cross-entropy only |
| CE + MCR² | cross-entropy plus MCR² regularization |
| CE + MCR² + Orth | plus an explicit subspace-orthogonality penalty |

The explicit orthogonality penalty was:

\[
L_{\mathrm{orth}}
=
\sum_{i\ne j}
\frac{\|\Sigma_i \Sigma_j\|_F^2}
{\|\Sigma_i\|_F \|\Sigma_j\|_F+\epsilon}
\]

where:

- \(\Sigma_i = Z_i^\top Z_i / n_i\), the within-class covariance of class \(i\);
- \(Z_i\), the representations of class \(i\) at the current layer;
- \(n_i\), the number of samples in class \(i\);
- \(\|\cdot\|_F\), the Frobenius norm;
- \(\epsilon\), a term to prevent division by zero;
- the smaller \(L_{\mathrm{orth}}\), the closer the class subspaces are to being orthogonal.

The recorded results were:

| Mode | Last Hidden ΔR | Last Hidden Coherence | Test Acc |
|---|---:|---:|---:|
| CE | 8.6702 | 0.9962 | 1.0000 |
| CE + MCR² | 24.1395 | 0.9731 | 1.0000 |
| CE + MCR² + Orth | 10.1174 | 0.9741 | 0.7943 |

---

### 3.5 Conclusion of this stage

The key conclusion was not that “MCR² completely succeeded,” but rather:

1. MCR² can indeed change hidden-layer geometry.  
   `Last Hidden ΔR` increased from `8.6702` to `24.1395`.

2. However, MCR² does not automatically create truly orthogonal inter-class subspaces.  
   Coherence remained high.

3. An explicit orthogonality penalty did not stably improve coherence and instead damaged classification performance.  
   Test Acc dropped from `1.0000` to `0.7943`.

4. Therefore, directly forcing hidden layers of neural networks to satisfy CE, MCR², and orthogonality constraints at the same time is not stable.

The core lesson from this step was:

> It is not enough to simply force “interpretation metrics” into the loss of a black-box network.  
> If white-box behavior is desired, the model structure itself may need to be reconstructed.

---

## 4. Attempted white-box GNN unfolding

### 4.1 Corresponding stage

Corresponding directory:

```text
02_whitebox_GNN_unfolding_before_G_series
```

Representative files:

```text
whitebox_gat2.py
whitebox_gcn2.py
whitebox_gat_crate.py
whitebox_gcn_crate.py
whitebox_gat_v3_d1.py
...
whitebox_gat_v3_f3.py
```

---

### 4.2 The problem being addressed at that time

In ordinary GNNs, message passing is trained and the internal mechanism is opaque.  
This stage attempted to write a GNN layer as an explicit unfolding similar to an optimization algorithm.

The rough form was:

\[
H_{\frac12}
=
H
+
\eta \nabla R
-
\eta \lambda_{\mathrm{lap}} L H
\]

followed by a proximal operator:

\[
H_{\mathrm{out}}=\operatorname{prox}(H_{\frac12})
\]

where:

- \(H\): the current node representation;
- \(\eta\): the step size;
- \(\nabla R\): a coding-rate or rate-reduction-related gradient;
- \(\lambda_{\mathrm{lap}}\): graph-Laplacian regularization strength;
- \(L\): the graph Laplacian;
- \(LH\): the graph-smoothing / graph-energy term;
- \(\operatorname{prox}\): the proximal operator, which in early implementations corresponded to soft-thresholding / LayerNorm and related operations.

---

### 4.3 The meaning of this stage

This stage proposed an important direction:

```text
A GNN layer should not be only a black-box neural layer;
it can be viewed as an unfolding of graph regularization + rate reduction + proximal step.
```

But several problems quickly appeared:

1. trainable neural projection was still retained;
2. the formula explanation and the actual forward path were not necessarily the same;
3. the mechanism was complex and not easy to audit;
4. it could not naturally generate the node-level atlas needed later.

Therefore, the project entered the G-series to test more explicit hop-selection mechanisms.

---

## 5. G-series: four white-box hop-selection routes

### 5.1 Corresponding stage

Corresponding directory:

```text
03_G_series_CRgain_frequency_variational_rank_routes
```

The core question was:

> If different graph hops are to be selected explicitly, what white-box quantity should determine the hop weights?

---

### 5.2 G1: CRGain

Representative files:

```text
run_g1_cr_gain.py
run_g1_fixed_small.py
run_g1_small_clean.py
```

Core idea:

\[
R_k
=
\frac12 \log\det(I+\alpha Z_k^\top Z_k)
\]

\[
w_k
=
\operatorname{softmax}\left(\frac{R_k}{\tau}\right)
\]

where:

- \(Z_k\): the node representation at hop \(k\);
- \(R_k\): the coding rate of hop \(k\);
- \(w_k\): the weight of hop \(k\);
- \(\tau\): the softmax temperature;
- \(\alpha\): the scale factor.

Intuitive explanation:

```text
The hop whose representation has a higher coding rate
receives a larger weight.
```

Research meaning:

- G1 showed that coding rate can provide a meaningful signal for choosing graph-propagation hops;
- it performed best in the early G-series;
- it therefore became a temporary main-route candidate.

---

### 5.3 G2: Frequency / Laplacian energy

Core idea:

\[
E_k
=
\frac{\operatorname{tr}(H_k^\top L H_k)}{nd}
\]

then:

\[
w_k
=
\operatorname{softmax}\left(-\frac{E_k}{\tau}\right)
\]

where:

- \(E_k\): the graph-frequency energy of hop \(k\);
- \(L\): the graph Laplacian;
- \(n\): the number of nodes;
- \(d\): the feature dimension.

Reason for failure:

```text
Laplacian energy can only indicate whether a representation is smooth,
but smoothness is not equal to discriminative power.
```

This was further verified later on Chameleon: in heterophilic graphs, low-pass smoothing can instead remove useful differences.

---

### 5.4 G3: Variational hop selection

Core idea:

\[
q_k
=
\operatorname{softmax}\left(\frac{R_k}{\tau}\right)
\]

and then adding:

\[
\mathrm{KL}(q\|u)
\]

where:

- \(q\): the hop distribution;
- \(u\): the uniform distribution;
- the KL term is used to prevent collapse of the hop distribution.

Reason for failure:

```text
The hop distribution can easily collapse;
the variational form increases complexity but does not bring stable discriminative gains.
```

---

### 5.5 G4: Effective rank / subspace complexity

Core idea:

\[
r_{\mathrm{eff}}
=
\frac{\operatorname{tr}(\Sigma)^2}
{\operatorname{tr}(\Sigma^2)}
\]

using effective rank to measure how much a hop representation expands dimensionally.

Reason for failure:

```text
A high effective rank only indicates that a representation is expanded;
it does not indicate that it is more separable by class.
```

---

### 5.6 The key transition in the G-series: the “fake white-box” problem of G1

Although G1 performed best in the early stage, later code audits revealed a crucial problem.

In theory, the forward pass should use:

```text
H_half_in = H + η * grad_contrib - η * λ_lap * lap_contrib
```

But in the actual implementation, the following was later executed:

```text
H_half = self.out_proj(Z_agg)
```

This means:

```text
The white-box gradient step H_half_in was computed,
but the quantity that actually entered the later soft-threshold / LayerNorm path was H_half after a trainable projection.
```

In other words:

> On paper G1 looked like a white-box unfolding, but the path that actually dominated the output still contained a neural projection layer.

This was a very important discovery. It meant that G1 could not become the final strictly white-box algorithm.

---

## 6. L-series: repairing the white-box GNN route

### 6.1 Corresponding stage

Corresponding directory:

```text
04_L_series_repairs_of_whitebox_GNN_route
```

Representative files:

```text
run_l1_expansion_compression.py
run_l1_ista_proximal.py
run_l1_signal_noise.py
run_l1_tau_annealing.py
run_l1_graph_aware_cr.py
run_l3_multiscale.py
run_l3_adaptive_gate.py
```

---

### 6.2 Directions explored

This stage continued trying to repair the mechanism problems of white-box GNNs. The main explored directions included:

1. ReduNet-style expansion + compression;
2. CRATE / ISTA-style proximal coding;
3. signal-noise separation;
4. temperature annealing;
5. graph-aware coding rate;
6. multi-scale GCN;
7. adaptive gate.

---

### 6.3 Results and reasons

The L-series showed that white-box-inspired neural models are promising.  
Some versions performed well on Cora-like datasets, especially the multi-scale GCN and adaptive-gate directions.

However, this route did not become the final route, because:

1. it still relied on trained neural layers;
2. it still contained gates / projections / learned weights;
3. it was not sufficiently closed-form;
4. it was not sufficiently suitable as a dataset diagnostic probe;
5. it could not naturally output the dense node-level atlas required by the final paper.

Therefore, the direction shifted from:

```text
repairing white-box-inspired GNNs
```

to:

```text
explicit graph signals + subspaces + closed-form classifiers
```

---

# Part II: From white-box GNN to closed-form subspace classifiers

---

## 7. src_v1: layered closed-form white-box pipeline

### 7.1 Corresponding stage

Corresponding directory:

```text
05_src_v1_layered_closed_form_whitebox_pipeline
```

Main modules:

```text
data_loader.py
layer1_tikhonov.py
layer2_subspace.py
layer3_response.py
layer4_impact.py
layer5_predictor.py
```

---

### 7.2 Goal

Construct a white-box pipeline that does not depend on black-box neural layers:

```text
graph smoothing
→ class subspace
→ node responses to each class
→ residuals and impact measurements
→ closed-form prediction
```

---

### 7.3 Layer 1: Tikhonov graph smoothing

Core formula:

\[
Z=(I+\lambda L)^{-1}X
\]

where:

- \(X\): the original node-feature matrix;
- \(L\): the graph Laplacian;
- \(\lambda\): the smoothing strength;
- \(I\): the identity matrix;
- \(Z\): the smoothed node features.

Equivalent optimization objective:

\[
\min_Z
\|Z-X\|_F^2
+
\lambda \operatorname{tr}(Z^\top LZ)
\]

Interpretation:

- the first term prevents \(Z\) from deviating too far from the original features;
- the second term makes neighboring node representations smoother;
- the larger \(\lambda\), the stronger the graph smoothing.

---

### 7.4 Layer 2: class PCA subspaces

For each class \(c\), fit:

\[
\mu_c,\quad B_c
\]

where:

- \(\mu_c\): the center of class \(c\);
- \(B_c\): the PCA-subspace basis of class \(c\);
- \(B_c^\top B_c=I\).

This module is the direct origin of the final class-wise PCA subspace in WG-SRC.

---

### 7.5 Layer 3 / 4: responses, residuals, and impact

The key residual form is:

\[
R_{\perp,c}(z)
=
\|(I-B_cB_c^\top)(z-\mu_c)\|^2
\]

where:

- \(I-B_cB_c^\top\): the projection operator to the orthogonal complement of the class subspace;
- \(z-\mu_c\): the displacement of the node representation from the class center;
- \(R_{\perp,c}(z)\): the residual showing how much node \(z\) cannot be explained by the subspace of class \(c\).

The smaller this residual, the closer the node is to the geometry of class \(c\).

---

### 7.6 The meaning and problem of src_v1

Meaning:

```text
src_v1 was the key bridge from white-box GNNs to closed-form white-box classifiers.
```

It moved the project from “training an interpretable GNN layer” to:

```text
explicit linear operators
+ class subspaces
+ geometric residuals
+ closed-form prediction
```

Problem:

```text
src_v1 still leaned toward graph smoothing and homophily assumptions;
heterophilic graphs such as Chameleon exposed the problem that neighbors are not necessarily of the same class.
```

Therefore, later stages had to study explicit decompositions of raw / smooth / high-pass graph signals.

---

## 8. src_v3 to src_v6: WhiteBox-GSD pipelining, HASDC, and raw/smooth coupling

### 8.1 Corresponding stage

Corresponding directory:

```text
05b_src_v3_to_src_v6_WhiteBox_GSD_pipeline_Layer3_coupling_and_packaging
```

This stage was not yet the final algorithm itself, but it completed the transition from a prototype to a more systematic framework.

---

### 8.2 src_v3: pipeline formation

Main files:

```text
data_loader.py
layer1_tikhonov.py
layer2_subspace.py
layer3_discriminative.py
framework/configs.py
framework/pipeline.py
run_all.py
```

Main meaning:

```text
It organized the layered white-box idea of src_v1 into a configurable pipeline.
```

The flow became:

```text
data loading
→ Layer1 graph smoothing
→ Layer2 class subspace
→ Layer3 discriminative response
→ pipeline/config management
→ one-click execution
```

Limitation:

```text
It still mainly relied on smooth representations,
and had not yet explicitly separated raw / low-pass / high-pass.
```

---

### 8.3 src_v4: Cora geometric-reactive validation

Recorded representative result:

```text
dataset: cora
pipeline: WhiteBoxPipeline-v3
lambda: 10.0
sub_dim: 12
eta_react: 0.1
train: 0.9786
val:   0.8000
test:  0.8110
time:  25.07s
```

Meaning:

```text
The WhiteBox-GSD pipeline could run on a real citation graph,
and it could produce structured results.
```

Limitation:

```text
Cora is a relatively homophilic citation graph;
working on Cora does not imply working on heterophilic graphs such as Chameleon.
```

---

### 8.4 src_v5: HASDC and Layer3 coupling

At src_v5, the project began to seriously address:

```text
How should overlapping class subspaces be handled?
Which is more important, raw features or smoothed features?
```

The older Layer3 idea leaned toward:

```text
subspace overlap -> suppress
```

That is, if class subspaces overlapped, the overlapping direction was suppressed.

The problem is:

```text
Not every overlapping direction is bad;
some shared directions may carry useful semantics.
```

Therefore HASDC / raw-smooth coupling was introduced:

```text
smooth channel: Z = (I + λL)^(-1)X
raw channel:    X
```

and the following were tested:

- raw_plain_residual;
- smooth_plain_residual;
- raw_dynamic_layer3;
- smooth_dynamic_layer3;
- hasdc_channel_select;
- hasdc_merged;
- hasdc_gated.

---

### 8.5 What the HASDC results revealed

#### Small WebKB graphs

In early Cornell results:

```text
raw_plain_residual test ≈ 0.7568
smooth_plain_residual test ≈ 0.5946
hasdc_gated test ≈ 0.4865 or lower
```

This showed that:

```text
On small WebKB graphs such as Cornell, raw features may be clearly more reliable than smooth features.
```

#### Cornell / Texas / Wisconsin Phase A

Representative results:

```text
Cornell:
hasdc_channel_select test ≈ 0.7514
raw_plain_residual   test ≈ 0.7514
smooth_plain_residual test ≈ 0.6973

Texas:
hasdc_channel_select test ≈ 0.8162
raw_plain_residual   test ≈ 0.8108
smooth_plain_residual test ≈ 0.7568

Wisconsin:
hasdc_channel_select test ≈ 0.8431
raw_plain_residual   test ≈ 0.8431
smooth_plain_residual test ≈ 0.7451
```

Conclusion:

```text
Both raw and smooth signals matter;
different datasets require different signals;
but in many cases channel-select merely picks the better single channel,
rather than forming a stable new coupling mechanism.
```

#### Chameleon / Squirrel / Actor

Early Chameleon results:

```text
smooth_plain_residual test ≈ 0.4327
hasdc_channel_select  test ≈ 0.4327
hasdc_merged          test ≈ 0.4050
raw_plain_residual    test ≈ 0.3238
```

This showed that:

```text
On heterophilic graphs it is not as simple as raw being better or smooth being better;
a more detailed graph-signal decomposition is needed.
```

#### Cora / Citeseer / PubMed

Representative results:

```text
Cora:
raw_plain_residual test ≈ 0.5880
smooth_plain_residual test ≈ 0.7870
hasdc_merged test ≈ 0.8070

Citeseer:
raw_plain_residual test ≈ 0.5980
smooth_plain_residual test ≈ 0.7090
hasdc_channel_select test ≈ 0.7090

PubMed:
raw_plain_residual test ≈ 0.7040
smooth_plain_residual test ≈ 0.7570
hasdc_merged test ≈ 0.7690
```

This showed that:

```text
On citation graphs, the smooth channel is usually stronger;
but HASDC was still not the final core, because it did not explicitly name raw / low-pass / high-pass blocks.
```

---

### 8.6 src_v6: packaging

Main files:

```text
model.py
pyproject.toml
requirements.txt
examples/grid_search.py
examples/synthetic_demo.py
```

Meaning:

```text
It turned WhiteBox-GSD into a more reusable package-style interface.
```

But the scientific breakthrough was still not in the packaging. It was in the next step:

```text
Before entering the PCA subspace, explicitly construct a graph-signal dictionary.
```

---

# Part III: Chameleon, high-pass graph signals, and Fisher + PCA + Ridge

---

## 9. src_v7: the key breakthrough on Chameleon

### 9.1 Corresponding stage

Corresponding directory:

```text
06_src_v7_Chameleon_highpass_Fisher_PCA_Ridge_breakthrough
```

This stage was the direct predecessor of the final WG-SRC.

---

### 9.2 Stage goal

Chameleon is a heterophilic graph.  
On such graphs:

```text
Neighbors are not necessarily of the same class;
pure low-pass smoothing may erase the discriminative information of the node itself;
but ignoring graph structure completely is also not enough.
```

Therefore the goal became:

```text
Explicitly construct low-pass and high-pass graph signals,
and then use white-box subspaces and closed-form classifiers to judge which signals are useful.
```

---

### 9.3 Key discovery: high-pass graph differences are useful

The project began to explicitly construct:

```text
X
PX
P²X
X - PX
PX - P²X
```

where:

- \(X\): the original features;
- \(P\): the graph propagation matrix;
- \(PX\): the one-hop low-pass signal;
- \(P^2X\): the two-hop low-pass signal;
- \(X-PX\): the difference between a node itself and the neighborhood average, a high-pass signal;
- \(PX-P^2X\): the difference between one-hop and two-hop propagation, a high-pass signal.

The key idea was:

```text
In heterophilic graphs, being different from the neighbors can itself be a discriminative signal.
```

---

### 9.4 The three basic algorithms of src_v7

#### Algorithm 1: Multihop-Full + PCA Subspace Residual

Representative file:

```text
algo1_multihop_pca_68pct.py
```

Result:

```text
Chameleon 10-split test_mean ≈ 68.16%
```

Core structure:

```text
F = [X, PX, P²X, X-PX, PX-P²X]
row-L2 normalize each block
fit class PCA subspaces
classify by residual argmin
dimension selected by validation
```

Meaning:

```text
Explicit multihop / high-pass features + PCA residual could push Chameleon to around 68%.
```

---

#### Algorithm 2: Multihop-Full + PCA + Ridge

Representative file:

```text
algo2_multihop_pca_ridge_70pct.py
```

Result:

```text
Chameleon 10-split test_mean ≈ 70.02%
```

Change:

```text
PCA residual handles class geometry;
Ridge classifier handles the discriminative boundary;
the two are normalized and fused.
```

Meaning:

```text
The PCA subspace and the Ridge boundary are not redundant;
they are complementary.
```

This was the early origin of the PCA–Ridge phase map in the final paper.

---

#### Algorithm 3: Fisher + PCA + Ridge

Representative file:

```text
algo3_fisher_ensemble_71pct.py
```

Result:

```text
Chameleon 10-split test_mean ≈ 71.10%
```

Change:

```text
First use Fisher score to select top-k coordinates,
then perform PCA + Ridge.
```

Meaning:

```text
Fisher coordinate selection substantially improved the effectiveness of the white-box feature dictionary.
```

At this point the core skeleton of the final WG-SRC had already formed:

```text
explicit graph signals
+ Fisher coordinate selection
+ class PCA residual
+ closed-form Ridge boundary
+ validation fusion
```

---

### 9.5 Limitations of src_v7

Although src_v7 was already close to the final method, several problems remained:

1. the graph-signal dictionary was not yet complete enough;
2. symmetric-normalized propagation had not yet been added;
3. there was no \(P^3X\);
4. the final nine-block dictionary had not yet formed;
5. node-level statistical audits were not yet complete;
6. analyses of error nodes, branches, class pairs, and dimensions still needed to be systematized.

Therefore the project moved to v8 and the later statistical-audit route.

---

# Part IV: Audit-driven iteration from v8 to v17

---

## 10. v8: error-node and parameter-effect statistics

### 10.1 Corresponding directory

```text
results_v8_ch_error_audit_top20_r0/
```

Containing files:

```text
audit_meta.json
error_node_summary.csv
grid_summary_with_error_counts.csv
node_param_effects.csv
param_effects_summary.csv
```

---

### 10.2 Goal of this stage

This stage was not meant to directly propose a new model, but to answer:

```text
Which nodes are repeatedly wrong?
Which parameter values systematically reduce errors?
Where does the model improvement actually come from?
```

This step was extremely important because it started to change the iteration style from:

```text
looking at overall accuracy
```

to:

```text
looking at error nodes, parameter effects, and mechanism changes
```

---

### 10.3 Meaning of the statistical tables

#### `error_node_summary.csv`

It records:

```text
how many times each node is wrong under multiple grid configurations
```

and is used to identify hard nodes.

#### `grid_summary_with_error_counts.csv`

For each parameter combination, it records:

```text
n_wrong_all
n_wrong_train
n_wrong_val
n_wrong_test
val
test
```

Therefore it does not only look at accuracy, but also at exactly how many nodes were misclassified.

#### `node_param_effects.csv`

This records the effect of a particular parameter value on the error rate of specific nodes.

#### `param_effects_summary.csv`

This aggregates node-level effects into parameter-level statistics.

---

### 10.4 Representative statistics

For example:

```text
dim = 8
mean_test = 0.461124
mean_n_wrong_test = 245.73

dim = 10
mean_test = 0.452485
mean_n_wrong_test = 249.67
```

This shows that the criterion was not simply the accuracy of one run, but rather:

```text
dim=8 statistically reduced more test wrong nodes.
```

---

### 10.5 Research meaning of v8

The model of v8 itself was not the final breakthrough, but it established the statistical style for all later iterations:

```text
Do not only ask: what is the accuracy?
Also ask:
- which nodes were fixed?
- which nodes were broken?
- are errors concentrated in a specific class?
- does a parameter change have a systematic directional effect?
- does the mechanism bring net gain?
```

This directly developed into the later fixed / broken / redirect audit.

---

## 11. Baseline subspace geometry: first test the class subspaces themselves

### 11.1 Corresponding directory

```text
results_baseline_subspace_vectors_chameleon/
```

Containing files:

```text
pairwise_geometry_baseline_subspaces.csv
run_summary_baseline_subspace_vectors.csv
split_summary_baseline_subspace_vectors.csv
subspace_summary_baseline_vectors.csv
```

---

### 11.2 Goal

Before adding more complex mechanisms, first test:

```text
Are the class subspaces themselves useful?
What are the class-center distances?
How large is the overlap between subspaces?
What do the principal angles look like?
What is the effective rank?
What is the retained variance?
```

---

### 11.3 Summary result

`run_summary_baseline_subspace_vectors.csv` records:

```text
val_mean  = 0.6897119341563785
test_mean = 0.6815789473684211
test_std  = 0.023141940277671778
```

This shows that:

```text
Class-subspace geometry alone already reached about 68.16% on Chameleon.
```

---

### 11.4 Research meaning

This baseline is very important:

1. the class-subspace route is not invalid;
2. the class geometry of Chameleon can indeed be captured to some extent by PCA subspaces;
3. but the error structure is complex and requires more fine-grained branch / dimension / boundary mechanisms;
4. all later complex mechanisms should be compared against this geometric baseline.

---

## 12. src_v9: failed adaptive-branch route

### 12.1 Corresponding directory

```text
results_src_v9_adaptive_branch_chameleon_records/
```

Containing files:

```text
branch_member_rows.csv
branch_subspace_summary.csv
misclassified_test_nodes.csv
run_summary.json
split_summary_adaptive_branch_records.csv
train_internal_misclassified_nodes.csv
```

---

### 12.2 Stage goal

The goal was:

```text
Beyond the root PCA subspace, build adaptive branches for difficult misclassification modes.
```

The intuition was:

```text
A class may not correspond to a single subspace;
misclassified nodes may correspond to local branches inside a class.
```

---

### 12.3 Results

The statistical package recorded:

```text
val_mean  = 0.6399176954732511
test_mean = 0.6300438596491228
avg_total_branches = 15.0
avg_extra_branches = 10.0
```

This is clearly lower than the baseline-subspace result of about 68.16%.

---

### 12.4 Reasons for failure

The problems of src_v9 were:

1. too many branches;
2. branches were too coarse;
3. branches could absorb error nodes;
4. there were not enough conservative triggering conditions;
5. branch complexity increased while overall discrimination was damaged.

Conclusion:

```text
It cannot be assumed that more branches are always better.
```

The next step therefore had to statistically measure, for each branch:

```text
How many nodes did it fix?
How many nodes did it break?
Where did it redirect the wrong nodes?
```

---

## 13. src_v10: safe branch and fixed / broken / redirect audit

### 13.1 Corresponding directory

```text
results_compare_branch_details_src_v9_src_v10_chameleon/
results_src_v10_branch_effect_audit_chameleon/
```

---

### 13.2 Overall comparison from v9 to v10

`comparison_mean_summary_src_v9_src_v10.csv` records:

| Version | val_mean | test_mean | avg_total_branches | avg_extra_branches |
|---|---:|---:|---:|---:|
| src_v9 | 0.6399177 | 0.6300439 | 15.0 | 10.0 |
| src_v10 | 0.6953361 | 0.6866228 | 12.8 | 7.8 |

Important conclusion:

```text
src_v10 used fewer branches but achieved higher test performance.
```

This means src_v10 did not improve by “stacking more branches,” but through more conservative and more effective branch mechanisms.

---

### 13.3 src_v10 branch-effect audit

`split_summary_src_v10_branch_effect_audit.csv` recorded, for each split:

```text
val_acc_full
test_acc_full
val_acc_root_same_dim
test_acc_root_same_dim
test_changed_vs_root
test_fixed_vs_root
test_broken_vs_root
test_wrong_redirect_vs_root
```

Average trend:

```text
test_acc_full          ≈ 0.6866
test_acc_root_same_dim ≈ 0.6805

average per split:
test_changed_vs_root ≈ 13.1
test_fixed_vs_root   ≈ 5.8
test_broken_vs_root  ≈ 3.0
test_wrong_redirect  ≈ 4.3
```

---

### 13.4 Statistical interpretation

Several concepts:

- `fixed_vs_root`: the root was wrong and the branch corrected it;
- `broken_vs_root`: the root was correct and the branch made it wrong;
- `wrong_redirect`: the root was wrong and the branch changed it to another wrong class.

The v10 statistics show:

```text
The branch has net gain because fixed > broken;
but wrong_redirect is not small, which shows that the branch is still dangerous.
```

So the next step was not to keep adding branches, but to make branches safer.

---

## 14. src_v11 / src_v12: geometry-safe and class-gated branch

### 14.1 src_v11: geometry-safe branch

Corresponding directory:

```text
results_src_v11_geometry_safe_branch_audit_chameleon/
```

Result:

```text
val_mean  ≈ 0.6948
test_mean ≈ 0.6855
```

Statistics:

```text
test_fixed_by_calibration  ≈ 2.0
test_broken_by_calibration ≈ 1.9

test_fixed_by_extra  ≈ 7.4
test_broken_by_extra ≈ 3.5
```

Interpretation:

```text
Extra branches were still useful;
root calibration had very small net gain because fixed and broken were close.
```

Therefore src_v11 did not become the breakthrough.

---

### 14.2 src_v12: class-gated geometry-safe branch

Corresponding directory:

```text
results_src_v12_classgated_geometry_safe_branch_audit_chameleon/
```

Result:

```text
val_mean  ≈ 0.6951
test_mean ≈ 0.6879
```

Statistics:

```text
test_fixed_by_calibration  ≈ 0.6
test_broken_by_calibration ≈ 0.3

test_fixed_by_extra  ≈ 5.7
test_broken_by_extra ≈ 2.8
```

Interpretation:

```text
Class-gated calibration was more conservative and reduced damage;
but the calibration itself had only a small effect;
the main gain still came from the extra branches.
```

---

### 14.3 Conclusion of this stage

src_v11 / src_v12 showed that:

1. the branch mechanism can indeed fix nodes;
2. calibration can reduce risk, but contributes little;
3. the real problem may not be branch triggering, but the dimension and expressive capacity of the class subspace itself;
4. the next stage therefore moved to per-class dimension.

---

## 15. src_v13 / src_v14: per-class PCA dimension

### 15.1 src_v13: per-class dimension + coverage objective

Corresponding directory:

```text
results_src_v13_perclass_dim_coverage_audit_chameleon/
```

Key files:

```text
dim_search_trace_src_v13.csv
split_summary_src_v13_perclass_dim_coverage_audit.csv
```

Result:

```text
val_mean  ≈ 0.6989
test_mean ≈ 0.6886
```

---

### 15.2 Key change in v13

Previously all classes used the same PCA dimension.  
v13 changed this to:

```text
each class searches for its own PCA dimension.
```

Reason:

```text
Different classes in Chameleon have different geometric complexity;
a shared dimension can underfit some classes and overfit others.
```

---

### 15.3 Statistical selection criteria in v13

`dim_search_trace_src_v13.csv` did not only track accuracy. It also recorded:

```text
objective
val_acc
balanced_norm_true_residual
balanced_true_explained_ratio
mean_fp_rate
mean_dim_frac
accepted
reason
```

That is, dimension selection simultaneously considered:

1. whether the true-class residual decreased;
2. whether the true-class explained ratio increased;
3. whether the false-positive rate increased;
4. whether the dimension fraction became too large;
5. whether validation improved.

---

### 15.4 Fixed / broken statistics in v13

Average trend:

```text
root_test_acc ≈ 0.6809
test_acc      ≈ 0.6886

fixed_vs_root  ≈ 6.7
broken_vs_root ≈ 3.2
redirect       ≈ 3.4
```

This shows that:

```text
Per-class dimension has net gain,
but there are still broken and redirect cases.
```

---

### 15.5 src_v14: dimension floor + global fallback

Corresponding directory:

```text
results_src_v14_perclass_dim_floor_coverage_audit_chameleon/
```

Result:

```text
val_mean  ≈ 0.6979
test_mean ≈ 0.6904
```

The change in v14 was:

```text
Add a floor to the per-class dimension;
when necessary, fall back to a global dimension.
```

Reason:

```text
In some splits of v13, the selected dimension became too small and specific classes were underfit.
```

Average trend in v14:

```text
root_test_acc ≈ 0.6827
test_acc      ≈ 0.6904

fixed_vs_root  ≈ 6.1
broken_vs_root ≈ 2.6
redirect       ≈ 3.7
```

Interpretation:

```text
There are slightly fewer fixed nodes, but also fewer broken nodes;
overall test improves, which shows that floor/fallback increased robustness.
```

---

### 15.6 Stage conclusion

src_v13 / src_v14 proved that:

```text
Class-subspace complexity differs;
per-class dimension is an effective mechanism;
but it still cannot by itself solve class-pair errors and boundary problems.
```

Therefore the next stage moved to class bias and pairwise specialists in src_v15.

---

## 16. src_v15: class bias, pairwise specialists, and validation overfitting

### 16.1 Corresponding directory

```text
results_src_v15_score_pairwise_audit_chameleon/
```

Key files:

```text
class_bias_trace_src_v15.csv
pairwise_specialist_trace_src_v15.csv
split_summary_src_v15_score_pairwise_audit.csv
extra_branch_effect_summary_src_v15.csv
extra_branch_rewrite_nodes_src_v15.csv
dim_search_trace_src_v15.csv
misclassified_test_nodes_src_v15.csv
```

---

### 16.2 Results

Statistics:

```text
base_val_acc  ≈ 0.6979
base_test_acc ≈ 0.6904

bias_val_acc  ≈ 0.7048
bias_test_acc ≈ 0.6912

final val_acc  ≈ 0.7387
final test_acc ≈ 0.6923
```

This means that:

```text
Validation jumped from about 69.79% to 73.87%,
but test only changed from about 69.04% to 69.23%.
```

---

### 16.3 Why this matters

If only validation were considered, src_v15 would look very strong.  
But the statistics show that:

```text
The pairwise specialist greatly improves validation,
but this improvement is barely transferred to test.
```

This is a典型 validation overfitting.

---

### 16.4 Class-bias trace

`class_bias_trace_src_v15.csv` records many candidate biases:

```text
accepted = 18
rejected = 760
```

This shows that:

```text
Class bias was not added casually;
most candidates were rejected by statistical rules.
```

---

### 16.5 Pairwise-specialist trace

`pairwise_specialist_trace_src_v15.csv` records candidate class-pair specialists:

```text
accepted = 61
rejected = 400
```

Each candidate records:

```text
fixed
broken
redirect
broken_rate
gain
accepted
reason
```

This means each pair specialist was audited for:

```text
How many nodes did it fix?
How many nodes did it break?
Where did it redirect the wrong nodes?
```

---

### 16.6 Final judgment on v15

The value of src_v15 is that:

```text
It proved that class-pair error structure is real.
```

But it could not become the final main method, because:

1. validation improvement was far larger than test improvement;
2. fixed and broken tended to offset each other;
3. the pairwise specialist was complex;
4. the interpretation chain was more complex than that of the main method;
5. it was not suitable as a unified white-box scaffold.

Therefore the next stage could not keep increasing complexity on pairwise specialists, but needed a new direction.

---

## 17. src_v16 / src_v16b: reliability map and anti-hub

### 17.1 The directory corresponding to src_v16

Corresponding directory:

```text
results_src_v16_reliability_antihub_audit_chameleon/
```

Key files:

```text
analysis_report_src_v16.md
reliability_map_src_v16.csv
antihub_class_stats_src_v16.csv
penalty_search_trace_src_v16.csv
misclassified_test_nodes_src_v16.csv
split_summary_src_v16_reliability_antihub_audit.csv
```

---

### 17.2 The goal of src_v16

After abandoning the pairwise specialist, src_v16 tried:

```text
Class Subspace Reliability Map + Anti-Hub Subspace Penalty
```

The core idea was:

1. keep both PCA residual and Ridge;
2. use OOF statistics to determine under which conditions PCA or Ridge is more reliable;
3. identify hub classes;
4. add an anti-hub penalty to classes that easily absorb error nodes.

---

### 17.3 Reliability map

`reliability_map_src_v16.csv` records:

```text
class
gap_bin
agree_bin
expo_bin
w_pca
count
pca_correct
ridge_correct
```

Its meaning:

- `gap_bin`: margin-size interval;
- `agree_bin`: whether PCA and Ridge agree;
- `expo_bin`: neighborhood-exposure interval;
- `w_pca`: the reliability weight of the PCA branch under this condition;
- `pca_correct` / `ridge_correct`: whether the two branches were correct in the OOF statistics.

---

### 17.4 Anti-hub statistics

`antihub_class_stats_src_v16.csv` records:

```text
class
hub_ratio
inflow
capture
```

This is used to determine whether a class excessively absorbs nodes from other classes.

---

### 17.5 Penalty-search trace

`penalty_search_trace_src_v16.csv` records:

```text
alpha_reliability
beta_antihub
gamma_exposure
val_acc
fixed_val
broken_val
gain_val
accepted
reason
```

This shows that src_v16 was still statistically driven rather than manually selected by test performance.

---

### 17.6 Results of src_v16

`analysis_report_src_v16.md` records:

```text
Reliability+AntiHub val_mean  = 0.7207 ± 0.0183
Reliability+AntiHub test_mean = 0.7055 ± 0.0183

Fixed-weight baseline val_mean  = 0.7112 ± 0.0183
Fixed-weight baseline test_mean = 0.7103 ± 0.0155

Improvement = -0.0048
```

The key conclusion is:

```text
Reliability + anti-hub itself did not surpass the fixed-weight baseline;
instead it was about 0.48 percentage points lower.
```

---

### 17.7 The real contribution of src_v16

Although src_v16 was not the final successful mechanism, it exposed a more important fact:

```text
The Fisher + PCA + Ridge base was already strong;
continuing to build more complex correction mechanisms was less useful than improving the graph-signal dictionary itself.
```

This is exactly the starting point of src_v16c.

---

### 17.8 src_v16b

Corresponding directory:

```text
results_src_v16b_perclass_reliability_greedydim_audit_chameleon/
```

Results:

```text
val_mean  = 0.7276
test_mean = 0.7072
elapsed   ≈ 4940 s
```

v16b continued with:

```text
per-class reliability
greedy dimension
```

But it still did not approach v16c, and its runtime was very long.

Conclusion:

```text
Reliability / greedy-dimension search is not the main breakthrough direction.
```

---

## 18. src_v16c: formation of the final WG-SRC core

### 18.1 Corresponding directory

```text
results_src_v16c_audit_chameleon/
```

Key files:

```text
run_summary_src_v16c.json
split_summary_src_v16c.csv
```

---

### 18.2 Results

`run_summary_src_v16c.json` records:

```text
val_mean  = 0.7279835391
val_std   = 0.0216895810
test_mean = 0.7247807018
test_std  = 0.0175780916
elapsed   = 641.286 s
```

Compared with the earlier versions, this was a clear jump:

```text
src_v16  test ≈ 70.55%
src_v16b test ≈ 70.72%
src_v16c test ≈ 72.48%
```

---

### 18.3 Why did src_v16c break through?

src_v16c did not continue adding complex corrections.  
Instead, it returned to the most basic question:

```text
Is the graph-signal dictionary itself good enough?
```

It therefore added:

1. \(P_{\mathrm{row}}^3X\);
2. symmetric-normalized propagation;
3. symmetric high-pass difference;
4. multi-alpha Ridge ensemble;
5. energy-based PCA dimension control;
6. a validation-selected fusion weight.

---

### 18.4 The final nine-block graph-signal dictionary

The graph-signal dictionary of src_v16c is:

\[
F^{(0)}
=
[
X,\ 
P_{\mathrm{row}}X,\ 
P_{\mathrm{row}}^2X,\ 
P_{\mathrm{row}}^3X,\ 
X-P_{\mathrm{row}}X,\ 
P_{\mathrm{row}}X-P_{\mathrm{row}}^2X,\ 
P_{\mathrm{sym}}X,\ 
P_{\mathrm{sym}}^2X,\ 
X-P_{\mathrm{sym}}X
]
\]

where:

- \(X\): original node features;
- \(P_{\mathrm{row}}=D^{-1}A\): row-normalized propagation;
- \(P_{\mathrm{sym}}=D^{-1/2}AD^{-1/2}\): symmetric-normalized propagation;
- \(P_{\mathrm{row}}X, P_{\mathrm{row}}^2X, P_{\mathrm{row}}^3X\): row-normalized low-pass multihop signals;
- \(X-P_{\mathrm{row}}X\): ego-neighbor high-pass difference;
- \(P_{\mathrm{row}}X-P_{\mathrm{row}}^2X\): one-hop/two-hop high-pass difference;
- \(P_{\mathrm{sym}}X, P_{\mathrm{sym}}^2X\): symmetric low-pass signals;
- \(X-P_{\mathrm{sym}}X\): symmetric high-pass difference.

If the original dimension is \(d\), then after concatenating the nine blocks the dimension is:

\[
p=9d
\]

In Chameleon, the original feature dimension is 2325, so after the nine blocks:

\[
9\times 2325 = 20925
\]

---

### 18.5 Fisher coordinate selection

For each coordinate \(j\), compute the Fisher score:

\[
q_j
=
\frac{
\sum_{c=1}^{C} n_c(\mu_{c,j}-\mu_j)^2
}{
\sum_{c=1}^{C}
\sum_{i\in T:y_i=c}
(F^{(0)}_{ij}-\mu_{c,j})^2
+
\epsilon
}
\]

where:

- \(q_j\): the Fisher score of coordinate \(j\);
- \(C\): the number of classes;
- \(n_c\): the number of samples of class \(c\) in the training set;
- \(\mu_{c,j}\): the mean of class \(c\) on coordinate \(j\);
- \(\mu_j\): the mean of all training nodes on coordinate \(j\);
- the numerator measures between-class difference;
- the denominator measures within-class scatter.

Select the top-\(K\) coordinates to obtain:

\[
F=F^{(0)}[:,S]
\]

---

### 18.6 PCA residual branch

For each class \(c\), fit PCA:

\[
\mu_c,\quad B_c
\]

Then for node \(i\) and class \(c\), compute:

\[
R^{\mathrm{pca}}_{ic}
=
\left\|
(I-B_cB_c^\top)(f_i-\mu_c)
\right\|_2^2
\]

where:

- \(f_i\): the selected feature vector of node \(i\);
- \(\mu_c\): the class center;
- \(B_c\): the PCA subspace of class \(c\);
- \(R^{\mathrm{pca}}_{ic}\): the residual of node \(i\) outside the subspace of class \(c\).

The smaller this is, the more the node resembles that class.

---

### 18.7 Multi-alpha Ridge branch

The Ridge branch uses multiple \(\alpha\) values:

\[
\beta_\alpha
=
(F_{\mathrm{tr}}F_{\mathrm{tr}}^\top+\alpha I)^{-1}Y
\]

\[
Z_\alpha
=
FF_{\mathrm{tr}}^\top \beta_\alpha
\]

Then the scores over multiple \(\alpha\) values are averaged and normalized.

Meaning:

```text
PCA residual captures class geometry;
Ridge captures the discriminative boundary;
multi-alpha makes Ridge more stable with respect to regularization strength.
```

---

### 18.8 Validation-selected fusion

The final score is:

\[
S_{ic}
=
w\tilde R^{\mathrm{pca}}_{ic}
+
(1-w)\tilde R^{\mathrm{ridge}}_{ic}
\]

Prediction:

\[
\hat y_i
=
\arg\min_c S_{ic}
\]

where:

- \(w\): the fusion weight selected by validation;
- \(\tilde R^{\mathrm{pca}}\): the normalized PCA residual;
- \(\tilde R^{\mathrm{ridge}}\): the normalized residual-like Ridge score.

---

### 18.9 Representative configuration of split 0

The JSON record for split 0 contains:

```text
Nodes = 2277
Raw features = 2325
Enhanced features = 20925
Classes = 5

top_k = 8000
dim = 96
energy = 0.95
alphas = [0.01, 0.1, 1.0]
w = 0.6
test = 0.725877
```

This shows that the improvement of src_v16c came from:

```text
a better explicit graph-signal dictionary
+ Fisher feature selection
+ PCA geometry
+ Ridge boundary
+ validation-selected fusion
```

rather than from complex post-hoc corrections.

---

## 19. v16d / v16e / v16f / v16g / v17: complex variants were excluded

### 19.1 Overall comparison

| Version | Main idea | val_mean | test_mean | elapsed | Conclusion |
|---|---|---:|---:|---:|---|
| src_v16 | reliability + anti-hub | 0.7207 | 0.7055 | 3784s | correction worse than fixed base |
| src_v16b | per-class reliability + greedy dim | 0.7276 | 0.7072 | 4940s | slow, limited gain |
| src_v16c | 9-block enhanced multihop + multi-ridge | 0.7280 | 0.7248 | 641s | final main method |
| src_v16d | 13-block max features | 0.7280 | 0.7193 | 180s | more features but lower test |
| src_v16e | iterative label features | 0.7283 | 0.7248 | 1152s | tied with v16c but more complex |
| src_v16f | diffusion stacking | 0.7171 | 0.7191 | 467s | worse than v16c |
| src_v17 | OOF label beam | 0.7288 | 0.7239 | 6347s | slightly higher val but not better test, and too slow |

---

### 19.2 v16d: more feature blocks were worse than v16c

v16d tried to add more feature blocks, for example:

```text
P²X-P³X
X*PX
|X-PX|
degree-weighted X
larger top_k
more feature diversity
```

Result:

```text
v16d test_mean = 0.7193
v16c test_mean = 0.7248
```

Conclusion:

```text
More features are not always better;
adding beyond the nine blocks can introduce noise and damage the stability of Fisher/PCA/Ridge.
```

---

### 19.3 v16e: iterative label features

The idea of v16e was:

```text
Round 1:
9-block features + Fisher + PCA + Ridge

Round 2:
Use the first-round soft predictions to construct:
- neighbor soft-label distribution
- prediction entropy
- agreement features
Then run PCA + Ridge again in the second round
```

Result:

```text
v16e test_mean = 0.7248
v16c test_mean = 0.7248
```

The two tied, but v16e was not chosen as the main method because:

1. it is slower;
2. it is more complex;
3. it uses predicted labels to construct features, so the interpretation chain is longer;
4. it is less clean than the pure graph-signal dictionary of v16c;
5. it is less direct for the main paper line of a “white-box graph signal atlas.”

---

### 19.4 v16f: diffusion stacking

v16f tried diffusion-style stacking such as PPR / Heat diffusion.

Result:

```text
v16f test_mean = 0.7191
```

which is lower than v16c.

Conclusion:

```text
A more complex diffusion kernel did not outperform the explicit nine-block multihop/high-pass dictionary.
```

---

### 19.5 v16g: local subspace

v16g tried local-subspace refinement.

Problem:

```text
It was too slow, with cases exceeding >40 min / split.
```

Therefore it was not suitable for the official six-dataset repeats and paper scaffold.

---

### 19.6 v17: OOF label beam

Corresponding directory:

```text
results_src_v17_oof_label_beam_audit_chameleon/
```

Key files:

```text
candidate_trace_src_v17.csv
rewrite_nodes_round1_to_final_src_v17.csv
run_summary_src_v17_oof_label_beam.json
split_summary_src_v17_oof_label_beam.csv
```

Results:

```text
val_mean  = 0.7288065844
test_mean = 0.7239035088
elapsed   = 6346.601 s
```

The validation of v17 was slightly higher than v16c, but test was slightly lower, and runtime was much larger.

The statistics also recorded:

```text
round1_scan = 4050
round2_scan = 3600
total candidate scans = 7650
```

The rewrite statistics showed that round 2 could fix some nodes, but it also created broken / wrong-redirect cases.

Conclusion:

```text
The OOF label beam has diagnostic meaning, but its complexity is too high;
it did not surpass v16c and is not suitable as the main method.
```

---

## 20. Why src_v16c was finally selected

The final choice of src_v16c was not because it was the only highest number, but because of a combined judgment:

1. **highest or tied-highest test_mean**  
   v16c and v16e tied on Chameleon, at about 72.48%.

2. **simpler than v16e**  
   v16e used iterative label features and had a longer interpretation chain.

3. **much faster than v17**  
   v17 was about 6347 s, while v16c was about 641 s.

4. **does not use pseudo-label / label-beam correction**  
   v16c remains a pure graph-signal dictionary + closed-form classifier.

5. **directly matches the paper mechanism**  
   the nine-block dictionary of v16c can directly become the named blocks in the paper and can support the node-level atlas.

6. **keeps both the PCA and Ridge branches**  
   which allows later analyses of:
   - class-subspace complexity;
   - PCA/Ridge complementarity;
   - decision phase map;
   - geometry-vs-boundary behavior.

7. **suitable as a general white-box scaffold**  
   it is not the specialist for every dataset, but it is the most suitable cross-dataset diagnostic probe.

Therefore the final selection was:

```text
src_v16c = final WG-SRC core
```

---

# Part V: Formal paper experiment package and the final shaping of WG-SRC

---

## 21. src_v16c_paper_experiments: the formal paper package

### 21.1 Corresponding directory

```text
src_v16c_paper_experiments/
```

Structure:

```text
src_v16c_paper_experiments/
├── README.md
├── run_all_experiments.py
├── run_src_v16c_fill_main_table_10repeats.py
├── run_src_v16c_actor_only.py
├── configs/
│   └── default_datasets.json
├── paperexp/
│   ├── core.py
│   └── __init__.py
├── scripts/
│   ├── run_E01_baseline_table_aggregate.py
│   ├── run_E03_random_split_stability.py
│   ├── run_E05_ablation.py
│   ├── run_E07_param_scan.py
│   ├── run_E08_interpretability.py
│   ├── run_E09_efficiency.py
│   └── run_E11_sample_efficiency.py
└── results/
```

---

### 21.2 Algorithm freezing

At this stage, `paperexp/core.py` was treated as the final core.  
The instructions at that time were explicit:

```text
Do not modify Fisher selection;
do not modify PCA fitting;
do not modify the Ridge classifier;
do not modify validation selection;
do not tune according to test accuracy;
do not change the split-generation logic;
do not touch the algorithmic core of paperexp/core.py.
```

Only the following were allowed to be fixed:

```text
duplicate CSV fields;
missing paths;
encoding issues;
dict multiple values;
errors in logging / result-output directories.
```

This shows that at that point the method had already been frozen as the paper version.

---

### 21.3 Formal experiment scripts

The formal experiments included:

| Script | Function |
|---|---|
| `run_E01_baseline_table_aggregate.py` | baseline table |
| `run_E03_random_split_stability.py` | paired random-split stability |
| `run_E05_ablation.py` | mechanistic ablation |
| `run_E07_param_scan.py` | parameter scan |
| `run_E08_interpretability.py` | atlas / interpretability |
| `run_E09_efficiency.py` | efficiency profile |
| `run_E11_sample_efficiency.py` | sample efficiency |
| `run_src_v16c_fill_main_table_10repeats.py` | the main table with 10 repeats |

---

### 21.4 Final submission package

The final results included files such as:

```text
src_v16c_final_6datasets_repeat_rows.csv
src_v16c_final_6datasets_summary.csv
src_v16c_main_table_10repeat_rows.csv
srcv16c_submission_package.zip
srcv16c_submission_package_20260421_105707.tar.gz
```

This shows that the final experiment package completed the transition from a single Chameleon version to a six-dataset paper experiment system.

---

## 22. The final positioning of WG-SRC in the paper

In the final paper, WG-SRC is not positioned as:

```text
a GNN replacement created to chase the highest possible accuracy
```

but as:

```text
an accuracy-preserving white-box diagnostic scaffold
```

That is, it simultaneously serves two roles:

1. as a predictor, to perform node classification;
2. as a diagnostic probe, to produce a node-level atlas and a dataset fingerprint.

---

### 22.1 The six-dataset main experiment

The six datasets in the paper are:

```text
Amazon-Computers
Amazon-Photo
Chameleon
Cornell
Texas
Wisconsin
```

The main experiment shows that:

```text
WG-SRC is on average +1.52 pp higher than the strongest aligned baseline.
```

The meaning of this result is not to claim that WG-SRC is the strongest among all graph-learning models, but rather:

```text
its white-box diagnostic results come from a genuinely effective predictor,
rather than from a weak post-hoc probe.
```

---

### 22.2 Mechanistic ablation and the generalist-vs-specialist judgment

The ablation in the paper shows that:

- on Amazon, no-high-pass may be stronger;
- on Cornell/Texas, ridge-only may be stronger;
- on Wisconsin, raw-only may be stronger;
- Chameleon requires the full / near-full structure;
- the full scaffold has the best average rank, the best worst-case performance, and is top-3 on all six datasets.

This is consistent with the iteration path:

```text
Some specialists can be better on individual datasets;
but only the full scaffold can preserve the complete diagnostic capability.
```

Therefore ridge-only or raw-only was not selected as the main method.

---

### 22.3 Node-level signal atlas

The final WG-SRC can record, for each node:

```text
raw signal share
low-pass signal share
high-pass signal share
PCA prediction
Ridge prediction
PCA margin
Ridge margin
final prediction
correctness
degree
decision quadrant
```

These records are aggregated into the dataset fingerprint:

\[
m(D)
=
[
R_D,\ L_D,\ H_D,\ C_D,\ Q_D^{\mathrm{ridge}},\ Q_D^{\mathrm{hard}},\ \Delta H_D
]
\]

where:

- \(R_D\): dataset-level raw share;
- \(L_D\): dataset-level low-pass share;
- \(H_D\): dataset-level high-pass share;
- \(C_D\): mean class-subspace complexity;
- \(Q_D^{\mathrm{ridge}}\): the Ridge-only correction quadrant;
- \(Q_D^{\mathrm{hard}}\): the hard fraction where both PCA and Ridge fail;
- \(\Delta H_D\): the high-pass shift of wrong nodes versus correct nodes.

This is exactly the final formalized form of the v8–v17 statistical-audit idea.

---

# Part VI: Summary table of the audit-driven iteration logic

---

## 23. Statistical logic of the later versions

| Stage | Statistical object | Key table | Judgment obtained | Next step |
|---|---|---|---|---|
| v8 | hard nodes / parameter effects | `error_node_summary.csv`, `param_effects_summary.csv` | Accuracy alone is not enough; error-node structure must be examined | do branch audit |
| baseline subspace | class-subspace geometry | `pairwise_geometry_baseline_subspaces.csv` | class subspace is useful, but complex error nodes remain | adaptive branch |
| src_v9 | branch structure | `branch_member_rows.csv` | too many branches and performance drops | safe branch |
| src_v10 | fixed/broken/redirect | `extra_branch_rewrite_nodes_src_v10.csv` | branch has net gain but redirect risk | geometry-safe |
| src_v11/v12 | calibration and gated branch | `root_calibration_rewrite_nodes_*` | calibration contributes little; branch remains the main effect | per-class dimension |
| src_v13/v14 | per-class PCA dimension | `dim_search_trace_*` | different classes need different dimensions; floor/fallback is more stable | class-pair specialist |
| src_v15 | bias / pairwise specialist | `class_bias_trace`, `pairwise_specialist_trace` | validation overfits and test does not follow | abandon specialist |
| src_v16 | reliability / anti-hub | `reliability_map`, `penalty_search_trace` | correction is worse than the fixed base | improve signal dictionary |
| src_v16c | 9-block dictionary | `run_summary_src_v16c.json` | signal quality improvement brings a stable breakthrough | paper core |
| v16d/e/f/v17 | complex variants | each run summary / candidate trace | more complexity is not necessarily better | keep v16c |

---

## 24. Where the final method components came from

| Final WG-SRC component | Source version | Iteration reason |
|---|---|---|
| MCR² / subspace-interpretation idea | initial tracing stage | show that internal mechanism matters, not only accuracy |
| class-wise PCA subspace | src_v1 | shift from white-box GNN to closed-form geometric residual |
| raw / smooth distinction | HASDC / src_v5 | discovery that raw and smooth have different value on different graphs |
| high-pass difference | src_v7 | Chameleon showed that \(X-PX\) and related differences are useful |
| Fisher coordinate selection | src_v7 Algorithm 3 | pushed performance from about 70% to about 71.1% |
| Ridge boundary | src_v7 Algorithm 2 | PCA geometry alone is insufficient; a discriminative boundary is needed |
| fixed / broken / redirect audit | from src_v10 onward | judge whether a mechanism really fixes nodes |
| per-class dimension | src_v13/v14 | class complexity differs |
| reliability map | src_v16 | tests conditional reliability of PCA/Ridge, but correction is unstable |
| nine-block dictionary | src_v16c | final breakthrough; signal quality mattered more than complex correction |
| node-level atlas | v8–v17 statistical audit + paper E08 | formalize error audit into a diagnostic output |

---

## 25. Key lessons from failure

### 25.1 A neural layer with a formula is not automatically truly white-box

The problem of G1 showed:

```text
In theory there is H + ηgrad - ηλLH,
but in the actual forward path it is overridden by out_proj(Z_agg).
```

Therefore the final method no longer relies on trained message-passing layers.

---

### 25.2 Low-pass is not enough for heterophily

Chameleon showed that:

```text
Being different from neighbors can itself be a signal;
X-PX and PX-P²X cannot simply be discarded.
```

This directly led to the inclusion of high-pass blocks in the final dictionary.

---

### 25.3 A branch can fix nodes, but it can also break nodes

From src_v10 onward, every branch had to be evaluated by:

```text
fixed
broken
wrong_redirect
```

This turned the research from accuracy-driven into mechanism-audited research.

---

### 25.4 Validation improvement is not necessarily a real improvement

The core lesson of src_v15 was:

```text
val:  about 69.79% → 73.87%
test: about 69.04% → 69.23%
```

The pairwise specialist mainly improved validation and could not serve as the main method.

---

### 25.5 The correction mechanism was less important than the signal dictionary

src_v16 showed that:

```text
reliability + anti-hub is interpretable,
but it did not outperform the fixed baseline.
```

The real breakthrough came from src_v16c:

```text
Get the graph-signal dictionary right.
```

---

### 25.6 More features are not always better

v16d showed that:

```text
13-block features < 9-block v16c
```

Therefore the final nine-block dictionary is the balance between simplicity and expressive power.

---

### 25.7 More complexity is not necessarily better as the main paper method

v16e / v17 showed that:

```text
iterative labels / OOF beam can get close or tie,
but they are more complex, slower, and have a longer interpretation chain.
```

The final main method should be:

```text
simple, closed-form, auditable, interpretable, and reproducible.
```

---

# Part VII: Final conclusion

---

## 26. One-sentence summary

The real iteration path of the whole project was not:

```text
run many versions and pick the one with the highest accuracy.
```

Instead, it was:

```text
start from MCR² geometric explanation;
first try white-box GNN unfolding;
discover that the neural-layer route is not strictly white-box enough;
shift toward a closed-form subspace classifier;
on Chameleon discover the importance of high-pass / multihop graph-signal decomposition;
through v8-v17 statistics on error nodes, fixed/broken/redirect, dimensions, reliability, and candidate traces, exclude complex but unstable correction mechanisms;
finally converge to src_v16c:
nine-block explicit graph-signal dictionary + Fisher coordinate selection + class PCA subspace + multi-alpha Ridge + validation fusion.
```

---

## 27. The final narrative most suitable for the repository `research notes`

This project can be described as:

```text
This project began as an attempt to inspect whether neural representations obey low-rank class-subspace geometry suggested by MCR², ReduNet, and CRATE. Early white-box GNN unfolding routes showed that merely adding interpretable formulas to neural layers was insufficient, especially after code audit revealed that the theoretically meaningful update path could be bypassed by trainable projections.

The project then shifted toward closed-form white-box graph learning. Through Tikhonov smoothing, class PCA subspaces, response/residual scoring, raw-vs-smooth channel experiments, and Chameleon heterophily studies, the development identified explicit high-pass graph differences and multihop graph signals as essential. Subsequent v8-v17 iterations were driven not only by accuracy but by statistical audits of error nodes, branch fixed/broken/redirect behavior, per-class dimension traces, class-pair specialists, reliability maps, anti-hub penalties, and OOF candidate rewrites.

These audits showed that complex corrective mechanisms could improve validation or fix some nodes, but often introduced broken nodes, wrong redirects, overfitting, or excessive runtime. The final breakthrough was therefore not a more complex correction layer, but a cleaner signal dictionary: the src_v16c nine-block explicit graph-signal dictionary combined with Fisher coordinate selection, class-wise PCA subspace residuals, multi-alpha closed-form Ridge classification, and validation-selected fusion. This became the final WG-SRC scaffold and later supported the node-level signal atlas and dataset fingerprint analyses in the paper.
```

---

## 28. Final version positioning

```text
Official core method:
src_v16c

Official paper package:
src_v16c_paper_experiments

Final conceptual name:
WG-SRC
White-Box Graph Subspace–Ridge Classifier

Main contribution:
A white-box graph classifier that is also a dataset diagnostic probe.

Core method:
9-block explicit graph-signal dictionary
+ Fisher coordinate selection
+ class-wise PCA subspaces
+ closed-form multi-alpha Ridge
+ validation-selected score fusion
+ node-level signal atlas
```

---

## 29. Suggested research-record directories to keep

To prove that the iteration was statistically driven, it is recommended to keep the following statistical directories:

```text
research_notes/version_statistical_summary_pack/
├── MANIFEST_statistical_summary_pack.txt
├── results_v8_ch_error_audit_top20_r0/
├── results_baseline_subspace_vectors_chameleon/
├── results_compare_branch_details_src_v9_src_v10_chameleon/
├── results_src_v10_branch_effect_audit_chameleon/
├── results_src_v13_perclass_dim_coverage_audit_chameleon/
├── results_src_v14_perclass_dim_floor_coverage_audit_chameleon/
├── results_src_v15_score_pairwise_audit_chameleon/
├── results_src_v16_reliability_antihub_audit_chameleon/
├── results_src_v16c_audit_chameleon/
├── results_src_v16e_audit_chameleon/
└── results_src_v17_oof_label_beam_audit_chameleon/
```

The most critical evidence files include:

```text
results_v8_ch_error_audit_top20_r0/
├── error_node_summary.csv
├── node_param_effects.csv
└── param_effects_summary.csv

results_src_v10_branch_effect_audit_chameleon/
├── extra_branch_effect_summary_src_v10.csv
└── extra_branch_rewrite_nodes_src_v10.csv

results_src_v13_perclass_dim_coverage_audit_chameleon/
└── dim_search_trace_src_v13.csv

results_src_v14_perclass_dim_floor_coverage_audit_chameleon/
└── dim_search_trace_src_v14.csv

results_src_v15_score_pairwise_audit_chameleon/
├── class_bias_trace_src_v15.csv
├── pairwise_specialist_trace_src_v15.csv
└── split_summary_src_v15_score_pairwise_audit.csv

results_src_v16_reliability_antihub_audit_chameleon/
├── reliability_map_src_v16.csv
├── penalty_search_trace_src_v16.csv
├── antihub_class_stats_src_v16.csv
└── analysis_report_src_v16.md

results_src_v17_oof_label_beam_audit_chameleon/
├── candidate_trace_src_v17.csv
└── rewrite_nodes_round1_to_final_src_v17.csv
```

Together, these files prove that:

```text
The later iterations were not simple accuracy search,
but an error-audit-driven, mechanism-audit-driven, statistical-summary-driven research process.
```

---

## 30. Final short summary suitable for the README / research-notes homepage

```text
WG-SRC was developed through a multi-stage mechanism-audited iteration process. The project began with MCR²-based representation geometry tracing and early white-box GNN unfolding attempts. These routes revealed that interpretable formulas inside trainable GNN layers were not sufficient to guarantee a truly white-box mechanism. The development then shifted to closed-form graph subspace classifiers.

On heterophilic Chameleon, explicit high-pass graph differences and multihop signals proved essential. Later versions were not selected by accuracy alone: the development used node-level error audits, branch fixed/broken/redirect statistics, per-class dimension traces, class-pair specialist traces, reliability maps, anti-hub penalty searches, and OOF label-beam candidate traces. These statistics showed that complex correction mechanisms often improved validation but did not transfer reliably to test or introduced broken/wrong-redirect nodes.

The final selected method, src_v16c, uses a nine-block explicit graph-signal dictionary, Fisher coordinate selection, class-wise PCA subspace residuals, multi-alpha closed-form Ridge classification, and validation-selected score fusion. It was chosen because it achieved the best or tied-best Chameleon performance while remaining simpler, faster, more interpretable, and better aligned with the final node-level atlas and dataset-fingerprint analysis. This became the official WG-SRC scaffold in the paper.
```
