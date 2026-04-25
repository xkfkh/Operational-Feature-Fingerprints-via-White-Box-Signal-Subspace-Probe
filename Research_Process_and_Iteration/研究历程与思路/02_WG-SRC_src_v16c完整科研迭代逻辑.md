# WG-SRC / src_v16c 完整科研迭代逻辑与统计驱动原因

> 本文件用于系统整理 WG-SRC / src_v16c 从最初思想到最终论文实验包的完整迭代路径。  
> 重点不是简单列出“哪个版本准确率最高”，而是整理每一代为什么产生、解决了什么问题、统计结果说明了什么、为什么继续或放弃，以及这些迭代如何最终收敛到 `src_v16c` 和论文中的 WG-SRC。

---

## 0. 文件定位

本文件整理的对象包括四类材料：

1. **早期 Claude JSONL 迭代记录**  
   记录了从 MCR²、ReduNet、CRATE、white-box GNN、G1/G2/G3/G4、L-series，到 `src_v16c` 的模型开发过程。

2. **主文件夹详细路径记录**  
   记录了项目中真实存在的版本目录、脚本目录、结果目录和最终 paper experiment 包路径。

3. **`version_statistical_summary_pack` 统计包**  
   包含 v8 到 v17 之间大量统计审计结果，例如：
   - 错误节点统计；
   - 参数影响统计；
   - branch fixed / broken / redirect 统计；
   - class-pair geometry；
   - per-class dimension search trace；
   - class bias trace；
   - pairwise specialist trace；
   - reliability map；
   - anti-hub penalty search；
   - OOF label beam candidate trace。

4. **最终论文内容**  
   最终论文中的 WG-SRC 被定义为一个白盒图分类器和图数据集诊断 probe：  
   显式图信号字典 + Fisher 坐标选择 + 类 PCA 子空间 + 闭式 Ridge 分类 + 验证集融合 + node-level signal atlas。

本文件的目的不是重新写论文，而是把**真实研究过程**整理成可读、可审计、可放入 GitHub `research_notes/` 的 Markdown 文档。

---

## 1. 总体研究目标的变化

### 1.1 最初目标

项目最初不是为了单纯提高图分类准确率，而是从一个更基础的问题出发：

> 机器学习模型内部到底是否形成了可解释的低维结构？  
> 是否可以用更白盒、更线性代数化的机制替代黑盒神经网络？

最初理论背景来自：

- MCR²；
- ReduNet；
- CRATE；
- 子空间学习；
- coding rate；
- 类内压缩与类间分离。

核心直觉是：

> 一个好的表示空间应该让同类样本落在相对低维的类子空间里，不同类别的子空间尽量可分；如果这种结构可被显式构造，就不一定需要完全依赖黑盒神经网络。

---

### 1.2 后来目标

随着实验推进，研究目标从：

```text
解释黑盒模型内部表示
```

逐渐变成：

```text
构造本身就是白盒的图分类器
```

最终形成：

```text
WG-SRC = White-Box Graph Subspace–Ridge Classifier
```

最终方法不是 GNN，而是一个由线性代数模块组成的白盒图分类框架：

```text
显式 graph-signal dictionary
→ Fisher coordinate selection
→ class-wise PCA subspace residual
→ closed-form multi-alpha Ridge
→ validation-selected PCA/Ridge score fusion
→ node-level atlas / dataset fingerprint
```

---

## 2. 总体版本谱系

整个项目的主干演化可以概括为：

```text
MCR² / ReduNet / CRATE 表示几何诊断
        ↓
layerwise geometric tracing
        ↓
whitebox_gat / whitebox_gcn / CRATE-like GNN unfolding
        ↓
G-series: G1 / G2 / G3 / G4 hop-selection routes
        ↓
G1 CRGain 暂时胜出
        ↓
发现 G1 的理论白盒梯度路径被 trainable projection 覆盖
        ↓
L-series: 修复 white-box GNN route
        ↓
转向闭式分层白盒算法
        ↓
src_v1: Tikhonov smoothing + class subspace + response/impact + closed-form predictor
        ↓
src_v3-v6: WhiteBox-GSD pipeline 化、HASDC、raw/smooth coupling、package 化
        ↓
src_v7: Chameleon 上显式 high-pass / multihop + Fisher + PCA + Ridge
        ↓
v8: 错误节点与参数效应统计审计
        ↓
src_v9-v15: adaptive branch、safe branch、per-class dimension、pairwise specialist
        ↓
src_v16: reliability map + anti-hub
        ↓
src_v16b: per-class reliability + greedy dimension
        ↓
src_v16c: 9-block enhanced multihop + Fisher + PCA + multi-alpha Ridge
        ↓
v16d/e/f/g/v17: 复杂变体继续探索但被排除
        ↓
src_v16c_paper_experiments: 正式论文实验包
        ↓
WG-SRC 论文版本
```

这条线说明：

> 最终 WG-SRC 不是直接从某个 accuracy 高的版本挑出来的，而是经过多轮机制失败、统计审计和结构简化后收敛出来的。

---

# Part I：从黑盒解释到 white-box GNN 展开

---

## 3. MCR² 与 layerwise geometric tracing

### 3.1 对应阶段

对应整理目录：

```text
01_MCR2_layerwise_geometric_tracing
```

代表文件包括：

```text
mcr2_orth.py
mcr2_trace_fashionmnist.py
```

---

### 3.2 当时想解决的问题

最早想验证的是：

1. 神经网络隐藏层是否发生类内压缩；
2. 不同类别是否逐渐分离；
3. 每个类别是否形成低维子空间；
4. 表示维度是否逐层变得结构化；
5. MCR² 是否能解释神经网络中的 representation geometry。

因此这一阶段主要是**诊断工具**，不是最终分类器。

---

### 3.3 核心指标

当时关注的主要指标包括：

- coding rate；
- MCR² delta；
- effective rank；
- principal angle；
- subspace coherence；
- within-class scatter；
- between-class scatter；
- layerwise representation trajectory。

MCR² 的基本思想可以写成：

\[
\Delta R(Z)=R(Z)-R_c(Z)
\]

其中：

- \(Z\)：模型某一层的样本表示矩阵；
- \(R(Z)\)：所有样本整体表示的编码率；
- \(R_c(Z)\)：按类别计算后的类内编码率；
- \(\Delta R(Z)\)：整体表示复杂度与类内复杂度之间的差值。

直观上：

- \(\Delta R(Z)\) 越大，表示整体类别之间更展开；
- 类内越紧凑，类间越分离；
- 但这不自动等于子空间正交，也不自动等于分类更稳。

---

### 3.4 MCR² + orthogonality penalty 反证实验

这一阶段还做了 synthetic 10-class low-rank Gaussian subspace 实验：

- 每类 100 个 train 样本；
- 每类 100 个 test 样本；
- MLP hidden dims 为 `[128, 64, 32]`；
- 训练 15 epoch。

比较三种训练方式：

| 训练方式 | 含义 |
|---|---|
| CE only | 只用交叉熵 |
| CE + MCR² | 交叉熵加 MCR² 正则 |
| CE + MCR² + Orth | 再加显式子空间正交惩罚 |

显式正交惩罚为：

\[
L_{\mathrm{orth}}
=
\sum_{i\ne j}
\frac{\|\Sigma_i \Sigma_j\|_F^2}
{\|\Sigma_i\|_F \|\Sigma_j\|_F+\epsilon}
\]

其中：

- \(\Sigma_i = Z_i^\top Z_i / n_i\)，表示第 \(i\) 类的类内协方差；
- \(Z_i\)：第 \(i\) 类样本在当前层的表示；
- \(n_i\)：第 \(i\) 类样本数；
- \(\|\cdot\|_F\)：Frobenius norm；
- \(\epsilon\)：防止分母为 0；
- \(L_{\mathrm{orth}}\) 越小，表示类别子空间越接近正交。

实验记录中的结果：

| Mode | Last Hidden ΔR | Last Hidden Coherence | Test Acc |
|---|---:|---:|---:|
| CE | 8.6702 | 0.9962 | 1.0000 |
| CE + MCR² | 24.1395 | 0.9731 | 1.0000 |
| CE + MCR² + Orth | 10.1174 | 0.9741 | 0.7943 |

---

### 3.5 这一阶段的结论

关键结论不是“MCR² 完全成功”，而是：

1. MCR² 确实能改变隐藏层几何结构。  
   `Last Hidden ΔR` 从 `8.6702` 提升到 `24.1395`。

2. 但是 MCR² 不自动产生真正的类间正交子空间。  
   coherence 仍然很高。

3. 显式 orthogonality penalty 没有稳定改善 coherence，反而破坏分类性能。  
   Test Acc 从 `1.0000` 降到 `0.7943`。

4. 因此，直接要求神经网络隐藏层同时满足 CE、MCR² 和正交约束并不稳。

这一步带来的核心教训是：

> 不能只把“解释指标”强行加到黑盒网络 loss 里。  
> 如果想要白盒性，可能要从模型结构本身开始重构。

---

## 4. white-box GNN 展开尝试

### 4.1 对应阶段

对应目录：

```text
02_whitebox_GNN_unfolding_before_G_series
```

代表文件：

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

### 4.2 当时想解决的问题

普通 GNN 的 message passing 是训练出来的，内部机制不透明。  
这一阶段尝试把 GNN 层写成类似优化算法的显式展开。

形式大致是：

\[
H_{\frac12}
=
H
+
\eta \nabla R
-
\eta \lambda_{\mathrm{lap}} L H
\]

再经过近端算子：

\[
H_{\mathrm{out}}=\operatorname{prox}(H_{\frac12})
\]

其中：

- \(H\)：当前层节点表示；
- \(\eta\)：步长；
- \(\nabla R\)：coding-rate 或 rate-reduction 相关梯度；
- \(\lambda_{\mathrm{lap}}\)：图拉普拉斯正则强度；
- \(L\)：图拉普拉斯矩阵；
- \(LH\)：图平滑/图能量项；
- \(\operatorname{prox}\)：近端算子，早期实现中对应 soft-thresholding / LayerNorm 等操作。

---

### 4.3 这一阶段的意义

这一阶段提出了一个重要方向：

```text
GNN layer 不应只是黑盒神经层；
它可以被看作 graph regularization + rate-reduction + proximal step 的展开。
```

但问题也很快出现：

1. 仍然保留 trainable neural projection；
2. 公式解释和实际 forward path 不一定一致；
3. 机制复杂，不容易审计；
4. 不能自然生成后面需要的 node-level atlas。

因此进入 G-series，测试更明确的 hop 选择机制。

---

## 5. G-series：四条 white-box hop selection 路线

### 5.1 对应阶段

对应目录：

```text
03_G_series_CRgain_frequency_variational_rank_routes
```

核心问题：

> 如果要显式选择不同 graph hop，那么 hop 权重应该由什么白盒量决定？

---

### 5.2 G1：CRGain

代表文件：

```text
run_g1_cr_gain.py
run_g1_fixed_small.py
run_g1_small_clean.py
```

核心思想：

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

其中：

- \(Z_k\)：第 \(k\) hop 的节点表示；
- \(R_k\)：第 \(k\) hop 的 coding rate；
- \(w_k\)：第 \(k\) hop 的权重；
- \(\tau\)：softmax 温度；
- \(\alpha\)：尺度系数。

直观解释：

```text
哪个 hop 的 representation coding rate 更高，
哪个 hop 就获得更大权重。
```

科研意义：

- G1 证明 coding-rate 可以作为图传播 hop 选择信号；
- 在早期 G-series 中表现最好；
- 因此一度成为主路线候选。

---

### 5.3 G2：Frequency / Laplacian energy

核心思想：

\[
E_k
=
\frac{\operatorname{tr}(H_k^\top L H_k)}{nd}
\]

然后：

\[
w_k
=
\operatorname{softmax}\left(-\frac{E_k}{\tau}\right)
\]

其中：

- \(E_k\)：第 \(k\) hop 的图频率能量；
- \(L\)：图拉普拉斯；
- \(n\)：节点数；
- \(d\)：特征维度。

失败原因：

```text
Laplacian energy 只能说明表示是否平滑，
但平滑程度不等于类别判别力。
```

这在后来的 Chameleon 异配图实验中被进一步验证：  
异配图中 low-pass 平滑可能反而抹掉有效差异。

---

### 5.4 G3：Variational hop selection

核心思想：

\[
q_k
=
\operatorname{softmax}\left(\frac{R_k}{\tau}\right)
\]

并加入：

\[
\mathrm{KL}(q\|u)
\]

其中：

- \(q\)：hop 分布；
- \(u\)：均匀分布；
- KL 项用于避免 hop 分布塌缩。

失败原因：

```text
hop distribution 容易塌缩；
变分形式增加复杂度，但没有带来稳定判别收益。
```

---

### 5.5 G4：Effective rank / subspace complexity

核心思想：

\[
r_{\mathrm{eff}}
=
\frac{\operatorname{tr}(\Sigma)^2}
{\operatorname{tr}(\Sigma^2)}
\]

用有效秩衡量 hop 表示的维度展开程度。

失败原因：

```text
有效秩高只说明表示展开，
不说明它对类别更可分。
```

---

### 5.6 G-series 的关键转折：G1 的“假白盒”问题

G1 虽然在早期表现最好，但后续代码审计发现关键问题。

理论上，forward 中应该使用：

```text
H_half_in = H + η * grad_contrib - η * λ_lap * lap_contrib
```

但实际后面又执行：

```text
H_half = self.out_proj(Z_agg)
```

这意味着：

```text
白盒梯度步 H_half_in 被计算出来了，
但真正进入后续 soft-threshold / LayerNorm 的是 trainable projection 后的 H_half。
```

换句话说：

> G1 公式上像白盒展开，但真正主导输出的路径仍然包含神经投影层。

这一步非常关键。它导致 G1 不能成为最终严格白盒算法。

---

## 6. L-series：修复 white-box GNN 路线

### 6.1 对应阶段

对应目录：

```text
04_L_series_repairs_of_whitebox_GNN_route
```

代表文件：

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

### 6.2 尝试方向

这一阶段继续尝试修复 white-box GNN 的机制问题，主要方向包括：

1. ReduNet-style expansion + compression；
2. CRATE / ISTA proximal；
3. signal-noise separation；
4. temperature annealing；
5. graph-aware coding rate；
6. multi-scale GCN；
7. adaptive gate。

---

### 6.3 结果与原因

L-series 说明 white-box-inspired neural models 是有潜力的。  
某些版本在 Cora 类数据上表现不错，例如 multi-scale GCN 和 adaptive gate 方向。

但是它没有成为最终路线，原因是：

1. 仍然依赖训练式 neural layer；
2. 仍有 gate / projection / learned weights；
3. 不够闭式；
4. 不够适合作为 dataset diagnostic probe；
5. 无法自然输出最终论文需要的 dense node-level atlas。

因此，研究方向从：

```text
修补 white-box-inspired GNN
```

转向：

```text
显式图信号 + 子空间 + 闭式分类器
```

---

# Part II：从 white-box GNN 转向闭式子空间分类器

---

## 7. src_v1：分层闭式白盒算法框架

### 7.1 对应阶段

对应目录：

```text
05_src_v1_layered_closed_form_whitebox_pipeline
```

主要模块：

```text
data_loader.py
layer1_tikhonov.py
layer2_subspace.py
layer3_response.py
layer4_impact.py
layer5_predictor.py
```

---

### 7.2 目标

构造一个不依赖黑盒神经层的白盒流水线：

```text
图平滑
→ 类子空间
→ 节点对每个类的响应
→ 残差和影响量
→ 闭式预测
```

---

### 7.3 Layer 1：Tikhonov 图平滑

核心公式：

\[
Z=(I+\lambda L)^{-1}X
\]

其中：

- \(X\)：原始节点特征矩阵；
- \(L\)：图拉普拉斯矩阵；
- \(\lambda\)：平滑强度；
- \(I\)：单位矩阵；
- \(Z\)：平滑后的节点特征。

等价优化目标：

\[
\min_Z
\|Z-X\|_F^2
+
\lambda \operatorname{tr}(Z^\top LZ)
\]

解释：

- 第一项让 \(Z\) 不要偏离原始特征；
- 第二项让相邻节点表示更平滑；
- \(\lambda\) 越大，图平滑越强。

---

### 7.4 Layer 2：类 PCA 子空间

对每个类 \(c\) 拟合：

\[
\mu_c,\quad B_c
\]

其中：

- \(\mu_c\)：类 \(c\) 的中心；
- \(B_c\)：类 \(c\) 的 PCA 子空间基；
- \(B_c^\top B_c=I\)。

这一模块是最终 WG-SRC 中 class-wise PCA subspace 的直接来源。

---

### 7.5 Layer 3 / 4：响应、残差和 impact

关键残差形式：

\[
R_{\perp,c}(z)
=
\|(I-B_cB_c^\top)(z-\mu_c)\|^2
\]

其中：

- \(I-B_cB_c^\top\)：投影到类子空间正交补的算子；
- \(z-\mu_c\)：节点表示相对类中心的偏移；
- \(R_{\perp,c}(z)\)：节点不能被类 \(c\) 子空间解释的残差。

残差越小，节点越接近类 \(c\) 的几何结构。

---

### 7.6 src_v1 的意义与问题

意义：

```text
src_v1 是从 white-box GNN 转向 closed-form white-box classifier 的关键节点。
```

它把研究从“训练一个可解释 GNN 层”转向：

```text
显式线性算子
+ 类子空间
+ 几何残差
+ 闭式预测
```

问题：

```text
src_v1 仍然偏向图平滑和同配图假设；
Chameleon 这类 heterophilic graph 会暴露出邻居不一定同类的问题。
```

因此后续必须研究 raw / smooth / high-pass graph signal 的显式分解。

---

## 8. src_v3 到 src_v6：WhiteBox-GSD 管线化、HASDC 和 raw/smooth coupling

### 8.1 对应阶段

对应目录：

```text
05b_src_v3_to_src_v6_WhiteBox_GSD_pipeline_Layer3_coupling_and_packaging
```

这一阶段不是最终算法本体，但它完成了从雏形到系统的过渡。

---

### 8.2 src_v3：pipeline 化

主要文件：

```text
data_loader.py
layer1_tikhonov.py
layer2_subspace.py
layer3_discriminative.py
framework/configs.py
framework/pipeline.py
run_all.py
```

主要意义：

```text
把 src_v1 的分层白盒思想整理成可配置 pipeline。
```

流程变成：

```text
数据加载
→ Layer1 图平滑
→ Layer2 类子空间
→ Layer3 判别性响应
→ pipeline/config 管理
→ 一键运行
```

局限：

```text
仍主要依赖 smooth representation，
尚未明确拆分 raw / low-pass / high-pass。
```

---

### 8.3 src_v4：Cora geometric-reactive 验证

关键结果记录：

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

意义：

```text
WhiteBox-GSD pipeline 在真实 citation graph 上可运行，
并且可以获得结构化结果。
```

局限：

```text
Cora 是相对同配的 citation graph；
Cora 上可行不说明 Chameleon 这类异配图上可行。
```

---

### 8.4 src_v5：HASDC 与 Layer3 coupling

src_v5 开始认真处理：

```text
不同类别子空间重叠应该怎么处理？
raw feature 和 smoothed feature 谁更重要？
```

旧 Layer3 的思想偏向：

```text
subspace overlap -> suppress
```

也就是说，如果类别子空间重叠，就抑制重叠方向。

问题是：

```text
不是所有重叠方向都是坏的；
有些共享方向可能携带有效语义。
```

因此引入 HASDC / raw-smooth coupling：

```text
smooth channel: Z = (I + λL)^(-1)X
raw channel:    X
```

并测试：

- raw_plain_residual；
- smooth_plain_residual；
- raw_dynamic_layer3；
- smooth_dynamic_layer3；
- hasdc_channel_select；
- hasdc_merged；
- hasdc_gated。

---

### 8.5 HASDC 结果说明的问题

#### WebKB 小图

早期 Cornell 结果中：

```text
raw_plain_residual test ≈ 0.7568
smooth_plain_residual test ≈ 0.5946
hasdc_gated test ≈ 0.4865 或更低
```

说明：

```text
在 Cornell 这类小型 WebKB 图上，raw feature 可能明显比 smooth feature 更可靠。
```

#### Cornell / Texas / Wisconsin Phase A

代表性结果：

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

结论：

```text
raw 和 smooth 都重要；
不同数据集需要不同信号；
但 channel-select 很多时候只是选中了更好的单通道，
没有形成稳定的新 coupling 机制。
```

#### Chameleon / Squirrel / Actor

Chameleon 早期结果：

```text
smooth_plain_residual test ≈ 0.4327
hasdc_channel_select  test ≈ 0.4327
hasdc_merged          test ≈ 0.4050
raw_plain_residual    test ≈ 0.3238
```

说明：

```text
异配图不是简单 raw 更好或 smooth 更好；
需要更细的图信号分解。
```

#### Cora / Citeseer / PubMed

代表结果：

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

说明：

```text
citation graph 上 smooth channel 通常更强；
但 HASDC 仍然不是最终核心，因为它没有显式命名 raw / low-pass / high-pass block。
```

---

### 8.6 src_v6：package 化

主要文件：

```text
model.py
pyproject.toml
requirements.txt
examples/grid_search.py
examples/synthetic_demo.py
```

意义：

```text
把 WhiteBox-GSD 变成更可复用的 package 风格接口。
```

但科研突破仍然不在 package 化，而在下一步：

```text
在进入 PCA 子空间之前，显式构造 graph-signal dictionary。
```

---

# Part III：Chameleon、高通图信号与 Fisher + PCA + Ridge

---

## 9. src_v7：Chameleon 上的关键突破

### 9.1 对应阶段

对应目录：

```text
06_src_v7_Chameleon_highpass_Fisher_PCA_Ridge_breakthrough
```

这一阶段是最终 WG-SRC 的直接前身。

---

### 9.2 阶段目标

Chameleon 是 heterophilic graph。  
在这种图上：

```text
邻居不一定同类；
单纯 low-pass smoothing 可能把节点自身判别信息抹掉；
但完全不用图结构也不够。
```

因此目标变成：

```text
显式构造 low-pass 和 high-pass graph signals，
再用白盒子空间与闭式分类器判断哪些信号有用。
```

---

### 9.3 关键发现：high-pass graph differences 有用

开始显式构造：

```text
X
PX
P²X
X - PX
PX - P²X
```

其中：

- \(X\)：原始特征；
- \(P\)：图传播矩阵；
- \(PX\)：一跳传播后的低通信号；
- \(P^2X\)：两跳传播后的低通信号；
- \(X-PX\)：节点自身与邻居均值之间的差异，高通信号；
- \(PX-P^2X\)：一跳与两跳传播之间的差异，高通信号。

关键思想：

```text
在 heterophilic graph 中，节点与邻居不同本身可能就是判别信号。
```

---

### 9.4 src_v7 的三个基础算法

#### Algorithm 1：Multihop-Full + PCA Subspace Residual

代表文件：

```text
algo1_multihop_pca_68pct.py
```

结果：

```text
Chameleon 10-split test_mean ≈ 68.16%
```

核心结构：

```text
F = [X, PX, P²X, X-PX, PX-P²X]
每个 block row-L2 normalize
每类 PCA 子空间
按 residual argmin 分类
dimension 用 validation 选择
```

意义：

```text
显式 multihop / high-pass 特征 + PCA residual 可以把 Chameleon 推到 68% 左右。
```

---

#### Algorithm 2：Multihop-Full + PCA + Ridge

代表文件：

```text
algo2_multihop_pca_ridge_70pct.py
```

结果：

```text
Chameleon 10-split test_mean ≈ 70.02%
```

变化：

```text
PCA residual 负责类几何；
Ridge classifier 负责判别边界；
二者归一化后融合。
```

意义：

```text
PCA 子空间和 Ridge 边界不是重复模块；
二者有互补性。
```

这就是最终论文中 PCA–Ridge phase map 的早期来源。

---

#### Algorithm 3：Fisher + PCA + Ridge

代表文件：

```text
algo3_fisher_ensemble_71pct.py
```

结果：

```text
Chameleon 10-split test_mean ≈ 71.10%
```

变化：

```text
先用 Fisher score 选择 top-k coordinates，
再进行 PCA + Ridge。
```

意义：

```text
Fisher 坐标选择显著提升了白盒特征字典的有效性。
```

这一步已经形成最终 WG-SRC 的核心骨架：

```text
显式图信号
+ Fisher coordinate selection
+ class PCA residual
+ closed-form Ridge boundary
+ validation fusion
```

---

### 9.5 src_v7 的局限

虽然 src_v7 已经很接近最终方法，但仍存在问题：

1. graph-signal dictionary 还不够完整；
2. 还没有 symmetric-normalized propagation；
3. 还没有 \(P^3X\)；
4. 没有形成最终 nine-block dictionary；
5. 节点级统计审计还不完整；
6. 对错误节点、branch、class-pair、dimension 的机制分析还需要系统化。

因此进入 v8 及后续统计审计路线。

---

# Part IV：统计审计驱动迭代：v8 到 v17

---

## 10. v8：错误节点和参数效应统计

### 10.1 对应目录

```text
results_v8_ch_error_audit_top20_r0/
```

包含文件：

```text
audit_meta.json
error_node_summary.csv
grid_summary_with_error_counts.csv
node_param_effects.csv
param_effects_summary.csv
```

---

### 10.2 这一阶段的目标

这一阶段不是为了直接提出新模型，而是为了回答：

```text
哪些节点反复错？
哪些参数值系统性减少错误？
模型提升到底来自哪里？
```

这一步非常重要，因为它开始把迭代方式从：

```text
看 overall accuracy
```

改成：

```text
看错误节点、参数效应和机制变化
```

---

### 10.3 统计表含义

#### `error_node_summary.csv`

记录：

```text
每个节点在多个 grid 配置下错了多少次
```

用于识别 hard nodes。

#### `grid_summary_with_error_counts.csv`

记录每个参数组合下：

```text
n_wrong_all
n_wrong_train
n_wrong_val
n_wrong_test
val
test
```

因此它不只看 accuracy，还看具体错了多少节点。

#### `node_param_effects.csv`

记录某个参数值对具体节点错误率的影响。

#### `param_effects_summary.csv`

把节点级影响聚合成参数级统计。

---

### 10.4 代表性统计

例如：

```text
dim = 8
mean_test = 0.461124
mean_n_wrong_test = 245.73

dim = 10
mean_test = 0.452485
mean_n_wrong_test = 249.67
```

说明判断依据不是单个 run 的 accuracy，而是：

```text
dim=8 在统计上减少了更多 test 错误节点。
```

---

### 10.5 v8 的科研意义

v8 的模型本身不是最终突破，但它建立了后面全部迭代的统计方式：

```text
不只问：准确率是多少？
还要问：
- 哪些节点被修复？
- 哪些节点被弄坏？
- 错误是否集中在某类？
- 参数改变是否有系统性方向？
- 机制是否带来净收益？
```

这直接发展为后面的 fixed / broken / redirect 审计。

---

## 11. baseline subspace geometry：先测类子空间本身

### 11.1 对应目录

```text
results_baseline_subspace_vectors_chameleon/
```

包含文件：

```text
pairwise_geometry_baseline_subspaces.csv
run_summary_baseline_subspace_vectors.csv
split_summary_baseline_subspace_vectors.csv
subspace_summary_baseline_vectors.csv
```

---

### 11.2 目标

在继续加复杂机制之前，先测：

```text
class subspace 本身到底有没有用？
类中心距离是多少？
子空间之间 overlap 多大？
principal angles 如何？
effective rank 如何？
retained variance 如何？
```

---

### 11.3 汇总结果

`run_summary_baseline_subspace_vectors.csv` 记录：

```text
val_mean  = 0.6897119341563785
test_mean = 0.6815789473684211
test_std  = 0.023141940277671778
```

这说明：

```text
单纯 class subspace geometry 在 Chameleon 上已经能达到约 68.16%。
```

---

### 11.4 科研意义

这个基线非常重要：

1. class-subspace route 不是无效路线；
2. Chameleon 的类几何结构确实能被 PCA 子空间捕捉一部分；
3. 但错点结构复杂，需要更细粒度的 branch / dimension / boundary 机制；
4. 后面所有复杂机制都应该和这个几何基线比较。

---

## 12. src_v9：adaptive branch 失败路线

### 12.1 对应目录

```text
results_src_v9_adaptive_branch_chameleon_records/
```

包含文件：

```text
branch_member_rows.csv
branch_subspace_summary.csv
misclassified_test_nodes.csv
run_summary.json
split_summary_adaptive_branch_records.csv
train_internal_misclassified_nodes.csv
```

---

### 12.2 阶段目标

目标是：

```text
在 root PCA 子空间之外，为困难错分模式建立 adaptive branch。
```

直觉是：

```text
一个类可能不是单一子空间；
错分节点可能对应类内部的局部分支。
```

---

### 12.3 结果

统计包中记录：

```text
val_mean  = 0.6399176954732511
test_mean = 0.6300438596491228
avg_total_branches = 15.0
avg_extra_branches = 10.0
```

这明显低于 baseline subspace 的约 68.16%。

---

### 12.4 失败原因

src_v9 的问题是：

1. branch 太多；
2. branch 太粗；
3. branch 容易吸收错误节点；
4. 没有足够保守的触发条件；
5. 增加分支复杂度，却破坏整体判别。

结论：

```text
不能简单地认为 branch 越多越好。
```

下一步必须统计每个 branch 到底：

```text
修了多少节点；
弄坏了多少节点；
把错节点 redirect 到哪里。
```

---

## 13. src_v10：safe branch 与 fixed / broken / redirect 审计

### 13.1 对应目录

```text
results_compare_branch_details_src_v9_src_v10_chameleon/
results_src_v10_branch_effect_audit_chameleon/
```

---

### 13.2 v9 到 v10 的整体对比

`comparison_mean_summary_src_v9_src_v10.csv` 记录：

| Version | val_mean | test_mean | avg_total_branches | avg_extra_branches |
|---|---:|---:|---:|---:|
| src_v9 | 0.6399177 | 0.6300439 | 15.0 | 10.0 |
| src_v10 | 0.6953361 | 0.6866228 | 12.8 | 7.8 |

重要结论：

```text
src_v10 用更少 branch，获得更高 test。
```

这说明 v10 不是靠“堆更多分支”，而是靠更保守、更有效的 branch 机制。

---

### 13.3 src_v10 branch effect audit

`split_summary_src_v10_branch_effect_audit.csv` 记录了每个 split：

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

平均趋势：

```text
test_acc_full          ≈ 0.6866
test_acc_root_same_dim ≈ 0.6805

平均每 split:
test_changed_vs_root ≈ 13.1
test_fixed_vs_root   ≈ 5.8
test_broken_vs_root  ≈ 3.0
test_wrong_redirect  ≈ 4.3
```

---

### 13.4 统计解释

几个概念：

- `fixed_vs_root`：root 原来错，branch 修对；
- `broken_vs_root`：root 原来对，branch 弄错；
- `wrong_redirect`：root 原来错，branch 改成另一个错类。

v10 的统计说明：

```text
branch 有净收益，因为 fixed > broken；
但 wrong_redirect 不小，说明 branch 仍然危险。
```

因此下一步不是继续加 branch，而是让 branch 更安全。

---

## 14. src_v11 / src_v12：geometry-safe 与 class-gated branch

### 14.1 src_v11：geometry-safe branch

对应目录：

```text
results_src_v11_geometry_safe_branch_audit_chameleon/
```

结果：

```text
val_mean  ≈ 0.6948
test_mean ≈ 0.6855
```

统计：

```text
test_fixed_by_calibration  ≈ 2.0
test_broken_by_calibration ≈ 1.9

test_fixed_by_extra  ≈ 7.4
test_broken_by_extra ≈ 3.5
```

解释：

```text
extra branch 仍然有用；
root calibration 净收益很小，因为 fixed 和 broken 接近。
```

因此 src_v11 没有成为突破。

---

### 14.2 src_v12：class-gated geometry-safe branch

对应目录：

```text
results_src_v12_classgated_geometry_safe_branch_audit_chameleon/
```

结果：

```text
val_mean  ≈ 0.6951
test_mean ≈ 0.6879
```

统计：

```text
test_fixed_by_calibration  ≈ 0.6
test_broken_by_calibration ≈ 0.3

test_fixed_by_extra  ≈ 5.7
test_broken_by_extra ≈ 2.8
```

解释：

```text
class-gated calibration 更保守，破坏减少；
但 calibration 本身影响很小；
主要收益仍来自 extra branch。
```

---

### 14.3 阶段结论

src_v11 / src_v12 说明：

1. branch 机制确实能修节点；
2. calibration 可以降低风险，但贡献小；
3. 真正的问题可能不是 branch 触发，而是 class subspace 本身的维度和表达能力；
4. 因此下一步进入 per-class dimension。

---

## 15. src_v13 / src_v14：per-class PCA dimension

### 15.1 src_v13：per-class dimension + coverage objective

对应目录：

```text
results_src_v13_perclass_dim_coverage_audit_chameleon/
```

关键文件：

```text
dim_search_trace_src_v13.csv
split_summary_src_v13_perclass_dim_coverage_audit.csv
```

结果：

```text
val_mean  ≈ 0.6989
test_mean ≈ 0.6886
```

---

### 15.2 v13 的关键变化

之前每个类使用相同 PCA dimension。  
v13 改为：

```text
每个 class 单独选择 PCA dimension。
```

原因：

```text
Chameleon 中不同类别的几何复杂度不同；
统一维度可能让某些类欠拟合，另一些类过拟合。
```

---

### 15.3 v13 的统计选择标准

`dim_search_trace_src_v13.csv` 不是只看 accuracy，而是记录：

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

也就是说，维度选择同时考虑：

1. true-class residual 是否降低；
2. true-class explained ratio 是否提高；
3. false positive rate 是否升高；
4. dimension fraction 是否过大；
5. validation 是否改善。

---

### 15.4 v13 的 fixed / broken 统计

平均趋势：

```text
root_test_acc ≈ 0.6809
test_acc      ≈ 0.6886

fixed_vs_root  ≈ 6.7
broken_vs_root ≈ 3.2
redirect       ≈ 3.4
```

说明：

```text
per-class dimension 有净收益，
但仍然存在 broken 和 redirect。
```

---

### 15.5 src_v14：dimension floor + global fallback

对应目录：

```text
results_src_v14_perclass_dim_floor_coverage_audit_chameleon/
```

结果：

```text
val_mean  ≈ 0.6979
test_mean ≈ 0.6904
```

v14 的变化：

```text
给 per-class dimension 加 floor；
必要时 fallback 到 global dimension。
```

原因：

```text
v13 某些 split 里 dimension 会选得过低，导致类子空间欠拟合。
```

v14 平均趋势：

```text
root_test_acc ≈ 0.6827
test_acc      ≈ 0.6904

fixed_vs_root  ≈ 6.1
broken_vs_root ≈ 2.6
redirect       ≈ 3.7
```

解释：

```text
fixed 略少，但 broken 也减少；
整体 test 提升，说明 floor/fallback 提高了稳健性。
```

---

### 15.6 阶段结论

src_v13 / src_v14 证明：

```text
class-subspace complexity 不同；
per-class dimension 是有效机制；
但它仍然不能单独解决 class-pair 错误和边界问题。
```

于是进入 src_v15 的 class bias 和 pairwise specialist。

---

## 16. src_v15：class bias、pairwise specialist 与 validation overfitting

### 16.1 对应目录

```text
results_src_v15_score_pairwise_audit_chameleon/
```

关键文件：

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

### 16.2 结果

统计：

```text
base_val_acc  ≈ 0.6979
base_test_acc ≈ 0.6904

bias_val_acc  ≈ 0.7048
bias_test_acc ≈ 0.6912

final val_acc  ≈ 0.7387
final test_acc ≈ 0.6923
```

这说明：

```text
validation 从 69.79% 暴涨到 73.87%，
但 test 只从 69.04% 到 69.23%。
```

---

### 16.3 为什么这很重要

如果只看 validation，src_v15 好像非常强。  
但统计表说明：

```text
pairwise specialist 大幅吃 validation，
但几乎没有转化为 test 提升。
```

这是典型 validation overfitting。

---

### 16.4 class bias trace

`class_bias_trace_src_v15.csv` 记录大量候选 bias：

```text
accepted = 18
rejected = 760
```

说明：

```text
class bias 不是随便加的；
大多数候选都被统计规则拒绝。
```

---

### 16.5 pairwise specialist trace

`pairwise_specialist_trace_src_v15.csv` 记录候选 class-pair specialist：

```text
accepted = 61
rejected = 400
```

每个候选记录：

```text
fixed
broken
redirect
broken_rate
gain
accepted
reason
```

这说明每个 pair specialist 都被审计：

```text
它修了多少节点？
它弄坏了多少节点？
它把错节点重定向到哪里？
```

---

### 16.6 v15 的最终判断

src_v15 的价值：

```text
它证明 class-pair error structure 真实存在。
```

但它不能作为最终主方法，因为：

1. validation 提升远大于 test 提升；
2. fixed 和 broken 接近抵消；
3. pairwise specialist 复杂；
4. 解释链条比主方法复杂；
5. 不适合作为统一 white-box scaffold。

因此下一步不能继续在 pairwise specialist 上堆复杂度，而要换思路。

---

## 17. src_v16 / src_v16b：reliability map 与 anti-hub

### 17.1 src_v16 对应目录

```text
results_src_v16_reliability_antihub_audit_chameleon/
```

关键文件：

```text
analysis_report_src_v16.md
reliability_map_src_v16.csv
antihub_class_stats_src_v16.csv
penalty_search_trace_src_v16.csv
misclassified_test_nodes_src_v16.csv
split_summary_src_v16_reliability_antihub_audit.csv
```

---

### 17.2 src_v16 的目标

在放弃 pairwise specialist 后，src_v16 尝试：

```text
Class Subspace Reliability Map + Anti-Hub Subspace Penalty
```

核心思想：

1. 同时保留 PCA residual 和 Ridge；
2. 用 OOF 统计判断不同条件下 PCA / Ridge 哪个更可靠；
3. 识别 hub class；
4. 对容易吸收错误节点的类加入 anti-hub penalty。

---

### 17.3 reliability map

`reliability_map_src_v16.csv` 记录：

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

含义：

- `gap_bin`：margin 大小区间；
- `agree_bin`：PCA 与 Ridge 是否一致；
- `expo_bin`：邻域 exposure 区间；
- `w_pca`：该条件下 PCA 分支的可靠权重；
- `pca_correct` / `ridge_correct`：OOF 统计中两个分支正确情况。

---

### 17.4 anti-hub statistics

`antihub_class_stats_src_v16.csv` 记录：

```text
class
hub_ratio
inflow
capture
```

用于判断某类是否过度吸收别的类节点。

---

### 17.5 penalty search trace

`penalty_search_trace_src_v16.csv` 记录：

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

这说明 src_v16 仍然是统计驱动，而不是手工挑 test。

---

### 17.6 src_v16 结果

`analysis_report_src_v16.md` 记录：

```text
Reliability+AntiHub val_mean  = 0.7207 ± 0.0183
Reliability+AntiHub test_mean = 0.7055 ± 0.0183

Fixed-weight baseline val_mean  = 0.7112 ± 0.0183
Fixed-weight baseline test_mean = 0.7103 ± 0.0155

Improvement = -0.0048
```

关键结论：

```text
reliability + anti-hub 本身没有超过 fixed-weight baseline，
反而低了约 0.48 个百分点。
```

---

### 17.7 src_v16 的真实贡献

src_v16 虽然不是最终成功机制，但它暴露了一个更重要事实：

```text
Fisher + PCA + Ridge 的 base 已经很强；
继续做复杂 correction 不如改善 graph-signal dictionary 本身。
```

这正是 src_v16c 的出发点。

---

### 17.8 src_v16b

对应目录：

```text
results_src_v16b_perclass_reliability_greedydim_audit_chameleon/
```

结果：

```text
val_mean  = 0.7276
test_mean = 0.7072
elapsed   ≈ 4940 s
```

v16b 继续做：

```text
per-class reliability
greedy dimension
```

但仍然没有接近 v16c，且运行时间很长。

结论：

```text
reliability / greedy dim 不是主突破方向。
```

---

## 18. src_v16c：最终 WG-SRC 核心成型

### 18.1 对应目录

```text
results_src_v16c_audit_chameleon/
```

关键文件：

```text
run_summary_src_v16c.json
split_summary_src_v16c.csv
```

---

### 18.2 结果

`run_summary_src_v16c.json` 记录：

```text
val_mean  = 0.7279835391
val_std   = 0.0216895810
test_mean = 0.7247807018
test_std  = 0.0175780916
elapsed   = 641.286 s
```

这相对前面版本是明显跃升：

```text
src_v16  test ≈ 70.55%
src_v16b test ≈ 70.72%
src_v16c test ≈ 72.48%
```

---

### 18.3 为什么 src_v16c 突破？

src_v16c 没有继续加复杂 correction。  
它回到最基础的问题：

```text
图信号字典本身是否足够好？
```

于是加入：

1. \(P_{\mathrm{row}}^3X\)；
2. symmetric-normalized propagation；
3. symmetric high-pass difference；
4. multi-alpha Ridge ensemble；
5. energy-based PCA dimension control；
6. validation-selected fusion weight。

---

### 18.4 最终九块图信号字典

src_v16c 的 graph-signal dictionary 为：

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

其中：

- \(X\)：原始节点特征；
- \(P_{\mathrm{row}}=D^{-1}A\)：row-normalized propagation；
- \(P_{\mathrm{sym}}=D^{-1/2}AD^{-1/2}\)：symmetric-normalized propagation；
- \(P_{\mathrm{row}}X, P_{\mathrm{row}}^2X, P_{\mathrm{row}}^3X\)：row-normalized low-pass multihop signals；
- \(X-P_{\mathrm{row}}X\)：ego-neighbor high-pass difference；
- \(P_{\mathrm{row}}X-P_{\mathrm{row}}^2X\)：one-hop/two-hop high-pass difference；
- \(P_{\mathrm{sym}}X, P_{\mathrm{sym}}^2X\)：symmetric low-pass signals；
- \(X-P_{\mathrm{sym}}X\)：symmetric high-pass difference。

如果原始维度为 \(d\)，九块拼接后维度是：

\[
p=9d
\]

Chameleon 中原始特征为 2325 维，因此九块后是：

\[
9\times 2325 = 20925
\]

---

### 18.5 Fisher coordinate selection

对每个坐标 \(j\) 计算 Fisher score：

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

其中：

- \(q_j\)：第 \(j\) 个坐标的 Fisher 分数；
- \(C\)：类别数；
- \(n_c\)：训练集中类 \(c\) 的样本数；
- \(\mu_{c,j}\)：类 \(c\) 在坐标 \(j\) 上的均值；
- \(\mu_j\)：所有训练节点在坐标 \(j\) 上的均值；
- 分子衡量类间差异；
- 分母衡量类内散度。

选择 top-\(K\) 坐标得到：

\[
F=F^{(0)}[:,S]
\]

---

### 18.6 PCA residual branch

对每个类 \(c\)，拟合 PCA：

\[
\mu_c,\quad B_c
\]

然后对节点 \(i\) 和类别 \(c\) 计算：

\[
R^{\mathrm{pca}}_{ic}
=
\left\|
(I-B_cB_c^\top)(f_i-\mu_c)
\right\|_2^2
\]

其中：

- \(f_i\)：节点 \(i\) 的 selected feature vector；
- \(\mu_c\)：类中心；
- \(B_c\)：类 \(c\) 的 PCA 子空间；
- \(R^{\mathrm{pca}}_{ic}\)：节点 \(i\) 对类 \(c\) 的子空间外残差。

越小越像该类。

---

### 18.7 Multi-alpha Ridge branch

Ridge 分支使用多个 \(\alpha\)：

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

再对多个 \(\alpha\) 的分数做平均和归一化。

意义：

```text
PCA residual 捕捉类几何；
Ridge 捕捉判别边界；
multi-alpha 让 Ridge 对正则强度更稳。
```

---

### 18.8 Validation-selected fusion

最终分数：

\[
S_{ic}
=
w\tilde R^{\mathrm{pca}}_{ic}
+
(1-w)\tilde R^{\mathrm{ridge}}_{ic}
\]

预测：

\[
\hat y_i
=
\arg\min_c S_{ic}
\]

其中：

- \(w\)：验证集选择的融合权重；
- \(\tilde R^{\mathrm{pca}}\)：归一化后的 PCA residual；
- \(\tilde R^{\mathrm{ridge}}\)：归一化后的 Ridge residual-like score。

---

### 18.9 split 0 代表配置

JSON 记录中 split 0 的配置：

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

这说明 src_v16c 的提升来自：

```text
更好的显式 graph-signal dictionary
+ Fisher feature selection
+ PCA geometry
+ Ridge boundary
+ validation-selected fusion
```

而不是复杂 post-hoc correction。

---

## 19. v16d / v16e / v16f / v16g / v17：复杂变体被排除

### 19.1 总体对比

| Version | 主要思想 | val_mean | test_mean | elapsed | 结论 |
|---|---|---:|---:|---:|---|
| src_v16 | reliability + anti-hub | 0.7207 | 0.7055 | 3784s | correction 不如 fixed base |
| src_v16b | per-class reliability + greedy dim | 0.7276 | 0.7072 | 4940s | 慢，收益有限 |
| src_v16c | 9-block enhanced multihop + multi-ridge | 0.7280 | 0.7248 | 641s | 最终主方法 |
| src_v16d | 13-block max features | 0.7280 | 0.7193 | 180s | 特征更多但 test 下降 |
| src_v16e | iterative label features | 0.7283 | 0.7248 | 1152s | 与 v16c 持平但更复杂 |
| src_v16f | diffusion stacking | 0.7171 | 0.7191 | 467s | 不如 v16c |
| src_v17 | OOF label beam | 0.7288 | 0.7239 | 6347s | val 略高但 test 不赢且太慢 |

---

### 19.2 v16d：更多 feature blocks 反而不如 v16c

v16d 尝试加入更多特征块，例如：

```text
P²X-P³X
X*PX
|X-PX|
degree-weighted X
更多 top_k
更多 feature diversity
```

结果：

```text
v16d test_mean = 0.7193
v16c test_mean = 0.7248
```

结论：

```text
特征不是越多越好；
超过九块后可能引入噪声，破坏 Fisher/PCA/Ridge 稳定性。
```

---

### 19.3 v16e：iterative label features

v16e 思路：

```text
Round 1:
9-block features + Fisher + PCA + Ridge

Round 2:
用第一轮 soft prediction 构造：
- neighbor soft-label distribution
- prediction entropy
- agreement features
再跑第二轮 PCA + Ridge
```

结果：

```text
v16e test_mean = 0.7248
v16c test_mean = 0.7248
```

二者持平，但 v16e 不作为主方法，因为：

1. 更慢；
2. 更复杂；
3. 使用预测标签再构造特征，解释链条更长；
4. 不如 v16c 的纯 graph-signal dictionary 干净；
5. 对论文主线“white-box graph signal atlas”不如 v16c 直接。

---

### 19.4 v16f：diffusion stacking

v16f 尝试 PPR / Heat diffusion 等扩散式 stacking。

结果：

```text
v16f test_mean = 0.7191
```

低于 v16c。

结论：

```text
更复杂 diffusion kernel 没有超过显式 nine-block multihop/high-pass dictionary。
```

---

### 19.5 v16g：local subspace

v16g 尝试 local subspace refinement。

问题：

```text
运行过慢，曾出现 >40 min / split 的情况。
```

因此不适合正式六数据集 repeat 和 paper scaffold。

---

### 19.6 v17：OOF label beam

对应目录：

```text
results_src_v17_oof_label_beam_audit_chameleon/
```

关键文件：

```text
candidate_trace_src_v17.csv
rewrite_nodes_round1_to_final_src_v17.csv
run_summary_src_v17_oof_label_beam.json
split_summary_src_v17_oof_label_beam.csv
```

结果：

```text
val_mean  = 0.7288065844
test_mean = 0.7239035088
elapsed   = 6346.601 s
```

v17 的 validation 略高于 v16c，但 test 略低，且慢很多。

统计中还记录：

```text
round1_scan = 4050
round2_scan = 3600
total candidate scans = 7650
```

rewrite 统计中显示 round2 能修一些节点，但也会制造 broken / wrong redirect。

结论：

```text
OOF label beam 有诊断意义，但复杂度过高；
没有超过 v16c，不适合作主方法。
```

---

## 20. 为什么最终选择 src_v16c

最终选择 src_v16c 不是因为它唯一数字最高，而是综合判断：

1. **test_mean 最高并列**  
   v16c 与 v16e 在 Chameleon 上 test_mean 持平，约 72.48%。

2. **比 v16e 更简单**  
   v16e 使用 iterative label features，解释链更长。

3. **比 v17 快很多**  
   v17 约 6347s，v16c 约 641s。

4. **不使用 pseudo-label / label beam correction**  
   v16c 保持纯 graph-signal dictionary + closed-form classifier。

5. **直接对应论文机制**  
   v16c 的九块字典正好可以变成论文中的 named blocks，并用于 node-level atlas。

6. **PCA 和 Ridge 双分支都保留**  
   这使得后续可以分析：
   - class-subspace complexity；
   - PCA/Ridge complementarity；
   - decision phase map；
   - geometry-vs-boundary behavior。

7. **适合作为 general white-box scaffold**  
   不是每个数据集上的 specialist，但最适合做跨数据集诊断 probe。

因此最终选：

```text
src_v16c = final WG-SRC core
```

---

# Part V：正式论文实验包与 WG-SRC 定型

---

## 21. src_v16c_paper_experiments：正式论文包

### 21.1 对应目录

```text
src_v16c_paper_experiments/
```

结构：

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

### 21.2 算法冻结

这一阶段中，`paperexp/core.py` 被视为最终核心。  
当时明确要求：

```text
不要修改 Fisher selection；
不要修改 PCA fitting；
不要修改 Ridge classifier；
不要修改 validation selection；
不要根据 test accuracy 调参；
不要改 split 生成逻辑；
不要动 paperexp/core.py 的算法核心。
```

只允许修复：

```text
CSV 字段重复；
路径不存在；
encoding 问题；
dict multiple values；
日志/结果输出目录错误。
```

这说明此时方法已经冻结为 paper version。

---

### 21.3 正式实验脚本

正式实验包括：

| 脚本 | 作用 |
|---|---|
| `run_E01_baseline_table_aggregate.py` | baseline 表 |
| `run_E03_random_split_stability.py` | random split paired stability |
| `run_E05_ablation.py` | 机制消融 |
| `run_E07_param_scan.py` | 参数扫描 |
| `run_E08_interpretability.py` | atlas / interpretability |
| `run_E09_efficiency.py` | efficiency profile |
| `run_E11_sample_efficiency.py` | sample efficiency |
| `run_src_v16c_fill_main_table_10repeats.py` | 主表 10 repeats |

---

### 21.4 最终提交包

最终结果中出现：

```text
src_v16c_final_6datasets_repeat_rows.csv
src_v16c_final_6datasets_summary.csv
src_v16c_main_table_10repeat_rows.csv
srcv16c_submission_package.zip
srcv16c_submission_package_20260421_105707.tar.gz
```

这说明最终实验包完成了从单一 Chameleon 版本到六数据集论文实验系统的迁移。

---

## 22. 论文中 WG-SRC 的最终定位

最终论文中，WG-SRC 的定位不是：

```text
一个为了刷最高准确率的 GNN 替代品
```

而是：

```text
一个 accuracy-preserving white-box diagnostic scaffold
```

也就是它同时承担两件事：

1. 作为 predictor，完成节点分类；
2. 作为 diagnostic probe，生成 node-level atlas 和 dataset fingerprint。

---

### 22.1 六数据集主实验

论文中六个数据集：

```text
Amazon-Computers
Amazon-Photo
Chameleon
Cornell
Texas
Wisconsin
```

主实验显示：

```text
WG-SRC 平均比 strongest aligned baseline 高 +1.52 pp。
```

这个结果的意义不是宣称 WG-SRC 是所有图学习模型中最强，而是：

```text
它的白盒诊断结果来自一个真实有效的 predictor，
而不是来自弱 post-hoc probe。
```

---

### 22.2 机制消融与 generalist-vs-specialist 判断

论文中的 ablation 显示：

- Amazon 上 no-high-pass 可能更强；
- Cornell/Texas 上 ridge-only 可能更强；
- Wisconsin 上 raw-only 可能更强；
- Chameleon 需要 full / near-full 结构；
- full scaffold 平均 rank 最好、worst-case 最好、六个数据集全部 top-3。

这和迭代路径一致：

```text
某些 specialist 可以在单数据集上更好；
但 full scaffold 才能保留完整诊断能力。
```

因此没有选 ridge-only 或 raw-only 作为主方法。

---

### 22.3 Node-level signal atlas

最终 WG-SRC 能记录每个节点：

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

这些记录聚合成 dataset fingerprint：

\[
m(D)
=
[
R_D,\ L_D,\ H_D,\ C_D,\ Q_D^{ridge},\ Q_D^{hard},\ \Delta H_D
]
\]

其中：

- \(R_D\)：dataset-level raw share；
- \(L_D\)：dataset-level low-pass share；
- \(H_D\)：dataset-level high-pass share；
- \(C_D\)：mean class-subspace complexity；
- \(Q_D^{ridge}\)：Ridge-only correction quadrant；
- \(Q_D^{hard}\)：PCA 和 Ridge 都失败的 hard fraction；
- \(\Delta H_D\)：wrong nodes 与 correct nodes 的 high-pass shift。

这正是 v8-v17 统计审计思想的最终形式化版本。

---

# Part VI：统计驱动迭代逻辑总表

---

## 23. 后期版本统计驱动逻辑

| 阶段 | 统计对象 | 关键表 | 得出的判断 | 下一步 |
|---|---|---|---|---|
| v8 | hard nodes / 参数影响 | `error_node_summary.csv`, `param_effects_summary.csv` | 不能只看 accuracy，要看错误节点结构 | 做 branch 审计 |
| baseline subspace | 类子空间几何 | `pairwise_geometry_baseline_subspaces.csv` | class subspace 有用，但仍有复杂错点 | adaptive branch |
| src_v9 | branch 结构 | `branch_member_rows.csv` | branch 太多且性能下降 | safe branch |
| src_v10 | fixed/broken/redirect | `extra_branch_rewrite_nodes_src_v10.csv` | branch 有净收益但有 redirect 风险 | geometry-safe |
| src_v11/v12 | calibration 和 gated branch | `root_calibration_rewrite_nodes_*` | calibration 贡献小，branch 仍是主效应 | per-class dimension |
| src_v13/v14 | 每类 PCA 维度 | `dim_search_trace_*` | 不同类需要不同维度，floor/fallback 更稳 | class-pair specialist |
| src_v15 | bias / pairwise specialist | `class_bias_trace`, `pairwise_specialist_trace` | validation overfitting，test 不跟随 | 放弃 specialist |
| src_v16 | reliability / anti-hub | `reliability_map`, `penalty_search_trace` | correction 不如 fixed base | 改信号字典 |
| src_v16c | 9-block dictionary | `run_summary_src_v16c.json` | 信号质量提升带来稳定突破 | paper core |
| v16d/e/f/v17 | 复杂变体 | 各 run summary / candidate trace | 更复杂不一定更好 | 保留 v16c |

---

## 24. 最终方法各组件的来源

| 最终 WG-SRC 组件 | 来源版本 | 迭代原因 |
|---|---|---|
| MCR² / 子空间解释思想 | 初始 tracing 阶段 | 证明需要看内部机制，不只看 accuracy |
| class-wise PCA subspace | src_v1 | 从 white-box GNN 转向闭式几何残差 |
| raw / smooth 区分 | HASDC / src_v5 | 发现 raw 和 smooth 在不同图上价值不同 |
| high-pass difference | src_v7 | Chameleon 证明 \(X-PX\) 等差分有用 |
| Fisher coordinate selection | src_v7 Algorithm 3 | 从 70% 推进到约 71.1% |
| Ridge boundary | src_v7 Algorithm 2 | PCA 几何不足，需要判别边界 |
| fixed/broken/redirect 审计 | src_v10 起 | 判断机制是否真的修节点 |
| per-class dimension | src_v13/v14 | 类复杂度不同 |
| reliability map | src_v16 | 验证 PCA/Ridge 条件可靠性，但 correction 不稳 |
| nine-block dictionary | src_v16c | 最终突破，信号质量优先于复杂修正 |
| node-level atlas | v8-v17 统计审计 + paper E08 | 把错误审计正式变成诊断输出 |

---

## 25. 关键失败教训

### 25.1 有公式的神经层不等于真正白盒

G1 的问题说明：

```text
理论上有 H + ηgrad - ηλLH，
但实际 forward 中被 out_proj(Z_agg) 覆盖。
```

因此最终不再依赖训练式 message-passing layer。

---

### 25.2 low-pass 不足以处理 heterophily

Chameleon 说明：

```text
邻居不同本身可能是信号；
X-PX 和 PX-P²X 不能被简单丢弃。
```

这直接导致 high-pass blocks 进入最终 dictionary。

---

### 25.3 branch 能修节点，也能弄坏节点

从 src_v10 开始，每个 branch 都必须看：

```text
fixed
broken
wrong_redirect
```

这使研究从 accuracy-driven 变成 mechanism-audited。

---

### 25.4 validation 提升不一定是真提升

src_v15 的核心教训：

```text
val: 约 69.79% → 73.87%
test: 约 69.04% → 69.23%
```

pairwise specialist 主要吃 validation，不能作为主方法。

---

### 25.5 correction mechanism 不如 signal dictionary

src_v16 说明：

```text
reliability + anti-hub 有解释性，
但没有超过 fixed baseline。
```

真正突破来自 src_v16c：

```text
把 graph-signal dictionary 做对。
```

---

### 25.6 特征不是越多越好

v16d 说明：

```text
13-block features < 9-block v16c
```

因此最终九块字典是简洁性和表达力之间的平衡。

---

### 25.7 更复杂不一定更适合论文主方法

v16e / v17 说明：

```text
iterative labels / OOF beam 可以接近或持平，
但复杂、慢、解释链更长。
```

最终主方法应该是：

```text
简单、闭式、可审计、可解释、可复现。
```

---

# Part VII：最终结论

---

## 26. 一句话总结

整个项目的真实迭代不是：

```text
跑很多版本，然后选准确率最高的。
```

而是：

```text
从 MCR² 几何解释出发；
先尝试 white-box GNN 展开；
发现神经层路线不够严格白盒；
转向闭式子空间分类器；
在 Chameleon 上通过 high-pass / multihop 发现图信号分解的重要性；
通过 v8-v17 的错误节点、fixed/broken/redirect、dimension、reliability、candidate trace 统计排除复杂但不稳定的修正机制；
最终收敛到 src_v16c：
九块显式图信号字典 + Fisher 坐标选择 + 类 PCA 子空间 + multi-alpha Ridge + 验证融合。
```

---

## 27. 最适合放入仓库 research notes 的最终叙事

可以将本项目描述为：

```text
This project began as an attempt to inspect whether neural representations obey low-rank class-subspace geometry suggested by MCR², ReduNet, and CRATE. Early white-box GNN unfolding routes showed that merely adding interpretable formulas to neural layers was insufficient, especially after code audit revealed that the theoretically meaningful update path could be bypassed by trainable projections.

The project then shifted toward closed-form white-box graph learning. Through Tikhonov smoothing, class PCA subspaces, response/residual scoring, raw-vs-smooth channel experiments, and Chameleon heterophily studies, the development identified explicit high-pass graph differences and multihop graph signals as essential. Subsequent v8-v17 iterations were driven not only by accuracy but by statistical audits of error nodes, branch fixed/broken/redirect behavior, per-class dimension traces, class-pair specialists, reliability maps, anti-hub penalties, and OOF candidate rewrites.

These audits showed that complex corrective mechanisms could improve validation or fix some nodes, but often introduced broken nodes, wrong redirects, overfitting, or excessive runtime. The final breakthrough was therefore not a more complex correction layer, but a cleaner signal dictionary: the src_v16c nine-block explicit graph-signal dictionary combined with Fisher coordinate selection, class-wise PCA subspace residuals, multi-alpha closed-form Ridge classification, and validation-selected fusion. This became the final WG-SRC scaffold and later supported the node-level signal atlas and dataset fingerprint analyses in the paper.
```

---

## 28. 最终版本定位

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

## 29. 建议保留的研究记录目录

为了证明迭代是统计驱动的，建议保留以下统计目录：

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

最关键的证据文件包括：

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

这些文件共同证明：

```text
后期迭代不是简单 accuracy search，
而是 error-audit-driven、mechanism-audit-driven、statistical-summary-driven 的科研过程。
```

---

## 30. 最终可放在 README / research notes 首页的短版摘要

```text
WG-SRC was developed through a multi-stage mechanism-audited iteration process. The project began with MCR²-based representation geometry tracing and early white-box GNN unfolding attempts. These routes revealed that interpretable formulas inside trainable GNN layers were not sufficient to guarantee a truly white-box mechanism. The development then shifted to closed-form graph subspace classifiers.

On heterophilic Chameleon, explicit high-pass graph differences and multihop signals proved essential. Later versions were not selected by accuracy alone: the development used node-level error audits, branch fixed/broken/redirect statistics, per-class dimension traces, class-pair specialist traces, reliability maps, anti-hub penalty searches, and OOF label-beam candidate traces. These statistics showed that complex correction mechanisms often improved validation but did not transfer reliably to test or introduced broken/wrong-redirect nodes.

The final selected method, src_v16c, uses a nine-block explicit graph-signal dictionary, Fisher coordinate selection, class-wise PCA subspace residuals, multi-alpha closed-form Ridge classification, and validation-selected score fusion. It was chosen because it achieved the best or tied-best Chameleon performance while remaining simpler, faster, more interpretable, and better aligned with the final node-level atlas and dataset-fingerprint analysis. This became the official WG-SRC scaffold in the paper.
```
