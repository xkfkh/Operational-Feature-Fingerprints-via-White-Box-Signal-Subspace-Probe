# WG-SRC / src_v16c 完整科研迭代邏輯與統計驅動原因

> 本文件用於系統整理 WG-SRC / src_v16c 從最初思想到最終論文實驗包的完整迭代路徑。  
> 重點不是簡單列出“哪個版本準確率最高”，而是整理每一代為什麼產生、解決了什麼問題、統計結果說明了什麼、為什麼繼續或放棄，以及這些迭代如何最終收斂到 `src_v16c` 和論文中的 WG-SRC。

---

## 0. 文件定位

本文件整理的對象包括四類材料：

1. **早期 Claude JSONL 迭代記錄**  
   記錄了從 MCR²、ReduNet、CRATE、white-box GNN、G1/G2/G3/G4、L-series，到 `src_v16c` 的模型開發過程。

2. **主文件夾詳細路徑記錄**  
   記錄了項目中真實存在的版本目錄、腳本目錄、結果目錄和最終 paper experiment 包路徑。

3. **`version_statistical_summary_pack` 統計包**  
   包含 v8 到 v17 之間大量統計審計結果，例如：
   - 錯誤節點統計；
   - 參數影響統計；
   - branch fixed / broken / redirect 統計；
   - class-pair geometry；
   - per-class dimension search trace；
   - class bias trace；
   - pairwise specialist trace；
   - reliability map；
   - anti-hub penalty search；
   - OOF label beam candidate trace。

4. **最終論文內容**  
   最終論文中的 WG-SRC 被定義為一個白盒圖分類器和圖數據集診斷 probe：  
   顯式圖信號字典 + Fisher 坐標選擇 + 類 PCA 子空間 + 閉式 Ridge 分類 + 驗證集融合 + node-level signal atlas。

本文件的目的不是重新寫論文，而是把**真實研究過程**整理成可讀、可審計、可放入 GitHub `research_notes/` 的 Markdown 文檔。

---

## 1. 總體研究目標的變化

### 1.1 最初目標

項目最初不是為了單純提高圖分類準確率，而是從一個更基礎的問題出發：

> 機器學習模型內部到底是否形成了可解釋的低維結構？  
> 是否可以用更白盒、更線性代數化的機制替代黑盒神經網絡？

最初理論背景來自：

- MCR²；
- ReduNet；
- CRATE；
- 子空間學習；
- coding rate；
- 類內壓縮與類間分離。

核心直覺是：

> 一個好的表示空間應該讓同類樣本落在相對低維的類子空間裡，不同類別的子空間盡量可分；如果這種結構可被顯式構造，就不一定需要完全依賴黑盒神經網絡。

---

### 1.2 後來目標

隨著實驗推進，研究目標從：

```text
解釋黑盒模型內部表示
```

逐漸變成：

```text
構造本身就是白盒的圖分類器
```

最終形成：

```text
WG-SRC = White-Box Graph Subspace–Ridge Classifier
```

最終方法不是 GNN，而是一個由線性代數模組組成的白盒圖分類框架：

```text
顯式 graph-signal dictionary
→ Fisher coordinate selection
→ class-wise PCA subspace residual
→ closed-form multi-alpha Ridge
→ validation-selected PCA/Ridge score fusion
→ node-level atlas / dataset fingerprint
```

---

## 2. 總體版本譜系

整個項目的主幹演化可以概括為：

```text
MCR² / ReduNet / CRATE 表示幾何診斷
        ↓
layerwise geometric tracing
        ↓
whitebox_gat / whitebox_gcn / CRATE-like GNN unfolding
        ↓
G-series: G1 / G2 / G3 / G4 hop-selection routes
        ↓
G1 CRGain 暫時勝出
        ↓
發現 G1 的理論白盒梯度路徑被 trainable projection 覆蓋
        ↓
L-series: 修復 white-box GNN route
        ↓
轉向閉式分層白盒算法
        ↓
src_v1: Tikhonov smoothing + class subspace + response/impact + closed-form predictor
        ↓
src_v3-v6: WhiteBox-GSD pipeline 化、HASDC、raw/smooth coupling、package 化
        ↓
src_v7: Chameleon 上顯式 high-pass / multihop + Fisher + PCA + Ridge
        ↓
v8: 錯誤節點與參數效應統計審計
        ↓
src_v9-v15: adaptive branch、safe branch、per-class dimension、pairwise specialist
        ↓
src_v16: reliability map + anti-hub
        ↓
src_v16b: per-class reliability + greedy dimension
        ↓
src_v16c: 9-block enhanced multihop + Fisher + PCA + multi-alpha Ridge
        ↓
v16d/e/f/g/v17: 複雜變體繼續探索但被排除
        ↓
src_v16c_paper_experiments: 正式論文實驗包
        ↓
WG-SRC 論文版本
```

這條線說明：

> 最終 WG-SRC 不是直接從某個 accuracy 高的版本挑出來的，而是經過多輪機制失敗、統計審計和結構簡化後收斂出來的。

---

# Part I：從黑盒解釋到 white-box GNN 展開

---

## 3. MCR² 與 layerwise geometric tracing

### 3.1 對應階段

對應整理目錄：

```text
01_MCR2_layerwise_geometric_tracing
```

代表文件包括：

```text
mcr2_orth.py
mcr2_trace_fashionmnist.py
```

---

### 3.2 當時想解決的問題

最早想驗證的是：

1. 神經網絡隱藏層是否發生類內壓縮；
2. 不同類別是否逐漸分離；
3. 每個類別是否形成低維子空間；
4. 表示維度是否逐層變得結構化；
5. MCR² 是否能解釋神經網絡中的 representation geometry。

因此這一階段主要是**診斷工具**，不是最終分類器。

---

### 3.3 核心指標

當時關注的主要指標包括：

- coding rate；
- MCR² delta；
- effective rank；
- principal angle；
- subspace coherence；
- within-class scatter；
- between-class scatter；
- layerwise representation trajectory。

MCR² 的基本思想可以寫成：

\[
\Delta R(Z)=R(Z)-R_c(Z)
\]

其中：

- \(Z\)：模型某一層的樣本表示矩陣；
- \(R(Z)\)：所有樣本整體表示的編码率；
- \(R_c(Z)\)：按類別計算後的類內編码率；
- \(\Delta R(Z)\)：整體表示複雜度與類內複雜度之間的差值。

直觀上：

- \(\Delta R(Z)\) 越大，表示整體類別之間更展開；
- 類內越緊湊，類間越分離；
- 但這不自動等於子空間正交，也不自動等於分類更穩。

---

### 3.4 MCR² + orthogonality penalty 反證實驗

這一階段還做了 synthetic 10-class low-rank Gaussian subspace 實驗：

- 每類 100 個 train 樣本；
- 每類 100 個 test 樣本；
- MLP hidden dims 為 `[128, 64, 32]`；
- 訓練 15 epoch。

比較三種訓練方式：

| 訓練方式 | 含義 |
|---|---|
| CE only | 只用交叉熵 |
| CE + MCR² | 交叉熵加 MCR² 正則 |
| CE + MCR² + Orth | 再加顯式子空間正交懲罰 |

顯式正交懲罰為：

\[
L_{\mathrm{orth}}
=
\sum_{i\ne j}
\frac{\|\Sigma_i \Sigma_j\|_F^2}
{\|\Sigma_i\|_F \|\Sigma_j\|_F+\epsilon}
\]

其中：

- \(\Sigma_i = Z_i^\top Z_i / n_i\)，表示第 \(i\) 類的類內協方差；
- \(Z_i\)：第 \(i\) 類樣本在當前層的表示；
- \(n_i\)：第 \(i\) 類樣本數；
- \(\|\cdot\|_F\)：Frobenius norm；
- \(\epsilon\)：防止分母為 0；
- \(L_{\mathrm{orth}}\) 越小，表示類別子空間越接近正交。

實驗記錄中的結果：

| Mode | Last Hidden ΔR | Last Hidden Coherence | Test Acc |
|---|---:|---:|---:|
| CE | 8.6702 | 0.9962 | 1.0000 |
| CE + MCR² | 24.1395 | 0.9731 | 1.0000 |
| CE + MCR² + Orth | 10.1174 | 0.9741 | 0.7943 |

---

### 3.5 這一階段的結論

關鍵結論不是“MCR² 完全成功”，而是：

1. MCR² 確實能改變隱藏層幾何結構。  
   `Last Hidden ΔR` 從 `8.6702` 提升到 `24.1395`。

2. 但是 MCR² 不自動產生真正的類間正交子空間。  
   coherence 仍然很高。

3. 顯式 orthogonality penalty 沒有穩定改善 coherence，反而破壞分類性能。  
   Test Acc 從 `1.0000` 降到 `0.7943`。

4. 因此，直接要求神經網絡隱藏層同時滿足 CE、MCR² 和正交約束並不穩。

這一步帶來的核心教訓是：

> 不能只把“解釋指標”強行加到黑盒網絡 loss 裡。  
> 如果想要白盒性，可能要從模型結構本身開始重構。

---

## 4. white-box GNN 展開嘗試

### 4.1 對應階段

對應目錄：

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

### 4.2 當時想解決的問題

普通 GNN 的 message passing 是訓練出來的，內部機制不透明。  
這一階段嘗試把 GNN 層寫成類似優化算法的顯式展開。

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

再經過近端算子：

\[
H_{\mathrm{out}}=\operatorname{prox}(H_{\frac12})
\]

其中：

- \(H\)：當前層節點表示；
- \(\eta\)：步長；
- \(\nabla R\)：coding-rate 或 rate-reduction 相關梯度；
- \(\lambda_{\mathrm{lap}}\)：圖拉普拉斯正則強度；
- \(L\)：圖拉普拉斯矩陣；
- \(LH\)：圖平滑/圖能量項；
- \(\operatorname{prox}\)：近端算子，早期實現中對應 soft-thresholding / LayerNorm 等操作。

---

### 4.3 這一階段的意義

這一階段提出了一個重要方向：

```text
GNN layer 不應只是黑盒神經層；
它可以被看作 graph regularization + rate-reduction + proximal step 的展開。
```

但問題也很快出現：

1. 仍然保留 trainable neural projection；
2. 公式解釋和實際 forward path 不一定一致；
3. 機制複雜，不容易審計；
4. 不能自然生成後面需要的 node-level atlas。

因此進入 G-series，測試更明確的 hop 選擇機制。

---

## 5. G-series：四條 white-box hop selection 路線

### 5.1 對應階段

對應目錄：

```text
03_G_series_CRgain_frequency_variational_rank_routes
```

核心問題：

> 如果要顯式選擇不同 graph hop，那麼 hop 權重應該由什麼白盒量決定？

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

- \(Z_k\)：第 \(k\) hop 的節點表示；
- \(R_k\)：第 \(k\) hop 的 coding rate；
- \(w_k\)：第 \(k\) hop 的權重；
- \(\tau\)：softmax 溫度；
- \(\alpha\)：尺度系數。

直觀解釋：

```text
哪個 hop 的 representation coding rate 更高，
哪個 hop 就獲得更大權重。
```

科研意義：

- G1 證明 coding-rate 可以作為圖傳播 hop 選擇信號；
- 在早期 G-series 中表現最好；
- 因此一度成為主路線候選。

---

### 5.3 G2：Frequency / Laplacian energy

核心思想：

\[
E_k
=
\frac{\operatorname{tr}(H_k^\top L H_k)}{nd}
\]

然後：

\[
w_k
=
\operatorname{softmax}\left(-\frac{E_k}{\tau}\right)
\]

其中：

- \(E_k\)：第 \(k\) hop 的圖頻率能量；
- \(L\)：圖拉普拉斯；
- \(n\)：節點數；
- \(d\)：特徵維度。

失敗原因：

```text
Laplacian energy 只能說明表示是否平滑，
但平滑程度不等於類別判別力。
```

這在後來的 Chameleon 異配圖實驗中被進一步驗證：  
異配圖中 low-pass 平滑可能反而抹掉有效差異。

---

### 5.4 G3：Variational hop selection

核心思想：

\[
q_k
=
\operatorname{softmax}\left(\frac{R_k}{\tau}\right)
\]

並加入：

\[
\mathrm{KL}(q\|u)
\]

其中：

- \(q\)：hop 分布；
- \(u\)：均勻分布；
- KL 項用於避免 hop 分布塌縮。

失敗原因：

```text
hop distribution 容易塌縮；
變分形式增加複雜度，但沒有帶來穩定判別收益。
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

用有效秩衡量 hop 表示的維度展開程度。

失敗原因：

```text
有效秩高只說明表示展開，
不說明它對類別更可分。
```

---

### 5.6 G-series 的關鍵轉折：G1 的“假白盒”問題

G1 雖然在早期表現最好，但後續程式碼審計發現關鍵問題。

理論上，forward 中應該使用：

```text
H_half_in = H + η * grad_contrib - η * λ_lap * lap_contrib
```

但實際後面又執行：

```text
H_half = self.out_proj(Z_agg)
```

這意味著：

```text
白盒梯度步 H_half_in 被計算出來了，
但真正進入後續 soft-threshold / LayerNorm 的是 trainable projection 後的 H_half。
```

換句話說：

> G1 公式上像白盒展開，但真正主導輸出的路徑仍然包含神經投影層。

這一步非常關鍵。它導致 G1 不能成為最終嚴格白盒算法。

---

## 6. L-series：修復 white-box GNN 路線

### 6.1 對應階段

對應目錄：

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

### 6.2 嘗試方向

這一階段繼續嘗試修復 white-box GNN 的機制問題，主要方向包括：

1. ReduNet-style expansion + compression；
2. CRATE / ISTA proximal；
3. signal-noise separation；
4. temperature annealing；
5. graph-aware coding rate；
6. multi-scale GCN；
7. adaptive gate。

---

### 6.3 結果與原因

L-series 說明 white-box-inspired neural models 是有潛力的。  
某些版本在 Cora 類數據上表現不錯，例如 multi-scale GCN 和 adaptive gate 方向。

但是它沒有成為最終路線，原因是：

1. 仍然依賴訓練式 neural layer；
2. 仍有 gate / projection / learned weights；
3. 不夠閉式；
4. 不夠適合作為 dataset diagnostic probe；
5. 無法自然輸出最終論文需要的 dense node-level atlas。

因此，研究方向從：

```text
修補 white-box-inspired GNN
```

轉向：

```text
顯式圖信號 + 子空間 + 閉式分類器
```

---

# Part II：從 white-box GNN 轉向閉式子空間分類器

---

## 7. src_v1：分層閉式白盒算法框架

### 7.1 對應階段

對應目錄：

```text
05_src_v1_layered_closed_form_whitebox_pipeline
```

主要模組：

```text
data_loader.py
layer1_tikhonov.py
layer2_subspace.py
layer3_response.py
layer4_impact.py
layer5_predictor.py
```

---

### 7.2 目標

構造一個不依賴黑盒神經層的白盒流水線：

```text
圖平滑
→ 類子空間
→ 節點對每個類的響應
→ 殘差和影響量
→ 閉式預測
```

---

### 7.3 Layer 1：Tikhonov 圖平滑

核心公式：

\[
Z=(I+\lambda L)^{-1}X
\]

其中：

- \(X\)：原始節點特徵矩陣；
- \(L\)：圖拉普拉斯矩陣；
- \(\lambda\)：平滑強度；
- \(I\)：單位矩陣；
- \(Z\)：平滑後的節點特徵。

等價優化目標：

\[
\min_Z
\|Z-X\|_F^2
+
\lambda \operatorname{tr}(Z^\top LZ)
\]

解釋：

- 第一項讓 \(Z\) 不要偏離原始特徵；
- 第二項讓相鄰節點表示更平滑；
- \(\lambda\) 越大，圖平滑越強。

---

### 7.4 Layer 2：類 PCA 子空間

對每個類 \(c\) 擬合：

\[
\mu_c,\quad B_c
\]

其中：

- \(\mu_c\)：類 \(c\) 的中心；
- \(B_c\)：類 \(c\) 的 PCA 子空間基；
- \(B_c^\top B_c=I\)。

這一模組是最終 WG-SRC 中 class-wise PCA subspace 的直接來源。

---

### 7.5 Layer 3 / 4：響應、殘差和 impact

關鍵殘差形式：

\[
R_{\perp,c}(z)
=
\|(I-B_cB_c^\top)(z-\mu_c)\|^2
\]

其中：

- \(I-B_cB_c^\top\)：投影到類子空間正交補的算子；
- \(z-\mu_c\)：節點表示相對類中心的偏移；
- \(R_{\perp,c}(z)\)：節點不能被類 \(c\) 子空間解釋的殘差。

殘差越小，節點越接近類 \(c\) 的幾何結構。

---

### 7.6 src_v1 的意義與問題

意義：

```text
src_v1 是從 white-box GNN 轉向 closed-form white-box classifier 的關鍵節點。
```

它把研究從“訓練一個可解釋 GNN 層”轉向：

```text
顯式線性算子
+ 類子空間
+ 幾何殘差
+ 閉式預測
```

問題：

```text
src_v1 仍然偏向圖平滑和同配圖假設；
Chameleon 這類 heterophilic graph 會暴露出鄰居不一定同類的問題。
```

因此後續必須研究 raw / smooth / high-pass graph signal 的顯式分解。

---

## 8. src_v3 到 src_v6：WhiteBox-GSD 管線化、HASDC 和 raw/smooth coupling

### 8.1 對應階段

對應目錄：

```text
05b_src_v3_to_src_v6_WhiteBox_GSD_pipeline_Layer3_coupling_and_packaging
```

這一階段不是最終算法本體，但它完成了從雏形到系統的過渡。

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

主要意義：

```text
把 src_v1 的分層白盒思想整理成可配置 pipeline。
```

流程變成：

```text
數據加載
→ Layer1 圖平滑
→ Layer2 類子空間
→ Layer3 判別性響應
→ pipeline/config 管理
→ 一鍵運行
```

局限：

```text
仍主要依賴 smooth representation，
尚未明確拆分 raw / low-pass / high-pass。
```

---

### 8.3 src_v4：Cora geometric-reactive 驗證

關鍵結果記錄：

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

意義：

```text
WhiteBox-GSD pipeline 在真實 citation graph 上可運行，
並且可以獲得結構化結果。
```

局限：

```text
Cora 是相對同配的 citation graph；
Cora 上可行不說明 Chameleon 這類異配圖上可行。
```

---

### 8.4 src_v5：HASDC 與 Layer3 coupling

src_v5 開始認真處理：

```text
不同類別子空間重疊應該怎麼處理？
raw feature 和 smoothed feature 誰更重要？
```

舊 Layer3 的思想偏向：

```text
subspace overlap -> suppress
```

也就是說，如果類別子空間重疊，就抑制重疊方向。

問題是：

```text
不是所有重疊方向都是壞的；
有些共享方向可能携帶有效語義。
```

因此引入 HASDC / raw-smooth coupling：

```text
smooth channel: Z = (I + λL)^(-1)X
raw channel:    X
```

並測試：

- raw_plain_residual；
- smooth_plain_residual；
- raw_dynamic_layer3；
- smooth_dynamic_layer3；
- hasdc_channel_select；
- hasdc_merged；
- hasdc_gated。

---

### 8.5 HASDC 結果說明的問題

#### WebKB 小圖

早期 Cornell 結果中：

```text
raw_plain_residual test ≈ 0.7568
smooth_plain_residual test ≈ 0.5946
hasdc_gated test ≈ 0.4865 或更低
```

說明：

```text
在 Cornell 這類小型 WebKB 圖上，raw feature 可能明顯比 smooth feature 更可靠。
```

#### Cornell / Texas / Wisconsin Phase A

代表性結果：

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

結論：

```text
raw 和 smooth 都重要；
不同數據集需要不同信號；
但 channel-select 很多時候只是選中了更好的單通道，
沒有形成穩定的新 coupling 機制。
```

#### Chameleon / Squirrel / Actor

Chameleon 早期結果：

```text
smooth_plain_residual test ≈ 0.4327
hasdc_channel_select  test ≈ 0.4327
hasdc_merged          test ≈ 0.4050
raw_plain_residual    test ≈ 0.3238
```

說明：

```text
異配圖不是簡單 raw 更好或 smooth 更好；
需要更細的圖信號分解。
```

#### Cora / Citeseer / PubMed

代表結果：

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

說明：

```text
citation graph 上 smooth channel 通常更強；
但 HASDC 仍然不是最終核心，因為它沒有顯式命名 raw / low-pass / high-pass block。
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

意義：

```text
把 WhiteBox-GSD 變成更可複用的 package 風格接口。
```

但科研突破仍然不在 package 化，而在下一步：

```text
在進入 PCA 子空間之前，顯式構造 graph-signal dictionary。
```

---

# Part III：Chameleon、高通圖信號與 Fisher + PCA + Ridge

---

## 9. src_v7：Chameleon 上的關鍵突破

### 9.1 對應階段

對應目錄：

```text
06_src_v7_Chameleon_highpass_Fisher_PCA_Ridge_breakthrough
```

這一階段是最終 WG-SRC 的直接前身。

---

### 9.2 階段目標

Chameleon 是 heterophilic graph。  
在這種圖上：

```text
鄰居不一定同類；
單純 low-pass smoothing 可能把節點自身判別信息抹掉；
但完全不用圖結構也不夠。
```

因此目標變成：

```text
顯式構造 low-pass 和 high-pass graph signals，
再用白盒子空間與閉式分類器判斷哪些信號有用。
```

---

### 9.3 關鍵發現：high-pass graph differences 有用

開始顯式構造：

```text
X
PX
P²X
X - PX
PX - P²X
```

其中：

- \(X\)：原始特徵；
- \(P\)：圖傳播矩陣；
- \(PX\)：一跳傳播後的低通信號；
- \(P^2X\)：两跳傳播後的低通信號；
- \(X-PX\)：節點自身與鄰居均值之間的差異，高通信號；
- \(PX-P^2X\)：一跳與两跳傳播之間的差異，高通信號。

關鍵思想：

```text
在 heterophilic graph 中，節點與鄰居不同本身可能就是判別信號。
```

---

### 9.4 src_v7 的三個基礎算法

#### Algorithm 1：Multihop-Full + PCA Subspace Residual

代表文件：

```text
algo1_multihop_pca_68pct.py
```

結果：

```text
Chameleon 10-split test_mean ≈ 68.16%
```

核心結構：

```text
F = [X, PX, P²X, X-PX, PX-P²X]
每個 block row-L2 normalize
每類 PCA 子空間
按 residual argmin 分類
dimension 用 validation 選擇
```

意義：

```text
顯式 multihop / high-pass 特徵 + PCA residual 可以把 Chameleon 推到 68% 左右。
```

---

#### Algorithm 2：Multihop-Full + PCA + Ridge

代表文件：

```text
algo2_multihop_pca_ridge_70pct.py
```

結果：

```text
Chameleon 10-split test_mean ≈ 70.02%
```

變化：

```text
PCA residual 負責類幾何；
Ridge classifier 負責判別邊界；
二者歸一化後融合。
```

意義：

```text
PCA 子空間和 Ridge 邊界不是重複模組；
二者有互補性。
```

這就是最終論文中 PCA–Ridge phase map 的早期來源。

---

#### Algorithm 3：Fisher + PCA + Ridge

代表文件：

```text
algo3_fisher_ensemble_71pct.py
```

結果：

```text
Chameleon 10-split test_mean ≈ 71.10%
```

變化：

```text
先用 Fisher score 選擇 top-k coordinates，
再進行 PCA + Ridge。
```

意義：

```text
Fisher 坐標選擇顯著提升了白盒特徵字典的有效性。
```

這一步已經形成最終 WG-SRC 的核心骨架：

```text
顯式圖信號
+ Fisher coordinate selection
+ class PCA residual
+ closed-form Ridge boundary
+ validation fusion
```

---

### 9.5 src_v7 的局限

雖然 src_v7 已經很接近最終方法，但仍存在問題：

1. graph-signal dictionary 還不夠完整；
2. 還沒有 symmetric-normalized propagation；
3. 還沒有 \(P^3X\)；
4. 沒有形成最終 nine-block dictionary；
5. 節點級統計審計還不完整；
6. 對錯誤節點、branch、class-pair、dimension 的機制分析還需要系統化。

因此進入 v8 及後續統計審計路線。

---

# Part IV：統計審計驅動迭代：v8 到 v17

---

## 10. v8：錯誤節點和參數效應統計

### 10.1 對應目錄

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

### 10.2 這一階段的目標

這一階段不是為了直接提出新模型，而是為了回答：

```text
哪些節點反復錯？
哪些參數值系統性減少錯誤？
模型提升到底來自哪裡？
```

這一步非常重要，因為它開始把迭代方式從：

```text
看 overall accuracy
```

改成：

```text
看錯誤節點、參數效應和機制變化
```

---

### 10.3 統計表含義

#### `error_node_summary.csv`

記錄：

```text
每個節點在多個 grid 配置下錯了多少次
```

用於識別 hard nodes。

#### `grid_summary_with_error_counts.csv`

記錄每個參數組合下：

```text
n_wrong_all
n_wrong_train
n_wrong_val
n_wrong_test
val
test
```

因此它不只看 accuracy，還看具體錯了多少節點。

#### `node_param_effects.csv`

記錄某個參數值對具體節點錯誤率的影響。

#### `param_effects_summary.csv`

把節點級影響聚合成參數級統計。

---

### 10.4 代表性統計

例如：

```text
dim = 8
mean_test = 0.461124
mean_n_wrong_test = 245.73

dim = 10
mean_test = 0.452485
mean_n_wrong_test = 249.67
```

說明判斷依據不是單個 run 的 accuracy，而是：

```text
dim=8 在統計上減少了更多 test 錯誤節點。
```

---

### 10.5 v8 的科研意義

v8 的模型本身不是最終突破，但它建立了後面全部迭代的統計方式：

```text
不只問：準確率是多少？
還要問：
- 哪些節點被修復？
- 哪些節點被弄壞？
- 錯誤是否集中在某類？
- 參數改變是否有系統性方向？
- 機制是否帶來淨收益？
```

這直接發展為後面的 fixed / broken / redirect 審計。

---

## 11. baseline subspace geometry：先測類子空間本身

### 11.1 對應目錄

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

### 11.2 目標

在繼續加複雜機制之前，先測：

```text
class subspace 本身到底有沒有用？
類中心距離是多少？
子空間之間 overlap 多大？
principal angles 如何？
effective rank 如何？
retained variance 如何？
```

---

### 11.3 彙總結果

`run_summary_baseline_subspace_vectors.csv` 記錄：

```text
val_mean  = 0.6897119341563785
test_mean = 0.6815789473684211
test_std  = 0.023141940277671778
```

這說明：

```text
單純 class subspace geometry 在 Chameleon 上已經能達到約 68.16%。
```

---

### 11.4 科研意義

這個基線非常重要：

1. class-subspace route 不是無效路線；
2. Chameleon 的類幾何結構確實能被 PCA 子空間捕捉一部分；
3. 但錯點結構複雜，需要更細粒度的 branch / dimension / boundary 機制；
4. 後面所有複雜機制都應該和這個幾何基線比較。

---

## 12. src_v9：adaptive branch 失敗路線

### 12.1 對應目錄

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

### 12.2 階段目標

目標是：

```text
在 root PCA 子空間之外，為困難錯分模式建立 adaptive branch。
```

直覺是：

```text
一個類可能不是單一子空間；
錯分節點可能對應類內部的局部分支。
```

---

### 12.3 結果

統計包中記錄：

```text
val_mean  = 0.6399176954732511
test_mean = 0.6300438596491228
avg_total_branches = 15.0
avg_extra_branches = 10.0
```

這明顯低於 baseline subspace 的約 68.16%。

---

### 12.4 失敗原因

src_v9 的問題是：

1. branch 太多；
2. branch 太粗；
3. branch 容易吸收錯誤節點；
4. 沒有足夠保守的觸發條件；
5. 增加分支複雜度，却破壞整體判別。

結論：

```text
不能簡單地認為 branch 越多越好。
```

下一步必須統計每個 branch 到底：

```text
修了多少節點；
弄壞了多少節點；
把錯節點 redirect 到哪裡。
```

---

## 13. src_v10：safe branch 與 fixed / broken / redirect 審計

### 13.1 對應目錄

```text
results_compare_branch_details_src_v9_src_v10_chameleon/
results_src_v10_branch_effect_audit_chameleon/
```

---

### 13.2 v9 到 v10 的整體對比

`comparison_mean_summary_src_v9_src_v10.csv` 記錄：

| Version | val_mean | test_mean | avg_total_branches | avg_extra_branches |
|---|---:|---:|---:|---:|
| src_v9 | 0.6399177 | 0.6300439 | 15.0 | 10.0 |
| src_v10 | 0.6953361 | 0.6866228 | 12.8 | 7.8 |

重要結論：

```text
src_v10 用更少 branch，獲得更高 test。
```

這說明 v10 不是靠“堆更多分支”，而是靠更保守、更有效的 branch 機制。

---

### 13.3 src_v10 branch effect audit

`split_summary_src_v10_branch_effect_audit.csv` 記錄了每個 split：

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

平均趨勢：

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

### 13.4 統計解釋

幾個概念：

- `fixed_vs_root`：root 原來錯，branch 修對；
- `broken_vs_root`：root 原來對，branch 弄錯；
- `wrong_redirect`：root 原來錯，branch 改成另一個錯類。

v10 的統計說明：

```text
branch 有淨收益，因為 fixed > broken；
但 wrong_redirect 不小，說明 branch 仍然危險。
```

因此下一步不是繼續加 branch，而是讓 branch 更安全。

---

## 14. src_v11 / src_v12：geometry-safe 與 class-gated branch

### 14.1 src_v11：geometry-safe branch

對應目錄：

```text
results_src_v11_geometry_safe_branch_audit_chameleon/
```

結果：

```text
val_mean  ≈ 0.6948
test_mean ≈ 0.6855
```

統計：

```text
test_fixed_by_calibration  ≈ 2.0
test_broken_by_calibration ≈ 1.9

test_fixed_by_extra  ≈ 7.4
test_broken_by_extra ≈ 3.5
```

解釋：

```text
extra branch 仍然有用；
root calibration 淨收益很小，因為 fixed 和 broken 接近。
```

因此 src_v11 沒有成為突破。

---

### 14.2 src_v12：class-gated geometry-safe branch

對應目錄：

```text
results_src_v12_classgated_geometry_safe_branch_audit_chameleon/
```

結果：

```text
val_mean  ≈ 0.6951
test_mean ≈ 0.6879
```

統計：

```text
test_fixed_by_calibration  ≈ 0.6
test_broken_by_calibration ≈ 0.3

test_fixed_by_extra  ≈ 5.7
test_broken_by_extra ≈ 2.8
```

解釋：

```text
class-gated calibration 更保守，破壞減少；
但 calibration 本身影響很小；
主要收益仍來自 extra branch。
```

---

### 14.3 階段結論

src_v11 / src_v12 說明：

1. branch 機制確實能修節點；
2. calibration 可以降低風險，但貢獻小；
3. 真正的問題可能不是 branch 觸發，而是 class subspace 本身的維度和表達能力；
4. 因此下一步進入 per-class dimension。

---

## 15. src_v13 / src_v14：per-class PCA dimension

### 15.1 src_v13：per-class dimension + coverage objective

對應目錄：

```text
results_src_v13_perclass_dim_coverage_audit_chameleon/
```

關鍵文件：

```text
dim_search_trace_src_v13.csv
split_summary_src_v13_perclass_dim_coverage_audit.csv
```

結果：

```text
val_mean  ≈ 0.6989
test_mean ≈ 0.6886
```

---

### 15.2 v13 的關鍵變化

之前每個類使用相同 PCA dimension。  
v13 改為：

```text
每個 class 單獨選擇 PCA dimension。
```

原因：

```text
Chameleon 中不同類別的幾何複雜度不同；
統一維度可能讓某些類欠擬合，另一些類過擬合。
```

---

### 15.3 v13 的統計選擇標準

`dim_search_trace_src_v13.csv` 不是只看 accuracy，而是記錄：

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

也就是說，維度選擇同時考慮：

1. true-class residual 是否降低；
2. true-class explained ratio 是否提高；
3. false positive rate 是否升高；
4. dimension fraction 是否過大；
5. validation 是否改善。

---

### 15.4 v13 的 fixed / broken 統計

平均趨勢：

```text
root_test_acc ≈ 0.6809
test_acc      ≈ 0.6886

fixed_vs_root  ≈ 6.7
broken_vs_root ≈ 3.2
redirect       ≈ 3.4
```

說明：

```text
per-class dimension 有淨收益，
但仍然存在 broken 和 redirect。
```

---

### 15.5 src_v14：dimension floor + global fallback

對應目錄：

```text
results_src_v14_perclass_dim_floor_coverage_audit_chameleon/
```

結果：

```text
val_mean  ≈ 0.6979
test_mean ≈ 0.6904
```

v14 的變化：

```text
給 per-class dimension 加 floor；
必要時 fallback 到 global dimension。
```

原因：

```text
v13 某些 split 裡 dimension 會選得過低，導致類子空間欠擬合。
```

v14 平均趨勢：

```text
root_test_acc ≈ 0.6827
test_acc      ≈ 0.6904

fixed_vs_root  ≈ 6.1
broken_vs_root ≈ 2.6
redirect       ≈ 3.7
```

解釋：

```text
fixed 略少，但 broken 也減少；
整體 test 提升，說明 floor/fallback 提高了穩健性。
```

---

### 15.6 階段結論

src_v13 / src_v14 證明：

```text
class-subspace complexity 不同；
per-class dimension 是有效機制；
但它仍然不能單獨解決 class-pair 錯誤和邊界問題。
```

於是進入 src_v15 的 class bias 和 pairwise specialist。

---

## 16. src_v15：class bias、pairwise specialist 與 validation overfitting

### 16.1 對應目錄

```text
results_src_v15_score_pairwise_audit_chameleon/
```

關鍵文件：

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

### 16.2 結果

統計：

```text
base_val_acc  ≈ 0.6979
base_test_acc ≈ 0.6904

bias_val_acc  ≈ 0.7048
bias_test_acc ≈ 0.6912

final val_acc  ≈ 0.7387
final test_acc ≈ 0.6923
```

這說明：

```text
validation 從 69.79% 暴漲到 73.87%，
但 test 只從 69.04% 到 69.23%。
```

---

### 16.3 為什麼這很重要

如果只看 validation，src_v15 好像非常強。  
但統計表說明：

```text
pairwise specialist 大幅吃 validation，
但幾乎沒有轉化為 test 提升。
```

這是典型 validation overfitting。

---

### 16.4 class bias trace

`class_bias_trace_src_v15.csv` 記錄大量候選 bias：

```text
accepted = 18
rejected = 760
```

說明：

```text
class bias 不是隨便加的；
大多數候選都被統計規則拒絕。
```

---

### 16.5 pairwise specialist trace

`pairwise_specialist_trace_src_v15.csv` 記錄候選 class-pair specialist：

```text
accepted = 61
rejected = 400
```

每個候選記錄：

```text
fixed
broken
redirect
broken_rate
gain
accepted
reason
```

這說明每個 pair specialist 都被審計：

```text
它修了多少節點？
它弄壞了多少節點？
它把錯節點重定向到哪裡？
```

---

### 16.6 v15 的最終判斷

src_v15 的價值：

```text
它證明 class-pair error structure 真實存在。
```

但它不能作為最終主方法，因為：

1. validation 提升遠大於 test 提升；
2. fixed 和 broken 接近抵消；
3. pairwise specialist 複雜；
4. 解釋鏈條比主方法複雜；
5. 不適合作為統一 white-box scaffold。

因此下一步不能繼續在 pairwise specialist 上堆複雜度，而要換思路。

---

## 17. src_v16 / src_v16b：reliability map 與 anti-hub

### 17.1 src_v16 對應目錄

```text
results_src_v16_reliability_antihub_audit_chameleon/
```

關鍵文件：

```text
analysis_report_src_v16.md
reliability_map_src_v16.csv
antihub_class_stats_src_v16.csv
penalty_search_trace_src_v16.csv
misclassified_test_nodes_src_v16.csv
split_summary_src_v16_reliability_antihub_audit.csv
```

---

### 17.2 src_v16 的目標

在放棄 pairwise specialist 後，src_v16 嘗試：

```text
Class Subspace Reliability Map + Anti-Hub Subspace Penalty
```

核心思想：

1. 同時保留 PCA residual 和 Ridge；
2. 用 OOF 統計判斷不同條件下 PCA / Ridge 哪個更可靠；
3. 識別 hub class；
4. 對容易吸收錯誤節點的類加入 anti-hub penalty。

---

### 17.3 reliability map

`reliability_map_src_v16.csv` 記錄：

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

含義：

- `gap_bin`：margin 大小區間；
- `agree_bin`：PCA 與 Ridge 是否一致；
- `expo_bin`：鄰域 exposure 區間；
- `w_pca`：該條件下 PCA 分支的可靠權重；
- `pca_correct` / `ridge_correct`：OOF 統計中兩個分支正確情況。

---

### 17.4 anti-hub statistics

`antihub_class_stats_src_v16.csv` 記錄：

```text
class
hub_ratio
inflow
capture
```

用於判斷某類是否過度吸收別的類節點。

---

### 17.5 penalty search trace

`penalty_search_trace_src_v16.csv` 記錄：

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

這說明 src_v16 仍然是統計驅動，而不是手工挑 test。

---

### 17.6 src_v16 結果

`analysis_report_src_v16.md` 記錄：

```text
Reliability+AntiHub val_mean  = 0.7207 ± 0.0183
Reliability+AntiHub test_mean = 0.7055 ± 0.0183

Fixed-weight baseline val_mean  = 0.7112 ± 0.0183
Fixed-weight baseline test_mean = 0.7103 ± 0.0155

Improvement = -0.0048
```

關鍵結論：

```text
reliability + anti-hub 本身沒有超過 fixed-weight baseline，
反而低了約 0.48 個百分點。
```

---

### 17.7 src_v16 的真實貢獻

src_v16 雖然不是最終成功機制，但它暴露了一個更重要事實：

```text
Fisher + PCA + Ridge 的 base 已經很強；
繼續做複雜 correction 不如改善 graph-signal dictionary 本身。
```

這正是 src_v16c 的出發點。

---

### 17.8 src_v16b

對應目錄：

```text
results_src_v16b_perclass_reliability_greedydim_audit_chameleon/
```

結果：

```text
val_mean  = 0.7276
test_mean = 0.7072
elapsed   ≈ 4940 s
```

v16b 繼續做：

```text
per-class reliability
greedy dimension
```

但仍然沒有接近 v16c，且運行時間很長。

結論：

```text
reliability / greedy dim 不是主突破方向。
```

---

## 18. src_v16c：最終 WG-SRC 核心成型

### 18.1 對應目錄

```text
results_src_v16c_audit_chameleon/
```

關鍵文件：

```text
run_summary_src_v16c.json
split_summary_src_v16c.csv
```

---

### 18.2 結果

`run_summary_src_v16c.json` 記錄：

```text
val_mean  = 0.7279835391
val_std   = 0.0216895810
test_mean = 0.7247807018
test_std  = 0.0175780916
elapsed   = 641.286 s
```

這相對前面版本是明顯躍升：

```text
src_v16  test ≈ 70.55%
src_v16b test ≈ 70.72%
src_v16c test ≈ 72.48%
```

---

### 18.3 為什麼 src_v16c 突破？

src_v16c 沒有繼續加複雜 correction。  
它回到最基礎的問題：

```text
圖信號字典本身是否足夠好？
```

於是加入：

1. \(P_{\mathrm{row}}^3X\)；
2. symmetric-normalized propagation；
3. symmetric high-pass difference；
4. multi-alpha Ridge ensemble；
5. energy-based PCA dimension control；
6. validation-selected fusion weight。

---

### 18.4 最終九塊圖信號字典

src_v16c 的 graph-signal dictionary 為：

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

- \(X\)：原始節點特徵；
- \(P_{\mathrm{row}}=D^{-1}A\)：row-normalized propagation；
- \(P_{\mathrm{sym}}=D^{-1/2}AD^{-1/2}\)：symmetric-normalized propagation；
- \(P_{\mathrm{row}}X, P_{\mathrm{row}}^2X, P_{\mathrm{row}}^3X\)：row-normalized low-pass multihop signals；
- \(X-P_{\mathrm{row}}X\)：ego-neighbor high-pass difference；
- \(P_{\mathrm{row}}X-P_{\mathrm{row}}^2X\)：one-hop/two-hop high-pass difference；
- \(P_{\mathrm{sym}}X, P_{\mathrm{sym}}^2X\)：symmetric low-pass signals；
- \(X-P_{\mathrm{sym}}X\)：symmetric high-pass difference。

如果原始維度為 \(d\)，九塊拼接後維度是：

\[
p=9d
\]

Chameleon 中原始特徵為 2325 維，因此九塊後是：

\[
9\times 2325 = 20925
\]

---

### 18.5 Fisher coordinate selection

對每個坐標 \(j\) 計算 Fisher score：

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

- \(q_j\)：第 \(j\) 個坐標的 Fisher 分數；
- \(C\)：類別數；
- \(n_c\)：訓練集中類 \(c\) 的樣本數；
- \(\mu_{c,j}\)：類 \(c\) 在坐標 \(j\) 上的均值；
- \(\mu_j\)：所有訓練節點在坐標 \(j\) 上的均值；
- 分子衡量類間差異；
- 分母衡量類內散度。

選擇 top-\(K\) 坐標得到：

\[
F=F^{(0)}[:,S]
\]

---

### 18.6 PCA residual branch

對每個類 \(c\)，擬合 PCA：

\[
\mu_c,\quad B_c
\]

然後對節點 \(i\) 和類別 \(c\) 計算：

\[
R^{\mathrm{pca}}_{ic}
=
\left\|
(I-B_cB_c^\top)(f_i-\mu_c)
\right\|_2^2
\]

其中：

- \(f_i\)：節點 \(i\) 的 selected feature vector；
- \(\mu_c\)：類中心；
- \(B_c\)：類 \(c\) 的 PCA 子空間；
- \(R^{\mathrm{pca}}_{ic}\)：節點 \(i\) 對類 \(c\) 的子空間外殘差。

越小越像該類。

---

### 18.7 Multi-alpha Ridge branch

Ridge 分支使用多個 \(\alpha\)：

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

再對多個 \(\alpha\) 的分數做平均和歸一化。

意義：

```text
PCA residual 捕捉類幾何；
Ridge 捕捉判別邊界；
multi-alpha 讓 Ridge 對正則強度更穩。
```

---

### 18.8 Validation-selected fusion

最終分數：

\[
S_{ic}
=
w\tilde R^{\mathrm{pca}}_{ic}
+
(1-w)\tilde R^{\mathrm{ridge}}_{ic}
\]

預測：

\[
\hat y_i
=
\arg\min_c S_{ic}
\]

其中：

- \(w\)：驗證集選擇的融合權重；
- \(\tilde R^{\mathrm{pca}}\)：歸一化後的 PCA residual；
- \(\tilde R^{\mathrm{ridge}}\)：歸一化後的 Ridge residual-like score。

---

### 18.9 split 0 代表配置

JSON 記錄中 split 0 的配置：

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

這說明 src_v16c 的提升來自：

```text
更好的顯式 graph-signal dictionary
+ Fisher feature selection
+ PCA geometry
+ Ridge boundary
+ validation-selected fusion
```

而不是複雜 post-hoc correction。

---

## 19. v16d / v16e / v16f / v16g / v17：複雜變體被排除

### 19.1 總體對比

| Version | 主要思想 | val_mean | test_mean | elapsed | 結論 |
|---|---|---:|---:|---:|---|
| src_v16 | reliability + anti-hub | 0.7207 | 0.7055 | 3784s | correction 不如 fixed base |
| src_v16b | per-class reliability + greedy dim | 0.7276 | 0.7072 | 4940s | 慢，收益有限 |
| src_v16c | 9-block enhanced multihop + multi-ridge | 0.7280 | 0.7248 | 641s | 最終主方法 |
| src_v16d | 13-block max features | 0.7280 | 0.7193 | 180s | 特徵更多但 test 下降 |
| src_v16e | iterative label features | 0.7283 | 0.7248 | 1152s | 與 v16c 持平但更複雜 |
| src_v16f | diffusion stacking | 0.7171 | 0.7191 | 467s | 不如 v16c |
| src_v17 | OOF label beam | 0.7288 | 0.7239 | 6347s | val 略高但 test 不贏且太慢 |

---

### 19.2 v16d：更多 feature blocks 反而不如 v16c

v16d 嘗試加入更多特徵塊，例如：

```text
P²X-P³X
X*PX
|X-PX|
degree-weighted X
更多 top_k
更多 feature diversity
```

結果：

```text
v16d test_mean = 0.7193
v16c test_mean = 0.7248
```

結論：

```text
特徵不是越多越好；
超過九塊後可能引入噪聲，破壞 Fisher/PCA/Ridge 穩定性。
```

---

### 19.3 v16e：iterative label features

v16e 思路：

```text
Round 1:
9-block features + Fisher + PCA + Ridge

Round 2:
用第一輪 soft prediction 構造：
- neighbor soft-label distribution
- prediction entropy
- agreement features
再跑第二輪 PCA + Ridge
```

結果：

```text
v16e test_mean = 0.7248
v16c test_mean = 0.7248
```

二者持平，但 v16e 不作為主方法，因為：

1. 更慢；
2. 更複雜；
3. 使用預測標簽再構造特徵，解釋鏈條更長；
4. 不如 v16c 的純 graph-signal dictionary 乾淨；
5. 對論文主線“white-box graph signal atlas”不如 v16c 直接。

---

### 19.4 v16f：diffusion stacking

v16f 嘗試 PPR / Heat diffusion 等擴散式 stacking。

結果：

```text
v16f test_mean = 0.7191
```

低於 v16c。

結論：

```text
更複雜 diffusion kernel 沒有超過顯式 nine-block multihop/high-pass dictionary。
```

---

### 19.5 v16g：local subspace

v16g 嘗試 local subspace refinement。

問題：

```text
運行過慢，曾出現 >40 min / split 的情況。
```

因此不適合正式六數據集 repeat 和 paper scaffold。

---

### 19.6 v17：OOF label beam

對應目錄：

```text
results_src_v17_oof_label_beam_audit_chameleon/
```

關鍵文件：

```text
candidate_trace_src_v17.csv
rewrite_nodes_round1_to_final_src_v17.csv
run_summary_src_v17_oof_label_beam.json
split_summary_src_v17_oof_label_beam.csv
```

結果：

```text
val_mean  = 0.7288065844
test_mean = 0.7239035088
elapsed   = 6346.601 s
```

v17 的 validation 略高於 v16c，但 test 略低，且慢很多。

統計中還記錄：

```text
round1_scan = 4050
round2_scan = 3600
total candidate scans = 7650
```

rewrite 統計中顯示 round2 能修一些節點，但也會制造 broken / wrong redirect。

結論：

```text
OOF label beam 有診斷意義，但複雜度過高；
沒有超過 v16c，不適合作主方法。
```

---

## 20. 為什麼最終選擇 src_v16c

最終選擇 src_v16c 不是因為它唯一數字最高，而是综合判斷：

1. **test_mean 最高並列**  
   v16c 與 v16e 在 Chameleon 上 test_mean 持平，約 72.48%。

2. **比 v16e 更簡單**  
   v16e 使用 iterative label features，解釋鏈更長。

3. **比 v17 快很多**  
   v17 約 6347s，v16c 約 641s。

4. **不使用 pseudo-label / label beam correction**  
   v16c 保持純 graph-signal dictionary + closed-form classifier。

5. **直接對應論文機制**  
   v16c 的九塊字典正好可以變成論文中的 named blocks，並用於 node-level atlas。

6. **PCA 和 Ridge 雙分支都保留**  
   這使得後續可以分析：
   - class-subspace complexity；
   - PCA/Ridge complementarity；
   - decision phase map；
   - geometry-vs-boundary behavior。

7. **適合作為 general white-box scaffold**  
   不是每個數據集上的 specialist，但最適合做跨數據集診斷 probe。

因此最終選：

```text
src_v16c = final WG-SRC core
```

---

# Part V：正式論文實驗包與 WG-SRC 定型

---

## 21. src_v16c_paper_experiments：正式論文包

### 21.1 對應目錄

```text
src_v16c_paper_experiments/
```

結構：

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

### 21.2 算法凍結

這一階段中，`paperexp/core.py` 被視為最終核心。  
當時明確要求：

```text
不要修改 Fisher selection；
不要修改 PCA fitting；
不要修改 Ridge classifier；
不要修改 validation selection；
不要根據 test accuracy 調參；
不要改 split 生成邏輯；
不要動 paperexp/core.py 的算法核心。
```

只允許修復：

```text
CSV 字段重複；
路徑不存在；
encoding 問題；
dict multiple values；
日志/結果輸出目錄錯誤。
```

這說明此時方法已經凍結為 paper version。

---

### 21.3 正式實驗腳本

正式實驗包括：

| 腳本 | 作用 |
|---|---|
| `run_E01_baseline_table_aggregate.py` | baseline 表 |
| `run_E03_random_split_stability.py` | random split paired stability |
| `run_E05_ablation.py` | 機制消融 |
| `run_E07_param_scan.py` | 參數掃描 |
| `run_E08_interpretability.py` | atlas / interpretability |
| `run_E09_efficiency.py` | efficiency profile |
| `run_E11_sample_efficiency.py` | sample efficiency |
| `run_src_v16c_fill_main_table_10repeats.py` | 主表 10 repeats |

---

### 21.4 最終提交包

最終結果中出現：

```text
src_v16c_final_6datasets_repeat_rows.csv
src_v16c_final_6datasets_summary.csv
src_v16c_main_table_10repeat_rows.csv
srcv16c_submission_package.zip
srcv16c_submission_package_20260421_105707.tar.gz
```

這說明最終實驗包完成了從單一 Chameleon 版本到六數據集論文實驗系統的遷移。

---

## 22. 論文中 WG-SRC 的最終定位

最終論文中，WG-SRC 的定位不是：

```text
一個為了刷最高準確率的 GNN 替代品
```

而是：

```text
一個 accuracy-preserving white-box diagnostic scaffold
```

也就是它同時承擔两件事：

1. 作為 predictor，完成節點分類；
2. 作為 diagnostic probe，生成 node-level atlas 和 dataset fingerprint。

---

### 22.1 六數據集主實驗

論文中六個數據集：

```text
Amazon-Computers
Amazon-Photo
Chameleon
Cornell
Texas
Wisconsin
```

主實驗顯示：

```text
WG-SRC 平均比 strongest aligned baseline 高 +1.52 pp。
```

這個結果的意義不是宣稱 WG-SRC 是所有圖學習模型中最強，而是：

```text
它的白盒診斷結果來自一個真實有效的 predictor，
而不是來自弱 post-hoc probe。
```

---

### 22.2 機制消融與 generalist-vs-specialist 判斷

論文中的 ablation 顯示：

- Amazon 上 no-high-pass 可能更強；
- Cornell/Texas 上 ridge-only 可能更強；
- Wisconsin 上 raw-only 可能更強；
- Chameleon 需要 full / near-full 結構；
- full scaffold 平均 rank 最好、worst-case 最好、六個數據集全部 top-3。

這和迭代路徑一致：

```text
某些 specialist 可以在單數據集上更好；
但 full scaffold 才能保留完整診斷能力。
```

因此沒有選 ridge-only 或 raw-only 作為主方法。

---

### 22.3 Node-level signal atlas

最終 WG-SRC 能記錄每個節點：

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

這些記錄聚合成 dataset fingerprint：

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
- \(Q_D^{hard}\)：PCA 和 Ridge 都失敗的 hard fraction；
- \(\Delta H_D\)：wrong nodes 與 correct nodes 的 high-pass shift。

這正是 v8-v17 統計審計思想的最終形式化版本。

---

# Part VI：統計驅動迭代邏輯總表

---

## 23. 後期版本統計驅動邏輯

| 階段 | 統計對象 | 關鍵表 | 得出的判斷 | 下一步 |
|---|---|---|---|---|
| v8 | hard nodes / 參數影響 | `error_node_summary.csv`, `param_effects_summary.csv` | 不能只看 accuracy，要看錯誤節點結構 | 做 branch 審計 |
| baseline subspace | 類子空間幾何 | `pairwise_geometry_baseline_subspaces.csv` | class subspace 有用，但仍有複雜錯點 | adaptive branch |
| src_v9 | branch 結構 | `branch_member_rows.csv` | branch 太多且性能下降 | safe branch |
| src_v10 | fixed/broken/redirect | `extra_branch_rewrite_nodes_src_v10.csv` | branch 有淨收益但有 redirect 風險 | geometry-safe |
| src_v11/v12 | calibration 和 gated branch | `root_calibration_rewrite_nodes_*` | calibration 貢獻小，branch 仍是主效應 | per-class dimension |
| src_v13/v14 | 每類 PCA 維度 | `dim_search_trace_*` | 不同類需要不同維度，floor/fallback 更穩 | class-pair specialist |
| src_v15 | bias / pairwise specialist | `class_bias_trace`, `pairwise_specialist_trace` | validation overfitting，test 不跟隨 | 放棄 specialist |
| src_v16 | reliability / anti-hub | `reliability_map`, `penalty_search_trace` | correction 不如 fixed base | 改信號字典 |
| src_v16c | 9-block dictionary | `run_summary_src_v16c.json` | 信號質量提升帶來穩定突破 | paper core |
| v16d/e/f/v17 | 複雜變體 | 各 run summary / candidate trace | 更複雜不一定更好 | 保留 v16c |

---

## 24. 最終方法各組件的來源

| 最終 WG-SRC 組件 | 來源版本 | 迭代原因 |
|---|---|---|
| MCR² / 子空間解釋思想 | 初始 tracing 階段 | 證明需要看內部機制，不只看 accuracy |
| class-wise PCA subspace | src_v1 | 從 white-box GNN 轉向閉式幾何殘差 |
| raw / smooth 區分 | HASDC / src_v5 | 發現 raw 和 smooth 在不同圖上價值不同 |
| high-pass difference | src_v7 | Chameleon 證明 \(X-PX\) 等差分有用 |
| Fisher coordinate selection | src_v7 Algorithm 3 | 從 70% 推進到約 71.1% |
| Ridge boundary | src_v7 Algorithm 2 | PCA 幾何不足，需要判別邊界 |
| fixed/broken/redirect 審計 | src_v10 起 | 判斷機制是否真的修節點 |
| per-class dimension | src_v13/v14 | 類複雜度不同 |
| reliability map | src_v16 | 驗證 PCA/Ridge 條件可靠性，但 correction 不穩 |
| nine-block dictionary | src_v16c | 最終突破，信號質量優先於複雜修正 |
| node-level atlas | v8-v17 統計審計 + paper E08 | 把錯誤審計正式變成診斷輸出 |

---

## 25. 關鍵失敗教訓

### 25.1 有公式的神經層不等於真正白盒

G1 的問題說明：

```text
理論上有 H + ηgrad - ηλLH，
但實際 forward 中被 out_proj(Z_agg) 覆蓋。
```

因此最終不再依賴訓練式 message-passing layer。

---

### 25.2 low-pass 不足以處理 heterophily

Chameleon 說明：

```text
鄰居不同本身可能是信號；
X-PX 和 PX-P²X 不能被簡單丟棄。
```

這直接導致 high-pass blocks 進入最終 dictionary。

---

### 25.3 branch 能修節點，也能弄壞節點

從 src_v10 開始，每個 branch 都必須看：

```text
fixed
broken
wrong_redirect
```

這使研究從 accuracy-driven 變成 mechanism-audited。

---

### 25.4 validation 提升不一定是真提升

src_v15 的核心教訓：

```text
val: 約 69.79% → 73.87%
test: 約 69.04% → 69.23%
```

pairwise specialist 主要吃 validation，不能作為主方法。

---

### 25.5 correction mechanism 不如 signal dictionary

src_v16 說明：

```text
reliability + anti-hub 有解釋性，
但沒有超過 fixed baseline。
```

真正突破來自 src_v16c：

```text
把 graph-signal dictionary 做對。
```

---

### 25.6 特徵不是越多越好

v16d 說明：

```text
13-block features < 9-block v16c
```

因此最終九塊字典是簡潔性和表達力之間的平衡。

---

### 25.7 更複雜不一定更適合論文主方法

v16e / v17 說明：

```text
iterative labels / OOF beam 可以接近或持平，
但複雜、慢、解釋鏈更長。
```

最終主方法應該是：

```text
簡單、閉式、可審計、可解釋、可復現。
```

---

# Part VII：最終結論

---

## 26. 一句話總結

整個項目的真實迭代不是：

```text
跑很多版本，然後選準確率最高的。
```

而是：

```text
從 MCR² 幾何解釋出發；
先嘗試 white-box GNN 展開；
發現神經層路線不夠嚴格白盒；
轉向閉式子空間分類器；
在 Chameleon 上通過 high-pass / multihop 發現圖信號分解的重要性；
通過 v8-v17 的錯誤節點、fixed/broken/redirect、dimension、reliability、candidate trace 統計排除複雜但不穩定的修正機制；
最終收斂到 src_v16c：
九塊顯式圖信號字典 + Fisher 坐標選擇 + 類 PCA 子空間 + multi-alpha Ridge + 驗證融合。
```

---

## 27. 最適合放入倉庫 research notes 的最終敘事

可以將本項目描述為：

```text
This project began as an attempt to inspect whether neural representations obey low-rank class-subspace geometry suggested by MCR², ReduNet, and CRATE. Early white-box GNN unfolding routes showed that merely adding interpretable formulas to neural layers was insufficient, especially after code audit revealed that the theoretically meaningful update path could be bypassed by trainable projections.

The project then shifted toward closed-form white-box graph learning. Through Tikhonov smoothing, class PCA subspaces, response/residual scoring, raw-vs-smooth channel experiments, and Chameleon heterophily studies, the development identified explicit high-pass graph differences and multihop graph signals as essential. Subsequent v8-v17 iterations were driven not only by accuracy but by statistical audits of error nodes, branch fixed/broken/redirect behavior, per-class dimension traces, class-pair specialists, reliability maps, anti-hub penalties, and OOF candidate rewrites.

These audits showed that complex corrective mechanisms could improve validation or fix some nodes, but often introduced broken nodes, wrong redirects, overfitting, or excessive runtime. The final breakthrough was therefore not a more complex correction layer, but a cleaner signal dictionary: the src_v16c nine-block explicit graph-signal dictionary combined with Fisher coordinate selection, class-wise PCA subspace residuals, multi-alpha closed-form Ridge classification, and validation-selected fusion. This became the final WG-SRC scaffold and later supported the node-level signal atlas and dataset fingerprint analyses in the paper.
```

---

## 28. 最終版本定位

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

## 29. 建議保留的研究記錄目錄

為了證明迭代是統計驅動的，建議保留以下統計目錄：

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

最關鍵的證據文件包括：

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

這些文件共同證明：

```text
後期迭代不是簡單 accuracy search，
而是 error-audit-driven、mechanism-audit-driven、statistical-summary-driven 的科研過程。
```

---

## 30. 最終可放在 README / research notes 首頁的短版摘要

```text
WG-SRC was developed through a multi-stage mechanism-audited iteration process. The project began with MCR²-based representation geometry tracing and early white-box GNN unfolding attempts. These routes revealed that interpretable formulas inside trainable GNN layers were not sufficient to guarantee a truly white-box mechanism. The development then shifted to closed-form graph subspace classifiers.

On heterophilic Chameleon, explicit high-pass graph differences and multihop signals proved essential. Later versions were not selected by accuracy alone: the development used node-level error audits, branch fixed/broken/redirect statistics, per-class dimension traces, class-pair specialist traces, reliability maps, anti-hub penalty searches, and OOF label-beam candidate traces. These statistics showed that complex correction mechanisms often improved validation but did not transfer reliably to test or introduced broken/wrong-redirect nodes.

The final selected method, src_v16c, uses a nine-block explicit graph-signal dictionary, Fisher coordinate selection, class-wise PCA subspace residuals, multi-alpha closed-form Ridge classification, and validation-selected score fusion. It was chosen because it achieved the best or tied-best Chameleon performance while remaining simpler, faster, more interpretable, and better aligned with the final node-level atlas and dataset-fingerprint analysis. This became the official WG-SRC scaffold in the paper.
```
