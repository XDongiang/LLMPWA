# kk_dis 画图（draw / plot）约束规范

> 本文档是 `generate_draw_weight` / `generate_draw_plot` 两个生成 stage 必须产出的
> 绘图代码的**唯一约束来源**。生成 agent 应严格按本文档产出 `run/draw_weight_script.py`
> 与 `run/draw_plot_script.py`；产出图应与 `tmp/picture_example/` 下的标准图一致
> （内容一致优先，文件命名/布局可在下述规范内微调）。
>
> 参考标准图：`tmp/picture_example/*_weight.png`（全分波叠加）与
> `tmp/picture_example/*_*_kk_weight.png`（单分波标注参数/拟合占比）。

---

## 0. 目标：两张图 / 每个物理量

对每个可绘制运动学变量（sbc）产出**两类** PNG：

| 类型 | 文件名模式 | 内容 |
|---|---|---|
| **A. 全分波叠加** | `<var>_weight.png` | Data + Fit total + 所有分波曲线在同一画布叠加，右上角图例列出各分波名 |
| **B. 单分波** | `<var><resonance>_kk_weight.png` | Data + Fit total + 该分波一条曲线；右上角标注该共振的**拟合参数结果**与 **fit fraction** |

其中 `<var>` 为物理量名（`phi_kk` / `f_kk` / `b123_kk` / `b124_kk`），
`<resonance>` 为共振标签（`phif0_980` / `phif0_1710` / `phif2_1270` / `phif2_1525` /
`phif2_2150` / `phif2_2340`）。

---

## 1. 分波集合与数据来源

### 1.1 分波（partial wave）集合 —— 从配置自动推导，不写死

标准图展示 **6 条**分波曲线，对应 `gen/fragments/classification.json` 的
`amplitude_classification`（resonance → mode）映射。生成 agent 应**从当前配置动态推导**
共振集合（见 `resonances_config.toml` 的 `[resonances.*]`），而不是硬编码 6 或 7 个名字：

```
phi_kk + f_kk + (b123_kk, b124_kk 由 [draw].extra_sbc 提供)
  └─ 每条曲线对应一个共振态：
      phif0_980   (phif0_kk_BW_flatte980)
      phif0_1710  (phif0_kk_BW_BW basis 0)
      phif2_1270  (phif2_kk_BW_BW basis 0)
      phif2_1525  (phif2_kk_BW_BW basis 1)
      phif2_2150  (phif2_kk_BW_BW basis 2)
      phif2_2340  (phif2_kk_BW_BW basis 3)
```

> 注：`phif0_2470` 在 `resonances_config.toml` 中存在，但标准图未单独画曲线；
> 当 `phif0_kk_BW_BW` 同时含 `phif0_1710` 与 `phif0_2470` 两个 basis 时，
> 是否额外画 `phif0_2470` 由生成代码按配置自动决定。**默认按示例**：不单独画
> `phif0_2470`，只把 `phif0_kk_BW_BW` 整体（basis 0）作为 `phif0_1710` 分波展示。

### 1.2 数据来源（读 `data/`，只读）

绘图脚本**不调用** `base_functions.load_data()` / `normalize_data()`（它们把 MC 重复
`n_repeat=3` 次、且依赖 jax/scipy，绘图脚本应保持纯 numpy + ROOT）。而是直接读：
- 真实数据：`data/real_data/<var>.npy`
- MC 评价样本：`data/mc_truth/<var>.npy`

每个物理量在画图前对其**数值**做 `np.sqrt(...)`（存储的是 s 值，X 轴用质量/动能
`sqrt(s)`，见 §3 轴标题）。`data_size` = `data/real_data/<var>.npy` 的长度。

权重文件（由 `draw_weight_script` 产出）：
- `output/draw/weight.npz`（mode="pass"，作用于 MC 全样本，长度 = MC 事件数）
- `output/draw/weight_truth.npz`（mode="truth"，作用于 truth-MC 150000 事件）

### 1.3 权重键

`weight.npz` 中包含：
- `all_mods_wt`：总强度权重（每事件）
- `"<mode>_<basis>"`：每个分波权重，如 `phif0_kk_BW_flatte980_0`、
  `phif0_kk_BW_BW_0`、`phif2_kk_BW_BW_0` ... `phif2_kk_BW_BW_3`
- `fit_value`、`sum_wt`

绘图脚本通过正则 `^(.*)_(\d+)$` 从 npz 键名动态发现存在的分波键，映射到
`MODE_RESONANCES`（mode → 共振标签列表），缺失的键/越界的 basis 优雅跳过。

---

## 2. 直方图构造（ROOT TH1D）

对每个物理量 `var`：
1. `hist_data`：用 `data_<var>`（真实数据）`Fill`，**不加权**。
2. `hist_fit`：用 `mc_<var>` 按 `all_mods_wt` 加权 `Fill`，再 `Scale(data_size / sum_wt)`
   归一化到真实数据。
3. 每条分波 `hist_comp`：用 `mc_<var>` 按对应 `"<mode>_<basis>"` 权重 `Fill`，
   再 `Scale(data_size / sum_wt)`。

- bin 数建议 60（示例用 100，可调）；X 轴范围取 data 与 mc 数值的并集 min/max，
  若上下界相等则上下各扩 1。
- **尖峰（近 delta）变量的 X 轴范围必须放宽**：当 data 与 mc 并集的相对跨度
  `(max-min)/max(|min|,|max|,1)` 很小（如 `M_φ` 所有事件都落在 ~1.019 GeV，
  跨度 ~1e-15）时，若直接用 min/max 作范围，X 轴会被压缩成一条线，ROOT 自动生成的
  坐标刻度（带很多小数）会**叠在一起看不清**。此时应把范围放宽为以峰为中心的一个
  物理窗口：`center ± 0.15*max(|center|,1)`；例如 `M_φ` 中心 ≈1.019 →
  范围 ≈0.87–1.17 GeV（与标准图一致），X 轴刻度（0.9/0.95/1.0/…）即可清晰显示。
  对其他正常跨度的变量（`M_KK`、`M_φK`），该规则不触发，保持原 min/max 范围。
- 归一化基准 `sum_wt = sum(all_mods_wt)`（用 pass 权重文件）。
- 每个 histogram 设置 `SetDirectory(0)` 避免 ROOT 垃圾回收清空。

---

## 3. 布局 / 样式（尽量与标准图一致）

| 项 | 值 |
|---|---|
| 画布 | `TCanvas(name, title, 900, 600)`，`gStyle.SetOptStat(0)` |
| Data 点 | 黑色实心点，`SetMarkerStyle(20)`，`SetMarkerSize(0.65)`，`Draw("E1")`（含误差棒） |
| Fit total | 红色实线 `kRed+1`，`SetLineWidth(3)`，`Draw("HIST SAME")` |
| 每条分波 | 依次取不同颜色（`kBlue+1`, `kGreen+2`, `kMagenta+1`, `kOrange+7`, `kCyan+1`, `kViolet+1`, `kAzure+2`, `kRed-4`），`SetLineWidth(2)`，线型 `1+(i//ncolors)%3`，`Draw("HIST SAME")` |
| X 轴标题 | 按物理量：`phi_kk`→`M_φ (GeV)`；`f_kk`→`M_KK (GeV)`；`b123_kk`/`b124_kk`→`M_φK (GeV)` |
| Y 轴标题 | `Events/<binsize> GeV`（binsize 由 X 轴范围/bin 数推导，如 `0.02 GeV`、`0.00 GeV`） |
| 图例 | `TLegend`，含 Data / Fit total / 各分波（分波图例可附 fit fraction） |
| 文件名 | `output/pictures/<type>/<name>.png`（建议 `output/pictures/`，子目录可选） |

X 轴标题符号（Hepep 风格 `M_φ`, `M_KK`, `M_φK`）需写成 ROOT 可渲染的
`M_{#phi}` / `M_{KK}` / `M_{#phi K}`，或用 `TLatex`。

---

## 4. 单分波图的右上角标注（对应示例）

单分波图（类型 B）右上角用 `TLatex`（`SetNDC(True)`，`SetTextSize(0.028)`）标注该共振的
**拟合结果**与 **fit fraction**：

```
kk_<param>_result:<value>          # 每行一个自由参数（如 kk_f980_m / kk_g_kk / kk_rg）
...
fit fraction : <value>
```

- **参数行只列该共振的传播子形状参数**（质量 `B_propagator.mass`、宽度 `B_propagator.width`、
  `g_kk`/`rg`），**不列 constN/thetaN**（示例只标共振态本身参数，不标振幅耦合）。
- **参数名做共振态特异处理**：把 `B_propagator.mass`→`kk_f<共振数字>_m`、
  `B_propagator.width`→`kk_f<共振数字>_w`、`B_propagator.g_kk`→`kk_g_kk`、
  `B_propagator.rg`→`kk_rg`。例如：
  - `phif0_980` → `kk_f980_m` / `kk_g_kk` / `kk_rg`
  - `phif0_1710` → `kk_f1710_m` / `kk_f1710_w`
  - `phif2_1270` → `kk_f1270_m` / `kk_f1270_w`
  - `phif2_1525` → `kk_f1525_m` / `kk_f1525_w`
  - `phif2_2150` → `kk_f2150_m` / `kk_f2150_w`
  - `phif2_2340` → `kk_f2340_m` / `kk_f2340_w`
  - （`phif0_2470` 若单独画，则 → `kk_f2470_m` / `kk_f2470_w`）
- **布局避免与图例重叠**：类型 B 一般不画图例（或把图例放右下、标注放右上）；若保留
  图例则需让 TLatex 起始 y 高（约 0.90）且每行递减 0.035，避免压到图例/曲线。
- `fit fraction` 用 truth 权重计算：
  `frac = sum(truth_wt["<mode>_<basis>"]) / sum(truth_wt["all_mods_wt"])`。
- 全分波叠加图（类型 A）的图例中，每个分波条目附 `(%.3f)` 的 fit fraction。

---

## 5. ROOT 运行环境（本地 vs 服务器）

绘图脚本用 **pyROOT**（`import ROOT`）。生成 agent 的**冒烟/试跑**需在有 pyROOT 的环境执行：

| 环境 | 位置 | 用法 |
|---|---|---|
| **本地** `analyses/kk_dis/.pyroot_envs/rootplot` | **已装** pyROOT 6.40.04 + numpy（见 `document/draw_plot_environment.md`） | `.pyroot_envs/rootplot/bin/python run/draw_plot_script.py`，**首选**试跑环境 |
| **本地** `analyses/kk_dis/.venv` | 通常**无** pyROOT | 仅做 `py_compile` 语法检查 |
| **远程** HEP1 conda `rootenv` | `~/miniconda3/envs/rootenv/bin/python`（pyROOT 6.40.04 + numpy 2.4.6） | 生成完成后在该环境跑 `run/draw_plot_script.py` 产出真实 PNG |

> 生成 agent **必须优先探测** `analyses/kk_dis/.pyroot_envs/rootplot/bin/python`
> （本机已安装，无需联网/写权限），其次远程 `rootenv`，最后 `.venv`。
> 只要任一 pyROOT 可用，就应**真实跑图验证**（而非仅语法检查）。
> 详见 `document/draw_plot_environment.md`。

---

## 6. 交付物自查清单

- [ ] `run/draw_weight_script.py` 可 import、可从 `run/free_params.toml` 推导 `extract_parameters`
- [ ] `run/draw_weight_script.py` 产出 `output/draw/weight.npz` 与 `weight_truth.npz`
- [ ] `run/draw_plot_script.py` 纯 numpy + ROOT，不依赖 jax/scipy/base_functions
- [ ] 每物理量产出 全分波叠加 + 每分波 单分波图，文件名与 §0 一致
- [ ] Data / fit / 分波直方图权重与归一化正确（`Scale(data_size/sum_wt)`）
- [ ] 单分波图右上角含参数结果与 fit fraction
- [ ] 在远程 `rootenv` 跑出真实 PNG，与 `tmp/picture_example/` 内容一致
- [ ] 不修改 `data/`、不修改 `resonances_config.toml`
