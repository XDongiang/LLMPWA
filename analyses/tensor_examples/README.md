# tensor_examples —— 文档公式例子的回归分析

本目录把 `analyses/kk_dis/document/ψ衰变到介子的分波分析协变张量形式.md`
里的衰变例子**变成可跑的回归测试**，用来验收框架级张量生成器
（[`agent/tensor/`](../../agent/tensor/)）。

它不是生产分析：没有拟合、没有 `llm_config_*.toml`、没有 `data/` 真实样本。
目的只有一个 —— 给「框架新加一条道」这件事一个**独立于 kk_dis** 的验收面。

目前覆盖：

| 例子 | 末态 | 道数 | 备注 |
|---|---|---|---|
| **例 3** `ψ→φπ⁺π⁻` | 4 | 10（ρ′ 2 + b₁ 8） | 深度 3，三表象齐全 |
| **例 5** `ψ→K⁺π⁻K⁻π⁺` | 4 | 10（K*₀ 2 + K*₂ 8） | 深度 3 |
| **例 1** `ψ→π⁺π⁻π⁰` | **3** | 6（ρ 3 + ρ₃ 3） | 真三体，只有 cascade 能 emit |
| **例 2** `ψ→K⁺K⁻π⁰` | **3** | 3（KK̄ 1 + Kπ 2） | 真三体，同上 |

合计 **29 条道**。例 1/2 是 2026-10-20 新增的（置换求和模型）。

## 现状

| 例子 | 文档行 | 状态 | 道 |
|---|---|---|---|
| 例 3 `ψ→φπ⁺π⁻` 的 ρ′ 道 | 374–387 | ✅ 已实现 | `u_rho_1` `u_rho_2`（Eq.46） |
| 例 3 `ψ→φπ⁺π⁻` 的 b₁ 道 | 391–416 | ✅ 已实现 | `u_{SS,SD,DS,DD}_{1,2}`（Eq.47–50） |
| 例 5 `ψ→K⁺π⁻K⁻π⁺` 的 K*₀/K*₂ 道 | 431–623 | ✅ 已实现（2026-10-19） | 10 条（`u_kst0_*` + `u_PP/PF/FP/FF_*`） |
| 例 1 `ψ→π⁺π⁻π⁰` 的 ρ 家族 | 200–273 | ✅ 已实现（2026-10-20） | `u_rho_{12,13,23}`（三体，3 条置换道） |
| 例 1 `ψ→π⁺π⁻π⁰` 的 ρ₃(3⁻) 家族 | 236–247 | ✅ 已实现（2026-10-20） | `u_rho3_{12,13,23}`（符号 (+,-,-)） |
| 例 1 `ψ→π⁺π⁻π⁰` 的 ρ₅(5⁻) | 249–263 | ❌ 超内核上限 | 需 J=5 与 l=5，`LAM_SLOTS`/`l_max` 到 4 |
| 例 2 `ψ→K⁺K⁻π⁰` 的 KK̄ 共振 | 274–304 | ✅ 已实现（2026-10-20） | `u_rho_K_12`（单项） |
| 例 2 `ψ→K⁺K⁻π⁰` 的 K\* 共振 | 274–304 | ✅ 已实现（2026-10-20） | `u_kstar_{13,23}`（两项同号） |
| 例 2 的 K\*₃ / K\*₅ | 274–304 | ⬜ 未做 | K\*₃ 可做；K\*₅ 超上限 |
| 例 7–11 辐射衰变 | 845–1383 | ⬜ 需新基建 | 秩 2 振幅 + 横向投影 |
| 例 4/6/12 | — | ❌ 不做 | 三体 / 深度 3 的 omega / 文档为空 |

可行性评估（含实测数据）见
[`documentation/tensor_examples_feasibility.md`](../../documentation/tensor_examples_feasibility.md)。

### 例 1/2 的「置换求和」：已实现（2026-10-20）

结论：**不需要新基建**。「一个衰变写成 N 项置换和」用现有的
「一份 spec 多条道 + 逐道 `g_i`」就能表达，`f^{(R)}`（BW 传播子）放在道外
所以各项自然共用 —— 与文档一致。

四份 spec + 一张验收表：

| spec | 例 | 族 | 道数 | 置换结构 |
|---|---|---|---|---|
| [`ex1_pipipi0_rho.toml`](specs/ex1_pipipi0_rho.toml) | 1 | ρ(1⁻) | 3 | `{12,13,23}` 全排列，三项**同号** |
| [`ex1_pipipi0_rho3.toml`](specs/ex1_pipipi0_rho3.toml) | 1 | ρ₃(3⁻) | 3 | 同上，但三项**(+,−,−)** |
| [`ex2_kkpi0_rho.toml`](specs/ex2_kkpi0_rho.toml) | 2 | KK̄ 共振 | 1 | **单项**（换 1↔3 就成 Kπ，不是同一个态） |
| [`ex2_kkpi0_kstar.toml`](specs/ex2_kkpi0_kstar.toml) | 2 | Kπ 共振 | 2 | `{13,23}` 两项**同号**（K⁺K⁻ 同种粒子，1↔2 无新配对） |

实测（三体相空间 N=64，`helicity_cascade`）：

| 检验 | 结果 |
|---|---|
| 置换协变性（重标号动量后等于另一条道，\|c\| = 1） | 残差 ≤ 5.4e-15，四份 spec 全过 |
| J=1 的逐道 `s·c·ε·B1(top)B1(pair)` | 残差 ≤ 4.7e-15，\|c\| = 0.191835806 各道一致 |
| 三条道相加 vs 文档括号 | 残差 5.2e-15，c 为全局常数 |
| `check` 三体拓扑 | ✅ 4/4 |
| `emit`（cascade）产物 vs 直接调内核 | **逐位相同**（0.0） |

验收脚本：[`agent/tensor/tests/test_ex12_permutation.py`](../../agent/tensor/tests/test_ex12_permutation.py)（7 项）
与探针 [`probes/verify_ex12_permutation.py`](probes/verify_ex12_permutation.py)。

四个必须知道的坑：

1. **树的子节点顺序会翻号**。`rho -> (Pip1, Pi0)` 与 `rho -> (Pi0, Pip1)`
   相差一个整体负号（实测 −0.191836 vs +0.191836）。这是 ε 张量指标配对
   顺序导致的，**不是物理差异**；各道顺序不统一时相加会得到
   `B12 − B13 + B23` 而不是文档的 `B12 + B13 + B23`。
   隐蔽之处：每条道单独看都自洽（都能精确写成 `s·c·ε·B`，残差 1e-15），
   **只有做置换检验或求和才暴露** —— 所以判据用置换协变性，不用"每道能跑通"。
2. **`ε·B` 恒等式只对 J=1 成立**。J=3 时文档的 `T̃⁽³⁾` 是秩 2 张量，
   `ε·p_ψ` 缩并后**不能**化成单个 `ε_{μνλσ}p₁p₂p₃`（实测残差 1.04，
   是结构性的）。ρ₃ 只验置换协变性。
3. **`cartesian`/`helicity` 出不来**：`emit` 需要
   `agent/tensor/chains/<chain>/<channel>.py` 里的**手写**道函数
   （`_emit_covariant` 按 `def <name>` 取函数体内联）。三体置换道没有手写
   函数，只有 `helicity_cascade` 能 emit（它按逐节点 `(l,s)` 描述道）。
   这不是缺陷 —— 级联本来就是三体该用的内核。

4. 🐛 **omega 对置换族是静默错的**（2026-10-20 发现，真缺陷）。
   `omega_from_momenta(mom, chain)` 只吃 **base chain**，返回**一份逐道共用**
   的 `Omega`；三条置换道的树不同却共用它，只有与 base 树一致的那条道对得上。
   实测 `ex1_pipipi0_rho`：`u_rho_12` 残差 2.4e-16（对），
   `u_rho_13`/`u_rho_23` 残差 0.30/0.27（**静默错**：emit 不报错、跑得出数）。
   ⚠️ 判据要取对：❌「顶层两子节点都衰变」会误判 `ex2_kkpi0_rho`
   （三体但**单道**，omega 实测残差 3.3e-16，**正确**）；❌「深度 ≤ 2」
   拦不住三体。✅ 正确判据是「**是否存在道，其 tree 与 base 树不同**」，
   `spec.py` 的 `validate` 已按此拦截。
   **结论：置换族一律用 `helicity_cascade`；单道三体用 omega 可以。**

超出内核上限、**没做**的两项：例 1 的 ρ₅(5⁻)、例 2 的 K\*₅ —— 需要
J=5 与 l=5，而 `LAM_SLOTS` 与 `l_max` 都只到 4。K\*₃(3⁻) 技术上可做，未做。

## 目录

```text
analyses/tensor_examples/
├── README.md               ← 本文件
├── chains/                 ← **每条链一份文档**（做了什么 / 还有什么问题）
│   ├── README.md           ← 链文档索引 + 三条边界
│   ├── gen_chain_docs.py   ← 从 spec 自动导出"事实"部分
│   ├── notes/              ← 手工"叙述"部分（每链一份 .notes.md）
│   ├── phif0.md            ← ψ→φf₀   （深度 2，三表象）  ← 生成物
│   ├── phif2.md            ← ψ→φf₂   （深度 2，三表象）  ← 生成物
│   ├── b1pi.md             ← ψ→b₁π   （深度 3，验收主力）← 生成物
│   ├── rhopi.md            ← ψ→ρ′π   （深度 3，边界）    ← 生成物
│   ├── rhof2_generic.md    ← ψ→ρf₂   （假想链）          ← 生成物
│   ├── ex5_kst2.md         ← ψ→K*₂π  （例 5，4/8 是级联）← 生成物
│   ├── ex5_kst0.md         ← ψ→K*₀π  （例 5，2/2 是级联）← 生成物
│   ├── ex1_rho.md          ← ψ→ρπ⁰   （例 1，三体，3 条置换道）← 生成物
│   ├── ex1_rho3.md         ← ψ→ρ₃π⁰  （例 1，J=3，符号 (+,-,-)）← 生成物
│   ├── ex2_rho.md          ← ψ→ρ′π⁰  （例 2，KK̄ 共振，1 条）  ← 生成物
│   └── ex2_kstar.md        ← ψ→K*K   （例 2，Kπ 共振，2 条）  ← 生成物
├── specs/
│   ├── ex3_rhopi.toml      ← 例 3 的 ρ′ 树（2 条道）
│   ├── ex3_b1pi.toml       ← 例 3 的 b₁ 树（8 条道）
│   ├── ex5_kst2.toml       ← 例 5 的 K*₂ 树（8 条道）
│   └── ex5_kst0.toml       ← 例 5 的 K*₀ 树（2 条道）
├── data/
│   └── make_momenta.py     ← 合成 ψ→φπ⁺π⁻→(K⁺K⁻)(π⁺π⁻) 四动量（自带守恒自检）
├── check_ex3.py            ← 端到端回归（**仅例 3**）：L1/L2/L3 + A1/A3/A7
├── probes/                 ← 诊断脚本（复现 roadmap §7 与边界实测）
│   ├── verify_ex5_port.py  ← 例 5 的 10 条道 vs jax 逐式对拍（10/10，1e-15）
│   ├── probe_ex5_identify.py ← 例 5 级联可表示性穷举 + 阳性对照（b₁ 8/8）
│   ├── probe_a1_sensitivity.py ← **判据 A1 自身的有效性**（错误注入）
│   └── …                   ← ρ′/b₁ 的边界探针、行序探针等
└── generated/  out/        ← 生成物与运行产物（.gitignore，可随时重建）
```

**每条链的详细文档在 [`chains/`](chains/README.md)** —— 覆盖十一条链
（`phif0` / `phif2` / `b1pi` / `rhopi` / `rhof2_generic` / `ex5_kst2` /
`ex5_kst0`）各自的拓扑、分波、表象可用性、验收数据与遗留问题。

这些文档是**生成物**：拓扑/道表/计数律/表象可用性从 TOML 自动导出，
"做了什么、还有什么问题"手写在 `chains/notes/`。改了 spec 要重跑：

```bash
python analyses/tensor_examples/chains/gen_chain_docs.py           # 重生成
python analyses/tensor_examples/chains/gen_chain_docs.py --check   # 只检查
python agent/tensor/tests/test_chain_docs.py                       # 已挂进测试
```

**为什么是四份 spec 而不是两份**：两个**不同**的理由，别混。

| spec 对 | 为什么要拆 |
|---|---|
| `ex3_rhopi` / `ex3_b1pi` | **树不同**（中间态分别是 `phi+π⁻` 与 `phi+π⁺`），一棵树只能描述一组配对 |
| `ex5_kst2` / `ex5_kst0` | **自旋相同不了**（K*₂ 是 $J=2$、K*₀ 是 $J=0$），而一个 `[particles.*]` 条目只能有一个 `J` |

前者是拓扑冲突，后者是量子数冲突。分拆是最简明的解法 —— 这也是用户指定的
结构（`chain.name/chain.channel.py` 存储、生成时组装）。

⚠️ 每份 spec 内部仍然有"两棵树塞在一起"的问题：`b1pi` 与 `ex5_kst2` 的
`_1`/`_2` 是**旁观者 π 对调**，靠逐道 `tree` 覆盖共存。**道函数体本身不读树**
（`agent/tensor/chains/{b1pi,kstpi}/u_*.py` 里没有 `tree`/`nodes`，各道函数只吃
`(psi, phi, f2, Kp, Km, Pip, Pim)` 七个四动量），所以能工作；
但真实拟合里"两棵树是否共享传播子/参数"仍未处理。

---

# 如何验证：三层验证 + 三条判据

这是本目录的**核心方法**，值得单独讲清楚。生成器出错的地方有很多种，
如果只用一个"总对拍"来判，失败了不知道错在哪。所以判据按**能回答什么问题**
分层 —— 每层只在一种可能性上失败，失败即可定位。

## 三层验证：管"生成流程本身没写错"

| 层 | 比什么 | 阈值 | 抓什么 bug |
|---|---|---|---|
| **L1** 生成物 vs 框架内核 | `emit` 出的产物脚本算出的 cache vs 直接调 `kernels/covariant.py` | rel < 1e-12 | emitter 组装：拼错道函数体、漏 import、内联顺序错、动错动量分量 |
| **L2** cartesian vs helicity | $W = \mathrm{EPS\_A}\cdot T$ | rel < 1e-12 | 表象变换：store 体写错、旋转约定、`EPS_A` 行序 |
| **L3** 有限性 | `max|T|` 有限且 > 0 | 无 NaN/Inf、非恒零 | `Qabc<0`、除零、未定义分支、**整条道被静默算成 0** |

三层是**递进**的，不能跳：

* L1 过而 L2 失败 ⇒ 产物组装没问题，错在表象变换；
* L1 失败 ⇒ 不必看 L2，先修 emitter；
* L3 恒零是个**特别容易漏**的情形 —— rel 类判据在两边都恒零时会给出
  `0/0` 或假 OK，所以它必须单独查 `max|T| > 0`。

## 三条判据：管"新能力真的推开了吗"

这三条针对的是 `helicity_cascade` 表象（见下一节），答案是"是/否"而不是
"数值够小"。

| 判据 | 问的问题 | 阈值 | 在哪 |
|---|---|---|---|
| **A1** | 级联 == 协变，逐道逐事例？ | rel < 1e-12 | `check_ex3.py`、`test_helicity_cascade.py` |
| **A3** | 8 条道真的两两不同，还是退化了？ | min pairwise rel > 1e-6 | 同上 |
| **A7** | cache 形状/dtype 与 `helicity` **精确**一致？ | 精确相等 | `check_ex3.py` |

* **A1 是最强的一条**：`kernels/covariant.py` 是**手写公式**，不经过 CG 系数、
  不经过欧拉角、不经过逐节点缩并 —— 与级联路线的实现**完全不共享代码**。
  所以它能同时抓住 CG 约定错、螺旋度槽错、缩并顺序错、参考系错。
* **A3 防的是"能力假象"**：omega 时代 b₁ 的 8 条道只映射到 2 组 $(l,s)$，
  28 对里有 **12 对逐位重合**。级联让 8 条全部线性独立，A3 就是钉这个的。
* **A7 防的是"能力推开但拟合侧崩了"**：`helicity_cascade` 的卖点之一是
  拟合侧**零改动**，所以这条必须是**精确**相等，不接受 `1e-12`。

## A1 的独立性是有限的（必须知道）

A1 是必要条件，**不是充分条件**。两条路线**共享**的东西，A1 抓不到：

* 同一份四动量、同一个 `EPS_A` 行序、同一个位垒 $B_l$ 与 $Q_0$；
* 同一套物理定义（"哪条道算 SS"、"$b_1$ 的 $J^P$"）；
* 同一个 $(l,s)\leftrightarrow$ 道名的对应关系（标错了两边一起错）。

**所以 A1 本身也需要被验证。** 做法是**错误注入**
（[`probes/probe_a1_sensitivity.py`](probes/probe_a1_sensitivity.py)）：
往级联侧故意注入已知错误，看 A1 是否报警。

| 注入的错误 | A1 的 rel | 抓住了 |
|---|---|---|
| 关掉 Chung Eq.52 协变项 | 2.012e-01 | ✅ |
| 用旧参考系（`chain_angles`） | 1.722e-01 | ✅ |
| 顶层 $(L,S)$ 写错 | 9.784e-01 | ✅ |
| 内层 $(l,s)$ 写错 | 1.039e+00 | ✅ |
| 把子粒子螺旋度压成 0 | 1.010e+00 | ✅ |

（探针共 7 组注入；另 2 组是"宇称禁止"的退化组合，振幅恒零、rel 为 `nan`
—— 那不构成对 A1 的检验，所以不进表。）

五类错全部报警 ⇒ A1 在这个灵敏度上是有效的。这是"判据的判据"。

## 单看一个 rel 是不够的

A1 的 `rel` 是**逐事例**比值。若要判断"是不是同一个振幅"，一个数字不够 ——
ρ′ 就是靠三个**互相独立**的量定的（见下面"已知边界"）：

| 量 | 它排除了什么 |
|---|---|
| 自由复系数最小二乘残差 | 排除"差一个常数因子"（正常情形会被 rel 掩盖） |
| 逐事例标量 $c(e)$ 的离散度 | 排除"差一个**事件依赖**的因子"（常数因子 vs 事件依赖因子的分水岭） |
| 两个子空间的主角 | 排除"维数不同"这种平凡解释 —— 同维不同向才是真的不同 |

## 完整的判据清单（roadmap 的 A1–A7）

| 判据 | 内容 | 阈值 | 状态 |
|---|---|---|---|
| **A1** | 新内核 vs `covariant.py` 逐道对拍（经 `EPS_A`） | rel < 1e-12 | ✅ b₁/φf₀/φf₂/例 5 的 6 条；❌ ρ′、例 5 的 `u_PP_*`/`u_FP_*`（**这是发现，不是 bug**） |
| **A2** | 逐节点组合的秩 $=\prod_n N_n$ | 整数精确 | ✅ `test_helicity_cascade.py` |
| **A3** | 8 条道两两不重合 | 全部 rel > 1e-6 | ✅ |
| **A4** | 深度 3 链对**最内层**角分布敏感 | 转动后 rel > 0.1 | ✅ `test_helicity_cascade.py` |
| **A5** | 与 jax 原版 `FUNC.py` 对拍 | rel < 1e-12 | ✅ 例 3（`test_covariant_vs_jax.py`，24 个对象）+ 例 5（`verify_ex5_port.py`，10/10） |
| **A5′** | 与**上游 tf-pwa** 对拍 | rel < 1e-12 | ✅ 例 3 b1pi 8/8（`test_helicity_vs_tfpwa.py`，三层判据，2026-10-20 实测） |
| **A6** | 现有 11 份 spec 全部 `check` 通过 | 无异常 | ✅ 11/11（2026-10-20 实测，含例 1/2 的四份）；⚠️ "旧/新 cache 逐位回归"是**另一件事**，见下 |
| **A7** | cache 形状与拟合侧 `_load_cached_tensor` 兼容 | 形状/dtype 精确 | ✅ |

⚠️ A6 的两个口径别混：**`check` 通过**（11/11，已实测）≠ **旧/新 cache 逐位回归**
（只实测了 3 份深度 2 的生产 spec 是 no-op，没有做全部 11 份的逐位对照）。
`probes/probe_a4_a6.py` 里的 A6 glob 的是 `agent/tensor/specs/*.toml` +
`analyses/*/specs/*.toml`（实测命中 11 份，与上面一致）。

⚠️ A5 有两处必须注意的坑：

1. **jax 默认 float32**。必须在**导入 `FUNC` 之前**
   `jax.config.update("jax_enable_x64", True)`，否则 $\varepsilon\approx1.2\times10^{-7}$，
   对拍只能到 1e-7，看起来像"移植错了"。
2. 参考实现的**公开名**与内部名不同（`_u_PP_l` → `u_PP_1`），见 `FUNC.py::Modifiers`。

### A5′：与上游 tf-pwa 对拍

A5 比的是**项目自己的 jax 参考实现**，A5′ 比的是**社区标准实现 tf-pwa**
（`tf_pwa.amp.core.HelicityDecay` + `DecayGroup.get_amp3`）。两者都过，
说明我们的螺度约定既忠实于原推导、也不是自成一派的方言。

测试在 `agent/tensor/tests/test_helicity_vs_tfpwa.py`，用**冻结的 fixture**
（`tfpwa_b1pi_golden.json`），所以本地没有 TF / tf-pwa 也能跑。分三层：

| 层 | 比什么 | 实测 |
|---|---|---|
| 1 | CG 矩阵 `cg_matrix` vs `get_cg_matrix()` | 逐元素 0（精确相等） |
| 2 | 逐节点 `(alpha, beta)` vs `cal_angle_from_momentum` | ≤ 5e-15 |
| 3 | 链振幅 `chain_amplitude` vs `get_amp3` | 8/8 道 ≤ 3e-15 |

⚠️ **第 3 层必须把位垒因子除掉**，这是 A5′ 最容易踩的坑：tf-pwa 的
Blatt-Weisskopf 用 $d=3\ \mathrm{GeV}^{-1}$

$$B'_l(q^2)=\sqrt{P_l(q_0^2 d^2)/P_l(q^2 d^2)}$$

而本仓库（和 `Tensor/BASE.py`）用 $Q_0=0.197321$ 的 $q^lB_l(q^2)$ ——
**两者是不同的约定，连 $q^2$ 的形状都不同**（实测把 $d$ 调成 $1/Q_0$ 也
对不上，比值随事例变 2.6 倍）。所以 fixture 存的是 tf-pwa
`has_barrier_factor=False` 的振幅，测试端乘上本仓库的位垒乘子再比。
`test_barrier_convention_is_documented` 守住这个前提。

另外 tf-pwa 在 `add_covariant_term=True` 时还会往位垒里塞一个逐事例的
`mass` 因子，这是关掉位垒的第二个理由。协变项本身要**打开** —— 它对应
本仓库 `chain_amplitude(..., covariant=True)` 的 Chung Eq.52 项。

比法用**一个全局复常数**（$W_{\text{ours}}\approx c\,W_{\text{tfpwa}}$，
最小二乘拟合 $c$ 后要求残差 < 1e-12），**不逐事例、不逐道**各拟合一个 ——
那样会把"只对得上形状"的错也放过。整体常数来自 tf-pwa `g_ls` 的归一约定。

fixture 的重新生成见 `agent/tensor/tests/make_tfpwa_golden.py` 的 docstring
（需要 tf-pwa 环境，本机没有）。

---

# 螺旋度级联表象（`kernel = "helicity_cascade"`）

`documentation/helicity_amplitude_roadmap.md` 的实现落点。它按 tf-pwa 的
`HelicityDecay` + `DecayChain.get_amp` 逐节点构造振幅、再链级 `einsum` 缩并内线
螺旋度，所以**对链的深度没有任何假设**，也**每个节点都可以有多个 (l,s)** ——
这正是 omega 的两条能力边界。

| | omega / helicity | helicity_cascade |
|---|---|---|
| 逐节点 (l,s) | ❌ 非顶层节点只以标量 $K_i$ 出现 | ✅ 每节点一个列表 |
| 深度 | ❌ 只算 `D0*D1*D2` | ✅ 任意深度 |
| 逐节点螺旋度参考系 | ❌ 全部从顶层静止系直接 boost | ✅ 每层在**母节点**系里量 |
| cache 形状 | `(n_ch, N, 3)` complex | **逐位相同**（A7） |
| 拟合侧 | — | **零改动**（`_load_cached_tensor` 不变） |

TOML 侧新增两个 `[[channel]]` 字段：

```toml
[[channel]]
name = "u_SD_1"
l = 0
s = 1
# 逐节点 (l,s)：顶层波 × 内层波才是完备标签（SS/SD/DS/DD）
ls   = { psi2s = [[0,1]], b1 = [[2,1]], phi = [[1,0]] }
# 镜像树：`_2` 族的旁观者是 Pip
tree = { psi2s = ["b1","Pim"], b1 = ["phi","Pip"] }
```

`ls` 缺省时，顶层用道表的 `(l,s)`、其余衰变节点用 `allowed_ls`（宇称 + 三角
的**全部**允许值）；`tree` 缺省时沿用 `[particles.*].daughters`。两者都只在
`helicity_cascade` 表象里被读，但**在读 TOML 时**就校验（未知节点、宇称禁止的
`(l,s)`、改变末态集合的"镜像树"都会报错，而不是运行时 KeyError）。

## 已知边界

> **三条边界一览**（细节各见其下）：
>
> | # | 边界 | 判据 | 状态 |
> |---|---|---|---|
> | 1 | omega 的**逐节点 $(l,s)$ 容量** | 非顶层节点在 cache 里只有标量 $K_i$ | ✅ 已由级联内核推开 |
> | 2 | omega 的**深度**（只算 `D0*D1*D2`） | 深度 ≥ 3 时最内层角分布被丢 | ✅ 已由级联内核推开 |
> | 3 | 某些**协变公式不是级联** | 落在级联子空间之外 | ❌ **未解决** |

### 边界 1/2：omega 的两条限制

| 坑 | 现象 | 现在 |
|---|---|---|
| 非顶层分波被压成标量 | $\Omega+K+M+g$：非顶层节点的 $(l,s)$ 只在顶层 $M$ 里，内层只有 $K_i$ | 级联每节点一个列表 |
| 只算 `D0*D1*D2` | 深度 ≥ 3 时最内层（$\phi\to KK$）的角分布被**静默丢弃** | 级联对深度无假设；`spec.py` 的深度分级**直接拒绝** omega（不是静默出错） |
| 逐节点参考系从顶层 boost | 差别只在深度 ≥ 3 时出现（落在**最内层**的子粒子上） | 每层在母节点系里量角 |

第三条最隐蔽：深度 2 的 φf₀/φf₂ 上两种参考系构造**逐位相同**（no-op），
所以这个 bug 直到 b₁ 才暴露。

### 边界 3：ρ′ 的 Eq.(46) 不是"逐节点核 + 常数复系数"的级联

判据 A1 对 b₁ 的 8 条道全部通过（rel 6e-15 … 2.5e-14），但对 ρ′ 的
`u_rho_1`/`u_rho_2` **不通过**（rel 0.24 … 0.32，随样本变 —— 这本身就是
"逐事例失配"的征兆）。三条独立证据：

| 证据 | b₁（控制） | ρ′ |
|---|---|---|
| 直接最小二乘（自由复系数） | 6e-15 | **2.72e-01** |
| 逐事例 $c(e)$ 的 $\lvert c/c_0-1\rvert_{\max}$ | 9e-13（常数） | **3.4e+01** |
| 级联族 vs 协变族的子空间主角 | 8 维 vs 8 维，$\sim10^{-6}$ 度 | 2 维 vs 2 维，**16 度** |

**维数相同**（ρ′ 两边都是 2）是关键：这不是"级联少了几维"，而是
"两个 2 维子空间张成的**方向**不同"。

公式结构上的原因：

* b₁ 的 Eq.(47)–(50) 含**中间态的投影算符**
  $\tilde g_{\mu\nu}(p_{b_1})$ / $\tilde t^{(2)}_{\mu\nu}(p_{b_1})$ ——
  这就是"逐节点级联"的协变写法，所以与 `helicity_cascade` 逐位一致。
* ρ′ 的 Eq.(46) 是 $\epsilon^\mu{}_{\alpha\beta\gamma}p_\psi^\alpha
  [\ldots \epsilon^{\gamma\delta\sigma\lambda}p_{\psi\delta}\ldots]$ ——
  两个 $\epsilon$ 张起来、**内层 $\epsilon$ 缩并的是 $p_\psi$ 而不是 $p_{\rho'}$**，
  中间没有投影算符。

排除过的替代（见 `probes/probe_rho_qfactor.py`、`probe_rho_rank.py`）：

* 内层 $\epsilon$ 换成 $p_{\rho'}$（rel 0.29 → 0.36，没改善）；
* 内层 $(l,s)$ 加项 —— ρ′ 内层由宇称 + 三角**只有** $(1,1)$ 一项；
* 各种事件依赖标量因子（$q_{\text{内}}$、$B_1$、$q_{\text{外}}$、$m^2_{\rho'}$、
  $\gamma_\phi$、$\gamma_\pi$、以及全部一起）—— 残差全部**不降反升**；
* 两棵树 × 两种参考系 × 四组内层 $(l,s)$ 穷举 —— 最好 2.1e-01。

**结论（收窄后的版本）**：ρ′ 的协变 Eq.(46) 与"当前逐节点核 + 常数复系数
的级联"是**两种不同的振幅**，这一条已由上面三条证据确立。
但**不能**说"ρ′ 不是任何意义上的级联"—— 一个带张量结构的事件依赖投影
并未被穷举排除。**Eq.(46) 与逐节点级联在物理上是否等价，目前未知**；
要澄清需要把 Eq.(46) **重写**成带 $\rho'$ 投影算符的级联形式，
而不是去拟合它。

所以 `check_ex3.py` 对 ρ′ 设 `"a1": False`（只验 A7 形状与有限性）。
独立复算脚本见
`analyses/kk_dis/task/20261019_螺旋度级联内核_审阅/material/verify.py`。

同类现象在例 5 里也出现了（`u_PP_*`/`u_FP_*` 共 4 条落在级联子空间之外），
详见 [`chains/ex5_kst2.md`](chains/ex5_kst2.md)。

---

# 物理要点

## 例 3（`ψ→φπ⁺π⁻`）

粒子编号与文档一致：`K⁺=1`、`K⁻=2`、`π⁺=3`、`π⁻=4`。

* **ρ′ 族**（`chains/rhopi/`）：`ψ→ρ′π` 与 `ρ′→φπ` 都限于 P 波，所以只有
  **一个**独立振幅；`u_rho_1`/`u_rho_2` 是"哪一个 π 是旁观者"的两种取法。
* **b₁ 族**（`chains/b1pi/`）：`ψ→b₁π` 与 `b₁→φπ` 都可以既有 S 波又有 D 波，
  所以有**四个**独立振幅，完备标签是"顶层波 × 内层波"（SS/SD/DS/DD）。
  对应对调旁观者 π 又多一倍 → 8 条道。
* ⚠️ `_1`/`_2` 的下标在**两个族里含义相反**（照抄参考实现的命名）：
  ρ′ 族 `_1` 的旁观者是粒子 3，b₁ 族 `_1` 的旁观者是粒子 4。
* ⚠️ 顶层 `(l,s)` **不是** b₁ 族的完备标签（8 条道只映射到 2 组 `(l,s)`），
  所以两份 spec 都设 `check_channels = false`。
* ⚠️ 三层树（`ψ→X π`，`X→φπ`，`φ→K⁺K⁻`）**不能出 omega 表象** ——
  omega 内核逐节点读树、只算 `D0*D1*D2`。这条限制由
  `agent/tensor/spec.py` 按 `[[basis]]` 分级执行。
  现在可以改出 `helicity_cascade`：它对深度没有假设。

## 例 5（`ψ→K⁺π⁻K⁻π⁺`）

拓扑 $\psi\to K^*\pi \to (\phi\pi)\pi \to (K^+K^-)(\pi\pi)$ —— 把 b₁ 换成 K*，
**深度同样是 3**，其余结构与例 3 的 b₁ 完全一样。所以 omega 同样不可用、
级联同样可用。10 条道在
[`agent/tensor/chains/kstpi/`](../../agent/tensor/chains/kstpi/)。

| spec | 道数 | 顶层 $(l,s)$ | 内层 $(l,s)$ |
|---|---|---|---|
| [`ex5_kst2.toml`](specs/ex5_kst2.toml) | 8 | $(1,2)$ / $(3,2)$ | $(1,1)$ / $(3,1)$ |
| [`ex5_kst0.toml`](specs/ex5_kst0.toml) | 2 | $(1,0)$ | $(1,1)$ |

**结果**（验证细节见 [`chains/ex5_kst2.md`](chains/ex5_kst2.md) 与
[`chains/ex5_kst0.md`](chains/ex5_kst0.md)）：

* **移植** A5：10/10 到 `rel ≈ 1e-15`；`emit` 三种表象全部通过。
* **级联可表示性**：`ex5_kst2` **4/8**（`u_PF_*`、`u_FF_*` 命中），
  `ex5_kst0` **2/2**。⚠️ 探针输出的是**合计 6/10**，引用单链时用 4/8。
* **阳性对照**：同一探针跑 b₁ 得 **8/8**，所以 4/8 不是探针坏了。

`u_PP_*`/`u_FP_*` 四条都含 $P^{(2)}_{\psi}$ 横向投影且内层是 P 波；但 K*₀ 的
内层**也是** P 波却命中，所以不是"P 波一律不行"，而是
$(J_{K^*}=2,\ l_{\text{内}}=1)$ 这个特定组合。**原因未查清**（同边界 3）。

⚠️ **例 5 还没有端到端脚本** —— `check_ex3.py` 只覆盖例 3，所以例 5 目前
**缺 L1/L2/L3 那三层**，只由 `test_ex5_channels.py` + `emit` 跑通来保证。

## 数据为什么是合成的

`chains/` 里的道函数是按道名**硬编码的公式**，对任意满足守恒的四动量都成立
（道函数体里不出现 `tree`/`nodes`，见 `agent/tensor/chains/` 下的 `u_*.py`）。所以验证
它们不需要真实样本，只需要物理四动量。`make_momenta.py` 用顺序两体衰变严格
生成，并自检：守恒、每个末态在壳、每个两体子系统过阈值（否则 `Qabc<0`，
`B1/B2` 出 NaN —— **jax 原版与 numpy 移植版都会 NaN**，那不是移植错误）。

`m(π⁺π⁻)` 在 `[2m_π, m_ψ-m_φ]` 上均匀抽样，使 ρ′/b₁ 的不变质量自然扫过宽区间。

---

# 如何运行

## 一键跑完全部自检（推荐）

```bash
cd <LLMPWA>

# 例 3 的端到端回归：生成 → 跑 → L1/L2/L3 + A1/A3/A7
python analyses/tensor_examples/check_ex3.py --n 500

# 只跑某一条树
python analyses/tensor_examples/check_ex3.py --n 500 --tree ex3_rhopi
```

它做三层验证 + 三条判据（定义见上文"如何验证"）：

1. **L1**：产物脚本内联的道函数体算出的 cache，必须与直接调
   `agent.tensor.kernels.covariant.compute_event` 逐位一致；
2. **L2**：`W = EPS_A · T`；
3. **L3**：没有 NaN/Inf，且不恒零；
4. **A1**：`helicity_cascade` 逐道对拍协变路线（经 `EPS_A`），rel < 1e-12；
5. **A3**：b₁ 的 8 条道两两不重合（min pairwise rel > 1e-6）；
6. **A7**：级联 cache 的形状/dtype 与 helicity **精确**一致。

## 分步跑

```bash
cd <LLMPWA>

# 1) 合成四动量（自带守恒 / 在壳 / 阈值自检）
python analyses/tensor_examples/data/make_momenta.py \
    --out analyses/tensor_examples/data/ex3_phipipi.npz --n 2000 --seed 7

# 2) 看链 / 道表 / 表象
python -m agent.tensor show  analyses/tensor_examples/specs/ex3_rhopi.toml
python -m agent.tensor check analyses/tensor_examples/specs/ex3_b1pi.toml

# 3) 生成 + 直接跑（emit 出的脚本是自足的，不依赖 agent.tensor）
python -m agent.tensor emit analyses/tensor_examples/specs/ex3_rhopi.toml
python analyses/tensor_examples/generated/ex3_rhopi/gen_tensor_ex3_rhopi_covariant.py \
    --npz analyses/tensor_examples/data/ex3_phipipi.npz \
    --out analyses/tensor_examples/out/ex3_rhopi --n 500
```

## 框架侧测试（不需要本目录的数据）

```bash
# 纯 numpy，永远能跑
python agent/tensor/tests/test_ex3_channels.py      # 8 项
python agent/tensor/tests/test_ex5_channels.py      # 6 项（例 5 的 10 条道）
python agent/tensor/tests/test_helicity_cascade.py  # 13 项：A1/A2/A3/A4 + 三条边界
python agent/tensor/tests/test_helicity_vs_tfpwa.py # 4 项：A5′（与 tf-pwa 对拍）
python agent/tensor/tests/test_ex12_permutation.py  # 7 项：例 1/2 的置换求和 + emit + omega 护栏
python agent/tensor/tests/test_chain_docs.py        # 5 项：链文档与 spec 同步
```

`test_helicity_vs_tfpwa.py` 用冻结 fixture，**不需要** TF / tf-pwa / jax。
fixture 的重新生成见 `agent/tensor/tests/make_tfpwa_golden.py`。

`test_ex12_permutation.py` 在文件内自己合成三体相空间，不依赖
`data/*.npz`（那些是生成物，不入库）。要手工造：

```bash
python analyses/tensor_examples/data/make_threebody.py ex1 --n 64
python analyses/tensor_examples/data/make_threebody.py ex2 --n 64
```

## 与 jax 原版对拍（A5，需要额外环境）

```bash
# 例 3 + phif0/phif2（需要 jax + Tensor/ 源码，缺了就跳过）
TENSOR_ROOT=<...>/1_运行代码/tensor_example \
    python agent/tensor/tests/test_covariant_vs_jax.py

# 例 5（同理）
analyses/kk_dis/.venv/bin/python analyses/tensor_examples/probes/verify_ex5_port.py
```

⚠️ 两个坑（见上文 A5 一节）：必须在导入 `FUNC` **之前**开
`jax_enable_x64`；参考实现的公开名与内部名不同（`_u_PP_l` → `u_PP_1`）。

## 诊断探针（一次性，用于复现结论）

```bash
# 例 5 的级联可表示性 + 阳性对照（不需要 jax）
python analyses/tensor_examples/probes/probe_ex5_identify.py
python analyses/tensor_examples/probes/probe_ex5_identify.py --control

# A1 判据自身的有效性（错误注入）
python analyses/tensor_examples/probes/probe_a1_sensitivity.py

# ρ′ 为何不是级联（四条证据）
python analyses/tensor_examples/probes/probe_rho_rank.py          # 子空间维数 / 主角
python analyses/tensor_examples/probes/probe_rho_qfactor.py       # 事件依赖标量因子
python analyses/tensor_examples/probes/probe_rho_decomposition.py # 逐节点分解（R 检验）
python analyses/tensor_examples/probes/probe_rho_structure.py     # 顶层张量 / omega 对照
```

⚠️ `probe_rho_structure.py` 曾长期跑不起来（`import harness`，而那个模块
**从未入库**、目录已删）。2026-10-20 已按现行内核 API
（`HC.node_angles` / `HC.chain_amplitude` / `RT.assemble`）重写。
