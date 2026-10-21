# 链文档索引

`analyses/tensor_examples/chains/` —— 每条**链**一份文档，写清
"目前做了什么、还有什么问题"。

本文档写于 2026-10-19。例 5 的两份（`ex5_kst0` / `ex5_kst2`）随后补入；
例 1/2 的七份于 2026-10-20 补入：`ex1_rho` / `ex1_rho3` / `ex1_rho5` /
`ex2_rho` / `ex2_kstar` / `ex2_kstar3` / `ex2_kstar5`。

## 这些文档是怎么来的

**每份链文档都是生成物**，分两半：

| 部分 | 来源 | 例子 |
|---|---|---|
| **事实**（会自动过期） | 由 [`gen_chain_docs.py`](gen_chain_docs.py) 从 TOML **自动导出** | 拓扑、粒子表、道表、逐节点 $(l,s)$、计数律、表象可用性、`emit` 实测 |
| **叙述**（只有人知道） | 手写在 [`notes/<chain>.notes.md`](notes/) | 为什么重要、踩过的坑、遗留问题、复现命令 |

所以**不要直接编辑 `chains/*.md`** —— 改 spec 或 `notes/*.notes.md`，然后：

```bash
# 重新生成全部 11 份
python analyses/tensor_examples/chains/gen_chain_docs.py

# 只检查是否同步（CI 用，不同步时非零退出）
python analyses/tensor_examples/chains/gen_chain_docs.py --check

# 打印某一份到 stdout（不写盘）
python analyses/tensor_examples/chains/gen_chain_docs.py --stdout b1pi
```

同步检查已挂进测试套件：

```bash
python agent/tensor/tests/test_chain_docs.py    # 5 项
```

它包含一条**反向验证**（故意弄脏文档，`--check` 必须报错），
防止"检查器永远返回 OK"的假绿。

### 自动导出的 6 个小节

| 小节 | 内容 |
|---|---|
| 1. 拓扑 | 树的文字图 + 深度 |
| 2. 粒子表 | $J$ / 宇称 / 质量 / 角色 / 允许的 $l$ |
| 3. 道表与逐节点 $(l,s)$ | 显式还是枚举；每条的逐节点 $(l,s)$ |
| 4. 计数律 | $\prod_n N_n$；非顶层是否多分波（⇒ omega 容量够不够）|
| 5. 表象与 `emit` 可行性 | **逐表象实测** `emitter.emit`，不靠文件存在推断 |
| 6. 叙述 | 手工笔记原文嵌入 |

`emit` 那一步写到**临时目录**，所以生成文档不会在仓库里留产物。

---

## 一张表看全

| 链 | 拓扑 | 深度 | 道数 | 表象 | `emit` | A1 | 状态 |
|---|---|---|---|---|---|---|---|
| [`phif0`](phif0.md) | ψ→φf₀→(KK)(ππ) | 2 | 2 | cart / hel / **omega** | ✅ | ✅ 5e-15 | 生产链 |
| [`phif2`](phif2.md) | ψ→φf₂→(KK)(ππ) | 2 | 5 | cart / hel / **omega** | ✅ | ✅ 3e-15 | 生产链 |
| [`b1pi`](b1pi.md) | ψ→b₁π→(φπ)π→(KK)(ππ) | **3** | 8 | cart / hel / **cascade** | ✅ | ✅ 6e-15 | **验收主力** |
| [`rhopi`](rhopi.md) | ψ→ρ′π→(φπ)π→(KK)(ππ) | **3** | 2 | cart / hel / **cascade** | ✅ | ❌ 0.28 | **边界** |
| [`rhof2_generic`](rhof2_generic.md) | ψ→ρf₂→(ππ)(ππ) | 2 | 5 | cart / hel / omega | ✅ | ✅ 2e-15 | **假想链** |
| [`ex5_kst2`](ex5_kst2.md) | ψ→K*₂π→(φπ)π→(KK)(ππ) | **3** | 8 | cart / hel / **cascade** | ✅ | ✅ 修正后全过 | **例 5** |
| [`ex5_kst2_paper`](ex5_kst2_paper.md) | 同上（**对照 spec**） | **3** | 8 | cart / hel / cascade | ✅ | ⚠️ 故意的 | **对照** |
| [`ex5_kst0`](ex5_kst0.md) | ψ→K*₀π→(φπ)π→(KK)(ππ) | **3** | 2 | cart / hel / **cascade** | ✅ | ✅ 2/2 | **例 5** |
| [`ex1_rho`](ex1_rho.md) | ψ→ρπ→(ππ)π | 2 | 3 | cart / hel / **cascade** | ✅ | ✅ 4e-15 | **例 1** |
| [`ex1_rho3`](ex1_rho3.md) | ψ→ρ₃π→(ππ)π | 2 | 3 | cart / hel / **cascade** | ✅ | ✅ 1e-14 | **例 1** |
| [`ex1_rho5`](ex1_rho5.md) | ψ→ρ₅π→(ππ)π | 2 | 3 | cart / hel / **cascade** | ✅ | ✅ 2e-14 | **例 1，J=5** |
| [`ex2_rho`](ex2_rho.md) | ψ→ρ′π→(KK)π | 2 | 1 | cart / hel / **cascade** | ✅ | ✅ 9e-16 | **例 2** |
| [`ex2_kstar`](ex2_kstar.md) | ψ→K*K→(Kπ)K | 2 | 2 | cart / hel / **cascade** | ✅ | ✅ 4e-15 | **例 2** |
| [`ex2_kstar3`](ex2_kstar3.md) | ψ→K*₃K→(Kπ)K | 2 | 2 | cart / hel / **cascade** | ✅ | ✅ 3e-15 | **例 2，J=3** |
| [`ex2_kstar5`](ex2_kstar5.md) | ψ→K*₅K→(Kπ)K | 2 | 2 | cart / hel / **cascade** | ✅ | ✅ 6e-15 | **例 2，J=5** |

⚠️ 表里 14 条链的 `emit` 列**全部是 ✅**（2026-10-21 起）—— 例 1/2 的 18 个
手写协变道补齐后，`cartesian`/`helicity` 两条路线也能出了。A1 列的数字来自
[`probes/verify_all_specs.py`](../probes/verify_all_specs.py)（全部 14 份 spec）。

"表象"列加粗的是该链**特有**的那个：

* `phif0`/`phif2` 特有 **omega**（深度 2，omega 可用）；
* `b1pi`/`rhopi`/`ex5_*` 特有 **cascade**（深度 3，omega 被禁止）；
* `rhof2_generic` 三个表象都列了（2026-10-21 起三个都能出）；
* 例 1/2 的七条链**原本**只有 **cascade** 能出（三体置换道没有手写协变道
  函数）。**2026-10-21 起 18 个手写道全部补齐**，所以这七条链现在也是
  `cart / hel / cascade` 三条路线都能出（`ex1_rho5` 是 2026-10-20 先行的
  那个，也是"协变张量路线能表达 J=5"的落点）。

"A1" 那列对例 1/2 的七条链留空（`—`）：它们不是 4 体 `φf_J` 链，
没有 `EPS_A` 逐道对拍的参考产物；判据改用**置换协变性**与
**三路线互相比**（见 `analyses/tensor_examples/README.md`）。
2026-10-21 起这三条路线之间也直接对了 A1（`probes/verify_all_specs.py`），
七条链全部到机器精度。

"A1" = 级联 vs 协变逐道对拍（经 `EPS_A`）。✅ 表示到机器精度。

`ex5_kst2` 的 "4/8" 是**级联可表示性**（本链 8 条道里 4 条逐位命中）：
命中 `u_PF_*`、`u_FF_*`；`u_PP_*`/`u_FP_*` 落在级联子空间之外
（rel ≈ 6e-3 / 2e-1）。见 [ex5_kst2.md](ex5_kst2.md) 的"级联判定"。

**2026-10-21：原因已定位，且生产代码已修。** 内层 P 波的无迹化用了
$\tilde g(p_\psi)$（$\psi$ 系）而不是 $\tilde g(p_{K^*_2})$（$K^*_2$ 系）
—— 与 `phif2` 的 (41)(42) 缺陷同型的**母系自旋投影错配**。
`chains/kstpi/` 里：

* `u_PP_1/2.py`、`u_FP_1/2.py` —— **修正版（生产）**，A1 到 ~5e-15；
* `u_PP_1/2_paper.py`、`u_FP_1/2_paper.py` —— 历史形式（逐式移植，A5 对拍用）。

完整推导与公式/代码对照见
[`agent/tensor/chains/kstpi/CORRECTION.md`](../../../agent/tensor/chains/kstpi/CORRECTION.md)。
内层 F 波命中是因为 `T3_cov` 自带 $K^*_2$ 系的无迹化；
$K^*_0$ 的内层也是 P 波却命中是因为 $J=0$ 耦合成标量、没有无迹化这一步。
`ex5_kst0` 的 2 条**都是**纯级联（2/2）。

⚠️ 探针输出的是**例 5 全部 10 条**的结果（**6 / 10**），因为它把
`ex5_kst0` 的 2 条也算进去了。**单看 `ex5_kst2` 是 4/8** —— 两个口径
不要混（第一版本表就写错过成 6/8）。这个 "4/8" 说的是**历史形式**
（= `*_paper.py`）；修正版是 **8/8**。

⚠️ 例 5 的 10 条道**已移植且与 jax 对拍通过（1e-15）** —— 所以"能不能算"
是 ✅；对拍现在落在 `*_paper.py` 上（那才是逐式移植）。

⚠️ 例 5 的移植对拍**必须**在导入 `FUNC` 前开 jax x64，否则全程 float32、
只能到 1e-7，会被误判成移植失败。这个坑第一次就踩了（见
[ex5_kst2.md](ex5_kst2.md)）。

### A5 的现状（2026-10-20 实测）

| 覆盖 | 命令 | 结果 |
|---|---|---|
| 例 3 的 10 条 + 7 条生产道 + 7 个 BASE 原语 | `TENSOR_ROOT=... python agent/tensor/tests/test_covariant_vs_jax.py` | ✅ 24/24，rel ≤ 1e-15 |
| 例 5 的 10 条 | `analyses/kk_dis/.venv/bin/python analyses/tensor_examples/probes/verify_ex5_port.py` | ✅ 10/10，rel ≤ 5e-15 |

两者都需要 `Tensor/` 源码（不在 git 里）；没有 jax 或没有源码时**跳过**而不是失败。
例 3 的对拍**不依赖例 5 移植**（它自己合成四动量），所以"需要例 5 移植"是错的。

### A5′ 的现状（与上游 tf-pwa 对拍，2026-10-20 实测）

A5 比的是项目自己的 jax 参考实现；A5′ 比的是社区标准实现 tf-pwa
（`HelicityDecay` + `DecayGroup.get_amp3`）。用冻结 fixture，
**本地没有 TF / tf-pwa / jax 也能跑**：

| 层 | 比什么 | 结果 |
|---|---|---|
| 1 | CG 矩阵 vs `get_cg_matrix()` | 逐元素 0（精确相等） |
| 2 | 逐节点 `(alpha, beta)` vs `cal_angle_from_momentum` | ≤ 5e-15 |
| 3 | 链振幅 vs `get_amp3`（**除掉位垒后**） | 8/8 道 ≤ 3e-15 |

```bash
python agent/tensor/tests/test_helicity_vs_tfpwa.py    # 4 项
```

⚠️ 第 3 层必须除掉位垒：tf-pwa 用 $d=3$ 的 Blatt-Weisskopf 参数化，本仓库
用 $Q_0=0.197321$，**两者是不同的约定**（连 $q^2$ 形状都不同，不是差常数）。
fixture 存的是 tf-pwa `has_barrier_factor=False` 的振幅。详见
[../README.md](../README.md) 的 A5′ 一节。

## 三条边界

### 1. 深度 ≥ 3 ⇒ omega 不可用

omega 只算 `D0*D1*D2`，会**静默丢掉**最内层的角分布。
现在由 [`spec.py`](../../../agent/tensor/spec.py) 的深度分级检查**强制报错**。

所以 `b1pi`/`rhopi` 的 spec **不出** omega —— 不是遗漏，是被禁止。

### 2. 非顶层多分波 ⇒ omega 容量不足

omega 的 cache 是 $\Omega+K+M+g$，非顶层节点的角动量只以**标量** $K_i$ 出现。
所以 $b_1\to\phi\pi$ 的 S/D 波之别**无处安放**（$b_1$ 有 2 个 $(l,s)$）。

计数律：独立振幅数 $=\prod_{\text{节点}}N_{\text{节点}}$。

| 链 | 非顶层 $(l,s)$ 数 | 独立振幅 | omega 能给 |
|---|---|---|---|
| `phif0` | 1 | 1 | ✅ 够 |
| `phif2` | 1 | 1 | ✅ 够 |
| `rhopi` | 1 | 1 | ✅ 够 |
| `b1pi` | **2** | **4** | ❌ 只给 2 |

### 3. 逐节点参考系（最隐蔽）

omega 的 `chain_angles` 把每个粒子都从**顶层**静止系直接 boost。
这在"中间共振不再衰变"的链上是对的，一旦中间共振自己还衰变就错了。

**实测规则**：差别只在链的最大深度 ≥ 3 时出现，落在最内层节点的子粒子上。

| 链 | 最大深度 | $\max\lvert$old − new$\rvert$（角）|
|---|---|---|
| `phif0` | 2 | 0.000e+00（no-op）|
| `phif2` | 2 | 0.000e+00（no-op）|
| `rhof2_generic` | 2 | 0.000e+00（no-op）|
| `b1pi` | 3 | 9.301e-01 |
| `rhopi` | 3 | 4.391e-01 |

⚠️ 注意 `rhopi` **也**受这个修正影响 —— 它的 A1 失配与"frame 修正"是
**两件独立的事**（见 [`rhopi.md`](rhopi.md) §3）。

## 一个反直觉的结论

`rhopi` 满足计数律、深度也只比 `phif2` 多一层，看起来"应该成功"。
但它的 A1 **不通过**（0.32），而 `b1pi`（分波更多、更复杂）**通过**（6e-15）。

原因不是内核有 bug，而是 **Eq.(46) 本身不是级联**：

* `b1pi` 的 Eq.(47)–(50) 含中间态**投影算符** ⇒ 天然是级联；
* `rhopi` 的 Eq.(46) 是两个 $\epsilon$ 张起来、内层缩并 $p_\psi$ 而非
  $p_{\rho'}$、中间没有投影算符 ⇒ **不是**级联。

所以正确的表述是"**两种形式的关系未知**"，**不是**"内核能力不足"。
详见 [`rhopi.md`](rhopi.md) §3 与 [README 的"已知边界"](../README.md)。

## 五条链各自缺什么（汇总）

| 缺口 | 影响哪些链 |
|---|---|
| 真实样本回归 | 全部 |
| 拟合侧端到端（`_load_cached_tensor` / toy likelihood）| 全部 |
| barrier 的 $q_0$ 依赖 | 全部（roadmap §8 风险 1）|
| `psi_mass` 口径（3.686 vs 3.685782956）| `phif0`/`phif2`/`b1pi`/`rhopi` |
| 节点级 Eq.52 消融 + 质量扫描 | `b1pi` |
| 与 tf-pwa `HelicityDecay` 逐节点对拍 | `b1pi`（frame 约定只有经验验证）|
| 构造性重写 Eq.(46) | `rhopi`（审阅意见的头号建议）|
| 手写协变道 | ✅ 已补齐（2026-10-21，48/48） |
| 一份 spec 装两棵树（设计） | `b1pi`/`rhopi` |

### 手写协变道：**已全部补齐**（2026-10-21）

"缺手写道"**不等于"协变路线算不了"** —— `chains/<chain>/<channel>.py` 放的
是**逐道的手工搬运**（把论文那一条道的缩并写成代码），`emitter.py` 按
`def <name>` 取体内联。缺了只是 `emit` 会 `LookupError`。

审计（[`probes/probe_missing_covariant.py`](../probes/probe_missing_covariant.py)）
先用**通用构造** `t_l` + ε 缩并（论文 Eq.258 + Eq.349）直接算那 18 条道，
与**级联路线**逐道对拍，**全部到机器精度（≤1.6e-14）** —— 确认缺的只是搬运。
随后把这 18 条搬进 `chains/`：

| 类别 | spec | 补齐的道数 | 手写目录 |
|---|---|---|---|
| **置换族** | `ex1_pipipi0_rho` / `_rho3` | 3 + 3 | `chains/ex1_pipipi0_rho{,3}/` |
| | `ex2_kkpi0_kstar` / `_kstar3` / `_kstar5` | 2 + 2 + 2 | `chains/ex2_kkpi0_kstar{,3,5}/` |
| **单道** | `ex2_kkpi0_rho` | 1 | `chains/ex2_kkpi0_rho/` |
| **纯改名** | `rhof2_generic` | 5 | `chains/rhof2/`（逐式改名自 `phif2`） |

合计 **18 个道、7 个 spec**。现在 **14 个 spec / 48 条道全部有手写协变道**，
`emit` 三条路线（`cartesian` / `helicity` / `helicity_cascade`）都能出。

实测（[`probes/verify_all_specs.py`](../probes/verify_all_specs.py)）：
14 份 spec 的 **L1**（产物 == 内核）、**L2**（`W = EPS_A·T`）、**L3**（有限）、
**A7**（形状/dtype）全过；**A1**（级联 vs 协变）只有两处已知失配
（`ex3_rhopi`、`ex5_kst2` 的 4 条）—— 两者都已定位到具体机制，见下。

⚠️ 别被"缺"字误导：这些态在**级联路线**上一直可用（`helicity_cascade`
按逐节点 `(l,s)` 描述道，不需要手写公式）。补的是 `cartesian`/`helicity`
两条路线的**产物**。这两条路线**共用同一批手写函数** —— 补一个道，
两条路线同时可用。

⚠️ 补写时**配对顺序必须逐道照抄 spec 的 `tree`**：`t_l` 里 $r = b - c$，
交换两个子节点会让振幅整体翻号；级联侧的口径由同一个 `tree` 定。
符号与相位的完整检验见
[`probes/probe_a1_signs.py`](../probes/probe_a1_signs.py)（各道 $c_i$ 同轴、
置换协变性 $|c|=1$ 全部通过）。

## 代码位置速查

| 东西 | 位置 |
|---|---|
| spec | `agent/tensor/specs/{phif0,phif2,rhof2_generic}.toml`<br>`analyses/tensor_examples/specs/{ex3_b1pi,ex3_rhopi}.toml` |
| 手写协变道 | `agent/tensor/chains/{phif0,phif2,b1pi,rhopi}/` |
| 级联内核 | [`agent/tensor/kernels/helicity_cascade.py`](../../../agent/tensor/kernels/helicity_cascade.py) |
| 协变内核 | [`agent/tensor/kernels/covariant.py`](../../../agent/tensor/kernels/covariant.py) |
| omega 内核 | [`agent/tensor/kernels/omega.py`](../../../agent/tensor/kernels/omega.py) |
| spec 解析 | [`agent/tensor/spec.py`](../../../agent/tensor/spec.py) |
| emitter | [`agent/tensor/emitter.py`](../../../agent/tensor/emitter.py) |
| **链文档生成器** | [`gen_chain_docs.py`](gen_chain_docs.py) |
| **链文档手工笔记** | [`notes/`](notes/) |
| 端到端回归 | [`check_ex3.py`](../check_ex3.py) |
| 诊断探针 | [`probes/`](../probes/) |
| 独立复算 | `analyses/kk_dis/task/20261019_螺旋度级联内核_审阅/material/verify.py` |

## 相关文档

* [`../README.md`](../README.md) —— 本目录总览 + 已知边界
* [`documentation/helicity_amplitude_roadmap.md`](../../../documentation/helicity_amplitude_roadmap.md) —— 实现规格
* [`documentation/tensor_examples_feasibility.md`](../../../documentation/tensor_examples_feasibility.md) —— 可行性评估
* 审阅记录：`analyses/kk_dis/task/20261019_螺旋度级联内核_审阅/gen/fragments/`
  （第一轮 `helicity_cascade_review.md` + 第二轮 `..._r2.md`）
