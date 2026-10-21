# 文档例子的可实现性评估（Chung 协变张量形式）

本文评估 `analyses/kk_dis/document/ψ衰变到介子的分波分析协变张量形式.md`
里各衰变例子，**在 `agent/tensor/` 框架下能否用来做测试**，以及各自的缺口。

> 目的：为「把文档例子补成回归测试」排序，不是物理推导。所有结论都是**实测**
> 出来的（拓扑试读、原语对拍、参考实现试调），命令与误差都记在文中。

## 0. 结论速览

文档共 **12 个小节**，其中 **11 个有公式**（`ψ→γφφ` 只有标题）。

| 优先级 | 例子 | 结论 |
|---|---|---|
| **P0 ✅** | 例 3 `ψ→φπ⁺π⁻` 的 ρ′/b₁ 道（Eq.46–50） | **已完成**（2026-10-19）：10 条道已移植、与 jax 原版逐式对拍通过。见 §7 |
| **P0 ✅** | 例 5 `ψ→K⁺π⁻K⁻π⁺` 的 K*₀/K*₂ 道（431–623 行） | **已完成**（2026-10-19）：10 条道已移植、与 jax 逐式对拍通过（1e-15）。见 §8 |
| ~~P1~~ 完成 | 例 1 `ψ→π⁺π⁻π⁰`、例 2 `ψ→K⁺K⁻π⁰` | ✅ **已实现**（2026-10-20）：四份 spec、9 条道、验收 4/4 |
| P2 | 例 7 `ψ→γ` + 两标量 | 需新基建（秩 2 振幅 + 横向投影） |
| P2 | 例 8/9/10/11 辐射 4–5 体 | 双重阻塞（秩 2 + 深度 3） |
| **P1 ✅** | 例 4 `ψ→ωK⁺K⁻`（418–430 行） | **已实现**（2026-10-21）：7 条道、手写 `omega_mu` 原语。**A1 仍缺** —— 见 §2.3.2 与 §2.5 |
| **P1 ✅** | 例 6 `ψ→φπ⁺π⁻π⁺π⁻`（624–795 行）的 **σσ 子分支** | **已实现**（2026-10-21）：14 条道、A1 = 6.97e-15。ρρ / f₂(1270)σ 未实现（见 §2.3.1） |
| **不做** | 例 12 `ψ→γφφ` | **文档里是空的**（只有标题） |

**最省力的突破口**：参考实现 `Tensor/FUNC.py` 里已经写好了 **20 个未移植的道
函数**，全部实测可调用。它们覆盖例 3 与例 5。

## 1. 参考实现的覆盖面（关键发现）

`analyses/kk_dis/task/20260929_螺旋度耦合振幅推导/1_运行代码/tensor_example/Tensor/FUNC.py`
共 30 个 `def`：2 个基础设施（`__init__`、`Modifiers`）+ 1 个辅助
（`_s1_vector`）+ **27 个道函数**。而 `agent/tensor/chains/` 只移植了 **7 条道**：

| `FUNC.py` | 文档 | 已移植？ |
|---|---|---|
| `_phif001` `_phif021` | Eq.(36)(37) `<φf₀\|01> <φf₀\|21>` | ✅ |
| `_phif201` `_phif221` `_phif222` `_phif223` `_phif243` | Eq.(41)–(45) `<φf₂\|…>` | ✅ |
| `_s1_vector` | Eq.(41) 修正版的辅助函数 | ✅（`kernels/covariant.py`） |
| `_u_rho_1` `_u_rho_2` | Eq.(46) `U_{ρ′}`（文档 374–387 行） | ✅ 2026-10-19 |
| `_u_SS_1/2` `_u_SD_1/2` `_u_DS_1/2` `_u_DD_1/2` | Eq.(47)–(50) `U_{b₁ SS/SD/DS/DD}`（391–416 行） | ✅ 2026-10-19 |
| `_u_kst0_r` `_u_kst0_l` | Eq.(75) | ❌ |
| `_u_PP/PF/FP/FF_r/l`（8 个） | 例 5 的 `KK*₂` 相关分波 | ❌ |

即：**例 3 共有 17 个振幅**（`<φf_J|LS>` 形式的 7 个 + ρ′/b₁ 的 10 个）。
截至 2026-10-19，**17 个全部已移植**（前 7 个在 `chains/phif{0,2}/`，
后 10 个在 `chains/rhopi/` 与 `chains/b1pi/`）。
例 5 另有 10 个待移植（`u_kst0_r/l` + `u_PP/PF/FP/FF_r/l`）。

### 实测：20 个未移植函数全部可运行

```
$ ./analyses/kk_dis/.venv/bin/python   # jax 0.11.1, CPU
  u_rho_1    OK (32, 4) finite=True max=148.6
  u_rho_2    OK (32, 4) finite=True max=176.5
  u_SS_1     OK (32, 4) finite=True max=1.993
  u_SS_2     OK (32, 4) finite=True max=2.009
  u_SD_1     OK (32, 4) finite=True max=17.46
  u_SD_2     OK (32, 4) finite=True max=19.46
  u_DS_1     OK (32, 4) finite=True max=16.19
  u_DS_2     OK (32, 4) finite=True max=14.5
  u_DD_1     OK (32, 4) finite=True max=105.1
  u_DD_2     OK (32, 4) finite=True max=126.5
  u_kst0_r   OK (32, 4) finite=True max=15.43
  u_kst0_l   OK (32, 4) finite=True max=13.77
  u_PP_1/2   OK  u_PF_1/2  OK  u_FP_1/2  OK  u_FF_1/2  OK
可调用 20 / 20
```

调用约定与已移植的道不同，**移植时要注意**：参考实现的签名是
`(phi, f2, Kp, Km, Pip, Pim)`，其中 `phi = Kp + Km`、`f2 = Pip + Pim`
由调用方先算好再传入（见 `Tensor/CacheTensor.py:51`）；而 `agent/tensor` 的
道函数签名是 `(psi, phi, f2, Kp, Km, Pip, Pim)`，`psi` 也显式传入。

### 参考实现里有一处**陈旧**（不要照抄）

`Tensor/CacheTensor.py` 注册了 4 个 `FUNC.py` 里**不存在**的函数：

```
u_kst2_12_l / u_kst2_12_r / u_kst2_32_l / u_kst2_32_r     -> 均缺失
```

且该文件开头 `from jax.config import config` 在现代 jax 上直接
`ModuleNotFoundError`。**以 `FUNC.py` 为准**，`CacheTensor.py` 只能当调用约定的
参考。

## 2. 逐例评估

「拓扑框架」= `python -m agent.tensor check` 能否读入该例的树描述；
「参考对拍」= `Tensor/FUNC.py` 里有没有对应实现可做黄金样本。

| # | 例子 | 文档行 | 拓扑框架 | 缺什么 | 参考对拍 | 结论 |
|---|---|---|---|---|---|---|
| 3 | `ψ→φπ⁺π⁻` 的 ρ′/b₁ 道 | 305–417 | ✅ | 无 | ✅ 10 个函数 | **P0 可直接做** |
| 5 | `ψ→K⁺π⁻K⁻π⁺` 的 K*₀/K*₂ 道 | 431–623 | ✅ | 无 | ✅ 10 个函数 | **P0 可直接做** |
| 1 | `ψ→π⁺π⁻π⁰` | 200–273 | ✅ | **已实现**（ρ 3 + ρ₃ 3 + ρ₅ 3；ρ₅ 需 J=5/l=5，已扩内核上限） | ❌ | ✅ 完成 |
| 2 | `ψ→K⁺K⁻π⁰` | 274–304 | ✅ | **已实现**（KK̄ 1 + Kπ 2 + Kπ₃ 2 + Kπ₅ 2） | ❌ | ✅ 完成 |
| 7 | `ψ→γ` + 两标量 | 845–895 | ⚠️ 需声明 γ | 秩 2 振幅 + 横向投影 | ❌ | P2 需新基建 |
| 8 | `ψ→γηπ⁺π⁻` | 896–1101 | ❌ 深度 3 | 秩 2 + 深度 | ❌ | P2 双重阻塞 |
| 9 | `ψ→γKK̄π` | 1102–1134 | ❌ 深度 3 | 同上 | ❌ | P2 双重阻塞 |
| 10 | `ψ→γπ⁺π⁻π⁺π⁻` | 1135–1262 | ❌ 深度 3 | 同上 | ❌ | P2 双重阻塞 |
| 11 | `ψ→γK⁺K⁻π⁺π⁻` | 1263–1383 | ❌ 深度 3 | 同上 | ❌ | P2 双重阻塞 |
| 4 | `ψ→ωK⁺K⁻` | 418–430 | ❌ | ω→3π **三体衰变** | ❌ | 原理性缺口 |
| 6 | `ψ→φπ⁺π⁻π⁺π⁻` | 624–844 | ❌ 深度 3 | f₂→σσ 是三层 | ❌ | 原理性缺口 |
| 12 | `ψ→γφφ` | 1384 | — | **文档里没有公式** | ❌ | 无内容 |

### 2.1 例 1、2：「置换求和」机制已验证（2026-10-20 更新）

文档的 `U_ρ^μ` 是三项之和（`ψ→π⁺π⁻π⁰`，文档 206–219 行）：

$$
U_\rho^\mu = \epsilon_{\mu\nu\lambda\sigma}p_1^\nu p_2^\lambda p_3^\sigma
  \bigl[B_1(Q_{\psi\rho 3})f^{(\rho)}_{(12)}B_1(Q_{\rho 12})
      + B_1(Q_{\psi\rho 2})f^{(\rho)}_{(13)}B_1(Q_{\rho 13})
      + B_1(Q_{\psi\rho 1})f^{(\rho)}_{(23)}B_1(Q_{\rho 23})\bigr]
$$

ρ 可以来自 12、13、23 **任意一对**。而框架的模型是**一棵树 = 一条链**：

```
$ 用 psi2s -> rho + Pip3, rho -> Pip + Pim 建树
OK   末态=3  自动枚举 1 条 [(1, 1)]
```

只覆盖第一项。**2026-10-20 复测更新**：此前判定的两个方案与结论如下 ——
方案 A **可行，不需要新基建**；"与文档物理不同"这个顾虑**不成立**。

* **方案 A（推荐，已实测）**：拆成三条链（`ρ→π₁π₂` / `ρ→π₁π₃` / `ρ→π₂π₃`），
  各自一条道。原来担心"耦合常数由拟合独立给出 `≠` 文档三项共用一个
  `f^{(ρ)}`"，实测证明**这个担心是错的**：
  - 三条道**各自**都等于 `s_i · c · ε_{μνλσ}p₁ᵛp₂ᵛp₃ᵛ · [B1(Q_ψρj)B1(Q_ρjk)]`，
    `c` 是**全局**复常数（逐事例跨度 ≤ 1e-13），即三条道共用同一个 `c`；
  - 三条道相加，与文档的括号 `Σ_i B1(Q_ψρi)B1(Q_ρi)` 残差 **4.6e-16**；
  - `f^{(ρ)}`（BW 传播子）本来就不在这三条道里 —— 框架把它放在道外
    （`g_i` / 拟合参数），三条道**自然共用**同一个 `f^{(ρ)}`，与文档一致；
  - `check` 三体拓扑通过；`emit` 的 `helicity_cascade` basis 产物与直接
    调内核**逐位相同**。

  ⚠️ 唯一的坑：**树的子节点顺序会整体翻号**
  （`rho -> (Pip1, Pi0)` 与 `rho -> (Pi0, Pip1)` 差一个负号，实测
  −0.191836 vs +0.191836）。这是 ε 张量指标配对顺序导致的，不是物理差异；
  三条道顺序若不一致，相加会得到 `B12 − B13 + B23`。统一顺序或用 `g_i = ±1`
  吸收即可。

  ⚠️ 另一个限制：`covariant`/`helicity` basis **emit 不出来**，因为
  `_emit_covariant` 要求 `agent/tensor/chains/<chain>/<channel>.py` 里有
  **手写**道函数（按 `def <name>` 取函数体内联）。三体置换道没有手写函数，
  只有 `helicity_cascade` 能 emit —— 而它正是三体该用的内核，所以不算缺陷。

* **方案 B**（仍可选，但不再必要）：让 channel 函数内部自己求和。要给 spec 加
  一个「非树」标记，且 `allowed_ls` 的宇称+三角枚举对多中间态不再唯一。
  既然方案 A 已经够用，**不建议**走这条。

  ⚠️ **第三个限制（J=3）**：`ε·B` 那个恒等式只对 J=1 成立。J=3 时文档的
  `T̃⁽³⁾` 是秩 2 张量，`ε·p_ψ` 缩并后**不能**化成单个 `ε_{μνλσ}p₁p₂p₃`
  （实测残差 1.04，结构性而非数值噪声）。ρ₃ 只验置换协变性。

  🐛 **第四个限制（omega 静默错，2026-10-20 发现）**：`omega_from_momenta`
  只吃 **base chain**，返回**一份逐道共用**的 `Omega`；置换族各道树不同
  却共用它，非 base 树的那几条道会被**静默算错**（`ex1_pipipi0_rho`：
  `u_rho_12` 残差 2.4e-16 对，`u_rho_13`/`u_rho_23` 残差 0.30/0.27 错；
  emit 不报错）。判据是「**是否存在道其 tree 与 base 树不同**」——
  ❌"顶层两子节点都衰变"会误伤 `ex2_kkpi0_rho`（三体但单道，omega 正确，
  残差 3.3e-16）；❌"深度 ≤ 2"拦不住三体。`spec.py` 已按正确判据拦截。

**已落地**（2026-10-20）：

| spec | 族 | 道数 |
|---|---|---|
| `ex1_pipipi0_rho.toml` | 例 1 ρ(1⁻) | 3（`{12,13,23}` 同号） |
| `ex1_pipipi0_rho3.toml` | 例 1 ρ₃(3⁻) | 3（`(+,−,−)`） |
| `ex2_kkpi0_rho.toml` | 例 2 KK̄ 共振 | 1（单项） |
| `ex2_kkpi0_kstar.toml` | 例 2 Kπ 共振 | 2（同号） |

验收：`agent/tensor/tests/test_ex12_permutation.py`（8/8）。

ρ₅(5⁻) / K\*₅(5⁻) / K\*₃(3⁻) 也已实现（2026-10-20）。J/l 上限从 4 扩到 6，
**两条路线都扩了**：

* **级联路线**：`LAM_SLOTS` 加 5/6、`Bl` 加 l=5 系数
  （`cg`/`small_d_weight`/`D_lambda` 本来就通用）；
* **协变张量路线**：`covariant.py` 补了论文 Eq.(258) 的通用 `t_l` 与
  `B5`/`B6`（原来只有写死的 `T1_cov..T4_cov`）。`t_l(1..4)` 与写死的
  四个函数**逐位一致**，所以这是"扩能力"不是"改公式"。

上限取 6（而不是更大）是因为 `small_d_weight` 的阶乘在 float64 下溢出
（J≈50 时直接 NaN）。

外部验证：ρ₅ 的协变张量振幅与 **tf-pwa** 对拍，overlap=1.00000000、
残差 2e-15（配对在壳；口径差异见下）。见
`agent/tensor/tests/test_j5_vs_tfpwa.py` 与 `test_covariant_general.py`。

⚠️ **位垒口径差异（既有，不是 bug）**：tf-pwa 的
`has_barrier_factor=False` 只关 `B_l`，仍然乘 `q^l`；而论文的
`t~^(l)` 是 `Q_abc^l B_l(Q_abc)`，本仓库的 `t_l` 只含 `B_l`，`q^l` 那一半
由拟合参数 `g_i` 吸收。所以与 tf-pwa 对拍时要把配对质量放在壳，
让这个因子退化成常数。

复现方式见
[`analyses/tensor_examples/README.md`](../analyses/tensor_examples/README.md)
的「例 1/2 的置换求和」一节。

### 2.2 例 7–11：辐射衰变需要新基建

文档明确说辐射衰变**不能**直接用 L-S 方案选独立振幅（843 行）：

> 对于 `ψ→γf₀`，两个振幅（L=0 与 2）在规范不变性下给出**相同**的角分布……
> 对于 ψ 辐射衰变，L-S 方案在选择独立振幅时不再有用。

我实测确认了这个陷阱：

```
ψ→γf₂  框架枚举 5 条 [(0,1),(2,1),(2,2),(2,3),(4,3)]
       文档：规范不变后只剩 3 个独立螺旋度振幅（1388 行）
```

所以**不能**把 `allowed_ls` 直接用在辐射例子上，否则会写出 5 条线性相关的振幅。

需要的新基建：

1. **秩 2 振幅** `U^{μν}`：多一个洛伦兹轴（光子）。现有 `_s1_vector` 返回 `(N,4)`，
   只有一阶。
2. **横向投影** `g^{(⊥⊥)}_{νν'}`：Coulomb 规范下对未观测光子螺旋度求和
   （文档 806–813 行）。
3. **`S_{μν} = ε_{μναβ}p_ψ^α q^β`** 与 **`K = p_ψ − q`**。
4. **按螺旋度数（而非 (l,s)）校验独立振幅数**。

好消息：`leci_four()` / `_leci_con()`（ε 张量）与 `Gt_cov` / `Gt_con`（$\tilde g$）
都已存在，可直接复用。

### 2.3 例 4、6：**2026-10-21 已补上**（曾判为原理性缺口）

两条曾经的"原理性缺口"，本轮都补掉了 —— 但补的方式不同，值得记清楚，
因为**其中一条暴露了一个真缺陷**（见 §2.3.3）。

#### 2.3.1 例 6：`深度 3` 不是真阻塞（当初判错了）

当初的理由是「`f₂→σσ` 使链变成三层，被深度检查拒绝」。**这条已经过时了**：

* `spec.py` 的深度上限**按表象分级**：只有 `omega` 表象拒绝深度 > 2
  （它逐节点读树、只算 `D0*D1*D2`）；`covariant` / `helicity` /
  `helicity_cascade` **完全不读树**，三层、四层都能算。
* 所以例 6 从来不需要"改内核的树模型"。

例 6 有**三条子分支**，拆成**三份 spec**（一份一个家族，共 38 条道）：

| spec | 子分支 | 文档行 | 道数 | A1 |
|---|---|---|---|---|
| `ex6_phi4pi_ss.toml` | $f_J\to\sigma\sigma$ | 630–674 | 14 | ✅ **7.35e-15** 全过 |
| `ex6_phi4pi_rr.toml` | $f_J\to\rho\rho$ | 676–730 | 14 | **混合**：$f_0$ 4 条 ✅ 5.67e-15；$f_2$ 10 条 ⚠️ 0.47–0.86 |
| `ex6_phi4pi_f2s.toml` | $f_2\to f_2(1270)\sigma$ | 732–775 | 10 | ⚠️ 0.47–1.08 全不过 |

**为什么拆开**：A1 是**逐道**判据。三条分支状态不同，混在一份 spec 里
"这份 spec 过不过 A1"就成了没法回答的问题 —— 报告里只能写"部分过"，
极易读错成"可用"。拆开后每份只有一个答案。

#### σσ 为什么精确

$\sigma$ 是标量，所以 $T^{(f_2)\mu\nu}_{(\sigma\sigma)}$ 就是普通的两体
$\tilde t^{(2)}$：`T2_ss(frame, a1, a2, b1, b2) = T2_cov(frame, a1+a2, b1+b2)`。
14 条道 worst = 7.35e-15。

#### ρρ 为什么一半过一半不过（本轮的关键发现）

这不是实现差异，而是**自旋结构**不同：

| | $f_0\to\rho\rho$ | $f_2\to\rho\rho$ |
|---|---|---|
| 需要的耦合 | $1\otimes1\to 0$（两个 $\rho$ 合成 $S=0$） | $1\otimes1\to 2$ |
| 正确写法 | $\tilde g(p_{f_0})^{\alpha\beta}\tilde t^{(1)}_{(34)\alpha}\tilde t^{(1)}_{(56)\beta}$ | $P^{(2)\mu\nu\alpha\beta}(p_{f_2})\tilde t^{(1)}_{(34)\alpha}\tilde t^{(1)}_{(56)\beta}$ |
| 落点 | `S_rr` | `T2_rr` |
| A1 | ✅ **5.67e-15** | ⚠️ 0.47–0.86 |

$f_0$ 那条**根本不需要 $P^{(2)}$** —— 它的横向投影已经把纵向分量排除，
所以不存在"无迹化在哪个系做"的问题。$f_2$ 要真的把两个秩-1 耦合成秩-2，
这时 $P^{(2)}(p_{f_2})$ 作用在**实验室系**张量上，缺少"先 boost 到 $f_2$ 系、
去迹、再 boost 回来"这一步的等价物。

> **一个易错的坑**：$f_0\to\rho\rho$ 如果写成**裸点积**
> $g^{\alpha\beta}u_\alpha v_\beta$（不夹 $\tilde g(p_{f_0})$），
> A1 残差是 1.02e-1 —— 看着"接近了"，其实是把 $\rho$ 的纵向分量算了进去。
> 夹上横向度规才是 5.67e-15。`tests/test_ex6_branches.py` 钉住了这一条。

#### f₂(1270)σ 为什么全不过

$2\otimes0\to2$：$f_2(1270)$ 的 $\tilde t^{(2)}$ 已经是秩 2，再与标量
$\sigma$ 耦合成 $J=2$ 时同样的 $P^{(2)}$ 参考系问题。A1 = 0.47–1.08。

#### 逐条实测（$f_2$ 的 20 条，**全部不过**）

| 道 | 01 | 21 | 22 | 23 | 43 |
|---|---|---|---|---|---|
| ρρ，配对 (34)(56) | 0.83 | 0.86 | 0.53 | 0.63 | 0.84 |
| ρρ，配对 (36)(45) | 0.66 | 0.62 | 0.47 | 0.52 | 0.72 |
| f₂(1270)σ，配对 (34)(56) | 0.98 | 0.99 | 1.08 | 1.06 | 0.98 |
| f₂(1270)σ，配对 (36)(45) | 0.60 | 0.47 | 1.05 | 0.56 | 0.66 |

**不是常数倍的差别** —— 若是整体归一化问题，逐道全局复常数 $|c|$ 会吸收掉。
试过但都不行的变体（残差 0.19–0.57）：加 $\tilde g(p_{f_2})$ 横向投影、
对称化 $u_\mu v_\nu+v_\mu u_\nu$、`T2_from_pair`、把 $P^{(2)}$ 换成
$f_2$ 系里的无迹化。

**这不是"文档错了"，是"我们还不知道怎么把它写成协变量"。**
公式已按文档 / tex 逐式直译，并由
`tests/test_ex6_branches.py::test_primitives_match_document` 钉住
（独立实现 vs 内核，rel 3.0e-16）。

#### 与原始 tex 的交叉核对（2026-10-21）

md 是从 tex 转写的，转写可能出错，所以所有公式都回到
**`documentation/jpsipwa1.tex`**（原始投稿文件，1594 行）再核对一遍：

| 内容 | tex 行 | md 行 | 一致？ |
|---|---|---|
| 例 6 引言（三条子分支） | 708–715 | 626 | ✅ |
| $\sigma\sigma$ 七条分波振幅 | 716–753 | 628–664 | ✅ 逐式 |
| $T^{(f_2)}_{(\sigma\sigma)}$ 定义 | 754–759 | 669–674 | ✅ |
| $\rho\rho$ 七条分波振幅 | 760–798 | 680–718 | ✅ 逐式 |
| $T^{(f_2)}_{(\rho\rho)}$ 定义 | 803–808 | 724–730 | ✅ |
| $f_2(1270)\sigma$ 五条分波振幅 | 810–837 | 736–758 | ✅ 逐式 |
| $T^{(f_2)}_{(f_2\sigma)}$ 定义 | 838–850 | 764–775 | ✅ |
| 例 4 的 $\omega^\mu$ | 493–507 | 418–429 | ✅ 逐式 |
| 例 3 $<\phi f_J|LS>$ 的 (l,s) 计数 | 376–492 | 307–369 | ✅ |

结论：**md 转写与 tex 一致**，本轮 20 条失败的道不是转写问题。
（核对方法：逐条 `sed -n '<行范围>p'` 对照，记号如
$\tilde t^{(1)\alpha\beta}$、$Q_{\omega\rho 0}$、$P^{(3)\mu\alpha\beta\gamma\delta\nu}$
逐字符比对。）

#### 这 20 条道怎么记账（关键）

**不是"缺道"** —— 道在、能 `emit`、L1/L2/L3 全过。所以不能用
缺道的方式记录，否则 `emit` 成功会让人以为可用。做法：

| 位置 | 表名 | 语义 |
|---|---|---|
| `probes/verify_all_specs.py` | `A1_KNOWN` | A1 单独记账，逐条打印 `A1_known_<道名>` |
| `agent/tensor/tests/test_handwritten_complete.py` | `A1_EXEMPT` | 同上，且断言豁免项**确实**失配 |

两处都要求"豁免项必须确实失配"：哪天修好了会**失败**并提醒删表 ——
免得豁免变成永久免罪符。另外 `test_handwritten_complete.py` 还断言
每条豁免都**真的被走到**（拼错道名会让豁免静默失效）。

⚠️ 必须与两种情况区分开：

* `A1_NO_CASCADE_SPECS`（`ex4_omega_kk`）：**没有级联产物**，A1 无对象；
* 真失配（本节的 20 条）：有产物，但对不上。

三者混在一起会把缺陷藏掉。

**⚠️ 不要拿这 20 条道去拟合。** 它们的作用是：(a) 把文档公式钉在代码里，
改公式时立刻能看出影响；(b) 给未来修无迹化参考系留下对照基线。

#### 2.3.2 例 4：能算，但靠的是手写原语而非树模型

当初的理由「ω 由三个 π 重建，实测 `粒子 omega 有 3 个 daughter，
只支持两体衰变`」—— **这条仍然成立**，只是它不再阻塞了。

关键洞察：$\omega^\mu$ 里的 $\epsilon^{\mu\nu\lambda\sigma}$ 缩并的是
$p_1,p_2,p_0$ **三个末态动量**。硬写成两体级联
（`ω→ρ+π⁰`、`ρ→π⁺π⁻`）时，那个节点的 ε 变成

$$
\epsilon^{\mu\nu\lambda\sigma}p_\omega{}_\nu p_\rho{}_\lambda p_0{}_\sigma=0
\quad(\text{因为 }p_\omega=p_\rho+p_0)
$$

实测 $|\epsilon(p_\rho,p_\pi,p_\omega)|<10^{-17}$ —— **恒为零**，不是数值小。
所以"用两体树拼出三体节点"在原理上不通。

补法是**加一个手写原语** `kernels/covariant.py::omega_mu`，与文档公式
逐位相同（rel = 0.00e+00）。`specs/ex4_omega_kk.toml` 出 7 条道
（$f_0$ 的 $01/21$ + $f_2$ 的 $01/21/22/23/43$），只有 `covariant` /
`helicity` 两种表象。验收：

| 检查 | 结果 |
|---|---|
| L1 covariant / helicity | 0.00e+00 / 1.15e-16 |
| L2 | 0.00e+00 |
| A1 | **不适用**（没有级联产物 —— 这正是缺陷） |

spec 里的树写成二叉**占位** `omega -> rho + Pi0`、`rho -> Pip + Pim`，
只为让 `validate` 的两条检查（两体、叶子 == `[input].keys`）通过。
**道函数完全不读这棵树**，所以占位不改任何数值。

#### 2.3.3 这条缺陷本身：缺一个真正的三体节点

例 4 能算，但**不是靠树模型算的** —— 靠的是"把三体对象整个塞进一个手写
道函数"。这就是缺口，逐条记在 §2.5。

### 2.4 例 12：文档里是空的

```
$ sed -n '1384,1385p' 文档
### $\psi\to\gamma\phi\phi\to\gamma K^+K^-K^+K^-$$
$
```

只有标题。**无从实现**，可能是原稿遗留。

### 2.5 缺陷：没有真正的三体节点（2026-10-21 记录）

上一节说例 4 "能算" —— 但它是**绕过**树模型算的，不是树模型支持了三体。
这是目前最实的一个缺陷，逐条写清楚。

#### 2.5.1 症状

`spec.py` 的 `validate` 里，`walk` 函数硬性要求每个衰变节点**恰好两个**
daughter：

```python
if len(ds) != 2:
    raise SpecError("粒子 %s 有 %d 个 daughter，只支持两体衰变" % (n, len(ds)))
```

实测：

```
$ # 把 ex4 spec 的占位树改成真三体
$ omega -> ["Pi0", "Pip", "Pim"]
SpecError: 粒子 omega 有 3 个 daughter，只支持两体衰变
```

同时 `_check_mirror_tree` 也有一份同样的检查（镜像树覆盖后也要两体）。

#### 2.5.2 为什么不能"用两体树凑"

三体节点不能分解成两体节点的**串联**，因为 $\epsilon$ 结构会退化：

$$
\epsilon^{\mu\nu\lambda\sigma}p_\omega{}_\nu p_\rho{}_\lambda p_\pi{}_\sigma=0
\qquad(p_\omega=p_\rho+p_\pi)
$$

实测 $|\epsilon^{\mu\nu\lambda\sigma}p_{\rho\nu}p_{\pi\lambda}p_{\omega\sigma}|
\approx10^{-17}$ —— 就是浮点零（$O(10^{-17})$ 是四动量本身的求和误差）。
而真三体的 $\epsilon^{\mu\nu\lambda\sigma}p_1{}_\nu p_2{}_\lambda p_0{}_\sigma$
是 $O(1)$。**这不是数值精度问题，是恒等式。**

也不能分解成"两体节点的并联"（三条置换道各写一条）：那些道里的 $\epsilon$
仍然只缩并**两个**末态动量，缺第三个。

#### 2.5.3 现在的绕过方式及其代价

| 绕过方式 | 用在哪 | 代价 |
|---|---|---|
| 手写道函数内置三体分析器 | 例 4（`omega_mu`）、例 1/2（各自的 `u_rho_*` 里内联） | **每个三体对象一个手写函数**；新例子要重写，不能复用 |
| 二叉占位树 | 例 4 的 spec | spec 不再描述真实拓扑；`validate` 的两体检查形同虚设 |
| 只出 `covariant`/`helicity` | 例 4 | **`helicity_cascade` 用不了**，于是没法用"级联 vs 协变"这条最有力的交叉验证（A1） |

第三条是最贵的：例 4 现在只有 L1/L2/L3，**没有 A1**。A1（级联逐道 == 协变
到一个全局复常数）是发现公式错误的唯一有效手段 —— 本轮例 6 的 $\rho\rho$
问题就是靠 A1 发现的。

#### 2.5.4 缺什么才算"真三体节点"

要让 `helicity_cascade` 支持 `omega -> Pi0 + Pip + Pim`，需要三样：

1. **树模型放宽**：`walk` / `_check_mirror_tree` / `cascade_channels`
   允许 `len(ds) >= 2`，并把"节点"从"两体"泛化成"n 体"。
2. **逐节点振幅**：`helicity_cascade.node_amplitude` 现在的签名是
   `A[lam_a, lam_b, lam_c]`（三个指标），三体要 `A[lam_a, lam_b, lam_c, lam_d]`，
   CG 组合从 $1\otimes1\to J$ 变成 $1\otimes1\otimes1\to J$。
3. **角分布**：现在的 `D_lambda` 是"父系里一个相对方向"，三体要
   **两个独立的相对方向**（三个末态有 2 个自由度），对应两个 $D$ 函数的
   乘积与一套新的欧拉角定义。

这是一次真正的内核改造，不是补几个函数。**建议在需要第二个三体例子时再做**
—— 目前只有例 4 一个，手写原语的边际成本还低。

#### 2.5.5 建议的记录方式

* 任何新的三体例子 → 先在本文档记一行"需要手写原语"；
* `specs/ex4_omega_kk.toml` 里的占位树注释必须保留（它标记了这个缺陷）；
* `probes/verify_all_specs.py` 对 `ex4_omega_kk` 只跑 L1/L2/L3，
  **A1 缺失要在输出里显式说明**，不能默默跳过。

已经钉住的断言（哪天缺陷修了，这些会失败并提醒更新本文档）：

| 断言 | 位置 | 若失败意味着 |
|---|---|---|
| 3 个 daughter 必须 `SpecError` | `tests/test_spec.py::test_three_daughter_node_rejected` | 三体节点已实现 → 改测试 + 更新 §2.5 |
| 两体节点的 ε 恒为零 | `tests/test_threebody_node.py::test_epsilon_degenerates_for_two_body_node` | 运动学退化，测试无意义 |
| `omega_mu` == 文档公式 | `tests/test_threebody_node.py::test_omega_mu_matches_document_formula` | 手写原语写错了 |
| `ex4_omega_kk` 没有 `helicity_cascade` 表象 | `tests/test_handwritten_complete.py`（`A1_NO_CASCADE_SPECS`） | 例 4 能被级联算了 → 缺陷已修 |

## 3. 已经验证过的两件事

在动手补例子之前，先确认了框架与文档是自洽的。

### 3.1 文档的核心恒等式成立

文档 193–196 行给出 $\tilde t^{(L)}$ 与投影算符的关系：

$$
\tilde t^{(L)}_{\mu_1\cdots\mu_L}
= (-1)^L P^{(L)}_{\mu_1\cdots\mu_L\mu'_1\cdots\mu'_L}
  r^{\mu'_1}\cdots r^{\mu'_L}B_L(Q_{abc}), \qquad r = p_b - p_c
$$

拿 `kernels/covariant.py` 的 `T1_cov`–`T4_cov` 与 `P2`–`P4` 对拍
（随机四动量，`a` 取 ψ 静止系）：

```
L=1  (-1)^1 P1.r.B1   vs T1_cov : 0.0
L=2  (+1) P2.rr.B2    vs T2_cov : 4.44e-16
L=3  (-1)^3 P3.rrr.B3 vs T3_cov : 3.55e-15
L=4  (+1) P4.rrrr.B4  vs T4_cov : 0.0
```

另外文档 137 行说 $P^{(L)}$ 是「无迹」的，也验证了：

```
P2 无迹 max|g_ij P2^ijkl| = 1.11e-16
P3 无迹 max               = 1.39e-17
P4 无迹 max               = 8.33e-17
```

**结论**：文档这套形式体系与已移植的内核完全一致，新例子可以照文档写。

### 3.2 独立振幅数对得上

用现有的 `allowed_ls()`（宇称 + 角动量三角）与文档声明的数目比对：

| 过程 | 文档声明 | 框架枚举 |
|---|---|---|
| `<φf₀>` | 2 | ✅ 2 `[(0,1),(2,1)]` |
| `<φf₂>` | 5 | ✅ 5 `[(0,1),(2,1),(2,2),(2,3),(4,3)]` |
| `<K*K̄*>` | 10, 11, 12, 32 | ✅ 4 条，正好这四组 |
| `<K*K*₀>` | 12, 32 | ✅ 2 |
| `<K*₀K*'₀>` | 10 | ✅ 1 |

## 4. 一个需要澄清的边界：深度 2 限制是 omega 专属

`spec.validate` 会拒绝级联深度 > 2 的树，理由是
「runtime 只算 `D0*D1*D2`」。但实测表明**这条限制只对 omega 内核成立**：

```
_phif243 参数: (psi, phi, f2, Kp, Km, Pip, Pim)
内部是否引用 chain/nodes: False
```

协变张量的道函数是**按道名硬编码的公式**，完全不读树结构；只有
`kernels/omega.py` 的 `chain_angles` / `D_lambda` 才依赖树。

这意味着：例 5 里 `PP/PF/FP/FF` 那些函数（内部是
`psi→K*₂+π, K*₂→φ+π, φ→K⁺K⁻`，深度 3）**用协变路线是可以算的**，
只是当前 TOML 会因为深度检查被拒。

**✅ 已实现**（2026-10-19）：深度检查改为按 `[[basis]]` 分级 —— 只含
`cartesian`/`helicity` 时放宽，含 `omega` 时维持拒绝。这与
「`J≤4`/`l≤4` 是 omega 专属约束」是同一类问题。实现与判据见
[`tensor_cache_generation.md` §10](tensor_cache_generation.md#10-深度限制按表象分级2026-10-19-修正)。

判据（`agent/tensor/tests/test_ex3_channels.py::test_depth_grading`）：
三层树 `ψ→ρ′π, ρ′→φπ, φ→K⁺K⁻` 在 cartesian/helicity 下读入成功，
追加一个 `[[basis]] name="omega"` 后被拒。

## 5. 建议的推进顺序

1. ~~**移植例 3 的 ρ′/b₁ 道**（Eq.46–50，10 个函数）。~~
   **✅ 已完成**，见 §7。
2. ~~**移植例 5 的 K*₀/K*₂ 道**（10 个函数）。~~
   **✅ 已完成**，见 §8。
3. ~~定「置换求和」模型（方案 A 还是 B）~~ —— **已完成**：方案 A 可行且已落地
   （§2.1，2026-10-20 实测残差 5.2e-15）。例 1、例 2 的四份 spec 与
   `agent/tensor/tests/test_ex12_permutation.py` 已入库。
4. **辐射衰变单独立项**：秩 2 振幅 + 横向投影 + 螺旋度数校验。
   例 7（两标量）是最小可做单元。

## 6. 复现本文结论

```bash
cd <LLMPWA>

# 拓扑试读（各例的树描述）
python -m agent.tensor check <spec.toml>

# 文档恒等式对拍（§3.1）
python -c "..."          # 见本文正文的 einsum

# 参考实现试调（§1）
./analyses/kk_dis/.venv/bin/python -c "
import sys, numpy as np, jax.numpy as jnp
T='analyses/kk_dis/task/20260929_螺旋度耦合振幅推导/1_运行代码/tensor_example'
sys.path.insert(0, T + '/Tensor')
import FUNC; f = FUNC.FUNCTION(); f.Modifiers()
z = np.load('analyses/kk_dis/tmp/bundledata/route1_cartesian/real_data/Momentum_kk.npz')
Kp,Km,Pip,Pim = [jnp.asarray(np.asarray(z[k])[:32]) for k in ('Kp','Km','Pip','Pim')]
phi, f2 = Kp+Km, Pip+Pim
print(np.asarray(f.u_rho_1(phi,Kp,Km,Pip,Pim)).shape)   # (32, 4)
"
```

## 7. 例 3 的落地结果（2026-10-19）

按 §5 第 1 条执行完毕。产物：

| 位置 | 内容 |
|---|---|
| `agent/tensor/chains/rhopi/` | `u_rho_1.py` `u_rho_2.py`（Eq.46） |
| `agent/tensor/chains/b1pi/` | `u_{SS,SD,DS,DD}_{1,2}.py`（Eq.47–50），共 8 条 |
| `analyses/tensor_examples/specs/` | `ex3_rhopi.toml` `ex3_b1pi.toml` |
| `analyses/tensor_examples/data/make_momenta.py` | 合成 `ψ→φπ⁺π⁻→(K⁺K⁻)(π⁺π⁻)`，自带守恒/在壳/阈值自检 |
| `analyses/tensor_examples/check_ex3.py` | 端到端回归（生成 → 跑 → 三层自检） |
| `agent/tensor/tests/test_ex3_channels.py` | 8 项纯 numpy 自洽性测试 |

**对拍结果**（`TENSOR_ROOT=... python agent/tensor/tests/test_covariant_vs_jax.py`）：

```
=== 例 3 的 rho'/b1 十条道：numpy 移植 vs jax 原版 ===
  u_rho_1  rel = 3.762e-15  OK      u_DS_1   rel = 2.558e-16  OK
  u_rho_2  rel = 1.799e-16  OK      u_DS_2   rel = 9.711e-16  OK
  u_SS_1   rel = 1.239e-16  OK      u_DD_1   rel = 8.798e-16  OK
  u_SS_2   rel = 1.203e-16  OK      u_DD_2   rel = 1.166e-15  OK
  u_SD_1   rel = 1.756e-16  OK
  u_SD_2   rel = 3.907e-16  OK
PASS：全部 24 个对象与原版一致（rel < 1e-12）
```

移植过程中确立的三条约定（已写进 `chains/*/__init__.py` 与 §4）：

1. **`_1`/`_2` 的下标在两个族里含义相反**（照抄参考实现的命名）：
   ρ′ 族 `_1` 的旁观者是粒子 3（π⁺），b₁ 族 `_1` 的旁观者是粒子 4（π⁻）。
2. **`(l,s)` 不是 b₁ 族的完备标签**：8 条道只映射到 2 组 `(l,s)`，
   所以 `check_channels = false`，且不能用 omega 表象。
3. **两份 spec 而不是一份**：ρ′ 与 b₁ 是两棵不同的树，一棵树只能描述一组配对。

另外实测确认（并已修）：`_momenta()` 那种 `ψ→φf₂→(K⁺K⁻)(π⁺π⁻)` 的四动量
**恰好**满足 `φ + π⁺ + π⁻ = ψ`，所以 `ρ′ = φ+π⁻`、`b₁ = φ+π⁺` 都是真两体衰变，
`Qabc > 0`，不需要另写相空间抽样。分析目录里的 `make_momenta.py` 仍独立生成
一批（`m(π⁺π⁻)` 均匀扫描），用于端到端回归。


## 8. 例 5 的落地结果（2026-10-19）

按 §5 第 2 条执行完毕。产物：

| 位置 | 内容 |
|---|---|
| `agent/tensor/chains/kstpi/` | 10 个道文件（`u_kst0_l/r` + `u_PP/PF/FP/FF_{1,2}`） |
| `analyses/tensor_examples/specs/ex5_kst2.toml` | K*₂ 树，8 条道，3 个表象 |
| `analyses/tensor_examples/specs/ex5_kst0.toml` | K*₀ 树，2 条道，3 个表象 |
| `analyses/tensor_examples/probes/verify_ex5_port.py` | 与 jax 逐式对拍（10/10，rel ≈ 1e-15） |
| `analyses/tensor_examples/probes/probe_ex5_identify.py` | 级联可表示性穷举 + 阳性对照 |
| `agent/tensor/tests/test_ex5_channels.py` | 纯 numpy 回归（6 项） |

### 8.1 结论

* **能算**：10 条道全部移植成功，`check` / `emit` / 端到端跑产物都通过。
  所以例 5 在**协变路线**上是完全可实现的。
* **拓扑**：$\psi\to K^*\pi\to(\phi\pi)\pi\to(K^+K^-)(\pi\pi)$，
  **深度 3**，与例 3 的 b₁ 同构。
* **必须拆两份 spec**：$K^*_0$ 的 $J=0$ 与 $K^*_2$ 的 $J=2$ 在一个
  `[particles]` 条目里冲突。
* **级联可表示性**：按链算是 `ex5_kst2` **4/8**、`ex5_kst0` **2/2**；
  合起来 **6/10**（探针输出的是后者，别与单链混）。
  命中的是内层 F 波的四条（`u_PF_*`、`u_FF_*`）加 $K^*_0$ 的两条；
  `u_PP_*`/`u_FP_*` 落在级联子空间之外（rel ≈ 6e-3 / 2e-1）。
  **阳性对照**：同一探针跑 b₁ 得 8/8。

### 8.2 两个坑（都踩过）

1. **jax 默认 float32**。移植对拍第一次 10/10 "失败"（rel 1e-6…1e-8），
   看着像移植错，其实是精度。必须在**导入 `FUNC` 之前**
   `jax.config.update("jax_enable_x64", True)`。同一个坑也污染了级联探针，
   还伪造出一个"`_1` 族全败、`_2` 族全中"的假不对称。
2. **K*₀ 的宇称**。$J=0$ + 宇称守恒 + 三角只允许 $l=s=1$，对应 $0^-$；
   而常见的 $K^*_0(1430)$ 是 $0^+$。所以这条道描述的**不是** $K^*_0(1430)$。
   疑点未解决。

### 8.3 遗留

| 项 | 状态 |
|---|---|
| `u_PP`/`u_FP` 为何不是级联 | ❓ 未解决（都含 $P^{(2)}_\psi$ 投影 + 内层 P 波） |
| 与文档 Eq 的逐条对应 | ⚠️ 未敲定（以参考实现为准移植） |
| $K^*_0$ 的 $0^-$/$0^+$ 归属 | ❓ 未解决 |
| 真实样本 / 拟合侧 | ❌ 未做 |

详见 [`../analyses/tensor_examples/chains/ex5_kst2.md`](../analyses/tensor_examples/chains/ex5_kst2.md)
与 [`ex5_kst0.md`](../analyses/tensor_examples/chains/ex5_kst0.md)。
