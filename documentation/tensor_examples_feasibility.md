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
| **不做** | 例 4 `ψ→ωK⁺K⁻`、例 6 `ψ→φπ⁺π⁻π⁺π⁻` | 三体衰变 / 深度 3，原理性缺口 |
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

### 2.3 例 4、6：原理性缺口

* **例 4**（`ψ→ωK⁺K⁻`）：文档要求把 `<φf_J|LS>` 里的 $\tilde t^{(1)\mu}_{(12)}$
  替换成 $\omega^\mu$（423–427 行），而 ω 由 **三个** π 重建。
  实测：`粒子 omega 有 3 个 daughter，只支持两体衰变`。
* **例 6**（`ψ→φπ⁺π⁻π⁺π⁻`）：`f₂→σσ`、`f₂→ρρ` 使链变成三层
  （`ψ→φf₂ → f₂σσ → σππ`）。实测被深度检查拒绝。

这两条不是「补几个函数」能解决的，需要改内核的树模型。

### 2.4 例 12：文档里是空的

```
$ sed -n '1384,1385p' 文档
### $\psi\to\gamma\phi\phi\to\gamma K^+K^-K^+K^-$$
$
```

只有标题。**无从实现**，可能是原稿遗留。

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
