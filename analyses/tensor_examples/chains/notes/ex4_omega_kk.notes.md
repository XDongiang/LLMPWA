### 这条链是什么

例 4 `ψ(2S)→ωK⁺K⁻→(π⁺π⁻π⁰)K⁺K⁻`（文档 418–430 行）。

| | 本链 |
|---|---|
| 末态 | **5 个**（ω 下面 3 个 π + KK 一对） |
| 文档给出的层级 | ω 是**单节点三体**；KK 那一对是 f₀ 或 f₂ |
| 可用的表象 | **只有** `covariant` / `helicity` |

### 关键：ω^μ 不是两体级联

文档的写法里，ω 那部分是一个**整体分析器**：

$$
\omega^\mu=\epsilon^{\mu\nu\lambda\sigma}p_1^\nu p_2^\lambda p_0^\sigma
\Bigl[B_1(Q_{\omega\rho0})f^{(\rho)}_{(12)}B_1(Q_{\rho12})
      +B_1(Q_{\omega\rho2})f^{(\rho)}_{(10)}B_1(Q_{\rho10})
      +B_1(Q_{\omega\rho1})f^{(\rho)}_{(20)}B_1(Q_{\rho20})\Bigr]
$$

三条方括号是 $(\pi^+\pi^-)$、$(\pi^+\pi^0)$、$(\pi^-\pi^0)$ 三对配成 ρ 的
**置换求和**。这与例 1 的三条置换道结构相同，但有一条决定性的差别：

> **例 1 可以被两体树表达，例 4 不能。**

原因是 $\omega^\mu$ 里的 $\epsilon^{\mu\nu\lambda\sigma}$ 缩并的是
$p_1,p_2,p_0$ **三个末态动量**。如果硬写成两体级联
（`ω → ρ + π⁰`、`ρ → π⁺π⁻`），那个节点的 ε 就变成

$$
\epsilon^{\mu\nu\lambda\sigma}p_\omega{}_\nu p_\rho{}_\lambda p_0{}_\sigma
=0 \quad(\text{因为 } p_\omega=p_\rho+p_0)
$$

实测 $|\epsilon(p_\rho,p_\pi,p_\omega)|<10^{-17}$ —— **恒为零**，不是数值小。
所以"用两体树拼出三体节点"这条路在原理上不通。

### 我们怎么算的

道函数直接调 `kernels/covariant.py` 新增的 `omega_mu` 原语：

| 道 | 公式 | 文档 |
|---|---|---|
| `u_omega_f0_01` | $\omega^\mu$ | 630 / 418–430 |
| `u_omega_f0_21` | $\tilde T^{(2)\mu\nu}_{(\omega f_0)}\omega_\nu$ | 634 |
| `u_omega_f2_01` | $\tilde t^{(2)\mu\nu}_{(34)}\omega_\nu$ | 639 |
| `u_omega_f2_21` | $\tilde T^{(2)\mu\alpha}\tilde t^{(2)}_{(34)\alpha\nu}\omega^\nu$ | 642 |
| `u_omega_f2_22` | 两个 ε 的 $2\otimes2\to1$ | 646 |
| `u_omega_f2_23` | $P^{(3)}$ 的 $2\otimes2\to3$ | 654 |
| `u_omega_f2_43` | $\tilde T^{(4)}$ 的 $L=4$ | 659 |

`omega_mu` 与文档公式**逐位相同**（rel = 0.00e+00，同一个 numpy 表达式）。

### spec 里的树为什么要"占位"

`spec.py` 的 `validate` 硬性要求：

* 每个衰变节点**恰好两个** daughter（`只支持两体衰变`）；
* 树的叶子集合 == `[input].keys`。

三体节点 `omega -> Pi0 + Pip + Pim` 两条都过不了。所以 spec 里写成
等价的二叉占位

```
omega -> rho + Pi0
rho   -> Pip + Pim
```

并把 `Pi0/Pip/Pim` 登记成叶子。**道函数完全不读这棵树**（它们按参数名
拿四动量、调 `omega_mu`），所以占位不改任何数值 —— 但这是"缺一个真三体
节点"这条缺陷的直接症状，见
[`documentation/tensor_examples_feasibility.md`](../../../documentation/tensor_examples_feasibility.md).
的 §「缺陷：三体节点」。

### 验收

`probes/verify_all_specs.py` 对 `ex4_omega_kk` 的结果：

| 检查 | 结果 |
|---|---|
| L1（产物 == 内核） | 0.00e+00 |
| L2（helicity == EPS_A·cartesian） | 0.00e+00 |
| L3（有限、非恒零） | 通过 |
| A1 | 不适用（无级联产物，这正是缺陷） |

### 运动学：为什么用虚拟 m_ω = 1.5

物理上 $m_\omega=0.782 < m_\rho+m_\pi=0.911$，$\omega\to\rho\pi$ **在阈下**，
$Q_{\omega\rho i}^2<0$。位垒因子 $B_1=\sqrt{2/(Q^2+Q_0^2)}$ 里的 $Q^2$ 为负
虽不致命，但 ρ 的两体衰变 $Q^2$ 也贴着 0，$B_1$ 会剧烈起伏。

`data/make_omega5.py` 用 $m_\omega=1.5$（与 `make_threebody.py` 对
$m_\rho$ 用 1.5 是同一套理由），保证所有 $Q^2>0$。
**这只影响演示数据，不影响公式** —— 道函数对任意四动量都成立。
