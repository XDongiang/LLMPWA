# 链 `ex2_kstar5` —— psi2s → kstar5 + Km2 ; kstar5 → Kp1 + Pi0

> **本文件是生成物**，不要直接编辑。
> 事实部分由 [`gen_chain_docs.py`](gen_chain_docs.py) 从 [`analyses/tensor_examples/specs/ex2_kkpi0_kstar5.toml`](../specs/ex2_kkpi0_kstar5.toml) 自动导出；
> 叙述部分来自 [`notes/ex2_kstar5.notes.md`](notes/ex2_kstar5.notes.md)。
> 改完重跑：`python analyses/tensor_examples/chains/gen_chain_docs.py`

---

## 1. 拓扑（自动导出）

```text
psi2s  J=1  P=-1
├── kstar5     J=5  P=-1  → Kp1 + Pi0
│  ├── Kp1        J=0  P=-1   （末态）
│  ├── Pi0        J=0  P=-1   （末态）
├── Km2        J=0  P=-1   （末态）
```

**深度 = 2**（最长路径上的边数）。

## 2. 粒子表（自动导出）

| 粒子 | $J$ | 宇称 | 质量 | 角色 | 允许的 $l$ |
|---|---|---|---|---|---|
| `psi2s` | 1 | -1 | 3.686 | 顶层 | 5 |
| `kstar5` | 5 | -1 | 1.9 | 节点 | 5 |
| `Kp1` | 0 | -1 | 0.493677 | 末态 | - |
| `Pi0` | 0 | -1 | 0.134977 | 末态 | - |
| `Km2` | 0 | -1 | 0.493677 | 末态 | - |

## 3. 道表与逐节点 $(l,s)$（自动导出）

道表来源：**显式 `[[channel]]`**（2 条）。

| 道 | 顶层 $(l,s)$ | `kstar5` |
|---|---|---|
| `u_kstar5_13` | (5,5) | (5,0) |
| `u_kstar5_23` | (5,5) | (5,0) |

## 4. 计数律（自动导出）

| 节点 | 允许的 $(l,s)$ | $N$ |
|---|---|---|
| `psi2s` | (5,5) | 1 |
| `kstar5` | (5,0) | 1 |
| **合计** | | **1** |

⚠️ 有道的树与 base chain 不同 ⇒ omega **不可用**（实测残差 0.17–1.36，**静默错**：emit 不报错、跑得出数）。`omega_from_momenta` 只对 base chain 算一份逐道共用的 `Omega`。置换族 / 旁观者交换用 `helicity_cascade`。

## 5. 表象与 `emit` 可行性（自动导出）

| 表象 | kernel | 需要手写道 | 实测 `emit` | 说明 |
|---|---|---|---|---|
| `helicity_cascade` | `helicity_cascade` | 否 | ✅ | 1 个脚本 |

（`emit` 一列是**实测**结果 —— 逐表象单独调 `emitter.emit`，不是靠文件存在推断。）

⚠️ 本链**没有**手写道目录，所以 `cartesian`/`helicity` 两条路线的 `emit` 会报 `LookupError`；只有 `omega` 能出产物。

深度 = 2 ⇒ omega **可用**。

---

## 6. 叙述：做了什么、还有什么问题（手工）

### 这条链是什么

例 2 `ψ→K⁺K⁻π⁰` 的 **K\*₅(5⁻) 家族**（文档 274–304 行的 `U_{K^*_5}^μ`）。

与 [`ex2_kstar`](ex2_kstar.md) 同构（只有 `(13)`/`(23)` 两对，因为 K⁺K⁻
是同种粒子），只是 J=1 → 5、l=1 → 5。

### 它和 rho5 一起证明了 J=5 的能力边界已扩

`ex1_rho5`（ρ₅）与这条链在 2026-10-20 之前都被 `spec.py` 拒绝。
扩展只改了三处：`LAM_SLOTS`、`Bl` 的系数、`spec.py` 的按表象分级边界。
详见 [`ex1_rho5` 的 notes](notes/ex1_rho5.notes.md) 与
[`probes/probe_j5_feasibility.py`](../probes/probe_j5_feasibility.py)。

### ⚠️ 文档的 `- (1↔2)` 不是道间相对符号

文档写的是

$$
U_{K^*_5}^\mu = \epsilon_{\mu\nu\lambda\sigma}p_\psi^\sigma
  [\tilde T^{(5)\cdots}_{(K^*_5 2)}\tilde t^{(5)\cdots}_{(13)}
    f^{(K^*_5)}_{(13)} - (1\leftrightarrow 2)]
$$

看起来后一项该反号。但**实测两条道在置换下是同号**：

```
A_13(交换 0<->1 的动量) / A_23(原动量) = +1.000000   残差 0.00e+00
```

与 J=1 的 `U_{K^*}` 完全一致（那里也是同号）。那个负号来自
`T̃^{(J)}` 张量缩并的**整体**因子（`-i` 那一类），不是道与道之间的
相对符号。所以本链的两条道**直接相加**，不需要 `g_i = ±1`。

这一点和 ρ₃/ρ₅ 不同：ρ 家族（例 1）的后两项**确实**要反号，
本链不用。判据始终是**置换协变性实测**，不是照抄文档的符号。

### 判据与实测

| 判据 | 结果 |
|---|---|
| 置换协变性（`_13` 交换 0↔1 的动量 vs `_23`） | 残差 0，`c = +1.000000`（同号） |
| 振幅有限 | ✅ `|A|max = 1.8e5` |
| `ε·B` 恒等式 | 不适用（J=5，同 ρ₅） |
| `check` / `emit` | ✅ |
| `cartesian`/`helicity` + J=5 | ✅ **可算**（`covariant.py` 已有通用 `t_l`/`B5`） |
| 协变张量路线的 emit | ⚠️ 需要手写道函数；本链没写，故只有 cascade 能出 |
