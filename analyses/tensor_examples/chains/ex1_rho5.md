# 链 `ex1_rho5` —— psi2s → rho5 + Pi0 ; rho5 → Pip1 + Pim2

> **本文件是生成物**，不要直接编辑。
> 事实部分由 [`gen_chain_docs.py`](gen_chain_docs.py) 从 [`analyses/tensor_examples/specs/ex1_pipipi0_rho5.toml`](../specs/ex1_pipipi0_rho5.toml) 自动导出；
> 叙述部分来自 [`notes/ex1_rho5.notes.md`](notes/ex1_rho5.notes.md)。
> 改完重跑：`python analyses/tensor_examples/chains/gen_chain_docs.py`

---

## 1. 拓扑（自动导出）

```text
psi2s  J=1  P=-1
├── rho5       J=5  P=-1  → Pip1 + Pim2
│  ├── Pip1       J=0  P=-1   （末态）
│  ├── Pim2       J=0  P=-1   （末态）
├── Pi0        J=0  P=-1   （末态）
```

**深度 = 2**（最长路径上的边数）。

## 2. 粒子表（自动导出）

| 粒子 | $J$ | 宇称 | 质量 | 角色 | 允许的 $l$ |
|---|---|---|---|---|---|
| `psi2s` | 1 | -1 | 3.686 | 顶层 | 5 |
| `rho5` | 5 | -1 | 2.3 | 节点 | 5 |
| `Pip1` | 0 | -1 | 0.13957 | 末态 | - |
| `Pim2` | 0 | -1 | 0.13957 | 末态 | - |
| `Pi0` | 0 | -1 | 0.134977 | 末态 | - |

## 3. 道表与逐节点 $(l,s)$（自动导出）

道表来源：**显式 `[[channel]]`**（3 条）。

| 道 | 顶层 $(l,s)$ | `rho5` |
|---|---|---|
| `u_rho5_12` | (5,5) | (5,0) |
| `u_rho5_13` | (5,5) | (5,0) |
| `u_rho5_23` | (5,5) | (5,0) |

## 4. 计数律（自动导出）

| 节点 | 允许的 $(l,s)$ | $N$ |
|---|---|---|
| `psi2s` | (5,5) | 1 |
| `rho5` | (5,0) | 1 |
| **合计** | | **1** |

⚠️ 有道的树与 base chain 不同 ⇒ omega **不可用**（实测残差 0.17–1.36，**静默错**：emit 不报错、跑得出数）。`omega_from_momenta` 只对 base chain 算一份逐道共用的 `Omega`。置换族 / 旁观者交换用 `helicity_cascade`。

## 5. 表象与 `emit` 可行性（自动导出）

| 表象 | kernel | 需要手写道 | 实测 `emit` | 说明 |
|---|---|---|---|---|
| `helicity_cascade` | `helicity_cascade` | 否 | ✅ | 1 个脚本 |
| `cartesian` | `covariant` | 是 | ✅ | 1 个脚本 |
| `helicity` | `helicity` | 是 | ✅ | 1 个脚本 |

（`emit` 一列是**实测**结果 —— 逐表象单独调 `emitter.emit`，不是靠文件存在推断。）

手写道（`agent/tensor/chains/rhopi5/`，3 条）：`u_rho5_12`, `u_rho5_13`, `u_rho5_23`

深度 = 2 ⇒ omega **可用**。

---

## 6. 叙述：做了什么、还有什么问题（手工）

### 这条链是什么

例 1 `ψ→π⁺π⁻π⁰` 的 **ρ₅(5⁻) 家族**（文档 249–263 行），也是**能力边界**的
证明：J=5 / l=5 原本被内核拒绝，2026-10-20 扩到了 6。

### 为什么它曾经不可能

`spec.py` 的 `validate` 里有三处硬上限（J>4、l>4、`l_max>4`），
理由是"`LAM_SLOTS` / Wigner-D 只到 J=4，`Bl` 只到 l=4"。

逐项实测（[`probes/probe_j5_feasibility.py`](../probes/probe_j5_feasibility.py)）
发现**这个理由只对了一半**：

| 组件 | J=5 是否可用 | 说明 |
|---|---|---|
| `small_d_weight(2J)` | ✅ 本来就行 | tf-pwa 的公式接受任意 `2J` |
| `small_d_matrix` 正交性 | ✅ 2.1e-15 | J=20 还有 4.2e-12 |
| `cg` 正交归一 | ✅ 4.4e-16 | Racah 求和精确 |
| `covariant_hel_term` | ✅ 有限 | Chung Eq.52 |
| `LAM_SLOTS` | ❌ 缺键 | 纯字典，加 5/6 即可 |
| `Bl` | ❌ 缺 l=5 支 | **唯一**需要新公式的 |
| `covariant.py` 的 `T5_cov` | ❌ 不存在 | 但级联路线不读它 |

### tf-pwa 怎么做的

`tf_pwa/breit_wigner.py`：

* `Bprime_polynomial` 的系数表**写到了 L=5**：
  `[1, 15, 315, 6300, 99225, 893025]`；
* 更大的 l 走 `get_bprime_coeff(l)` —— 用**反向 Bessel 多项式**
  `theta_l(x)` 满足 `|theta_l(iw)|² = Σ cᵢ w^(2i)` 现算系数；
* `cg_coef` 优先用 sympy 的 `CG`，不设 J 上限（`cg_table.json` 只到 4
  是**无 sympy 时的退化路径**，不是能力上限）。

所以 tf-pwa 没有 J=5 的墙 —— 它的"上限"只是便利表的大小。

### 我们怎么扩的

只改三处（原语本来就是通用的）：

1. `omega.py::LAM_SLOTS` 加 `5: range(-5,6)`、`6: range(-6,7)`；
2. `omega.py::Bl` 改成查系数表 `_BPRIME_COEFF`（l≤5，系数与 tf-pwa 逐项
   一致），l>5 由 `_bprime_coeff` 用反向 Bessel 多项式**现算**；
   `num = Σ coeff = P_l(1)`，与原有 `BASE.py::B1..B4` 的 num 完全相同；
3. `spec.py` 的边界扩到 **6**（两条路线都扩，见下）。

后来（同日）`covariant.py` 也补上了论文 Eq.(258) 的**通用** `t_l` 与
`B5`/`B6`，所以**协变张量路线也能算 ρ₅** 了 —— `chains/rhopi5/` 就是为它
写的三只手写道函数。`t_l(1..4)` 与写死的 `T1_cov..T4_cov` 逐位一致。

### 判据与实测

| 判据 | 结果 |
|---|---|
| 置换协变性（`_12` 换动量 vs `_13`） | 残差 4.5e-15，`c = -1.000000` |
| 振幅有限 | ✅ `|A|max = 0.24`（相空间 `resonance_mass=2.5`） |
| `Bl` l=5 系数 vs tf-pwa `Bprime_polynomial` | 形状一致（比值与 q 无关） |
| 协变张量路线（`cartesian`/`helicity`） | ✅ 可 emit（`chains/rhopi5/`），与级联 rel ~ 4e-15 |
| 协变张量 vs **tf-pwa**（配对在壳） | overlap=1.00000000，残差 **2e-15** |
| `_bprime_coeff` 复现写死的表 | l=1..5 逐项一致（含 l=6 的独立复算） |
| `check` / `emit` | ✅ |
| `cartesian`/`helicity` + J=5 | ✅ **可算**（`covariant.py` 已有通用 `t_l`/`B5`） |

⚠️ **旧 cache 不受影响**：`Bl` 重构后 l≤4 与旧实现的最大相对差是
~4e-16（机器精度，Horner 的乘法结合顺序变了），`test_production_cache`
（对 jax 参考，容差 1e-12）仍通过。

### 真正的墙在哪

不是 J=5，而是 `small_d_weight` 的**阶乘溢出**：

| J | d 矩阵正交性 |
|---|---|
| 20 | 4.2e-12 |
| 30 | 1.6e-09 |
| 40 | 1.7e-06 |
| 50 | **NaN**（`171!` 超 float64） |

所以上限取 6 是**保守**的 —— 物理上 PWA 用不到 J>5，而 6 留了余量。
