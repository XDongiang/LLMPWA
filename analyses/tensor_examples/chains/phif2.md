# 链 `phif2` —— ψ(2S) → φ f₂ → (K⁺K⁻)(π⁺π⁻)

> 状态：**生产链**，三表象全通（`cartesian` / `helicity` / `omega`），深度 2。
> spec：[`agent/tensor/specs/phif2.toml`](../../../agent/tensor/specs/phif2.toml)
> 道：`agent/tensor/chains/phif2/`（5 条）
> 本文档写于 2026-10-19。

---

## 1. 这条链是什么

```
ψ(2S)  J=1  P=-1
  ├── φ   J=1  P=-1   → K⁺ K⁻      （P 波，l=1）
  └── f₂  J=2  P=+1   → π⁺ π⁻      （D 波，l=2）
```

拓扑是**两层**：顶层 + 两个共振。两个共振都直接衰变到末态，**不再有下层**。
这一点是理解这条链全部行为的钥匙 —— 见 §4。

粒子编号与文档一致：`K⁺=1`、`K⁻=2`、`π⁺=3`、`π⁻=4`。

## 2. 做了什么

### 2.1 五条道（协变张量）

| 道 | 文档公式 | 顶层 $(l,s)$ |
|---|---|---|
| `phif201` | Eq.(41) 修正版 | $(0,1)$ |
| `phif221` | Eq.(42) 修正版 | $(2,1)$ |
| `phif222` | Eq.(43) | $(2,2)$ |
| `phif223` | Eq.(44) | $(2,3)$ |
| `phif243` | Eq.(45) | $(4,3)$ |

每条道是 `agent/tensor/chains/phif2/<name>.py` 里一个手写的 `compute_event`
风格函数，**逐式**对照参考实现 `Tensor/FUNC.py` 移植（走 `_backend.xp`，
jax / numpy 两后端通用）。

道表在 spec 里**显式**给出（与 `method4_omega/basis.py::LS_F2` 一致）。
若删掉 `[[channel]]` 段，`specgen` 会按宇称 + 角动量三角**自动枚举**出同一组
$(l,s)$ —— 这条已被 `test_spec` 覆盖。

### 2.2 三表象

一份 TOML 同时出三种 tensor：

| 表象 | kernel | 内容 |
|---|---|---|
| `cartesian` | `covariant` | Chung 协变张量 $T_i(e,\mu)$，形状 $(5,N,4)$ |
| `helicity` | `helicity` | $W_i=\mathrm{EPS\_A}\cdot T_i$，形状 $(5,N,3)$ |
| `omega` | `omega` | $\Omega/K/M/g$ 预计算，形状 $(N,5,3,1)$ |

末轴长度：φf₂ 的末态是 $\phi+\pi\pi$，$\lambda_\phi$ 有 3 槽，所以
`helicity` 的末轴是 **3**（φf₀ 那边 $f_0$ 自旋为 0，末轴是 **1**；
两者共用同一句 `einsum("iab,enab->ien")`，靠长度自适应）。

### 2.3 验收

| 判据 | 结果 | 在哪 |
|---|---|---|
| 生成物 vs 框架内核（逐位） | rel = 0 | `test_emitter`、`check_ex3.py` 第 1 层 |
| `cartesian` vs `helicity`（$W=\mathrm{EPS\_A}\cdot T$） | rel = 0 | `test_cross_basis` |
| 与 jax 原版逐式对拍 | 通过 | `test_covariant_vs_jax` |
| 生产 cache 一致性 | 4/4 | `test_production_cache` |
| **级联 vs 协变**（A1，经 `EPS_A`） | **3.3e-15 … 6.3e-15** | `probe_lambda_row_order.py` |
| **omega vs 协变** | **3.3e-15 … 6.3e-15** | 实测 |

（上表为 5 条道的区间，样本 $n=64$、seed=11。两条路线的区间**重合**，
因为在这条链上它们算的是同一个东西 —— 见 §3.2。）

两条独立路线（omega 与 helicity_cascade）都能逐位重现手写协变道，
这是这条链最强的证据。

### 2.4 对螺旋度级联内核的意义

φf₂ 是**行序约定**的主要检验场之一。`LAM_SLOTS[J] = [-J..J]` 看起来与
`EPS_A` 的 $(\lambda{=}+1,0,-1)$ 相反，但实测**不需要翻轴**。

⚠️ 但要注意：**φf₂ 检验不了 $J>1$ 的行序**。`phif222`/`phif243` 里的
2、4 是**顶层 $L$**，而母粒子 ψ 的 $J$ 仍是 1，所以末轴还是 3 行。
真正的 $J>1$ 检验要用 `X(J)→0+0` 链（见
[`probes/probe_row_order_J2.py`](../probes/probe_row_order_J2.py)）。
这是审阅意见指出的一处**报告初稿错误**，已改正。

## 3. 还有什么问题

### 3.1 `omega` 的深度限制与这条链无关（因为它是两层）

omega 内核逐节点读树、只算 `D0*D1*D2`，所以**深度 > 2 的树会静默漏掉内层
角分布**。φf₂ 恰好是 2 层，所以 omega 是对的、可用的。

这条限制现在由 [`agent/tensor/spec.py`](../../../agent/tensor/spec.py)
的深度分级检查强制执行：深度 > 2 且要出 omega 时**直接报错**，
而不是静默出错。φf₂ 不受影响。

### 3.2 级联表象对这条链**是多余的**

`helicity_cascade` 在这条链上能给出与 omega 相同的结果（都到机器精度），
但 **spec 里没加这条 basis**。原因：

* omega 的 cache 是**一份 $\Omega$ 服务所有道**，而级联 cache 是
  **逐道一份**、复数，体积大得多（roadmap §8 明确列为代价）；
* 这条链的非顶层节点恰好只有 1 个允许的 $(l,s)$（$f_2\to\pi\pi$ 是
  D 波、$\phi\to KK$ 是 P 波），**omega 的信息容量够用**。

所以结论是：**φf₂ 不需要级联内核**。级联的价值在 b₁（非顶层多分波）
和深度 ≥ 3 的链上。

### 3.3 未做的事

| 项 | 状态 |
|---|---|
| 真实样本回归 | ❌ 只用合成四动量验过；没跑 `data/` 真实样本 |
| 拟合侧端到端 | ❌ 没跑 `_load_cached_tensor` / toy likelihood |
| barrier 的 $q_0$ 依赖 | ⚠️ 未解决（roadmap §8 风险 1）。真实拟合里共振质量自由时它会显现 |
| `psi_mass` 口径 | ⚠️ spec 里写死 3.686（与 `Tensor/FUNC.py` 一致）。**真实样本**要用 3.685782956，用错会让相对差从 1.7e-15 掉到 4.2e-04（见 spec 内注释） |

## 4. 为什么这条链"太顺利"

φf₂ 在整条螺旋度路线里扮演了一个**误导性**的角色：它三种表象全通、
所有判据都是机器精度，所以看上去"框架没问题"。

真相是：**它的结构恰好绕开了全部三个坑**。

| 坑 | φf₂ 为什么绕开 |
|---|---|
| 子粒子螺旋度被压成 0 | φf₂ 的共振（φ、f₂）都是**末态**，本来就没有下层螺旋度 |
| 只算 `D0*D1*D2` | 树只有 2 层，omega 的截断正好够 |
| 逐节点参考系从顶层 boost | φ 是末态、不衰变，两种构造**逐位相同** |

实测确认：`max|chain_angles − node_angles| = 0.000e+00`。

所以 φf₂ 的"全绿"**不能**外推成"框架对任意链都对"。这正是 b₁
（深度 3、非顶层多分波）才把三个坑一起暴露出来的原因。

## 5. 复现

```bash
cd <LLMPWA>

# 看链 / 道表 / 表象
python -m agent.tensor show  agent/tensor/specs/phif2.toml
python -m agent.tensor check agent/tensor/specs/phif2.toml

# 生成 + 跑
python -m agent.tensor emit agent/tensor/specs/phif2.toml

# 框架测试（不需要本目录数据）
python agent/tensor/tests/test_spec.py
python agent/tensor/tests/test_emitter.py
python agent/tensor/tests/test_cross_basis.py
python agent/tensor/tests/test_covariant_vs_jax.py   # 需要 TENSOR_ROOT
```

## 6. 相关文档

* [链 `phif0`](phif0.md) —— 同拓扑，$f_0$ 自旋 0 的退化情形
* [链 `b1pi`](b1pi.md) —— 把三个坑一起暴露的链
* [链索引](README.md)
* 实现规格：[`documentation/helicity_amplitude_roadmap.md`](../../../documentation/helicity_amplitude_roadmap.md)
