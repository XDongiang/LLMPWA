### 这条链为什么重要

φf₂ 是**行序约定**的主要检验场之一。`LAM_SLOTS[J] = [-J..J]` 看起来与
`EPS_A` 的 $(\lambda{=}+1,0,-1)$ 相反，但实测**不需要翻轴**。

⚠️ 但要注意：**φf₂ 检验不了 $J>1$ 的行序**。`phif222`/`phif243` 里的
2、4 是**顶层 $L$**，而母粒子 ψ 的 $J$ 仍是 1，所以末轴还是 3 行。
真正的 $J>1$ 检验要用 `X(J)→0+0` 链（见
[`probes/probe_row_order_J2.py`](../probes/probe_row_order_J2.py)）。
这是审阅意见指出的一处**报告初稿错误**，已改正。

### 末轴长度：φf₂ 与 φf₀ 共用一个 einsum

末轴长度：φf₂ 的末态是 $\phi+\pi\pi$，$\lambda_\phi$ 有 3 槽，所以
`helicity` 的末轴是 **3**；φf₀ 那边 $f_0$ 自旋为 0，末轴是 **1**。
两者共用同一句 `einsum("iab,enab->ien")`，靠长度自适应。

### 验收

| 判据 | 结果 | 在哪 |
|---|---|---|
| 生成物 vs 框架内核（逐位） | rel = 0 | `test_emitter` |
| `cartesian` vs `helicity`（$W=\mathrm{EPS\_A}\cdot T$） | rel = 0 | `test_cross_basis` |
| 与 jax 原版逐式对拍 | 通过 | `test_covariant_vs_jax` |
| 生产 cache 一致性 | 4/4 | `test_production_cache` |
| 级联 vs 协变（A1，经 `EPS_A`） | 3.3e-15 … 6.3e-15 | `probe_lambda_row_order.py` |
| omega vs 协变 | 3.3e-15 … 6.3e-15 | 实测 |

（区间来自 5 条道，样本 $n=64$、seed=11。两条路线的区间**重合**，
因为在这条链上它们算的是同一个东西 —— 见下。）

### 级联表象对这条链**是多余的**

`helicity_cascade` 在这条链上能给出与 omega 相同的结果（都到机器精度），
但 spec 里**没加**这条 basis。原因：

* omega 的 cache 是**一份 $\Omega$ 服务所有道**，而级联 cache 是
  **逐道一份**、复数，体积大得多（roadmap §8 明确列为代价）；
* 这条链的非顶层节点恰好只有 1 个允许的 $(l,s)$（$f_2\to\pi\pi$ 是
  D 波、$\phi\to KK$ 是 P 波），**omega 的信息容量够用**。

结论：**φf₂ 不需要级联内核**。级联的价值在 b₁（非顶层多分波）
和深度 ≥ 3 的链上。

### 为什么这条链"太顺利"

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

### 还有什么问题

| 项 | 状态 |
|---|---|
| 真实样本回归 | ❌ 只用合成四动量验过；没跑 `data/` 真实样本 |
| 拟合侧端到端 | ❌ 没跑 `_load_cached_tensor` / toy likelihood |
| barrier 的 $q_0$ 依赖 | ⚠️ 未解决（roadmap §8 风险 1）。真实拟合里共振质量自由时它会显现 |
| `psi_mass` 口径 | ⚠️ spec 里写死 3.686（与 `Tensor/FUNC.py` 一致）。**真实样本**要用 3.685782956，用错会让相对差从 1.7e-15 掉到 4.2e-04 |

### 复现

```bash
cd <LLMPWA>
python -m agent.tensor show  agent/tensor/specs/phif2.toml
python -m agent.tensor check agent/tensor/specs/phif2.toml
python -m agent.tensor emit  agent/tensor/specs/phif2.toml

python agent/tensor/tests/test_spec.py
python agent/tensor/tests/test_emitter.py
python agent/tensor/tests/test_cross_basis.py
python agent/tensor/tests/test_covariant_vs_jax.py   # 需要 TENSOR_ROOT
```

### 相关

* [链 `phif0`](phif0.md) —— 同拓扑，$f_0$ 自旋 0 的退化情形
* [链 `b1pi`](b1pi.md) —— 把三个坑一起暴露的链
* [链索引](README.md)
