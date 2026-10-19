# 链 `phif0` —— ψ(2S) → φ f₀ → (K⁺K⁻)(π⁺π⁻)

> 状态：**生产链**，三表象全通（`cartesian` / `helicity` / `omega`），深度 2。
> spec：[`agent/tensor/specs/phif0.toml`](../../../agent/tensor/specs/phif0.toml)
> 道：`agent/tensor/chains/phif0/`（2 条）
> 本文档写于 2026-10-19。

---

## 1. 这条链是什么

```
ψ(2S)  J=1  P=-1
  ├── φ   J=1  P=-1   → K⁺ K⁻      （P 波，l=1）
  └── f₀  J=0  P=+1   → π⁺ π⁻      （S 波，l=0）
```

与 [`phif2`](phif2.md) **同拓扑、同末态**，唯一区别是中间共振的自旋：
$f_0$ 是 $J=0$，$f_2$ 是 $J=2$。

粒子编号：`K⁺=1`、`K⁻=2`、`π⁺=3`、`π⁻=4`。

## 2. 做了什么

### 2.1 两条道

| 道 | 文档公式 | 顶层 $(l,s)$ |
|---|---|---|
| `phif001` | Eq.(36) | $(0,1)$ |
| `phif021` | Eq.(37) | $(2,1)$ |

$f_0$ 自旋为 0，$S$ 只能来自 $\phi$，所以 **$s$ 恒为 1、$l$ 取偶数** ——
只有上面两条（与 `method4_omega/basis.py::LS_F0` 一致）。

`phif001` 是**纯 S 波**：$B_0 \equiv 1$，没有 $q^0$ 因子，实现上是
整个系列里最简单的一条道。

### 2.2 三表象与一个长度细节

| 表象 | 形状 | 备注 |
|---|---|---|
| `cartesian` | $(2,N,4)$ | |
| `helicity` | $(2,N,3)$ | |
| `omega` | $(N,2,3,1)$ | |

⚠️ **注意末轴长度**：$f_0$ 自旋为 0，所以 `helicity` 的**中间轴**（$\lambda_{f_0}$）
长度是 1，而 φf₂ 那边是 5（$J=2$）。但 `cartesian`/`helicity` 的**末轴**
（$\lambda_\phi$）在两条链上都是 3。

这也是 φf₀ 与 φf₂ 能**共用同一句 einsum**
（`einsum("iab,enab->ien")`）的原因 —— 收缩轴的长度自适应，不需要分叉代码。

### 2.3 验收

| 判据 | 结果 | 在哪 |
|---|---|---|
| 生成物 vs 框架内核 | rel = 0 | `test_emitter` |
| 与 jax 原版逐式对拍 | 通过 | `test_covariant_vs_jax` |
| 生产 cache 一致性 | 4/4 | `test_production_cache` |
| **级联 vs 协变**（A1） | **5.4e-15 / 7.4e-15** | `test_helicity_cascade::test_phi_f0_channels_unregressed` |
| **omega vs 协变** | **4.1e-15 / 5.8e-15** | 实测（$n=64$、seed=11）|

（两条路线在 φf₀ 上都到机器精度，且**区间重合** —— 同 φf₂ 的道理。）

### 2.4 它对"级联内核"的独立价值

φf₀ 在级联验收里是**回归哨兵**：它是内层自旋为 0 的最简情形，
所以 `test_helicity_cascade.py::test_phi_f0_channels_unregressed`
用它的 2 条道钉住"最简单的情形没有退化"。

## 3. 还有什么问题

### 3.1 与 φf₂ 相同的问题

深度 2 ⇒ omega 可用（不是限制）；级联表象**多余**（omega 容量够用，
且级联 cache 逐道一份、体积大）。真实样本回归、拟合侧端到端、
barrier 的 $q_0$ 依赖 —— **都没做**，与 φf₂ 相同。

### 3.2 一个特有的、容易写错的点：$f_0$ 的质量

spec 里 `[particles.f0] mass = 0.99`。这个值**只用于拓扑/阈值校验**，
不影响协变张量（道函数是硬编码公式、不读质量），但会影响：
* `omega` 内核的 $K_i$（通过 $Q$）；
* 级联内核的 $B_l(q)$ 与 $\gamma$ 因子。

⚠️ 所以 φf₀ 的 `omega vs 协变` 之所以能到 4e-15，是因为**两边用同一个
$m_{f_0}$**。真实拟合里 $f_0$ 质量自由时，omega 与协变的比值**不再是
逐道常数** —— 这是 roadmap §8 风险 1 的具体实例，**未解决**。

### 3.3 `psi_mass` 口径（与 φf₂ 相同的坑）

spec 写死 3.686（与 `Tensor/FUNC.py` 一致）。真实样本要用 3.685782956。
用错会让协变张量与生产 cache 的相对差从 1.7e-15 掉到 4.2e-04。
**部署前必须确认**。

## 4. 与 φf₂ 的关系（一句话）

φf₀ 和 φf₂ 是**同一份拓扑的两个自旋特例**，验收上互为对照：
φf₀ 验"$J=0$ 的退化 + 末轴长度自适应"，φf₂ 验"$J=2$ 的多槽缩并"。
两者**都不能**用来验 $J>1$ 的母粒子行序（母粒子都是 $J=1$ 的 ψ）。

## 5. 复现

```bash
cd <LLMPWA>
python -m agent.tensor show  agent/tensor/specs/phif0.toml
python -m agent.tensor check agent/tensor/specs/phif0.toml
python -m agent.tensor emit  agent/tensor/specs/phif0.toml

python agent/tensor/tests/test_crosscheck.py        # φf0 + φf2
python agent/tensor/tests/test_production_cache.py
python agent/tensor/tests/test_helicity_cascade.py  # 含 φf0 回归哨兵
```

## 6. 相关文档

* [链 `phif2`](phif2.md)
* [链索引](README.md)
