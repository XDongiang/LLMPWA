# 链 `phif0` —— psi2s → phi + f0 ; phi → Kp + Km ; f0 → Pip + Pim

> **本文件是生成物**，不要直接编辑。
> 事实部分由 [`gen_chain_docs.py`](gen_chain_docs.py) 从 [`agent/tensor/specs/phif0.toml`](../../../agent/tensor/specs/phif0.toml) 自动导出；
> 叙述部分来自 [`notes/phif0.notes.md`](notes/phif0.notes.md)。
> 改完重跑：`python analyses/tensor_examples/chains/gen_chain_docs.py`

---

## 1. 拓扑（自动导出）

```text
psi2s  J=1  P=-1
├── phi        J=1  P=-1  → Kp + Km
│  ├── Kp         J=0  P=-1   （末态）
│  ├── Km         J=0  P=-1   （末态）
├── f0         J=0  P=+1  → Pip + Pim
│  ├── Pip        J=0  P=-1   （末态）
│  ├── Pim        J=0  P=-1   （末态）
```

**深度 = 2**（最长路径上的边数）。

## 2. 粒子表（自动导出）

| 粒子 | $J$ | 宇称 | 质量 | 角色 | 允许的 $l$ |
|---|---|---|---|---|---|
| `psi2s` | 1 | -1 | 3.686 | 顶层 | 0, 2 |
| `phi` | 1 | -1 | 1.019 | 节点 | 1 |
| `Kp` | 0 | -1 | 0.493677 | 末态 | - |
| `Km` | 0 | -1 | 0.493677 | 末态 | - |
| `f0` | 0 | 1 | 0.99 | 节点 | 0 |
| `Pip` | 0 | -1 | 0.13957 | 末态 | - |
| `Pim` | 0 | -1 | 0.13957 | 末态 | - |

## 3. 道表与逐节点 $(l,s)$（自动导出）

道表来源：**显式 `[[channel]]`**（2 条）。

| 道 | 顶层 $(l,s)$ | `phi` | `f0` |
|---|---|---|---|
| `phif001` | (0,1) | (1,0) | (0,0) |
| `phif021` | (2,1) | (1,0) | (0,0) |

## 4. 计数律（自动导出）

| 节点 | 允许的 $(l,s)$ | $N$ |
|---|---|---|
| `psi2s` | (0,1)  (2,1) | 2 |
| `phi` | (1,0) | 1 |
| `f0` | (0,0) | 1 |
| **合计** | | **2** |

✅ 非顶层节点都只有 **1 个** $(l,s)$，且深度 = 2 ≤ 2 ⇒ omega 的信息容量够用。

## 5. 表象与 `emit` 可行性（自动导出）

| 表象 | kernel | 需要手写道 | 实测 `emit` | 说明 |
|---|---|---|---|---|
| `cartesian` | `covariant` | 是 | ✅ | 1 个脚本 |
| `helicity` | `helicity` | 是 | ✅ | 1 个脚本 |
| `omega` | `omega` | 否 | ✅ | 1 个脚本 |

（`emit` 一列是**实测**结果 —— 逐表象单独调 `emitter.emit`，不是靠文件存在推断。）

手写道（`agent/tensor/chains/phif0/`，2 条）：`phif001`, `phif021`

深度 = 2 ⇒ omega **可用**。

---

## 6. 叙述：做了什么、还有什么问题（手工）

### 与 φf₂ 的关系

φf₀ 和 φf₂ 是**同一份拓扑的两个自旋特例**，验收上互为对照：
φf₀ 验"$J=0$ 的退化 + 末轴长度自适应"，φf₂ 验"$J=2$ 的多槽缩并"。
两者**都不能**用来验 $J>1$ 的母粒子行序（母粒子都是 $J=1$ 的 ψ）。

### 一个特有的、容易写错的点：$f_0$ 的质量

spec 里 `[particles.f0] mass = 0.99`。这个值**只用于拓扑/阈值校验**，
不影响协变张量（道函数是硬编码公式、不读质量），但会影响：

* `omega` 内核的 $K_i$（通过 $Q$）；
* 级联内核的 $B_l(q)$ 与 $\gamma$ 因子。

⚠️ 所以 φf₀ 的 `omega vs 协变` 之所以能到 4e-15，是因为**两边用同一个
$m_{f_0}$**。真实拟合里 $f_0$ 质量自由时，omega 与协变的比值**不再是
逐道常数** —— 这是 roadmap §8 风险 1 的具体实例，**未解决**。

### 验收

| 判据 | 结果 | 在哪 |
|---|---|---|
| 生成物 vs 框架内核 | rel = 0 | `test_emitter` |
| 与 jax 原版逐式对拍 | 通过 | `test_covariant_vs_jax` |
| 生产 cache 一致性 | 4/4 | `test_production_cache` |
| 级联 vs 协变（A1） | 5.4e-15 / 7.4e-15 | `test_helicity_cascade::test_phi_f0_channels_unregressed` |
| omega vs 协变 | 4.1e-15 / 5.8e-15 | 实测（$n=64$、seed=11）|

（两条路线在 φf₀ 上都到机器精度，且**区间重合** —— 同 φf₂ 的道理。）

### 它对"级联内核"的独立价值

φf₀ 在级联验收里是**回归哨兵**：它是内层自旋为 0 的最简情形，
所以 `test_helicity_cascade.py::test_phi_f0_channels_unregressed`
用它的 2 条道钉住"最简单的情形没有退化"。

### `phif001` 是纯 S 波

$B_0 \equiv 1$，没有 $q^0$ 因子，实现上是整个系列里**最简单**的一条道。

### 还有什么问题

* 深度 2 ⇒ omega 可用（不是限制）；级联表象**多余**（omega 容量够用，
  且级联 cache 逐道一份、体积大）。
* 真实样本回归、拟合侧端到端、barrier 的 $q_0$ 依赖 —— **都没做**，
  与 φf₂ 相同。
* `psi_mass` 口径同 φf₂（spec 写死 3.686；真实样本要用 3.685782956）。

### 复现

```bash
cd <LLMPWA>
python -m agent.tensor show  agent/tensor/specs/phif0.toml
python -m agent.tensor check agent/tensor/specs/phif0.toml
python -m agent.tensor emit  agent/tensor/specs/phif0.toml

python agent/tensor/tests/test_crosscheck.py        # φf0 + φf2
python agent/tensor/tests/test_production_cache.py
python agent/tensor/tests/test_helicity_cascade.py  # 含 φf0 回归哨兵
```

### 相关

* [链 `phif2`](phif2.md)
* [链索引](README.md)
