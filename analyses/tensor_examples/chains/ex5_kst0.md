# 链 `ex5_kst0` —— psi2s → Kstar + Pim ; Kstar → phi + Pip ; phi → Kp + Km

> **本文件是生成物**，不要直接编辑。
> 事实部分由 [`gen_chain_docs.py`](gen_chain_docs.py) 从 [`analyses/tensor_examples/specs/ex5_kst0.toml`](../specs/ex5_kst0.toml) 自动导出；
> 叙述部分来自 [`notes/ex5_kst0.notes.md`](notes/ex5_kst0.notes.md)。
> 改完重跑：`python analyses/tensor_examples/chains/gen_chain_docs.py`

---

## 1. 拓扑（自动导出）

```text
psi2s  J=1  P=-1
├── Kstar      J=0  P=-1  → phi + Pip
│  ├── phi        J=1  P=-1  → Kp + Km
│  │  ├── Kp         J=0  P=-1   （末态）
│  │  ├── Km         J=0  P=-1   （末态）
│  ├── Pip        J=0  P=-1   （末态）
├── Pim        J=0  P=-1   （末态）
```

**深度 = 3**（最长路径上的边数）。

## 2. 粒子表（自动导出）

| 粒子 | $J$ | 宇称 | 质量 | 角色 | 允许的 $l$ |
|---|---|---|---|---|---|
| `psi2s` | 1 | -1 | 3.686 | 顶层 | 1 |
| `Kstar` | 0 | -1 | 1.43 | 节点 | 1 |
| `phi` | 1 | -1 | 1.019 | 节点 | 1 |
| `Kp` | 0 | -1 | 0.493677 | 末态 | - |
| `Km` | 0 | -1 | 0.493677 | 末态 | - |
| `Pip` | 0 | -1 | 0.13957 | 末态 | - |
| `Pim` | 0 | -1 | 0.13957 | 末态 | - |

## 3. 道表与逐节点 $(l,s)$（自动导出）

道表来源：**显式 `[[channel]]`**（2 条）。

| 道 | 顶层 $(l,s)$ | `Kstar` | `phi` |
|---|---|---|---|
| `u_kst0_l` | (1,0) | (1,1) | (1,0) |
| `u_kst0_r` | (1,0) | (1,1) | (1,0) |

## 4. 计数律（自动导出）

| 节点 | 允许的 $(l,s)$ | $N$ |
|---|---|---|
| `psi2s` | (1,0) | 1 |
| `Kstar` | (1,1) | 1 |
| `phi` | (1,0) | 1 |
| **合计** | | **1** |

✅ 非顶层节点都只有 **1 个** $(l,s)$ ⇒ omega 的信息容量够用。

## 5. 表象与 `emit` 可行性（自动导出）

| 表象 | kernel | 需要手写道 | 实测 `emit` | 说明 |
|---|---|---|---|---|
| `cartesian` | `covariant` | 是 | ✅ | 1 个脚本 |
| `helicity` | `helicity` | 是 | ✅ | 1 个脚本 |
| `helicity_cascade` | `helicity_cascade` | 否 | ✅ | 1 个脚本 |

（`emit` 一列是**实测**结果 —— 逐表象单独调 `emitter.emit`，不是靠文件存在推断。）

手写道（`agent/tensor/chains/kstpi/`，10 条）：`u_FF_1`, `u_FF_2`, `u_FP_1`, `u_FP_2`, `u_PF_1`, `u_PF_2`, `u_PP_1`, `u_PP_2`, `u_kst0_l`, `u_kst0_r`

深度 = 3 ⇒ omega **不可用**（`spec.py` 的分级检查会直接报错，不是静默出错）。

---

## 6. 叙述：做了什么、还有什么问题（手工）

### 这条链是例 5 的 K*₀ 部分

$\psi\to K^+\pi^-K^-\pi^+$，文档 431–623 行。拓扑
$\psi\to K^*_0\pi\to(\phi\pi)\pi\to(K^+K^-)(\pi\pi)$ —— **深度 3**。

两条道 `u_kst0_l` / `u_kst0_r` 已从参考实现逐式移植到
[`agent/tensor/chains/kstpi/`](../../../agent/tensor/chains/kstpi/)，
与 jax 逐式对拍通过（rel ≈ 1e-15）。

### 为什么与 K*₂ 分开

$K^*_0$ 的 $J=0$ 与 $K^*_2$ 的 $J=2$ **冲突** —— 一个
`[particles.Kstar]` 条目只能有一个 $J$。协变道函数不读 $J$，
所以"能不能算"不受影响，但拓扑校验需要确定的 $J$，只能拆两份 spec。

### 量子数：这里的 K*₀ 是 $0^-$

宇称 + 三角：$K^*_0\to\phi\pi$ 要求
$\eta_{K^*_0}=\eta_\phi\eta_\pi(-1)^l=(-1)^l$。若取 $l=1$（唯一满足
$J=0$ 的三角关系），则 $\eta=-1$，即 $0^-$。

⚠️ 这与常见的 $K^*_0(1430)$（$0^+$）**不同**。本链的公式是参考实现
`_u_kst0_l/r` 的硬编码形式，其内层 $\tilde t^{(1)}$ 就是 $l=1$。
若物理上要 $0^+$，需要 $l$ 为偶数，而 $J=0$ 的三角关系只允许 $l=s=1$ ——
**矛盾**，说明这条道描述的不是 $0^+$ 的 $K^*_0(1430)$。

这个疑点**未解决**，见下方"还有什么问题"。

### 级联判定：2 / 2 命中

| 道 | 顶层 $(l,s)$ | 内层 $(l,s)$ | rel | 结论 |
|---|---|---|---|---|
| `u_kst0_l` | (1,0) | (1,1) | 1.9e-14 | ✅ 是级联 |
| `u_kst0_r` | (1,0) | (1,1) | 4.1e-15 | ✅ |

这两条是**纯级联**：三个 $\tilde t^{(1)}$ 依次缩并成矢量，中间**没有**
投影算符（对比 $u_{PP}$ 那些含 $P^{(2)}_\psi$ 的）。

有意思的是：本链内层是 **P 波**却命中了，而 `ex5_kst2` 里内层 P 波
且 $K^*$ 自旋为 2 的 $u_{PP}$/$u_{FP}$ 却**不**命中。所以规律不是
"P 波不行"，而是 $(J_{K^*}=2,\ l_{\text{内}}=1)$ 这个特定组合不行。

### 还有什么问题

| 项 | 状态 |
|---|---|
| **$0^-$ vs $0^+$ 的物理归属** | ❓ **未解决**。见上：$J=0$ + 宇称守恒只允许 $l=s=1$，对应 $0^-$；若这条道本意是 $K^*_0(1430)$（$0^+$），则公式与物理不符 |
| 与文档 Eq 的逐条对应 | ⚠️ 未敲定（同 `ex5_kst2`） |
| $K^*$ 质量 | ⚠️ spec 写 1.43，只是占位 |
| 真实样本 / 拟合侧 | ❌ 未做 |

### 复现

```bash
cd <LLMPWA>
python agent/tensor/tests/test_ex5_channels.py
python -m agent.tensor check analyses/tensor_examples/specs/ex5_kst0.toml
python -m agent.tensor emit  analyses/tensor_examples/specs/ex5_kst0.toml
```

### 相关

* [链 `ex5_kst2`](ex5_kst2.md) —— 同一移植的 $J=2$ 部分（含 6/10 级联判定）
* [链索引](README.md)
