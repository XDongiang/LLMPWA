# 链 `ex6_phi4pi_ss` —— psi2s → phi + f2 ; phi → Kp + Km ; f2 → sigA + sigB ; sigA → Pip + Pim ; sigB → Pip2 + Pim2

> **本文件是生成物**，不要直接编辑。
> 事实部分由 [`gen_chain_docs.py`](gen_chain_docs.py) 从 [`analyses/tensor_examples/specs/ex6_phi4pi_ss.toml`](../specs/ex6_phi4pi_ss.toml) 自动导出；
> 叙述部分来自 [`notes/ex6_phi4pi_ss.notes.md`](notes/ex6_phi4pi_ss.notes.md)。
> 改完重跑：`python analyses/tensor_examples/chains/gen_chain_docs.py`

---

## 1. 拓扑（自动导出）

```text
psi2s  J=1  P=-1
├── phi        J=1  P=-1  → Kp + Km
│  ├── Kp         J=0  P=-1   （末态）
│  ├── Km         J=0  P=-1   （末态）
├── f2         J=2  P=+1  → sigA + sigB
│  ├── sigA       J=0  P=+1  → Pip + Pim
│  │  ├── Pip        J=0  P=-1   （末态）
│  │  ├── Pim        J=0  P=-1   （末态）
│  ├── sigB       J=0  P=+1  → Pip2 + Pim2
│  │  ├── Pip2       J=0  P=-1   （末态）
│  │  ├── Pim2       J=0  P=-1   （末态）
```

**深度 = 3**（最长路径上的边数）。

## 2. 粒子表（自动导出）

| 粒子 | $J$ | 宇称 | 质量 | 角色 | 允许的 $l$ |
|---|---|---|---|---|---|
| `psi2s` | 1 | -1 | 3.686 | 顶层 | 0, 2, 4 |
| `phi` | 1 | -1 | 1.01946 | 节点 | 1 |
| `Kp` | 0 | -1 | 0.493677 | 末态 | - |
| `Km` | 0 | -1 | 0.493677 | 末态 | - |
| `f2` | 2 | 1 | 1.9 | 节点 | 2 |
| `sigA` | 0 | 1 | 0.5 | 节点 | 0 |
| `Pip` | 0 | -1 | 0.13957 | 末态 | - |
| `Pim` | 0 | -1 | 0.13957 | 末态 | - |
| `sigB` | 0 | 1 | 0.5 | 节点 | 0 |
| `Pip2` | 0 | -1 | 0.13957 | 末态 | - |
| `Pim2` | 0 | -1 | 0.13957 | 末态 | - |

## 3. 道表与逐节点 $(l,s)$（自动导出）

道表来源：**显式 `[[channel]]`**（14 条）。

| 道 | 顶层 $(l,s)$ | `phi` | `f0` | `sigA` | `sigB` |
|---|---|---|---|---|---|
| `u_phif0_ss_01` | (0,1) | (1,0) | (0,0) | (0,0) | (0,0) |
| `u_phif0_ss_01_b` | (0,1) | (1,0) | (0,0) | (0,0) | (0,0) |
| `u_phif0_ss_21` | (2,1) | (1,0) | (0,0) | (0,0) | (0,0) |
| `u_phif0_ss_21_b` | (2,1) | (1,0) | (0,0) | (0,0) | (0,0) |
| `u_phif2_ss_01` | (0,1) | (1,0) | - | (0,0) | (0,0) |
| `u_phif2_ss_01_b` | (0,1) | (1,0) | - | (0,0) | (0,0) |
| `u_phif2_ss_21` | (2,1) | (1,0) | - | (0,0) | (0,0) |
| `u_phif2_ss_21_b` | (2,1) | (1,0) | - | (0,0) | (0,0) |
| `u_phif2_ss_22` | (2,2) | (1,0) | - | (0,0) | (0,0) |
| `u_phif2_ss_22_b` | (2,2) | (1,0) | - | (0,0) | (0,0) |
| `u_phif2_ss_23` | (2,3) | (1,0) | - | (0,0) | (0,0) |
| `u_phif2_ss_23_b` | (2,3) | (1,0) | - | (0,0) | (0,0) |
| `u_phif2_ss_43` | (4,3) | (1,0) | - | (0,0) | (0,0) |
| `u_phif2_ss_43_b` | (4,3) | (1,0) | - | (0,0) | (0,0) |

## 4. 计数律（自动导出）

| 节点 | 允许的 $(l,s)$ | $N$ |
|---|---|---|
| `psi2s` | (0,1)  (2,1)  (2,2)  (2,3)  (4,3) | 5 |
| `phi` | (1,0) | 1 |
| `f2` | (2,0) | 1 |
| `sigA` | (0,0) | 1 |
| `sigB` | (0,0) | 1 |
| **合计** | | **5** |

⚠️ 深度 = 3 > 2 ⇒ omega **不可用**（它只算 `D0*D1*D2`，会静默丢掉最内层的角分布）；即使非顶层节点都只有一个 $(l,s)$ 也一样。

## 5. 表象与 `emit` 可行性（自动导出）

| 表象 | kernel | 需要手写道 | 实测 `emit` | 说明 |
|---|---|---|---|---|
| `helicity_cascade` | `helicity_cascade` | 否 | ✅ | 1 个脚本 |
| `cartesian` | `covariant` | 是 | ✅ | 1 个脚本 |
| `helicity` | `helicity` | 是 | ✅ | 1 个脚本 |

（`emit` 一列是**实测**结果 —— 逐表象单独调 `emitter.emit`，不是靠文件存在推断。）

手写道（`agent/tensor/chains/ex6_phi4pi/`，38 条）：`u_phif0_rr_01`, `u_phif0_rr_01_b`, `u_phif0_rr_21`, `u_phif0_rr_21_b`, `u_phif0_ss_01`, `u_phif0_ss_01_b`, `u_phif0_ss_21`, `u_phif0_ss_21_b`, `u_phif2_f2s_01`, `u_phif2_f2s_01_b`, `u_phif2_f2s_21`, `u_phif2_f2s_21_b`, `u_phif2_f2s_22`, `u_phif2_f2s_22_b`, `u_phif2_f2s_23`, `u_phif2_f2s_23_b`, `u_phif2_f2s_43`, `u_phif2_f2s_43_b`, `u_phif2_rr_01`, `u_phif2_rr_01_b`, `u_phif2_rr_21`, `u_phif2_rr_21_b`, `u_phif2_rr_22`, `u_phif2_rr_22_b`, `u_phif2_rr_23`, `u_phif2_rr_23_b`, `u_phif2_rr_43`, `u_phif2_rr_43_b`, `u_phif2_ss_01`, `u_phif2_ss_01_b`, `u_phif2_ss_21`, `u_phif2_ss_21_b`, `u_phif2_ss_22`, `u_phif2_ss_22_b`, `u_phif2_ss_23`, `u_phif2_ss_23_b`, `u_phif2_ss_43`, `u_phif2_ss_43_b`

深度 = 3 ⇒ omega **不可用**（`spec.py` 的分级检查会直接报错，不是静默出错）。

---

## 6. 叙述：做了什么、还有什么问题（手工）

### 这条链是什么

例 6 的 **σσ** 子分支（f_J→σσ$）。这是**拆分后**的三份 spec 之一。

例 6 原本写成一份 spec，2026-10-21 拆成三份，因为 **A1 是逐道判据**，
而三条子分支状态不同 —— 混在一份里"这份 spec 过不过 A1"没法回答。

| 项 | 值 |
|---|---|
| spec | `analyses/tensor_examples/specs/ex6_phi4pi_ss.toml` |
| 文档 | 见下表的行号 |
| 道数 | 14 |
| A1 | ✅ 全过（worst 7.35e-15） |

### 详细的推导、失败原因与记账方式

全部写在**共用的那份**笔记里：
[`ex6_phi4pi.notes.md`](notes/ex6_phi4pi.notes.md) —— 三条子分支放在一起讲，
因为它们共用同一套拓扑（$\psi\to\phi f_J$、\to$ 四个 $\pi$、$\phi\to K^+K^-$），
分开写会重复三遍。

那份笔记里有：为什么 $\sigma\sigma$ 精确、为什么 $\rho\rho$ 一半过一半不过、
为什么 $ 的 20 条不过、以及它们怎么在
`A1_KNOWN`/`A1_EXEMPT` 里记账。

### 三份 spec 的状态对照

| spec | 子分支 | 道数 | A1 |
|---|---|---|---|
| `ex6_phi4pi_ss` | \to\sigma\sigma$ | 14 | ✅ **7.35e-15** |
| `ex6_phi4pi_rr` | \to\rho\rho$ | 14 | f₀ ✅ **5.67e-15**；f₂ ⚠️ 0.47–0.86 |
| `ex6_phi4pi_f2s` | \to f_2(1270)\sigma$ | 10 | ⚠️ 0.47–1.08 |

⚠️ **不要拿 `rr` 里 $ 的 10 条、以及 `f2s` 的全部 10 条去拟合** ——
公式是文档逐式直译，但与级联对不上（无迹化的参考系问题）。

### 与原始 tex 的交叉核对

公式都回到 `documentation/jpsipwa1.tex`（原始投稿文件）核对过，
转写与 tex **一致**。行号对照见共用笔记的「与原始 tex 的交叉核对」一节。
