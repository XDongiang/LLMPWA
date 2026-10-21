# 链 `ex5_kst2_paper` —— psi2s → Kstar + Pim ; Kstar → phi + Pip ; phi → Kp + Km

> **本文件是生成物**，不要直接编辑。
> 事实部分由 [`gen_chain_docs.py`](gen_chain_docs.py) 从 [`analyses/tensor_examples/specs/ex5_kst2_paper.toml`](../specs/ex5_kst2_paper.toml) 自动导出；
> 叙述部分来自 [`notes/ex5_kst2_paper.notes.md`](notes/ex5_kst2_paper.notes.md)。
> 改完重跑：`python analyses/tensor_examples/chains/gen_chain_docs.py`

---

## 1. 拓扑（自动导出）

```text
psi2s  J=1  P=-1
├── Kstar      J=2  P=-1  → phi + Pip
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
| `psi2s` | 1 | -1 | 3.686 | 顶层 | 1, 3 |
| `Kstar` | 2 | -1 | 1.43 | 节点 | 1, 3 |
| `phi` | 1 | -1 | 1.019 | 节点 | 1 |
| `Kp` | 0 | -1 | 0.493677 | 末态 | - |
| `Km` | 0 | -1 | 0.493677 | 末态 | - |
| `Pip` | 0 | -1 | 0.13957 | 末态 | - |
| `Pim` | 0 | -1 | 0.13957 | 末态 | - |

## 3. 道表与逐节点 $(l,s)$（自动导出）

道表来源：**显式 `[[channel]]`**（8 条）。

| 道 | 顶层 $(l,s)$ | `Kstar` | `phi` |
|---|---|---|---|
| `u_PP_1` | (1,2) | (1,1) | (1,0) |
| `u_PP_2` | (1,2) | (1,1) | (1,0) |
| `u_FP_1` | (3,2) | (1,1) | (1,0) |
| `u_FP_2` | (3,2) | (1,1) | (1,0) |
| `u_PP_1_paper` | (1,2) | (1,1) | (1,0) |
| `u_PP_2_paper` | (1,2) | (1,1) | (1,0) |
| `u_FP_1_paper` | (3,2) | (1,1) | (1,0) |
| `u_FP_2_paper` | (3,2) | (1,1) | (1,0) |

## 4. 计数律（自动导出）

| 节点 | 允许的 $(l,s)$ | $N$ |
|---|---|---|
| `psi2s` | (1,2)  (3,2) | 2 |
| `Kstar` | (1,1)  (3,1) | 2 |
| `phi` | (1,0) | 1 |
| **合计** | | **4** |

⚠️ 非顶层节点 `Kstar` **有多个** $(l,s)$ ⇒ omega 的信息容量**不够**（它的 cache 里非顶层节点只有标量 $K_i$）。

## 5. 表象与 `emit` 可行性（自动导出）

| 表象 | kernel | 需要手写道 | 实测 `emit` | 说明 |
|---|---|---|---|---|
| `cartesian` | `covariant` | 是 | ✅ | 1 个脚本 |
| `helicity` | `helicity` | 是 | ✅ | 1 个脚本 |
| `helicity_cascade` | `helicity_cascade` | 否 | ✅ | 1 个脚本 |

（`emit` 一列是**实测**结果 —— 逐表象单独调 `emitter.emit`，不是靠文件存在推断。）

手写道（`agent/tensor/chains/kstpi/`，14 条）：`u_FF_1`, `u_FF_2`, `u_FP_1`, `u_FP_1_paper`, `u_FP_2`, `u_FP_2_paper`, `u_PF_1`, `u_PF_2`, `u_PP_1`, `u_PP_1_paper`, `u_PP_2`, `u_PP_2_paper`, `u_kst0_l`, `u_kst0_r`

深度 = 3 ⇒ omega **不可用**（`spec.py` 的分级检查会直接报错，不是静默出错）。

---

## 6. 叙述：做了什么、还有什么问题（手工）

### 这是一份**对照 spec**，不是生产链

`ex5_kst2_paper` 与 [`ex5_kst2`](ex5_kst2.md) 共用同一条链目录
（`agent/tensor/chains/kstpi/`）与同一套拓扑，**唯一区别是道名**：

| 道名 | 取哪个文件 | 形式 | A1 |
|---|---|---|---|
| `u_PP_1` / `u_PP_2` / `u_FP_1` / `u_FP_2` | `u_*.py` | **修正版（生产）** | ✅ $\sim10^{-14}$ |
| `u_PP_1_paper` / … | `u_*_paper.py` | 历史形式（`FUNC.py` 逐式移植） | ❌ $10^{-2}\sim10^{-1}$ |

`emitter` 按 **`channel.name`** 去 `chains/<chain_name>/<channel.name>.py`
取函数体，所以"生成论文形式"**不需要改任何代码** —— 只要把 spec 里的
道名写成 `*_paper`。本 spec 把两种形式放进**同一份产物**，便于逐道对照。

### 为什么这份 spec 的 A1 必然失败

4 条 `*_paper` 道**就是缺陷本身**（内层无迹化用错参考系）。
所以：

* `agent/tensor/tests/test_handwritten_complete.py` 把它列入
  `A1_SKIP_SPECS`（整份跳过 A1，只验 L1/L2/L3）；
* `analyses/tensor_examples/probes/verify_all_specs.py` 也跳过它。

**生产请用 `ex5_kst2`**（`u_PP_*`/`u_FP_*` 已改为修正版）。

### 生成与对照

```bash
cd <LLMPWA>

# 生成两种形式共存的产物
python -m agent.tensor check analyses/tensor_examples/specs/ex5_kst2_paper.toml
python -m agent.tensor emit  analyses/tensor_examples/specs/ex5_kst2_paper.toml

# 三形式对拍：历史形式（应失败）vs 修正版（应到机器精度）
python analyses/tensor_examples/probes/probe_ex5_paper_form.py
```

### 机制与公式

完整推导、公式对照、代码逐行对照见
[`agent/tensor/chains/kstpi/CORRECTION.md`](../../../agent/tensor/chains/kstpi/CORRECTION.md)。

### 相关

* [链 `ex5_kst2`](ex5_kst2.md) —— 生产链（修正版）
* [链 `ex5_kst0`](ex5_kst0.md) —— 同一移植的 $J=0$ 部分
* [链索引](README.md)
