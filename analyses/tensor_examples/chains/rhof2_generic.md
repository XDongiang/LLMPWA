# 链 `rhof2_generic` —— psi2s → rho + f2 ; rho → Pip + Pim ; f2 → Pip2 + Pim2

> **本文件是生成物**，不要直接编辑。
> 事实部分由 [`gen_chain_docs.py`](gen_chain_docs.py) 从 [`agent/tensor/specs/rhof2_generic.toml`](../../../agent/tensor/specs/rhof2_generic.toml) 自动导出；
> 叙述部分来自 [`notes/rhof2_generic.notes.md`](notes/rhof2_generic.notes.md)。
> 改完重跑：`python analyses/tensor_examples/chains/gen_chain_docs.py`

---

## 1. 拓扑（自动导出）

```text
psi2s  J=1  P=-1
├── rho        J=1  P=-1  → Pip + Pim
│  ├── Pip        J=0  P=-1   （末态）
│  ├── Pim        J=0  P=-1   （末态）
├── f2         J=2  P=+1  → Pip2 + Pim2
│  ├── Pip2       J=0  P=-1   （末态）
│  ├── Pim2       J=0  P=-1   （末态）
```

**深度 = 2**（最长路径上的边数）。

## 2. 粒子表（自动导出）

| 粒子 | $J$ | 宇称 | 质量 | 角色 | 允许的 $l$ |
|---|---|---|---|---|---|
| `psi2s` | 1 | -1 | 3.686 | 顶层 | 0, 2, 4 |
| `rho` | 1 | -1 | 0.775 | 节点 | 1 |
| `Pip` | 0 | -1 | 0.13957 | 末态 | - |
| `Pim` | 0 | -1 | 0.13957 | 末态 | - |
| `f2` | 2 | 1 | 1.275 | 节点 | 2 |
| `Pip2` | 0 | -1 | 0.13957 | 末态 | - |
| `Pim2` | 0 | -1 | 0.13957 | 末态 | - |

## 3. 道表与逐节点 $(l,s)$（自动导出）

道表来源：**自动枚举**（5 条）。

| 道 | 顶层 $(l,s)$ | `rho` | `f2` |
|---|---|---|---|
| `rhof2_l0_s1` | (0,1) | (1,0) | (2,0) |
| `rhof2_l2_s1` | (2,1) | (1,0) | (2,0) |
| `rhof2_l2_s2` | (2,2) | (1,0) | (2,0) |
| `rhof2_l2_s3` | (2,3) | (1,0) | (2,0) |
| `rhof2_l4_s3` | (4,3) | (1,0) | (2,0) |

## 4. 计数律（自动导出）

| 节点 | 允许的 $(l,s)$ | $N$ |
|---|---|---|
| `psi2s` | (0,1)  (2,1)  (2,2)  (2,3)  (4,3) | 5 |
| `rho` | (1,0) | 1 |
| `f2` | (2,0) | 1 |
| **合计** | | **5** |

✅ 非顶层节点都只有 **1 个** $(l,s)$ ⇒ omega 的信息容量够用。

## 5. 表象与 `emit` 可行性（自动导出）

| 表象 | kernel | 需要手写道 | 实测 `emit` | 说明 |
|---|---|---|---|---|
| `cartesian` | `covariant` | 是 | ❌ | LookupError |
| `helicity` | `helicity` | 是 | ❌ | LookupError |
| `omega` | `omega` | 否 | ✅ | 1 个脚本 |

（`emit` 一列是**实测**结果 —— 逐表象单独调 `emitter.emit`，不是靠文件存在推断。）

⚠️ 本链**没有**手写道目录，所以 `cartesian`/`helicity` 两条路线的 `emit` 会报 `LookupError`；只有 `omega` 能出产物。

深度 = 2 ⇒ omega **可用**。

---

## 6. 叙述：做了什么、还有什么问题（手工）

### 这是一条**假想链**

⚠️ 物理上 $\psi\to\rho f_2$ 并不是一条真实的衰变道。它存在的唯一目的是
**验证生成器的通用性**。spec 文件头写得很清楚：

> 目的：验证 specgen 不写死 KK 道。这里
> * 顶层 daughters 自旋是 1 与 2（而非 1 与 0/2 的特例）
> * 道表**完全自动枚举**（没有 `[[channel]]` 段）
> * 末态是四个 pi，键名与 KK 道不同
>
> 只用来说明生成器可迁移；不是生产配置。

### 它测什么（三条通用性）

与 `phif0`/`phif2` 的对比，这条链专门覆盖三个"写死就会坏"的地方：

| 通用性维度 | φf₀ / φf₂ | `rhof2_generic` |
|---|---|---|
| 顶层 daughters 自旋 | $(1,0)$ / $(1,2)$ | $(1,2)$ —— 两个都非零 |
| 道表 | 显式 `[[channel]]` | **完全自动枚举**（0 条显式）|
| 末态键名 | `Kp/Km/Pip/Pim` | `Pip/Pim/Pip2/Pim2` |

自动枚举出的 $(l,s)$ 与 φf₂ **同一组** —— 因为顶层
$\psi\to(\text{自旋 }1)+(\text{自旋 }2)$ 的角动量结构与 $\psi\to\phi f_2$ 相同。
所以这条链其实是在验：**枚举逻辑只依赖自旋/宇称，不依赖粒子身份**。

测试覆盖：`test_generic_chain.py`（3 项）。

### 关键限制：能 `check`，不能 `emit`

这是它与其它四条链**本质的区别**，也是本次写文档时**新发现**的一条设计边界。

| 操作 | 结果 |
|---|---|
| `python -m agent.tensor check` | ✅ **OK**（拓扑 / 量子数 / 道表 / 表象校验通过）|
| `python -m agent.tensor emit`（`cartesian`） | ❌ `LookupError` |
| `python -m agent.tensor emit`（`helicity`） | ❌ `LookupError` |
| `python -m agent.tensor emit`（`omega`） | ✅ OK |

失败原因：

```
LookupError: 在 agent/tensor/kernels/covariant.py 里找不到 def rhof2_l0_s1
```

**协变/螺度路线的产物脚本需要逐道的手写公式**，而这条链没有
（`agent/tensor/chains/` 下没有 `rhof2` 目录）。`omega` 路线不需要手写道
（它走 $\Omega/K/M/g$），所以能出。

#### 这暴露的设计边界

**"spec 合法" ≠ "能生成产物"**。合法性检查（`check`）只看拓扑、量子数、
道表、表象约束，**不看手写道是否存在**。手写道是 emit 阶段才查找的。

所以这条链的正确定位是：

* ✅ **生成器逻辑的单元测试**（枚举、拓扑校验、表象分级）；
* ❌ **端到端回归**（跑不出产物脚本）。

如果将来要把它变成可 emit 的链，需要**手写 5 条协变道**
（`rhof2_l0_s1` … `rhof2_l4_s3`），并先用参考实现验证公式。

### 螺旋度级联内核能用它吗

**能算，但没有对拍对象。**

实测：级联内核可以正常算这条链（`cascade_channels` 给 5 条道，
第一个是 `rhof2_l0_s1`，逐节点 $(l,s)$ =
`{psi2s: [(0,1)], rho: [(1,0)], f2: [(2,0)]}`），振幅有限、非零。

但因为**没有手写协变道**，**没有 A1 可做** —— 无法验证级联算得对不对。
所以它**不能**作为级联内核的验收链，只能验"内核不崩溃"。

### 还有什么问题

| 项 | 状态 |
|---|---|
| 手写协变道 | ❌ **不存在**（这是它不能 emit 的原因）|
| 物理真实性 | ⚠️ **假想链**，不是真实衰变道 |
| 与参考实现对拍 | ❌ 无参考实现（`Tensor/FUNC.py` 里没有它）|
| 级联内核验证 | ❌ 无对拍对象，只能验"有限、非零" |
| 真实样本 / 拟合 | ❌ 不适用 |

### 它的价值（一句话）

`rhof2_generic` 是**唯一**一条验证"生成器不依赖 KK 道"的链。
它的存在保证了 `phif0`/`phif2` 的通过**不是**因为代码里写死了 `Kp/Km`。

它同时也划出了那条**设计边界**：`check` 通过不代表能 `emit` ——
手写道是 emit 阶段才查找的。

### 复现

```bash
cd <LLMPWA>

# 合法性检查（通过）
python -m agent.tensor check agent/tensor/specs/rhof2_generic.toml

# 端到端生成 —— **会失败**（没有手写道）
python -m agent.tensor emit agent/tensor/specs/rhof2_generic.toml

# 只有 omega 表象能出
python - <<'EOF'
import sys; sys.path.insert(0, ".")
from agent.tensor import spec as S, emitter as E
s = S.load_spec("agent/tensor/specs/rhof2_generic.toml")
for b in s["bases"]:
    s2 = dict(s); s2["bases"] = [b]
    try:
        r = E.emit(s2); print("  %-12s OK" % b["name"])
    except Exception as e:
        print("  %-12s FAIL: %s" % (b["name"], type(e).__name__))
EOF

# 框架测试
python agent/tensor/tests/test_generic_chain.py
```

### 相关

* [链 `phif2`](phif2.md) —— 枚举出同一组 $(l,s)$ 的生产链
* [链索引](README.md)
