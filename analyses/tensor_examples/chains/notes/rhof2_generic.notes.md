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

### 三条路线都能 `emit`（2026-10-21 起）

**2026-10-21 之前**这条链只能 `check`、不能 `emit` —— 协变/螺度路线需要
逐道的手写公式，而 `agent/tensor/chains/` 下当时没有 `rhof2` 目录，
`emit` 会报 `LookupError`。

**现在补上了**：`agent/tensor/chains/rhof2/` 有 5 条道，
是 `chains/phif2/` 的**逐式改名**（$\phi\to\rho$、$K^\pm\to\pi^\pm$），
公式一个数字都没动 —— 两条链**拓扑完全相同**（同 $(L,S)$、同 $J$）。
先由 [`probes/probe_missing_covariant.py`](../probes/probe_missing_covariant.py)
用改名后的 `phif2` 函数与级联逐道对拍到 1.2e-15，确认拓扑等价，才写成文件。

| 操作 | 结果 |
|---|---|
| `python -m agent.tensor check` | ✅ OK（拓扑 / 量子数 / 道表 / 表象校验通过）|
| `python -m agent.tensor emit`（`cartesian`） | ✅ OK |
| `python -m agent.tensor emit`（`helicity`） | ✅ OK |
| `python -m agent.tensor emit`（`omega`） | ✅ OK |

#### 仍然成立的设计边界

**"spec 合法" ≠ "能生成产物"**。合法性检查（`check`）只看拓扑、量子数、
道表、表象约束，**不看手写道是否存在**。手写道是 emit 阶段才查找的 ——
这条边界没有变，只是这条链现在把两边都满足了。

### 螺旋度级联内核能用它吗

**能算，且现在有对拍对象了。**

实测：级联内核可以正常算这条链（`cascade_channels` 给 5 条道，
第一个是 `rhof2_l0_s1`，逐节点 $(l,s)$ =
`{psi2s: [(0,1)], rho: [(1,0)], f2: [(2,0)]}`），振幅有限、非零。

补上手写协变道后，**A1 可做了**：级联 vs 协变逐道到 **1.7e-15**
（见 [`probes/probe_a1_full.py`](../probes/probe_a1_full.py)）。
所以它现在**能**作为级联内核的验收链 —— 只是它验证的是"改名后拓扑仍自洽"，
不是"新的物理分波"。

### 还有什么问题

| 项 | 状态 |
|---|---|
| 手写协变道 | ✅ 5 条（2026-10-21，`chains/rhof2/`，改名自 `phif2`）|
| 物理真实性 | ⚠️ **假想链**，不是真实衰变道 |
| 与参考实现对拍 | ❌ 无参考实现（`Tensor/FUNC.py` 里没有它）；对拍对象是**级联内核** |
| 级联内核验证 | ✅ A1 到 1.7e-15 |
| 真实样本 / 拟合 | ❌ 不适用 |

### 它的价值（一句话）

`rhof2_generic` 是**唯一**一条验证"生成器不依赖 KK 道"的链。
它的存在保证了 `phif0`/`phif2` 的通过**不是**因为代码里写死了 `Kp/Km`。

它同时也划出了那条**设计边界**：`check` 通过不代表能 `emit` ——
手写道是 emit 阶段才查找的（这条链 2026-10-21 补齐后才两边都满足）。

### 复现

```bash
cd <LLMPWA>

# 合法性检查（通过）
python -m agent.tensor check agent/tensor/specs/rhof2_generic.toml

# 端到端生成（三条路线都能出）
python -m agent.tensor emit agent/tensor/specs/rhof2_generic.toml

# 手写协变道 vs 级联（A1）
python analyses/tensor_examples/probes/probe_a1_full.py
```
