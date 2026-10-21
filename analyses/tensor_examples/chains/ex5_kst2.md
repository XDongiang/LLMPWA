# 链 `ex5_kst2` —— psi2s → Kstar + Pim ; Kstar → phi + Pip ; phi → Kp + Km

> **本文件是生成物**，不要直接编辑。
> 事实部分由 [`gen_chain_docs.py`](gen_chain_docs.py) 从 [`analyses/tensor_examples/specs/ex5_kst2.toml`](../specs/ex5_kst2.toml) 自动导出；
> 叙述部分来自 [`notes/ex5_kst2.notes.md`](notes/ex5_kst2.notes.md)。
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
| `u_PF_1` | (1,2) | (3,1) | (1,0) |
| `u_PF_2` | (1,2) | (3,1) | (1,0) |
| `u_FP_1` | (3,2) | (1,1) | (1,0) |
| `u_FP_2` | (3,2) | (1,1) | (1,0) |
| `u_FF_1` | (3,2) | (3,1) | (1,0) |
| `u_FF_2` | (3,2) | (3,1) | (1,0) |

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

### 这条链是例 5 的 K*₂ 部分

$\psi\to K^+\pi^-K^-\pi^+$，文档 431–623 行。拓扑
$\psi\to K^*_2\pi\to(\phi\pi)\pi\to(K^+K^-)(\pi\pi)$ —— **深度 3**，
与 `b1pi` 同构（把 $b_1$ 换成 $K^*_2$）。

10 条道已从参考实现 `Tensor/FUNC.py` 逐式移植到
[`agent/tensor/chains/kstpi/`](../../../agent/tensor/chains/kstpi/)，
**与 jax 逐式对拍通过**（rel ≈ 1e-15，见
[`probes/verify_ex5_port.py`](../probes/verify_ex5_port.py)）。

⚠️ **2026-10-21 起，其中 4 条的对拍对象是 `*_paper.py`** —— `u_PP_*`/
`u_FP_*` 的生产文件已改为修正版（见下文"原因已定位"与
[`CORRECTION.md`](../../../agent/tensor/chains/kstpi/CORRECTION.md)）。
逐式移植保留在 `*_paper.py`，A5 的"逐位一致"落在它身上；
`verify_ex5_port.py` 会把"生产文件 vs 历史形式"的差作为阳性对照打出来。

### 移植时踩到的坑：jax 默认 float32

第一次对拍 10/10 全部"失败"，rel 在 1e-6…1e-8。看着像移植错了，
其实是 **jax 默认 float32**（$\varepsilon\approx1.2\times10^{-7}$）。

必须在**导入 `FUNC` 之前**：

```python
import jax
jax.config.update("jax_enable_x64", True)
```

（`test_covariant_vs_jax.py` 里早有这一步，我漏抄了。）
打开 x64 后立刻 10/10 到 1e-15。

⚠️ **这个坑同样污染了级联可表示性探针** —— 见 §"级联判定"。

### 为什么拆成两份 spec

$K^*_0$ 的 $J=0$ 与 $K^*_2$ 的 $J=2$ **冲突**：一个 `[particles.Kstar]`
条目只能有一个 $J$。协变道函数本身不读 $J$（公式硬编码），所以"能不能算"
不受影响，但拓扑校验需要一个确定的 $J$，只能拆。

本文件是 $K^*_2$ 的 8 条；$K^*_0$ 的 2 条在
[`ex5_kst0`](ex5_kst0.md)。

### 验收：移植 10/10

```bash
# 移植正确性（需要 jax + Tensor/，注意下面那个 x64 坑）
# 现在对 `*_paper.py` 比对 —— 那才是逐式移植
analyses/kk_dis/.venv/bin/python analyses/tensor_examples/probes/verify_ex5_port.py

# 历史形式 vs 修正版（三形式对拍）
python analyses/tensor_examples/probes/probe_ex5_paper_form.py

# 纯 numpy 回归（永远能跑，不需要 jax）
python agent/tensor/tests/test_ex5_channels.py
```

* **移植**：10/10 到 `rel ≈ 1e-15`（`verify_ex5_port.py`）。
* **生成**：`check` / `emit` / 端到端跑产物，**三种表象**
  （`cartesian` / `helicity` / `helicity_cascade`）全部通过。

### 与 `b1pi` 同样的设计瑕疵

8 条道是**两棵树**（旁观者 $\pi$ 不同），靠逐道 `tree` 覆盖塞进一份 TOML。
协变张量道函数完全不读树，所以能共存；但真实拟合里"两棵树是否共享
传播子/参数"仍未处理。见 [`b1pi`](b1pi.md) 的同名小节。

### 级联判定：本链 4 / 8 命中

用与 `b1pi` 相同的手法穷举「树 × 顶层 $(l,s)$ × 内层 $(l,s)$」，
判据是"$W$ 是否落在该 $K^*$ 自旋下全部允许 $(l,s)$ 组合张成的子空间里"：

```bash
python analyses/tensor_examples/probes/probe_ex5_identify.py
python analyses/tensor_examples/probes/probe_ex5_identify.py --control
```

| 道 | 顶层 $(l,s)$ | 内层 $(l,s)$ | rel | 结论 |
|---|---|---|---|---|
| `u_PF_1` | (1,2) | (3,1) | 1.8e-14 | ✅ 是级联 |
| `u_PF_2` | (1,2) | (3,1) | 3.7e-15 | ✅ 是级联 |
| `u_FF_1` | (3,2) | (3,1) | 6.3e-15 | ✅ 是级联 |
| `u_FF_2` | (3,2) | (3,1) | 3.7e-15 | ✅ 是级联 |
| `u_PP_1` | (1,2) | (1,1) | 6.7e-03 | ❌ 不是 |
| `u_PP_2` | (1,2) | (1,1) | 1.7e-01 | ❌ 不是 |
| `u_FP_1` | (3,2) | (1,1) | 5.7e-03 | ❌ 不是 |
| `u_FP_2` | (3,2) | (1,1) | 2.3e-01 | ❌ 不是 |

**本链 4 / 8。** 规律很清楚：内层是 **F 波**（$l=3$）的四条全部精确命中；
内层是 **P 波**（$l=1$）且 $K^*$ 自旋为 2 的四条落在级联子空间之外。

⚠️ 别把两个数搞混：探针输出的是**例 5 全部 10 条道**的结果（**6 / 10**），
它把 `ex5_kst0` 的 2 条也算进去了。本链只有 8 条，所以是 **4 / 8**。
$K^*_0$ 那 2 条见 [`ex5_kst0`](ex5_kst0.md)。

也就是说：$K^*_0$ 的内层**也是** P 波却命中 —— 所以不是"P 波一律不行"，
而是 $(J_{K^*}=2,\ l_{\text{内}}=1)$ 这个特定组合不行。

#### 原因已定位（2026-10-21）：内层无迹化用错了参考系

**结论**：这四条道与 `phif2` 的 (41)(42) 缺陷**同型** ——
`ZB_psi_to_phif2_原式缺陷与修正.md` §4 的**母系自旋投影错配**。

内层是 P 波（$l=1$）时，"两个秩-1 分析器耦合成 $K^*_2$ 的自旋 2"这一步
需要一次**无迹化**（$1\otimes1\to2$ 要去掉自旋 0 的迹）。原式把这个迹
用 $\tilde g(p_\psi)$ 去掉 —— 那是 **$\psi$ 系**的度规，而耦合发生在
**$K^*_2$ 系**（$K^*_2$ 在 $\psi$ 静止系里是动的，$|p|\approx0.55$ GeV，
实测 $\max|\tilde g(p_{K^*_2})-\tilde g(p_\psi)| = 0.32$）。

把内层换成用 $\tilde g(p_{K^*_2})$ 无迹化的 $T^{(2)}$：

| 道 | 原式 rel | 修正后 rel |
|---|---|---|
| `u_PP_1` | 6.99e-03 | **7.17e-15** |
| `u_PP_2` | 2.47e-01 | **5.53e-15** |
| `u_FP_1` | 6.55e-03 | **5.95e-15** |
| `u_FP_2` | 2.68e-01 | **3.51e-15** |

（四个种子 3/7/11/23 上都是这个量级，见
[`probes/probe_ex5_why.py`](../probes/probe_ex5_why.py)。）

**为什么只有内层 P 波出问题**：内层 F 波用 `T3_cov(kst2, phi, π)`，
它**自带** $rrr-\frac15\tilde r^2(\tilde g\tilde r+\ldots)$，其中的
$\tilde g$ 就是 $\tilde g(p_{K^*_2})$ —— 无迹化已经在**正确的系**里做完，
$P^{(2)}(p_\psi)$ 只剩顶层的作用，不再引入错配。

**为什么 $K^*_0$ 的内层也是 P 波却命中**：$J_{K^*}=0$ ⇒ 内层
$(l,s)=(1,1)$ 耦合成 **0**，是标量，**根本没有秩-2 无迹化这一步**。

⚠️ 注意与 `rhopi` 的区别：`rhopi` 的 Eq.(46) 里内层 $\epsilon$ 缩并的是
$p_\psi$ 而不是 $p_{\rho'}$、中间**没有投影算符**，把它换成
$\tilde g(p_{\rho'})$ 也不落回级联（最好 6.4e-02，见
[`probes/probe_rhopi_why.py`](../probes/probe_rhopi_why.py)）——
那是**另一种**失配。本链是"投影装在了错的系里"，`rhopi` 是"根本没装"。

⚠️ 起初这个修正**只作为诊断结论**记录在这里（`chains/kstpi/` 那 10 条是
`Tensor/FUNC.py` 的逐式移植，A5 要求逐位一致，改了会与参考实现分家）。

**2026-10-21 已落地，用的是"两种形式共存"的方案** —— 不改 A5、又让生产
用修正版：

| 文件 | 形式 | A1 | 用途 |
|---|---|---|---|
| `u_PP_{1,2}.py`、`u_FP_{1,2}.py` | **修正版** | ✅ ~5e-15 | **生产** |
| `u_PP_{1,2}_paper.py`、`u_FP_{1,2}_paper.py` | 历史形式（逐式移植） | ❌ 6e-3…2.7e-1 | A5 对拍、对照基线 |

`emitter` 按 **`channel.name`** 取文件，所以两种形式**不需要改代码**就能
分别生成 —— 把 spec 里的道名写成 `*_paper` 即得历史形式。

* 完整推导 + 公式/代码逐行对照：
  [`agent/tensor/chains/kstpi/CORRECTION.md`](../../../agent/tensor/chains/kstpi/CORRECTION.md)
* 两种形式共存的对照 spec：
  [`specs/ex5_kst2_paper.toml`](../specs/ex5_kst2_paper.toml)
* 三形式对拍：
  [`probes/probe_ex5_paper_form.py`](../probes/probe_ex5_paper_form.py)
* A5（现在对 `*_paper` 文件比对）：
  [`probes/verify_ex5_port.py`](../probes/verify_ex5_port.py)

新增的内核原语：`kernels/covariant.py::T2_from_pair(frame, u, v)`
—— 把两个**任意**秩-1 分析器耦合成 $J=2$，并**在指定系里**无迹化。

#### 阳性对照（这个判据可信的前提）

同一套机器跑 `b1pi` 的 8 条道：**8/8 到 6e-15**。
所以"4/8"不是探针坏了 —— 见
[`probes/probe_ex5_identify.py`](../probes/probe_ex5_identify.py) 的 `control()`。

#### 已知的坑（会伪造出"未命中"）

探针第一版**没有**打开 jax x64，于是拿 float32 的参考实现比 float64 的
级联，得到一堆 1e-6…1e-9 的**假未命中**，还伪造出一个"`_1` 族全败、
`_2` 族全中"的假不对称。改用已移植的协变内核（本仓库、float64）后
不对称消失。

⚠️ **教训**：跨库对拍时，精度设置是承重约定。

### 还没有端到端脚本

`check_ex3.py` **只覆盖例 3**（读 `data/` 的 `phi f2` 动量、验 L1/L2/L3 + A1/A3/A7）。
例 5 目前只由 `test_ex5_channels.py`（纯 numpy 回归）+ `emit` 产物跑通来保证，
**没有**等价的 `check_ex5.py` —— 也就是说例 5 缺 L1/L2/L3 那三层。

真实样本、拟合侧端到端也都没做（与例 3 相同）。

### 还有什么问题

| 项 | 状态 |
|---|---|
| **$u_{PP}$/$u_{FP}$ 为何不是级联** | ✅ **已定位并已修（2026-10-21）**：内层 P 波的无迹化用了 $\tilde g(p_\psi)$ 而不是 $\tilde g(p_{K^*_2})$（母系自旋投影错配，与 `phif2` (41)(42) 同型）。生产文件已改为修正版（A1 ~5e-15），历史形式保留为 `*_paper.py`。见 [`CORRECTION.md`](../../../agent/tensor/chains/kstpi/CORRECTION.md) |
| 与文档 Eq 的逐条对应 | ⚠️ **未敲定**。文档 431–623 行给的是 $\langle K^*K^*_J|LS\rangle$ 一族，而参考实现按自己的 $P/F$、$l/r$ 命名。移植以**参考实现为准** |
| $\psi\to K^*_2\pi$ 的宇称 | ⚠️ spec 里写 $K^*_2$ 为 $2^-$（宇称+三角要求 $l$ 为奇数）。若实际取 $2^+$（$l$ 偶），道表要重列 |
| $K^*$ 质量 | ⚠️ spec 写 1.43，**只是占位**；协变道函数对任意四动量都成立 |
| 真实样本 / 拟合侧 | ❌ 未做（与例 3 相同） |
| 节点级 Eq.52 消融 | ❌ 未做 |

### 复现

```bash
cd <LLMPWA>

# 移植正确性（需要 jax + Tensor/，注意 x64）
analyses/kk_dis/.venv/bin/python analyses/tensor_examples/probes/verify_ex5_port.py

# 纯 numpy 回归（永远能跑）
python agent/tensor/tests/test_ex5_channels.py

# 级联可表示性 + 阳性对照
python analyses/tensor_examples/probes/probe_ex5_identify.py
python analyses/tensor_examples/probes/probe_ex5_identify.py --control

# 未命中的根因（内层无迹化用错参考系）+ 修正验证
python analyses/tensor_examples/probes/probe_ex5_why.py

# 端到端生成
python -m agent.tensor check analyses/tensor_examples/specs/ex5_kst2.toml
python -m agent.tensor emit  analyses/tensor_examples/specs/ex5_kst2.toml
```

### 相关

* [链 `ex5_kst0`](ex5_kst0.md) —— 同一移植的 $J=0$ 部分
* [链 `b1pi`](b1pi.md) —— 同构拓扑，级联验收的主力
* [链索引](README.md)
