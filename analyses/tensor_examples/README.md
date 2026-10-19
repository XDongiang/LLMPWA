# tensor_examples —— 文档公式例子的回归分析

本目录把 `analyses/kk_dis/document/ψ衰变到介子的分波分析协变张量形式.md`
里的衰变例子**变成可跑的回归测试**，用来验收框架级张量生成器
（[`agent/tensor/`](../../agent/tensor/)）。

它不是生产分析：没有拟合、没有 `llm_config_*.toml`、没有 `data/` 真实样本。
目的只有一个 —— 给「框架新加一条道」这件事一个**独立于 kk_dis** 的验收面。

目前覆盖**例 3**（`ψ→φπ⁺π⁻`，ρ′ + b₁ 共 10 条道）与**例 5**
（`ψ→K⁺π⁻K⁻π⁺`，K*₀ + K*₂ 共 10 条道），合计 **20 条道**。

## 现状

| 例子 | 文档行 | 状态 | 道 |
|---|---|---|---|
| 例 3 `ψ→φπ⁺π⁻` 的 ρ′ 道 | 374–387 | ✅ 已实现 | `u_rho_1` `u_rho_2`（Eq.46） |
| 例 3 `ψ→φπ⁺π⁻` 的 b₁ 道 | 391–416 | ✅ 已实现 | `u_{SS,SD,DS,DD}_{1,2}`（Eq.47–50） |
| 例 5 `ψ→K⁺π⁻K⁻π⁺` 的 K*₀/K*₂ 道 | 431–623 | ✅ 已实现（2026-10-19） | 10 条（`u_kst0_*` + `u_PP/PF/FP/FF_*`） |
| 例 1/2 三体（置换求和模型） | 200–304 | ⬜ 待定模型 | — |
| 例 7–11 辐射衰变 | 845–1383 | ⬜ 需新基建 | 秩 2 振幅 + 横向投影 |
| 例 4/6/12 | — | ❌ 不做 | 三体 / 深度 3 的 omega / 文档为空 |

可行性评估（含实测数据）见
[`documentation/tensor_examples_feasibility.md`](../../documentation/tensor_examples_feasibility.md)。

## 目录

```text
analyses/tensor_examples/
├── README.md               ← 本文件
├── chains/                 ← **每条链一份文档**（做了什么 / 还有什么问题）
│   ├── README.md           ← 链文档索引 + 三条边界
│   ├── gen_chain_docs.py   ← 从 spec 自动导出"事实"部分（见下）
│   ├── notes/              ← 手工"叙述"部分（每链一份 .notes.md）
│   ├── phif0.md            ← ψ→φf₀   （深度 2，三表象）  ← 生成物
│   ├── phif2.md            ← ψ→φf₂   （深度 2，三表象）  ← 生成物
│   ├── b1pi.md             ← ψ→b₁π   （深度 3，验收主力）← 生成物
│   ├── rhopi.md            ← ψ→ρ′π   （深度 3，边界）    ← 生成物
│   ├── rhof2_generic.md    ← ψ→ρf₂   （假想链）          ← 生成物
│   ├── ex5_kst2.md         ← ψ→K*₂π  （例 5，4/8 是级联）← 生成物
│   └── ex5_kst0.md         ← ψ→K*₀π  （例 5，2/2 是级联）← 生成物
├── specs/
│   ├── ex3_rhopi.toml      ← 例 3 的 ρ′ 树（2 条道）
│   ├── ex3_b1pi.toml       ← 例 3 的 b₁ 树（8 条道）
│   ├── ex5_kst2.toml       ← 例 5 的 K*₂ 树（8 条道）
│   └── ex5_kst0.toml       ← 例 5 的 K*₀ 树（2 条道）
├── data/
│   └── make_momenta.py     ← 合成 ψ→φπ⁺π⁻→(K⁺K⁻)(π⁺π⁻) 四动量（自带守恒自检）
├── check_ex3.py            ← 端到端回归：生成 → 跑 → 三层自检 + A1/A3/A7（**仅例 3**）
├── probes/                 ← 一次性诊断脚本（复现 roadmap §7 与边界实测）
│   ├── verify_ex5_port.py  ← 例 5 的 10 条道 vs jax 逐式对拍（10/10，1e-15）
│   ├── probe_ex5_identify.py ← 例 5 级联可表示性穷举 + 阳性对照（b₁ 8/8）
│   └── …                   ← ρ′/b₁ 的边界探针、行序探针、A1 错误注入等
└── generated/  out/        ← 生成物与运行产物（.gitignore，可随时重建）
```

**每条链的详细文档在 [`chains/`](chains/README.md)** —— 覆盖七条链
（`phif0` / `phif2` / `b1pi` / `rhopi` / `rhof2_generic` / `ex5_kst2` /
`ex5_kst0`）各自的拓扑、分波、表象可用性、验收数据与遗留问题。

这些文档是**生成物**：拓扑/道表/计数律/表象可用性从 TOML 自动导出，
"做了什么、还有什么问题"手写在 `chains/notes/`。改了 spec 要重跑：

```bash
python analyses/tensor_examples/chains/gen_chain_docs.py           # 重生成
python analyses/tensor_examples/chains/gen_chain_docs.py --check   # 只检查
python agent/tensor/tests/test_chain_docs.py                       # 已挂进测试
```

**为什么是四份 spec 而不是两份**：两个**不同**的理由，别混。

| spec 对 | 为什么要拆 |
|---|---|
| `ex3_rhopi` / `ex3_b1pi` | **树不同**（中间态分别是 `phi+π⁻` 与 `phi+π⁺`），一棵树只能描述一组配对 |
| `ex5_kst2` / `ex5_kst0` | **自旋相同不了**（K*₂ 是 $J=2$、K*₀ 是 $J=0$），而一个 `[particles.*]` 条目只能有一个 `J` |

前者是拓扑冲突，后者是量子数冲突。分拆是最简明的解法 —— 这也是用户指定的
结构（`chain.name/chain.channel.py` 存储、生成时组装）。

⚠️ 每份 spec 内部仍然有"两棵树塞在一起"的问题：`b1pi` 与 `ex5_kst2` 的
`_1`/`_2` 是**旁观者 π 对调**，靠逐道 `tree` 覆盖共存。协变道函数不读树
（`grep -c "chain\|nodes" agent/tensor/kernels/covariant.py` == 0），所以能工作；
但真实拟合里"两棵树是否共享传播子/参数"仍未处理。

## 跑

```bash
cd <LLMPWA>

# 1) 合成四动量（自带守恒 / 在壳 / 阈值自检）
analyses/kk_dis/.venv/bin/python analyses/tensor_examples/data/make_momenta.py \
    --out analyses/tensor_examples/data/ex3_phipipi.npz --n 2000 --seed 7

# 2) 看链 / 道表 / 表象
python -m agent.tensor show  analyses/tensor_examples/specs/ex3_rhopi.toml
python -m agent.tensor check analyses/tensor_examples/specs/ex3_b1pi.toml

# 3) 生成 + 直接跑（emit 出的脚本是自足的，不依赖 agent.tensor）
python -m agent.tensor emit analyses/tensor_examples/specs/ex3_rhopi.toml
analyses/kk_dis/.venv/bin/python \
    analyses/tensor_examples/generated/ex3_rhopi/gen_tensor_ex3_rhopi_covariant.py \
    --momenta analyses/tensor_examples/data/ex3_phipipi.npz \
    --out analyses/tensor_examples/out/ex3_rhopi --n 500

# 4) 一条命令跑完全部自检（推荐）
analyses/kk_dis/.venv/bin/python analyses/tensor_examples/check_ex3.py --n 500
```

`check_ex3.py` 做三层验证 + 三条判据：

1. **生成物 vs 框架内核**：产物脚本内联的道函数体算出的 cache，必须与直接调
   `agent.tensor.kernels.covariant.compute_event` 逐位一致（抓 emitter 组装 bug）；
2. **cartesian vs helicity**：`W = EPS_A · T`（抓 store 体写错）；
3. **有限性**：没有 NaN/Inf；
4. **A1**（新）：`helicity_cascade` 逐道对拍协变路线（经 `EPS_A`），rel < 1e-12；
5. **A3**（新）：b₁ 的 8 条道两两不重合（min pairwise rel > 1e-6）；
6. **A7**（新）：级联 cache 的形状/dtype 与 helicity **精确**一致。

框架侧的对应测试（不需要本目录的数据）：

```bash
python agent/tensor/tests/test_ex3_channels.py      # 8 项，纯 numpy，永远能跑
python agent/tensor/tests/test_ex5_channels.py      # 6 项，纯 numpy（例 5 的 10 条道）
python agent/tensor/tests/test_helicity_cascade.py  # 13 项：A1/A2/A3/A4 + 三条边界
python agent/tensor/tests/test_chain_docs.py        # 5 项：链文档与 spec 同步

# 与 jax 原版逐式对拍（需要 jax + Tensor/ 源码，缺了就跳过）
TENSOR_ROOT=<...>/tensor_example \
    python agent/tensor/tests/test_covariant_vs_jax.py   # phif0/phif2 的 7 条 + 例 3 的 10 条
analyses/kk_dis/.venv/bin/python analyses/tensor_examples/probes/verify_ex5_port.py
                                                         # 例 5 的 10 条
```

⚠️ 跑 jax 对拍**必须**在导入 `FUNC` 之前开 x64：

```python
import jax
jax.config.update("jax_enable_x64", True)
```

否则 jax 全程 float32（$\varepsilon\approx1.2\times10^{-7}$），对拍只能到 1e-7，
看起来像"移植错了"。这个坑第一次就踩了 —— 见下面例 5 那节。

## 螺旋度级联表象（`kernel = "helicity_cascade"`）

`documentation/helicity_amplitude_roadmap.md` 的实现落点。它按 tf-pwa 的
`HelicityDecay` + `DecayChain.get_amp` 逐节点构造振幅、再链级 `einsum` 缩并内线
螺旋度，所以**对链的深度没有任何假设**，也**每个节点都可以有多个 (l,s)** ——
这正是 omega 的两条能力边界（§1.1 / §1.2）。

| | omega / helicity | helicity_cascade |
|---|---|---|
| 逐节点 (l,s) | ❌ 非顶层节点只以标量 $K_i$ 出现 | ✅ 每节点一个列表 |
| 深度 | ❌ 只算 `D0*D1*D2` | ✅ 任意深度 |
| 逐节点螺旋度参考系 | ❌ 全部从顶层静止系直接 boost | ✅ 每层在**母节点**系里量 |
| cache 形状 | `(n_ch, N, 3)` complex | **逐位相同** |
| 拟合侧 | — | **零改动**（`_load_cached_tensor` 不变） |

TOML 侧新增两个 `[[channel]]` 字段：

```toml
[[channel]]
name = "u_SD_1"
l = 0
s = 1
# 逐节点 (l,s)：顶层波 × 内层波才是完备标签（SS/SD/DS/DD）
ls   = { psi2s = [[0,1]], b1 = [[2,1]], phi = [[1,0]] }
# 镜像树：`_2` 族的旁观者是 Pip
tree = { psi2s = ["b1","Pim"], b1 = ["phi","Pip"] }
```

`ls` 缺省时，顶层用道表的 `(l,s)`、其余衰变节点用 `allowed_ls`（宇称 + 三角
的**全部**允许值）；`tree` 缺省时沿用 `[particles.*].daughters`。两者都只在
`helicity_cascade` 表象里被读，但**在读 TOML 时**就校验（未知节点、宇称禁止的
`(l,s)`、改变末态集合的"镜像树"都会报错，而不是运行时 KeyError）。

### 已知边界：ρ′ 的 Eq.(46) 不是"逐节点核 + 常数复系数"的级联

> **三条边界一览**（细节各见其下）：
>
> | # | 边界 | 判据 | 状态 |
> |---|---|---|---|
> | 1 | omega 的**逐节点 $(l,s)$ 容量** | 非顶层节点在 cache 里只有标量 $K_i$ | 已由级联内核推开 |
> | 2 | omega 的**深度**（只算 `D0*D1*D2`） | 深度 ≥ 3 时最内层角分布被丢 | 已由级联内核推开 |
> | 3 | 某些**协变公式不是级联** | 落在级联子空间之外（ρ′ Eq.46；例 5 的 `u_PP_*`/`u_FP_*`） | **未解决** |

判据 A1 对 b₁ 的 8 条道全部通过（rel 6e-15 … 2.5e-14），但对 ρ′ 的
`u_rho_1`/`u_rho_2` **不通过**（rel 0.24 … 0.32，随样本变 —— 这本身就是
"逐事例失配"的征兆）。三条独立证据（都不依赖反演、不依赖秩）：

| 证据 | b₁（控制） | ρ′ |
|---|---|---|
| 直接最小二乘（自由复系数） | 6e-15 | **2.72e-01** |
| 逐事例 $c(e)$ 的 $\lvert c/c_0-1\rvert_{\max}$ | 9e-13（常数） | **3.4e+01** |
| 级联族 vs 协变族的子空间主角 | 8 维 vs 8 维，$\sim10^{-6}$ 度 | 2 维 vs 2 维，**16 度** |

**维数相同**（ρ′ 两边都是 2）是关键：这不是"级联少了几维"，而是
"两个 2 维子空间张成的**方向**不同"。

公式结构上的原因：

* b₁ 的 Eq.(47)–(50) 含**中间态的投影算符**
  $\tilde g_{\mu\nu}(p_{b_1})$ / $\tilde t^{(2)}_{\mu\nu}(p_{b_1})$ ——
  这就是"逐节点级联"的协变写法，所以与 `helicity_cascade` 逐位一致。
* ρ′ 的 Eq.(46) 是 $\epsilon^\mu{}_{\alpha\beta\gamma}p_\psi^\alpha
  [\ldots \epsilon^{\gamma\delta\sigma\lambda}p_{\psi\delta}\ldots]$ ——
  两个 $\epsilon$ 张起来、**内层 $\epsilon$ 缩并的是 $p_\psi$ 而不是 $p_{\rho'}$**，
  中间没有投影算符。

排除过的替代（见 `probes/probe_rho_qfactor.py`、`probe_rho_rank.py`）：

* 内层 $\epsilon$ 换成 $p_{\rho'}$（rel 0.29 → 0.36，没改善）；
* 内层 $(l,s)$ 加项 —— ρ′ 内层由宇称 + 三角**只有** $(1,1)$ 一项，没有别的可选；
* 各种事件依赖标量因子（$q_{\text{内}}$、$B_1$、$q_{\text{外}}$、$m^2_{\rho'}$、
  $\gamma_\phi$、$\gamma_\pi$、以及全部一起）—— 残差全部**不降反升**；
* 两棵树 × 两种参考系 × 四组内层 $(l,s)$ 穷举 —— 最好 2.1e-01。

**结论（收窄后的版本）**：ρ′ 的协变 Eq.(46) 与"当前逐节点核 + 常数复系数
的级联"是**两种不同的振幅**，这一条已由上面三条证据确立。
但**不能**说"ρ′ 不是任何意义上的级联"—— 一个带张量结构的事件依赖投影
并未被穷举排除。**Eq.(46) 与逐节点级联在物理上是否等价，目前未知**；
要澄清需要把 Eq.(46) **重写**成带 $\rho'$ 投影算符的级联形式，
而不是去拟合它。

所以 `check_ex3.py` 对 ρ′ 设 `"a1": False`（只验 A7 形状与有限性）。
独立复算脚本见
`analyses/kk_dis/task/20261019_螺旋度级联内核_审阅/material/verify.py`。

## 例 3 的物理要点（读道函数前先看这段）

粒子编号与文档一致：`K⁺=1`、`K⁻=2`、`π⁺=3`、`π⁻=4`。

* **ρ′ 族**（`chains/rhopi/`）：`ψ→ρ′π` 与 `ρ′→φπ` 都限于 P 波，所以只有
  **一个**独立振幅；`u_rho_1`/`u_rho_2` 是"哪一个 π 是旁观者"的两种取法。
* **b₁ 族**（`chains/b1pi/`）：`ψ→b₁π` 与 `b₁→φπ` 都可以既有 S 波又有 D 波，
  所以有**四个**独立振幅，完备标签是"顶层波 × 内层波"（SS/SD/DS/DD）。
  对应对调旁观者 π 又多一倍 → 8 条道。
* ⚠️ `_1`/`_2` 的下标在**两个族里含义相反**（照抄参考实现的命名）：
  ρ′ 族 `_1` 的旁观者是粒子 3，b₁ 族 `_1` 的旁观者是粒子 4。
* ⚠️ 顶层 `(l,s)` **不是** b₁ 族的完备标签（8 条道只映射到 2 组 `(l,s)`），
  所以两份 spec 都设 `check_channels = false`。
* ⚠️ 三层树（`ψ→X π`，`X→φπ`，`φ→K⁺K⁻`）**不能出 omega 表象** ——
  omega 内核逐节点读树、只算 `D0*D1*D2`。这条限制由
  `agent/tensor/spec.py` 按 `[[basis]]` 分级执行。
  现在可以改出 `helicity_cascade`：它对深度没有假设。

## 例 5（`ψ→K⁺π⁻K⁻π⁺`）的落地（2026-10-19）

10 条道已从参考实现 `Tensor/FUNC.py` 逐式移植到
[`agent/tensor/chains/kstpi/`](../../agent/tensor/chains/kstpi/)
（`u_kst0_{l,r}` + `u_{PP,PF,FP,FF}_{1,2}`）。

### 拓扑：深度 3，与 b₁ 同构

$\psi\to K^*\pi \to (\phi\pi)\pi \to (K^+K^-)(\pi\pi)$ —— 把 b₁ 换成 K*，
**其余结构与例 3 的 b₁ 完全一样**。所以 omega 同样不可用、级联同样可用。

### 为什么拆成 K*₂ / K*₀ 两份

K*₂ 是 $J=2$、K*₀ 是 $J=0$，一个 `[particles.*]` 条目只能有一个 `J`。
协变道函数本身不读 `J`（公式硬编码），所以"能不能算"不受影响，但拓扑校验
需要一个确定的 $J$，只能拆。

| spec | 道数 | 顶层 $(l,s)$ | 内层 $(l,s)$ |
|---|---|---|---|
| [`ex5_kst2.toml`](specs/ex5_kst2.toml) | 8 | $(1,2)$ / $(3,2)$ | $(1,1)$ / $(3,1)$ |
| [`ex5_kst0.toml`](specs/ex5_kst0.toml) | 2 | $(1,0)$ | $(1,1)$ |

### 验收：移植 10/10，级联 4/8 + 2/2

```bash
# 移植正确性（需要 jax + Tensor/，注意上面那个 x64 坑）
analyses/kk_dis/.venv/bin/python analyses/tensor_examples/probes/verify_ex5_port.py

# 纯 numpy 回归
python agent/tensor/tests/test_ex5_channels.py

# 级联可表示性 + 阳性对照
python analyses/tensor_examples/probes/probe_ex5_identify.py
python analyses/tensor_examples/probes/probe_ex5_identify.py --control
```

* **移植**：10/10 到 `rel ≈ 1e-15`。`check` / `emit` / 端到端跑产物，
  三种表象全部通过。
* **级联可表示性**：`ex5_kst2` **4/8**（`u_PF_*`、`u_FF_*` 命中），
  `ex5_kst0` **2/2**。合起来 **6/10** —— ⚠️ 探针输出的是这个合计数，
  **引用单链时用 4/8**，两个口径别混。
* **阳性对照**：同一探针跑 b₁ 得 **8/8**。所以 4/8 不是探针坏了。

`u_PP_*` / `u_FP_*` 落在级联子空间之外（rel ≈ 6e-3 / 2e-1），四条都含
$P^{(2)}_{\psi}$ 横向投影且内层是 P 波。但 K*₀ 的内层**也是** P 波却命中，
所以不是"P 波一律不行"，而是 $(J_{K^*}=2,\ l_{\text{内}}=1)$ 这个特定组合。
**原因未查清**。

### 两个坑（都踩过）

1. **jax 默认 float32** —— 见上文"框架侧的对应测试"。移植对拍第一次
   10/10 全"失败"（rel 1e-6…1e-8），打开 x64 后立刻 10/10 到 1e-15。
   同一个坑还**污染了级联探针**，伪造出一个"`_1` 族全败、`_2` 族全中"的
   假不对称；改用本仓库的协变内核（float64）后消失。
2. **K*₀ 的宇称** —— $J=0$ + 宇称守恒 + 三角只允许 $l=s=1$，对应 $0^-$；
   而常见的 $K^*_0(1430)$ 是 $0^+$。所以这条道描述的**不是** $K^*_0(1430)$。
   **疑点未解决**。

### 例 5 还没有端到端脚本

`check_ex3.py` 只覆盖例 3（读 `data/` 的 `phi f2` 动量、验 A1/A3/A7）。
例 5 目前只由 `test_ex5_channels.py`（纯 numpy）+ `emit` 产物跑通来保证，
**没有**等价的 `check_ex5.py`。真实样本、拟合侧端到端也都没做。

## 数据为什么是合成的

`chains/` 里的道函数是按道名**硬编码的公式**，对任意满足守恒的四动量都成立
（`grep -c "chain\|nodes" agent/tensor/kernels/covariant.py` == 0）。所以验证
它们不需要真实样本，只需要物理四动量。`make_momenta.py` 用顺序两体衰变严格
生成，并自检：守恒、每个末态在壳、每个两体子系统过阈值（否则 `Qabc<0`，
`B1/B2` 出 NaN —— **jax 原版与 numpy 移植版都会 NaN**，那不是移植错误）。

`m(π⁺π⁻)` 在 `[2m_π, m_ψ-m_φ]` 上均匀抽样，使 ρ′/b₁ 的不变质量自然扫过宽区间。
