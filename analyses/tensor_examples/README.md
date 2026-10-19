# tensor_examples —— 文档公式例子的回归分析

本目录把 `analyses/kk_dis/document/ψ衰变到介子的分波分析协变张量形式.md`
里的衰变例子**变成可跑的回归测试**，用来验收框架级张量生成器
（[`agent/tensor/`](../../agent/tensor/)）。

它不是生产分析：没有拟合、没有 `llm_config_*.toml`、没有 `data/` 真实样本。
目的只有一个 —— 给「框架新加一条道」这件事一个**独立于 kk_dis** 的验收面。

## 现状

| 例子 | 文档行 | 状态 | 道 |
|---|---|---|---|
| 例 3 `ψ→φπ⁺π⁻` 的 ρ′ 道 | 374–387 | ✅ 已实现 | `u_rho_1` `u_rho_2`（Eq.46） |
| 例 3 `ψ→φπ⁺π⁻` 的 b₁ 道 | 391–416 | ✅ 已实现 | `u_{SS,SD,DS,DD}_{1,2}`（Eq.47–50） |
| 例 5 `ψ→K⁺π⁻K⁻π⁺` 的 K*₀/K*₂ 道 | 431–623 | ⬜ 待做 | 10 条（`u_kst0_*` + `u_PP/PF/FP/FF_*`） |
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
├── check_ex3.py            ← 端到端回归：生成 → 跑 → 三层自检 + A1/A3/A7
├── probes/                 ← 一次性诊断脚本（复现 roadmap §7 与边界实测）
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

**为什么两份 spec 而不是一份**：ρ′ 与 b₁ 的**树不同**（中间态分别是
`phi+π⁻` 与 `phi+π⁺`），一棵树只能描述一组配对。分成两份 TOML 最简明 ——
这也是用户指定的结构（`chain.name/chain.channel.py` 存储、生成时组装）。

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
python agent/tensor/tests/test_helicity_cascade.py  # 9 项：A1/A2/A3/A4 + 边界
TENSOR_ROOT=<...>/tensor_example python agent/tensor/tests/test_covariant_vs_jax.py
                                                     # 与 jax 原版逐式对拍（24 个对象）
```

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

## 数据为什么是合成的

`chains/` 里的道函数是按道名**硬编码的公式**，对任意满足守恒的四动量都成立
（`grep -c "chain\|nodes" agent/tensor/kernels/covariant.py` == 0）。所以验证
它们不需要真实样本，只需要物理四动量。`make_momenta.py` 用顺序两体衰变严格
生成，并自检：守恒、每个末态在壳、每个两体子系统过阈值（否则 `Qabc<0`，
`B1/B2` 出 NaN —— **jax 原版与 numpy 移植版都会 NaN**，那不是移植错误）。

`m(π⁺π⁻)` 在 `[2m_π, m_ψ-m_φ]` 上均匀抽样，使 ρ′/b₁ 的不变质量自然扫过宽区间。
