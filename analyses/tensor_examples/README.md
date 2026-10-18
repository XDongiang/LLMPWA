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
├── specs/
│   ├── ex3_rhopi.toml      ← 例 3 的 ρ′ 树（2 条道）
│   └── ex3_b1pi.toml       ← 例 3 的 b₁ 树（8 条道）
├── data/
│   └── make_momenta.py     ← 合成 ψ→φπ⁺π⁻→(K⁺K⁻)(π⁺π⁻) 四动量（自带守恒自检）
├── check_ex3.py            ← 端到端回归：生成 → 跑 → 三层自检
└── generated/  out/        ← 生成物与运行产物（.gitignore，可随时重建）
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

`check_ex3.py` 做三层验证：

1. **生成物 vs 框架内核**：产物脚本内联的道函数体算出的 cache，必须与直接调
   `agent.tensor.kernels.covariant.compute_event` 逐位一致（抓 emitter 组装 bug）；
2. **cartesian vs helicity**：`W = EPS_A · T`（抓 store 体写错）；
3. **有限性**：没有 NaN/Inf。

框架侧的对应测试（不需要本目录的数据）：

```bash
python agent/tensor/tests/test_ex3_channels.py     # 8 项，纯 numpy，永远能跑
TENSOR_ROOT=<...>/tensor_example python agent/tensor/tests/test_covariant_vs_jax.py
                                                    # 与 jax 原版逐式对拍（24 个对象）
```

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
* ⚠️ 三层树（`ψ→X π`，`X→φπ`，`φ→K⁺K⁻`）**只能出协变/螺度表象**，
  不能出 omega —— omega 内核逐节点读树、只算 `D0*D1*D2`。这条限制现在由
  `agent/tensor/spec.py` 按 `[[basis]]` 分级执行。

## 数据为什么是合成的

`chains/` 里的道函数是按道名**硬编码的公式**，对任意满足守恒的四动量都成立
（`grep -c "chain\|nodes" agent/tensor/kernels/covariant.py` == 0）。所以验证
它们不需要真实样本，只需要物理四动量。`make_momenta.py` 用顺序两体衰变严格
生成，并自检：守恒、每个末态在壳、每个两体子系统过阈值（否则 `Qabc<0`，
`B1/B2` 出 NaN —— **jax 原版与 numpy 移植版都会 NaN**，那不是移植错误）。

`m(π⁺π⁻)` 在 `[2m_π, m_ψ-m_φ]` 上均匀抽样，使 ρ′/b₁ 的不变质量自然扫过宽区间。
