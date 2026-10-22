# case `psi_pipipi0`

ψ(2S)→π⁺π⁻π⁰：ρ/ρ₃/ρ₅ 三族相干

> 本文件由 `common/case.py build <case> --only readme` 生成。人手写的输入在 `case.toml`；实测事实在 `tensors/index.toml`。

## 共振族（相干求和）

| 族 | spec | 道数 | 复耦合 | m (GeV) | Γ (GeV) | 传播子 |
|---|---|---|---|---|---|---|
| `ex1_rho` | `ex1_pipipi0_rho` | 3 | 3 | 0.775 | 0.149 | BW |
| `ex1_rho3` | `ex1_pipipi0_rho3` | 3 | 3 | 1.69 | 0.19 | BW |
| `ex1_rho5` | `ex1_pipipi0_rho5` | 3 | 3 | 2.35 | 0.32 | BW |

合计 **9 道 -> 18 个自由实参**（每道一个复耦合 `g = const + i·theta`）。

## 四动量（tf-pwa 相空间）

`data/momenta.npz`，布局 `x,y,z,E`，键 `Pip1, Pim2, Pi0`。

| 项 | 值 |
|---|---|
| 事例数 | 20000 |
| 生成器 | `tf_pwa.phasespace.generate_phsp` |
| 耗时 | 0.07 s（2.91e+05 事例/秒）|
| 四动量守恒 | 9.55e-15（绝对）/ 5.23e-15（相对）|
| 在壳 | max 1.06e-14 GeV |

最紧的两体阈值余量：`Pip1+Pi0`，min = 0.277358 GeV，阈值 0.274547，余量 2.81e-03 GeV。

余量为正是**必要条件**：`Qabc` 变负会让 `B1/B2` 的 `sqrt` 出 NaN。

## 张量 cache

| 族 | 表象 | 形状 | dtype | max abs | 生成 |
|---|---|---|---|---|---|
| `ex1_rho` | `cartesian` | `[3, 20000, 3]` | `float64` | 28.5 | 0.30 s |
| `ex1_rho` | `helicity_cascade` | `[3, 20000, 3]` | `complex128` | 1.37 | 1.01 s |
| `ex1_rho3` | `cartesian` | `[3, 20000, 3]` | `float64` | 2.74e+04 | 0.43 s |
| `ex1_rho3` | `helicity_cascade` | `[3, 20000, 3]` | `complex128` | 249 | 1.13 s |
| `ex1_rho5` | `cartesian` | `[3, 20000, 3]` | `float64` | 2.43e+08 | 16.18 s |
| `ex1_rho5` | `helicity_cascade` | `[3, 20000, 3]` | `complex128` | 4.58e+05 | 1.34 s |

槽位语义是**承重约定**：协变侧末轴 = `(x, y, z)`，级联侧末轴 = `(λ=+1, 0, -1)`。横向强度取 `(x,y)` 与 `(±1)`，两边由 `W = EPS_A · T` 联系。

### 跨表象标定

`c_i` 定义为 `W_i = c_i · C_i`（协变侧经 `EPS_A` 后与级联侧比较）。残差 ~1e-15 说明两份 cache 是**同一振幅**、只差逐道复常数；残差大说明形状不同（真问题）。

| 族 | max_rel | 结论 |
|---|---|---|
| `ex1_rho` | 1.67e-07 | 只差常数（~1e-7：中间态离壳） |
| `ex1_rho3` | 9.15e-08 | 只差常数（~1e-7：中间态离壳） |
| `ex1_rho5` | 1.15e-07 | 只差常数（~1e-7：中间态离壳） |

## 参数表

`params/free_params.toml`：18 项，其中 0 项冻结。
初值是 `coupling_scale = 0.1` 下的**随机相位**，不是拟合结果。

`params/resonances_config.toml`：只写传播子（质量/宽度先冻结）。

## 似然测试结果

设备 `cpu:0`，jax 0.11.1（x64=True）。

| 表象 | 值+梯度 (s) | 事例/秒 |
|---|---|---|
| `cartesian` | 3.86e-03 | 1.04e+07 |
| `helicity_cascade` | 3.93e-03 | 1.02e+07 |

| 表象 | grad vs FD | HVP vs FD |
|---|---|---|
| `cartesian` | 4.6e-09 | 6.8e-07 |
| `helicity_cascade` | 4.4e-09 | 6.1e-06 |

跨表象强度残差（仅可比族）**1.19e-07**。

短跑 L-BFGS-B（18 次迭代，**未收敛是预期的**）：L 从初值到 11571.5，0.62 s。

## 重跑

```bash
cd LLMPWA   # cwd 必须是仓库根
python analyses/fit_examples/common/case.py build analyses/fit_examples/cases/psi_pipipi0
analyses/kk_dis/.venv/bin/python analyses/fit_examples/common/likelihood.py analyses/fit_examples/cases/psi_pipipi0 --basis both --check --fit
```

