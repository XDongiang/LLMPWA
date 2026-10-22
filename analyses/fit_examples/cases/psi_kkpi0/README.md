# case `psi_kkpi0`

ψ(2S)→K⁺K⁻π⁰：KK̄ + K*/K*₃/K*₅ 四族相干

> 本文件由 `common/case.py build <case> --only readme` 生成。人手写的输入在 `case.toml`；实测事实在 `tensors/index.toml`。

## 共振族（相干求和）

| 族 | spec | 道数 | 复耦合 | m (GeV) | Γ (GeV) | 传播子 |
|---|---|---|---|---|---|---|
| `ex2_kkbar` | `ex2_kkpi0_rho` | 1 | 1 | 1.71 | 0.14 | BW |
| `ex2_kstar` | `ex2_kkpi0_kstar` | 2 | 2 | 0.892 | 0.051 | BW |
| `ex2_kstar3` | `ex2_kkpi0_kstar3` | 2 | 2 | 1.78 | 0.16 | BW |
| `ex2_kstar5` | `ex2_kkpi0_kstar5` | 2 | 2 | 2.38 | 0.32 | BW |

合计 **7 道 -> 14 个自由实参**（每道一个复耦合 `g = const + i·theta`）。

## 四动量（tf-pwa 相空间）

`data/momenta.npz`，布局 `x,y,z,E`，键 `Kp1, Km2, Pi0`。

| 项 | 值 |
|---|---|
| 事例数 | 20000 |
| 生成器 | `tf_pwa.phasespace.generate_phsp` |
| 耗时 | 0.12 s（1.7e+05 事例/秒）|
| 四动量守恒 | 2.39e-15（绝对）/ 1.37e-15（相对）|
| 在壳 | max 7.47e-15 GeV |

最紧的两体阈值余量：`Kp1+Km2`，min = 0.989485 GeV，阈值 0.987354，余量 2.13e-03 GeV。

余量为正是**必要条件**：`Qabc` 变负会让 `B1/B2` 的 `sqrt` 出 NaN。

## 张量 cache

| 族 | 表象 | 形状 | dtype | max abs | 生成 |
|---|---|---|---|---|---|
| `ex2_kkbar` | `cartesian` | `[1, 20000, 3]` | `float64` | 28.5 | 0.19 s |
| `ex2_kkbar` | `helicity_cascade` | `[1, 20000, 3]` | `complex128` | 1.37 | 0.44 s |
| `ex2_kstar` | `cartesian` | `[2, 20000, 3]` | `float64` | 28.5 | 0.24 s |
| `ex2_kstar` | `helicity_cascade` | `[2, 20000, 3]` | `complex128` | 1.37 | 0.75 s |
| `ex2_kstar3` | `cartesian` | `[2, 20000, 3]` | `float64` | 2.14e+04 | 0.38 s |
| `ex2_kstar3` | `helicity_cascade` | `[2, 20000, 3]` | `complex128` | 196 | 0.84 s |
| `ex2_kstar5` | `cartesian` | `[2, 20000, 3]` | `float64` | 1.98e+08 | 11.57 s |
| `ex2_kstar5` | `helicity_cascade` | `[2, 20000, 3]` | `complex128` | 3.72e+05 | 0.96 s |

槽位语义是**承重约定**：协变侧末轴 = `(x, y, z)`，级联侧末轴 = `(λ=+1, 0, -1)`。横向强度取 `(x,y)` 与 `(±1)`，两边由 `W = EPS_A · T` 联系。

### 跨表象标定

`c_i` 定义为 `W_i = c_i · C_i`（协变侧经 `EPS_A` 后与级联侧比较）。残差 ~1e-15 说明两份 cache 是**同一振幅**、只差逐道复常数；残差大说明形状不同（真问题）。

| 族 | max_rel | 结论 |
|---|---|---|
| `ex2_kkbar` | 1.48e-07 | 只差常数（~1e-7：中间态离壳） |
| `ex2_kstar` | 3.03e-07 | 只差常数（~1e-7：中间态离壳） |
| `ex2_kstar3` | 1.14e-07 | 只差常数（~1e-7：中间态离壳） |
| `ex2_kstar5` | 1.19e-07 | 只差常数（~1e-7：中间态离壳） |

## 参数表

`params/free_params.toml`：14 项，其中 0 项冻结。
初值是 `coupling_scale = 0.1` 下的**随机相位**，不是拟合结果。

`params/resonances_config.toml`：只写传播子（质量/宽度先冻结）。

## 似然测试结果

设备 `cpu:0`，jax 0.11.1（x64=True）。

| 表象 | 值+梯度 (s) | 事例/秒 |
|---|---|---|
| `cartesian` | 2.73e-03 | 1.47e+07 |
| `helicity_cascade` | 2.63e-03 | 1.52e+07 |

| 表象 | grad vs FD | HVP vs FD |
|---|---|---|
| `cartesian` | 1.2e-09 | 6.0e-10 |
| `helicity_cascade` | 1.2e-09 | 3.2e-10 |

跨表象强度残差（仅可比族）**1.69e-07**。

短跑 L-BFGS-B（30 次迭代，**未收敛是预期的**）：L 从初值到 13932.3，0.70 s。

## 重跑

```bash
cd LLMPWA   # cwd 必须是仓库根
python analyses/fit_examples/common/case.py build analyses/fit_examples/cases/psi_kkpi0
analyses/kk_dis/.venv/bin/python analyses/fit_examples/common/likelihood.py analyses/fit_examples/cases/psi_kkpi0 --basis both --check --fit
```

