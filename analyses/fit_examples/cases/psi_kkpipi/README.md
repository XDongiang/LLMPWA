# case `psi_kkpipi`

ψ→K⁺K⁻π⁺π⁻：ρ′/b₁/K*₀/K*₂ 四族相干

> 本文件由 `common/case.py build <case> --only readme` 生成。人手写的输入在 `case.toml`；实测事实在 `tensors/index.toml`。

## 共振族（相干求和）

| 族 | spec | 道数 | 复耦合 | m (GeV) | Γ (GeV) | 传播子 |
|---|---|---|---|---|---|---|
| `ex3_rhopi` | `ex3_rhopi` | 2 | 2 | 1.46 | 0.31 | BW |
| `ex3_b1pi` | `ex3_b1pi` | 8 | 8 | 1.23 | 0.142 | BW |
| `ex5_kst0` | `ex5_kst0` | 2 | 2 | 1.43 | 0.29 | BW |
| `ex5_kst2` | `ex5_kst2` | 8 | 8 | 1.43 | 0.109 | BW |

合计 **20 道 -> 40 个自由实参**（每道一个复耦合 `g = const + i·theta`）。

## 四动量（tf-pwa 相空间）

`data/momenta.npz`，布局 `x,y,z,E`，键 `Kp, Km, Pip, Pim`。

| 项 | 值 |
|---|---|
| 事例数 | 20000 |
| 生成器 | `tf_pwa.phasespace.generate_phsp` |
| 耗时 | 0.11 s（1.77e+05 事例/秒）|
| 四动量守恒 | 7.99e-15（绝对）/ 4.80e-15（相对）|
| 在壳 | max 7.63e-15 GeV |

最紧的两体阈值余量：`Km+Pim`，min = 0.634040 GeV，阈值 0.633247，余量 7.93e-04 GeV。

余量为正是**必要条件**：`Qabc` 变负会让 `B1/B2` 的 `sqrt` 出 NaN。

## 张量 cache

| 族 | 表象 | 形状 | dtype | max abs | 生成 |
|---|---|---|---|---|---|
| `ex3_rhopi` | `cartesian` | `[2, 20000, 3]` | `float64` | 312 | 4.76 s |
| `ex3_rhopi` | `helicity_cascade` | `[2, 20000, 3]` | `complex128` | 1.26 | 1.06 s |
| `ex3_b1pi` | `cartesian` | `[8, 20000, 3]` | `float64` | 257 | 0.50 s |
| `ex3_b1pi` | `helicity_cascade` | `[8, 20000, 3]` | `complex128` | 11.7 | 3.83 s |
| `ex5_kst0` | `cartesian` | `[2, 20000, 3]` | `float64` | 25.6 | 0.25 s |
| `ex5_kst0` | `helicity_cascade` | `[2, 20000, 3]` | `complex128` | 3.19 | 1.04 s |
| `ex5_kst2` | `cartesian` | `[8, 20000, 3]` | `float64` | 9.67e+03 | 0.91 s |
| `ex5_kst2` | `helicity_cascade` | `[8, 20000, 3]` | `complex128` | 187 | 4.06 s |

槽位语义是**承重约定**：协变侧末轴 = `(x, y, z)`，级联侧末轴 = `(λ=+1, 0, -1)`。横向强度取 `(x,y)` 与 `(±1)`，两边由 `W = EPS_A · T` 联系。

### 跨表象标定

`c_i` 定义为 `W_i = c_i · C_i`（协变侧经 `EPS_A` 后与级联侧比较）。残差 ~1e-15 说明两份 cache 是**同一振幅**、只差逐道复常数；残差大说明形状不同（真问题）。

| 族 | max_rel | 结论 |
|---|---|---|
| `ex3_rhopi` | 3.13e-01 | **形状不同** |
| `ex3_b1pi` | 1.20e-07 | 只差常数（~1e-7：中间态离壳） |
| `ex5_kst0` | 1.29e-07 | 只差常数（~1e-7：中间态离壳） |
| `ex5_kst2` | 1.13e-07 | 只差常数（~1e-7：中间态离壳） |

## 参数表

`params/free_params.toml`：40 项，其中 0 项冻结。
初值是 `coupling_scale = 0.1` 下的**随机相位**，不是拟合结果。

`params/resonances_config.toml`：只写传播子（质量/宽度先冻结）。

## 似然测试结果

设备 `cpu:0`，jax 0.11.1（x64=True）。

| 表象 | 值+梯度 (s) | 事例/秒 |
|---|---|---|
| `cartesian` | 7.48e-03 | 5.35e+06 |
| `helicity_cascade` | 7.51e-03 | 5.33e+06 |

| 表象 | grad vs FD | HVP vs FD |
|---|---|---|
| `cartesian` | 2.4e-09 | 6.1e-10 |
| `helicity_cascade` | 6.2e-09 | 3.1e-10 |

跨表象强度残差（仅可比族）**1.65e-07**。

已排除（两份 cache 形状不同，不可比）：['ex3_rhopi'] —— 见 `../../README.md` 的「已知边界」。

## 重跑

```bash
cd LLMPWA   # cwd 必须是仓库根
python analyses/fit_examples/common/case.py build analyses/fit_examples/cases/psi_kkpipi
analyses/kk_dis/.venv/bin/python analyses/fit_examples/common/likelihood.py analyses/fit_examples/cases/psi_kkpipi --basis both --check --fit
```

