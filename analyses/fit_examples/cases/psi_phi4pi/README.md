# case `psi_phi4pi`

ψ(2S)→φπ⁺π⁻π⁺π⁻：σ 交换 / ρρ / f₂σ 三子分支

> 本文件由 `common/case.py build <case> --only readme` 生成。人手写的输入在 `case.toml`；实测事实在 `tensors/index.toml`。

## 共振族（相干求和）

| 族 | spec | 道数 | 复耦合 | m (GeV) | Γ (GeV) | 传播子 |
|---|---|---|---|---|---|---|
| `ex6_ss` | `ex6_phi4pi_ss` | 14 | 14 | 1.019 | 0.0042 | BW |
| `ex6_rr` | `ex6_phi4pi_rr` | 14 | 14 | 1.275 | 0.186 | BW |
| `ex6_f2s` | `ex6_phi4pi_f2s` | 10 | 10 | 1.275 | 0.186 | BW |

合计 **38 道 -> 76 个自由实参**（每道一个复耦合 `g = const + i·theta`）。

## 四动量（tf-pwa 相空间）

`data/momenta.npz`，布局 `x,y,z,E`，键 `Kp, Km, Pip, Pim, Pip2, Pim2`。

| 项 | 值 |
|---|---|
| 事例数 | 5000 |
| 生成器 | `tf_pwa.phasespace.generate_phsp` |
| 耗时 | 5.50 s（908 事例/秒）|
| 四动量守恒 | 2.00e-15（绝对）/ 1.48e-15（相对）|
| 在壳 | max 5.08e-15 GeV |

最紧的两体阈值余量：`Pip+Pim`，min = 0.279498 GeV，阈值 0.279141，余量 3.57e-04 GeV。

余量为正是**必要条件**：`Qabc` 变负会让 `B1/B2` 的 `sqrt` 出 NaN。

## 张量 cache

| 族 | 表象 | 形状 | dtype | max abs | 生成 |
|---|---|---|---|---|---|
| `ex6_ss` | `cartesian` | `[14, 5000, 3]` | `float64` | 1.94e+04 | 3.14 s |
| `ex6_ss` | `helicity_cascade` | `[14, 5000, 3]` | `complex128` | 389 | 2.88 s |
| `ex6_rr` | `cartesian` | `[14, 5000, 3]` | `float64` | 3.24e+05 | 3.62 s |
| `ex6_rr` | `helicity_cascade` | `[14, 5000, 3]` | `complex128` | 53 | 3.05 s |
| `ex6_f2s` | `cartesian` | `[10, 5000, 3]` | `float64` | 3.2e+05 | 3.67 s |
| `ex6_f2s` | `helicity_cascade` | `[10, 5000, 3]` | `complex128` | 53 | 2.26 s |

槽位语义是**承重约定**：协变侧末轴 = `(x, y, z)`，级联侧末轴 = `(λ=+1, 0, -1)`。横向强度取 `(x,y)` 与 `(±1)`，两边由 `W = EPS_A · T` 联系。

### 跨表象标定

`c_i` 定义为 `W_i = c_i · C_i`（协变侧经 `EPS_A` 后与级联侧比较）。残差 ~1e-15 说明两份 cache 是**同一振幅**、只差逐道复常数；残差大说明形状不同（真问题）。

| 族 | max_rel | 结论 |
|---|---|---|
| `ex6_ss` | 1.13e-07 | 只差常数（~1e-7：中间态离壳） |
| `ex6_rr` | 9.81e-01 | **形状不同** |
| `ex6_f2s` | 1.00e+00 | **形状不同** |

## 参数表

`params/free_params.toml`：76 项，其中 0 项冻结。
初值是 `coupling_scale = 0.1` 下的**随机相位**，不是拟合结果。

`params/resonances_config.toml`：只写传播子（质量/宽度先冻结）。

## 似然测试结果

设备 `cpu:0`，jax 0.11.1（x64=True）。

| 表象 | 值+梯度 (s) | 事例/秒 |
|---|---|---|
| `cartesian` | 2.53e-03 | 3.96e+06 |
| `helicity_cascade` | 2.64e-03 | 3.78e+06 |

| 表象 | grad vs FD | HVP vs FD |
|---|---|---|
| `cartesian` | 1.0e-09 | 4.6e-10 |
| `helicity_cascade` | 7.5e-10 | 3.4e-08 |

跨表象强度残差（仅可比族）**9.33e-08**。

已排除（两份 cache 形状不同，不可比）：['ex6_rr', 'ex6_f2s'] —— 见 `../../README.md` 的「已知边界」。

短跑 L-BFGS-B（30 次迭代，**未收敛是预期的**）：L 从初值到 1727.2，0.64 s。

## 重跑

```bash
cd LLMPWA   # cwd 必须是仓库根
python analyses/fit_examples/common/case.py build analyses/fit_examples/cases/psi_phi4pi
analyses/kk_dis/.venv/bin/python analyses/fit_examples/common/likelihood.py analyses/fit_examples/cases/psi_phi4pi --basis both --check --fit
```

