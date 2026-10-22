# case `psi_omega_kk`

ψ(2S)→ωK⁺K⁻：ω(三体节点) + f₀/f₂ 七道

> 本文件由 `common/case.py build <case> --only readme` 生成。人手写的输入在 `case.toml`；实测事实在 `tensors/index.toml`。

## 共振族（相干求和）

| 族 | spec | 道数 | 复耦合 | m (GeV) | Γ (GeV) | 传播子 |
|---|---|---|---|---|---|---|
| `ex4_omega` | `ex4_omega_kk` | 7 | 7 | 0.782 | 0.0085 | BW |

合计 **7 道 -> 14 个自由实参**（每道一个复耦合 `g = const + i·theta`）。

## 四动量（tf-pwa 相空间）

`data/momenta.npz`，布局 `x,y,z,E`，键 `Kp, Km, Pi0, Pip, Pim`。

| 项 | 值 |
|---|---|
| 事例数 | 20000 |
| 生成器 | `tf_pwa.phasespace.generate_phsp` |
| 耗时 | 0.90 s（2.23e+04 事例/秒）|
| 四动量守恒 | 3.33e-15（绝对）/ 2.10e-15（相对）|
| 在壳 | max 6.66e-15 GeV |

最紧的两体阈值余量：`Km+Pi0`，min = 0.628946 GeV，阈值 0.628654，余量 2.92e-04 GeV。

余量为正是**必要条件**：`Qabc` 变负会让 `B1/B2` 的 `sqrt` 出 NaN。

## 张量 cache

| 族 | 表象 | 形状 | dtype | max abs | 生成 |
|---|---|---|---|---|---|
| `ex4_omega` | `cartesian` | `[7, 20000, 3]` | `float64` | 3.82e+04 | 6.21 s |

本 case 只有 `cartesian` 一份 cache（spec 未声明另一个 `[[basis]]`），因此没有跨表象比较。

## 参数表

`params/free_params.toml`：14 项，其中 0 项冻结。
初值是 `coupling_scale = 0.1` 下的**随机相位**，不是拟合结果。

`params/resonances_config.toml`：只写传播子（质量/宽度先冻结）。

## 似然测试结果

设备 `cpu:0`，jax 0.11.1（x64=True）。

| 表象 | 值+梯度 (s) | 事例/秒 |
|---|---|---|
| `cartesian` | 2.94e-03 | 1.36e+07 |

| 表象 | grad vs FD | HVP vs FD |
|---|---|---|
| `cartesian` | 5.3e-10 | 5.5e-10 |

短跑 L-BFGS-B（30 次迭代，**未收敛是预期的**）：L 从初值到 13585.3，0.53 s。

## 重跑

```bash
cd LLMPWA   # cwd 必须是仓库根
python analyses/fit_examples/common/case.py build analyses/fit_examples/cases/psi_omega_kk
analyses/kk_dis/.venv/bin/python analyses/fit_examples/common/likelihood.py analyses/fit_examples/cases/psi_omega_kk --basis cartesian --check --fit
```

