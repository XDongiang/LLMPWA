# 随机初值扰动算法（Random Initial Perturbation）

> 用途：多起点拟合（`run/multi_start_fit.py`）的随机初始值生成，降低收敛到局部最优的概率。
> 参照旧版单节点脚本的抖动逻辑：const/theta 耦合重新采样到半径 0.1 的圆上，其余参数做小幅乘性抖动。
> 每次试验使用独立 `numpy.random.RandomState(seed)`，seed 由统一的 `base_seed + trial_idx` 派生，
> 所有进程生成完全一致的偏移，**无需通信**（天然适配多进程/分布式场景）。

---

## 1. 动机

单起点拟合可能收敛到某个局部极小。随机初值多起点扫描可提高找到更低 NLL 解的概率。

## 2. 算法描述

对基线参数向量 `base_args`（联合参数 `run/free_params_total.toml` 的自由参数，按 `arg_index` 排序）
施加两类扰动：

### 2.1 Amplitude 耦合（const / theta）——圆上重采样

`free_params_total.toml` 中 path 含 `.Amplitude.constN` / `.Amplitude.thetaN` 的参数成对出现，
表示复数耦合的实部（const）与虚部（theta）。扰动方式：

```
φ_i = 2π · U(0, 1)                    # 随机相位（每个耦合对独立）
const_i = 0.1 · sin(φ_i)              # 实部 = 0.1·sin
theta_i = 0.1 · cos(φ_i)              # 虚部 = 0.1·cos
```

- 幅值**固定为 0.1**，只随机化相对相位 → 落在半径 0.1 的圆上。

### 2.2 全部参数——统一乘性抖动

```
args_j *= (U(-0.5, +0.5) / disturb + 1.0)      # disturb=50 → 抖动幅度 ±1%
```

### 2.3 说明

- 不做边界 clamp：`free_params*.toml` 的 `range` 字段语义是**高斯约束 `(center, sigma)`**，
  拟合时似然约束项自然会拉回合理区间。
- 参数顺序：`find_const_theta_indices` 按 `arg_index` 收集索引，保证同一耦合对对齐。

## 3. 代码

```python
def find_const_theta_indices(paths):
    const_idx = [i for i, p in enumerate(paths) if ".Amplitude.const" in p]
    theta_idx = [i for i, p in enumerate(paths) if ".Amplitude.theta" in p]
    return onp.array(const_idx, dtype=int), onp.array(theta_idx, dtype=int)


def perturb_args(base_args, const_idx, theta_idx, seed, disturb=50.0):
    rng = onp.random.RandomState(seed)
    args = onp.array(base_args, dtype=float).copy()

    if const_idx.size > 0:
        theta = 2 * onp.pi * rng.rand(theta_idx.shape[0])
        args[theta_idx] = 0.1 * onp.cos(theta)
        args[const_idx] = 0.1 * onp.sin(theta)

    args = args * ((rng.rand(args.shape[0]) - 0.5) / disturb + 1.0)
    return args
```

> 约定：`onp` = `numpy`；`paths` 为与 `base_args` 等长的 path 列表（按 `arg_index` 排序）。

## 4. 在 runner 中的集成（`run/multi_start_fit.py`）

```python
values, _, data = load_free_params()
paths = [d["path"] for d in data]
const_idx, theta_idx = find_const_theta_indices(paths)

if start_idx == 0:
    x0 = values
else:
    seed = base_seed + start_idx * 1000
    x0 = perturb_args(values, const_idx, theta_idx, seed, disturb=args.disturb)
```

CLI：`--disturb 50`（默认），对应抖动幅度 ±1%。

## 5. 可复现性

- 每个起点使用独立 `RandomState(seed)`，`seed = base_seed + start_idx * 1000`。
- 同一 `--seed` + `--starts` 重跑，每个起点的初值完全一致 → 拟合可复现。
- 多进程/分布式下：各进程用同一 `base_seed + trial_idx` 算出相同 seed，生成一致偏移，无需通信。
