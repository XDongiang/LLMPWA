# 随机初始值扰动算法（Random Initial Perturbation）

> 用途：多起点拟合（`run/multi_start_fit.py`）的随机初始值生成，降低收敛到局部最优的概率。
> 参照旧版单节点脚本的抖动逻辑：const/theta 耦合重新采样到半径 0.1 的圆上，其余参数做小幅乘性抖动。
> 每次试验使用独立 `numpy.random.RandomState(seed)`，seed 由统一的 `base_seed + trial_idx` 派生，
> 所有进程生成完全一致的偏移，**无需通信**（天然适配多进程/分布式场景）。

---

## 1. 动机

单起点拟合可能收敛到某个局部极小（NLL 并非全局最优）。随机初值多起点扫描可提高找到更低
NLL 解的概率。实测（HEP1，2 起点）：

| 起点 | 初值扰动 | 最终 NLL | 说明 |
|------|---------|---------|------|
| start 0 | 基线（无扰动） | -3950.0025 | 原最优解 |
| start 1 | ±5% 均匀扰动（旧算法） | -3789.0429 | 落入另一个局部极小 |

不同初值确实会收敛到不同局部极小（NLL 差 ~161），因此多起点 + 选优是必要的。

---

## 2. 算法描述

对基线参数向量 `base_args`（`run/free_params.toml` 的 65 个自由参数，按 `arg_index` 排序）
施加两类扰动：

### 2.1 Amplitude 耦合（const / theta）——圆上重采样

`free_params.toml` 中 path 含 `.Amplitude.constN` / `.Amplitude.thetaN` 的参数成对出现，
表示复数耦合的实部（const）与虚部（theta）。扰动方式：

```
φ_i = 2π · U(0, 1)                    # 随机相位（每个耦合对独立）
const_i = 0.1 · sin(φ_i)              # 实部 = 0.1·sin
theta_i = 0.1 · cos(φ_i)              # 虚部 = 0.1·cos
```

- 幅值**固定为 0.1**（与物理约定一致），只随机化相对相位 → 落在半径 0.1 的圆上
- 覆盖全部 25 对 const/theta（phif0_980 / phif0_1710 / phif2_1270 / phif2_1525 / phif2_2150 / phif2_2340 / phif0_2470）

### 2.2 全部参数——统一乘性抖动

```
args_j *= (U(-0.5, +0.5) / disturb + 1.0)      # disturb=50 → 抖动幅度 ±1%
```

- 对包括 const/theta 在内的**所有**参数再乘一个独立的小扰动
- 乘性（而非加性）：与参数量纲/量级自洽，大参数（如质量 ~1-2.5 GeV）相对偏移与
  小参数（如耦合 ~0.01）一致

### 2.3 说明

- 不做边界 clamp：`free_params.toml` 的 `range` 字段语义是**高斯约束 `(center, sigma)`**
  （见 `run/likelihood_function.py` 的 `GAUSSIAN_CONSTRAINTS`），不是参数边界；
  拟合时似然中的约束项自然会拉回合理区间，扰动阶段无需（也不应）干预。
- 参数顺序：`find_const_theta_indices` 按 `arg_index`（= free_params 文件顺序）收集索引，
  保证 `const_idx[i]` 与 `theta_idx[i]` 属于同一耦合对（constN/thetaN 在文件中相邻成对）。

---

## 3. 代码

```python
# =============================================================================
# 随机初值扰动
#
# 参照旧版单节点脚本的抖动逻辑：
#   theta = 2*pi*rand()
#   theta_val = 0.1 * cos(theta); const_val = 0.1 * sin(theta)   （落在半径 0.1 的圆上）
#   args_float *= (rand() - 0.5) / disturb + 1.0                 （disturb=50 → 整体乘性抖动 ±1%）
# 这里改为对每次试验使用独立的 numpy RandomState(seed)，seed 由所有分布式进程
# 用同一个 base_seed + trial_idx 算出，因此各进程生成完全一致的偏移，无需通信。
# =============================================================================

def find_const_theta_indices(paths):
    """从 free_params.toml 的 path 字段中找出所有 Amplitude.constN / Amplitude.thetaN 的 arg_index"""
    const_idx = [i for i, p in enumerate(paths) if ".Amplitude.const" in p]
    theta_idx = [i for i, p in enumerate(paths) if ".Amplitude.theta" in p]
    return onp.array(const_idx, dtype=int), onp.array(theta_idx, dtype=int)


def perturb_args(base_args, const_idx, theta_idx, seed, disturb=50.0):
    """
    对初始参数做随机偏移，返回新的参数数组（不修改 base_args）。
    - const/theta：重新采样到半径 0.1 的圆上（幅值固定为 0.1，相位随机）
    - 全部参数：额外乘以一个统一的随机乘性抖动 (rand()-0.5)/disturb + 1.0
    """
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

---

## 4. 在 runner 中的集成（`run/multi_start_fit.py`）

```python
# 每个起点：
values, _, data = load_free_params()
paths = [d["path"] for d in data]
const_idx, theta_idx = find_const_theta_indices(paths)

if start_idx == 0:
    x0 = values                       # 基线起点，不扰动
else:
    seed = base_seed + start_idx * 1000
    x0 = perturb_args(values, const_idx, theta_idx, seed, disturb=args.disturb)
```

CLI：`--disturb 50`（默认），对应抖动幅度 ±1%；改小则抖动更大（如 `--disturb 10` → ±5%）。

---

## 5. 可复现性

- 每个起点使用独立 `RandomState(seed)`，`seed = base_seed + start_idx * 1000`
- 随机数消费顺序固定：先 `theta_idx.shape[0]` 个相位，再 `args.shape[0]` 个乘性因子
- 同一 `--seed` + `--starts` 重跑，每个起点的初值完全一致 → 拟合可复现
- 多进程/分布式下：各进程用同一 `base_seed + trial_idx` 算出相同 seed，生成一致偏移，无需通信

---

## 6. 与旧算法（multi_start_fit v1）的差异

| 方面 | v1（旧） | v2（本文） |
|------|---------|-----------|
| 扰动类型 | 全体 Uniform(±5%) | const/theta 圆上重采样 + 全体乘性 ±1% |
| const/theta 处理 | 当作普通参数 | 成对重置到半径 0.1 的圆（保留物理幅值） |
| range 字段 | 误当 [lo,hi] 做 clamp/fixed | 不干预（range 是高斯约束 center/sigma） |
| RNG | `default_rng` | `RandomState`（与旧版单节点脚本一致） |
| 扰动幅度 | 5% | 1%（disturb=50），可通过 `--disturb` 调 |
