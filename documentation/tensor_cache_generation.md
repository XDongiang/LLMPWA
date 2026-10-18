# 框架级 tensor 缓存代码生成（`agent/tensor/`）

本文说明 `agent/tensor/` 的设计、用法与验收判据。它是 `analyses/*` 的**上层**
能力：跨分析复用，**一份 TOML 同时产出协变张量与螺度两种 tensor 的生成代码**。

> 边界：框架（`agent/`）由这里维护；每个 analysis 内部的接线（数据路径、似然
> 主体、拟合脚本）由该 analysis 自己的 agent 管理。

## 1. 为什么要提到框架层

此前 tensor 缓存的生成逻辑散在 `analyses/kk_dis/task/.../4_三条路线/` 里：
四条路线各自有一份 `basis.py` 和 `gen_tensor.py`，其中
`method1/gen_tensor.py` 与 `method2/gen_tensor.py` 是**逐字节相同**的 46 行包装。
把它们提到 `agent/tensor/` 之后：

* 四条路线的差别收敛成 `agent/tensor/basis.py` 里的一张注册表；
* 一份 TOML 通过 `[[basis]]` 选择要产出哪几种表象；
* 协变张量内核成为框架资产，可以被别的分析直接复用。

## 2. 目录

```
agent/tensor/
  spec.py            统一 TOML schema：粒子/级联 + 道表 + [[basis]]
  basis.py           4 个表象钩子（precompute/amplitude/intensity/env）
  chains/            chain.<name>/chain.<channel>.py 存储层（生成时组装）
    phif0/ phif2/    psi->phi f_J 的 7 条道（Eq.36-45）
    rhopi/ b1pi/     例 3 的 10 条 rho'/b1 道（Eq.46-50，见 §11）
  kernels/
    _backend.py      数值后端：jax 优先，无 jax 时回退 numpy（TENSOR_BACKEND）
    omega.py         tf-pwa 无关的螺旋度级联运行时（CG/Wigner-D/Chung Eq.52）
    covariant.py     Chung 协变张量（原 jax 版的逐式移植）
    helicity.py      EPS_A ∘ covariant
  emitter.py         按 [[basis]] 产出自足、**jax 优先**的生成脚本
  cli.py             python -m agent.tensor show|check|emit|run|bases|chains
  tests/             spec / 跨表象 / emitter / 后端一致性 + jax 对拍 + 性能基准
```

## 3. 一份 TOML，多种表象

```toml
[chain]
name   = "phif2"
top    = "psi2s"
l_max  = 4
suffix = ""            # f2 族：Omega.npy / K.npy / M.npy / g.npy

[input]
keys   = ["Kp", "Km", "Pip", "Pim"]
layout = "x,y,z,E"

[particles.psi2s]
J = 1; mass = 3.686; parity = -1; daughters = ["phi", "f2"]
# ... phi / f2 / Kp / Km / Pip / Pim

# 省略 [[channel]] 时按宇称 + 角动量三角自动枚举：
#   [(0,1), (2,1), (2,2), (2,3), (4,3)]

[[basis]]
name   = "cartesian"
kernel = "covariant"
env    = { TENSOR_BASIS = "cartesian", HELICITY_LAYER = "0" }

[[basis]]
name   = "helicity"
kernel = "helicity"
env    = { TENSOR_BASIS = "helicity", HELICITY_LAYER = "1" }
```

`f_0` 与 `f_2` 分成**两个配置文件**（`specs/phif0.toml` / `specs/phif2.toml`）：
f0 自旋为 0，S 只能来自 phi，故 `s ≡ 1`、`l` 取偶数，只有 2 条道；与 f2 的
5 条道分开写更简明。两者共用同一句 `einsum("iab,enab->ien")`，因为 f0 的
末轴 `lambda_f` 长度为 1。

## 4. `chain.<name>/chain.<channel>.py` 存储约定

协变张量的**每条道一个文件**，按链名分目录；`emitter` 在生成时把道表里点到的
文件取出、内联进产物，所以产物是自足的（运行时不 import 框架）：

```
agent/tensor/chains/
  phif0/   __init__.py  phif001.py  phif021.py
  phif2/   __init__.py  phif201.py  phif221.py  phif222.py  phif223.py  phif243.py
  rhopi/   __init__.py  u_rho_1.py  u_rho_2.py                  # 例 3 的 rho'
  b1pi/    __init__.py  u_SS_{1,2}.py  u_SD_{1,2}.py            # 例 3 的 b1
                        u_DS_{1,2}.py  u_DD_{1,2}.py
```

每个文件导出与文件同名的函数：

```python
def phif223(psi, phi, f2, Kp, Km, Pip, Pim) -> (N, 4)
```

`chains.load_chain("phif2")` 在进程内动态装配；`chains.list_chains()` /
`list_channels(chain)` 用于发现，`chains.find_channel(name)` 跨目录按道名找。
若某个道在 `chains/` 里没有文件，`emitter` 会退回 `kernels/covariant.py` 里的
实现并在返回值里记下 `fallback`。

### 4.1 一份 spec 可以跨多个 chain 目录

例 3 的 ρ′ 与 b₁ 是**两棵不同的树**（中间态分别是 `φ+π⁻` 与 `φ+π⁺`），一棵树
只能描述一组配对。所以它们分成两个 chain 目录、两份 TOML，而不是硬塞进一份：

```
analyses/tensor_examples/specs/ex3_rhopi.toml   chain_name = "rhopi"
analyses/tensor_examples/specs/ex3_b1pi.toml    chain_name = "b1pi"
```

`emitter` 先按 `[chain].chain_name` 指定的目录找道，找不到再跨目录按道名找
（`chains.all_channels()`），并在返回值里给出 `from_chain: {道名: 链名}`。
道名全局唯一，所以跨目录不会歧义。

### 4.2 新增道只写在 `chains/` 里

`kernels/covariant.py` 里那 7 条 `_phif*` 是 A1 移植的落点，属于历史；
**之后新增的道（例 3、例 5）只写在 `chains/` 里**，由
`kernels/covariant.channel_fn()` 惰性回退过去：

```python
COV.channel_fn("phif001")   # 命中 _CHANNELS（内核里的副本）
COV.channel_fn("u_rho_1")   # 回退 chains/rhopi/u_rho_1.py
COV.available_channels()    # 两边合起来
```

这样同一份公式不会存两处，避免日后改一处漏一处。

## 5. 四个表象

| 表象 | `CACHE_BASIS` | 拟合时旋转 | 强度 | 存几个分量 |
|---|---|---|---|---|
| `cartesian` | cartesian | 否 | `\|T_x\|^2+\|T_y\|^2` | 3 |
| `cov_to_helicity` | cartesian（同一份文件） | **是** | `Σ_{λ=±1}\|W_λ\|^2` | 3 |
| `helicity` | helicity | 否 | `Σ_{λ=±1}\|W_λ\|^2` | 3 |
| `omega` | omega | 否 | `Σ_{λ=±1}\|W_λ\|^2` | 3 |

`TENSOR_BASIS` / `HELICITY_LAYER` 在**调用时**从环境读取，所以一份 `run/` 可以
服务多条路线（这是既有生产代码的约定，被保留）。

### EPS_A 是承重约定

```python
EPS_A = [[ 1/√2, -i/√2, 0],     # λ = +1
         [ 0,     0,    1],     # λ =  0
         [-1/√2, -i/√2, 0]]     # λ = -1
```

把第三行取正号、或换成 `(1, +i)` 那一支，逐事例与 tf-pwa 的重叠度会从
1.000000 掉到 0.62–0.67。详见
`analyses/kk_dis/task/20260929_螺旋度耦合振幅推导/4_三条路线/eps_a_provenance.md`。
`kernels/helicity.py` 与 `gen/templates/basis_layer.py` 里各有一份，**不要简化**。

## 6. 用法

```bash
# 看链、道表、自动枚举的 (l,s)
python -m agent.tensor show  specs/phif2.toml

# 校验（拓扑 / 量子数 / 道表 / 表象），非法拓扑给可读报错
python -m agent.tensor check specs/phif2.toml

# 列出四个表象与一致性自检
python -m agent.tensor bases

# 列出 chain.<name>/chain.<channel>.py
python -m agent.tensor chains

# 按 [[basis]] 产出脚本
python -m agent.tensor emit  specs/phif2.toml --out-dir generated

# emit 并直接跑一遍
python -m agent.tensor run   specs/phif2.toml --momenta mom.npz --out out --n 20000
```

生成脚本单独跑：

```bash
python generated/gen_tensor_phif2_covariant.py --momenta mom.npz --out data/gen --batch 128
python generated/gen_tensor_phif2_helicity.py  --momenta mom.npz --out data/gen --batch 128
python generated/gen_tensor_phif2.py           --momenta mom.npz --out data/gen --batch 128  # Omega
```

`--batch` 是必需的：`P4` 是 4⁸ 张量，单事例 float64 就 512 KB，5865 事例一次
算要 ~3 GB。生产用 `--batch 128`。三个脚本都支持 `--batch`，且都会按 batch
做 `jit`（见 §7）。

后端由环境变量 `TENSOR_BACKEND` 控制（`auto` / `jax` / `numpy`，默认 `auto`）：
有 jax 就用 jax，没有就回退 numpy。产物目录自带 `_backend.py`，所以脚本
**不依赖 `agent.tensor` 包**，在任意工作目录都能跑。

## 7. 后端：统一到 jax（numpy 为回退）

`kernels/_backend.py` 是唯一的后端开关：

```bash
TENSOR_BACKEND=auto    # 默认：有 jax 就用 jax，否则 numpy
TENSOR_BACKEND=jax     # 强制 jax（没有 jax 直接报错，不静默降级）
TENSOR_BACKEND=numpy   # 强制 numpy（对拍 / 无 GPU 环境）
```

**生产路径是 jax**：生成脚本对每个 batch 做一次 `jit`，XLA 负责常量折叠与
GPU 并行；jax 侧强制 `jax_enable_x64`（float32 会把相对误差抬到 1e-7，够不着
生产 cache 的 1e-12 验收线）。numpy 路径保留为**回退与数值基准**：

* 生成器本身（`emit` / `check` / `show`）不 import jax，在没有 jax 的机器上
  照样能出代码；
* 拟合侧与生产 cache 的一致性检验仍以 numpy 结果为准（它没有后端依赖）。

远程 HEP1（2×RTX3090，float64）上，同一份 5865 事例、同一份物理量、
`--batch 128`，三种后端的实测：

| 后端 | 总时间 | 吞吐 | 相对 numpy |
|---|---|---|---|
| numpy（CPU） | 1.492 s | 3 930 ev/s | 1.00× |
| jax（CPU） | 0.362 s | 16 199 ev/s | **4.1×** |
| jax（GPU, cuda:0） | 0.071 s | 82 865 ev/s | **21.1×** |

三者的 checksum **完全相同**（5.361629e+07），即后端切换没有改变数值。
GPU 相对 CPU 有明确提升（jax GPU 比 jax CPU 快 5.1×，比 numpy 快 21×），
所以统一用 jax、与拟合部分保持一致。逐道看，`phif243`（P4）在三者里都是
主导项。

### 移植到 jax 时踩到的坑（都在 `test_backend.py` 里锁住）

jax 数组**不可变**、`jit` 里不能用 numpy、`jnp.einsum` 不接受 numpy 的
`einsum_path` 格式。逐条对应：

1. **逐项赋值**：`D_lambda` / `node_factors` / `cg_matrix` / `small_d_weight`
   原来用 `out[i, j] = ...` 建数组。与事例无关的常量表（CG、Wigner-d、度规、
   ε 张量）改成 **host numpy 常量**（XLA 编译期折叠）；`D_lambda` 那种逐事例
   的改成向量化 `take` + `where` 掩码。
2. **tracer 上不能用 numpy**：道函数体与内核里的 `np.` 一律走 `xp`
   （`_backend.xp`）。投影算子缓存遇到 tracer 时**不缓存**（tracer 的
   `tobytes()` 拿不到值），直接交给 XLA 折叠。
3. **jit 的 bool 参数**：`omega(covariant=...)` 里的 `if covariant:` 会让
   jit 抛 `TracerBoolConversionError`，必须 `static_argnames=("covariant",)`。
4. **`einsum` 路径**：numpy 侧仍缓存 `optimize="optimal"` 路径；jax 侧交给
   XLA 自己的收缩规划。

### numpy 侧保留的两个热点优化

1. **对角度规升指标**：`T^{o...} = Σ G^{oi}…T_{i...}`，G 是对角的，求和只剩
   `i=o, j=p, …` 一项，等价于乘符号外积。P3 那种 6 指标的升指标从
   3.1 s 降到 2e-6 s（**~10⁶×**），逐位相同。
2. **多操作数收缩路径**：`phif222` 的两条 7 操作数 `einsum` 默认路径会物化巨大
   的中间张量。缓存 `optimize="optimal"` 的路径后从 0.10 s 降到 0.0001 s
   （**~1000×**），逐位相同。

合计：整批 5865 事例从 9.75 s 降到 1.49 s（numpy 侧），checksum 不变。

## 8. 验收

| # | 判据 | 状态 |
|---|---|---|
| 1 | **重算生产 cache**：`covariant` → `phif{0,2}_kk.npy`，`omega` → `Omega/K/M.npy` | PASS（rel ≤ 6.6e-15） |
| 1b | 协变内核与原 jax 版逐式一致（rel < 1e-12） | PASS（14/14 对象，最大 1.3e-15） |
| 1c | **jax 与 numpy 两后端逐道、逐对象一致**（17 道 + Omega/K） | PASS（rel ≤ 2.7e-15） |
| 1d | **生成的 3 类脚本在 jax 下跑得动，且与 numpy 后端逐文件一致** | PASS（rel ≤ 1.3e-15） |
| 2 | 跨表象恒等式 `\|W₊₁\|²+\|W₋₁\|² == \|T_x\|²+\|T_y\|²` | PASS（≤ 4.4e-16） |
| 3 | `precompute` 与拟合时 `amplitude` 给出同一 W | PASS（0.0） |
| 4 | 同一 TOML 两次 emit 字节一致 | PASS |
| 5 | 产物自足（不 import 框架）且可跑 | PASS |
| 6 | 非法拓扑给可读报错而非 `KeyError` | PASS（J>4 / l>4 / 末态不匹配 / 环） |
| 7 | sdist 内含 `agent/tensor/**/*.py` 与 `gen/templates/*.py` | PASS（33 + 16 文件） |
| 8 | `(l,s)` 自动枚举是通用的，不是硬编码 phif2 | PASS（换量子数仍枚举） |

跑测试：

```bash
python agent/tensor/tests/test_spec.py              # 11/11
python agent/tensor/tests/test_cross_basis.py       # 5/5
python agent/tensor/tests/test_emitter.py           # 4/4（含"产物在 jax 下跑"）
python agent/tensor/tests/test_backend.py           # 3/3（jax vs numpy，17 道）
python agent/tensor/tests/test_production_cache.py  # 4/4（需 tmp/bundledata/）
TENSOR_ROOT=<...>/tensor_example python agent/tensor/tests/test_covariant_vs_jax.py

# 想看 jax 路径的测试：用带 jax 的解释器 + TENSOR_BACKEND=jax
TENSOR_BACKEND=jax <带 jax 的 python> agent/tensor/tests/test_production_cache.py
```

### 生产 cache 回归（最硬的一条）

`analyses/kk_dis/tmp/bundledata/` 下有生产参照 cache 时，
`test_production_cache.py` 会重算并与它逐位比较：

```
route1_cartesian/real_data/phif0_kk.npy   rel=1.707e-15   (ref max 24.98)
route1_cartesian/real_data/phif2_kk.npy   rel=1.993e-15   (ref max 12780)
route4_omega/real_data/Omega.npy          rel=6.557e-15
route4_omega/real_data/K.npy              rel=6.183e-15
route4_omega/real_data/M.npy              rel=3.792e-16
端到端（emit 出的脚本）phif0_kk            rel=1.610e-15
```

⚠️ **真实样本的 psi 质量必须是 `3.685782956`**（不是 `Tensor/FUNC.py` 里硬编码
的 3.686）。用错会让 phif0 的 rel 从 1.7e-15 掉到 4.2e-04 —— 数值仍然"看着像样"，
所以测试里专门有一条 `test_wrong_psi_mass_is_detectable` 保证这个测试有区分力。
在 TOML 里用 `[chain].psi_mass` 控制。

另外，协变内核的原生分量序是 **`(x, y, z, E)`**（与 `Tensor/BASE.py` 一致），
而 `kernels/omega.py` 内部用 `(E,x,y,z)`。生成物曾经在这里照抄了 omega 的置换，
导致与生产 cache 差 0.98（完全错位）—— 端到端那条测试就是抓这个的。

## 9. 表象层模板（`analyses/kk_dis/gen/templates/basis_layer.py`）

框架侧定义在 `agent/tensor/basis.py`；分析侧的**可生成模板**是
`gen/templates/basis_layer.py`，含 `# SECTION: BASIS_LAYER`，已接进
`llm_config_fit.toml`：

```toml
[ref]
code_template = [..., "gen/templates/basis_layer.py"]

[stages.assemble_help_functions]
input = {"list"=[..., "ref.code_template.BASIS_LAYER", ...]}
```

**为什么必须做这件事**：这 251 行（`EPS_A` / `apply_helicity` /
`intensity_from_amplitude` / `_assemble_omega_cache` / `_load_cached_tensor` /
`_tile_cached_tensor` / `_cached_tensor_magnitude` 等）此前不在任何模板里，
重放 `assemble_help_functions` 只得到 206 行 / 18 个符号，而
`run/likelihood_function.py:4-20` 直接 import 其中 5 个 —— **任何一次全量重跑
生成器都会让 likelihood `ImportError`**。接进模板后重放覆盖 33/33 个符号
（28 个来自模板 + 5 个来自 stage）。

11 个函数的函数体与 `run/base_functions.py` **逐字节相同**（用脚本核对）。

## 9b. 审计报告里 `agent/` 侧问题的修复（2026-10-18）

来源：`analyses/kk_dis/task/20261017_多表象拟合流程标准化_多GPU/3_报告/LLMPWA代码生成能力审计.md`

| 审计项 | 问题 | 处理 |
|---|---|---|
| §2.2 | `psi2s -> phi + pi`（daughter 是末态）`check` 通过但运行时 `KeyError: 'daughters'` | **修**：`kernels/omega.py` 把末态节点的衰变因子当 1（== `D^{0*}_{0,0}`），`node_factors` 跳过末态节点。推荐做法（"在 runtime 里支持它"） |
| §2.3 | `J>4` / `l>4` 会运行时 `KeyError`；三层级联静默漏掉内层角分布 | **修**：`spec.validate` 前置报可读 `SpecError`，并新增**级联深度 ≤ 2** 检查 |
| §2.4 | `specgen` 零测试 | **修**：新增 `test_kernels.py`（9 条：CG 闭式/正交性、Wigner-D 正交性、`cg_matrix` shape、Blatt-Weisskopf 边界）+ `test_generic_chain.py`（`rhof2_generic.toml` 端到端 smoke，合成四动量） |
| §3.5 | LLM 调用无重试，429 直接终止整条流水线 | **修**：`agent/easytrans_client.py` 加 `_call_with_retry`（指数退避 + 抖动，`is_retryable` 区分 429/5xx/网络错误 vs 401/400），三个 API 入口都走它；8 条离线测试 |
| §4.1 | 模块 docstring 把 `foreach` 写成一种 `kind`，照抄立刻 `ConfigError` | **修**：docstring 改为正确写法；`kind="foreach"` 给**说清正确写法**的报错；`foreach` 用在非 `llm` stage 上也报错（原先静默忽略） |
| §2.1 | `verify` 硬编码本机路径 | **已消除**：`agent/tensor/` 不再带 `verify.py`（那份是分析专用、一次性验证），改为通用的 `test_production_cache.py`（可缺省跳过） |

新增测试计数：

```
test_spec              10/10
test_kernels            9/9     <- 新增（§2.4）
test_cross_basis        5/5
test_emitter            3/3
test_generic_chain      3/3     <- 新增（§2.4）
test_production_cache   4/4
test_covariant_vs_jax   PASS
agent/test_easytrans_client.py  8 条全过（含 §3.5 的 5 条重试测试）
```

### §3.5 重试的可调参数

```
EASYTRANS_MAX_RETRIES   默认 5
EASYTRANS_RETRY_BASE    退避基数秒，默认 1.0
EASYTRANS_RETRY_MAX     单次退避上限秒，默认 30.0
```

可重试：`408/409/425/429/500/502/503/504/522/524`、`APIConnectionError`/超时、
网关业务码含 `concurrency_limit` 等。不可重试（立刻抛出，不浪费时间）：
`400/401/403/404/422` 与参数错误。

## 9c. 道表完整性交叉检查（`[chain].check_channels`）

**背景**：`resolve_channels()` 原本是 `if 显式道表: 用它 else: 自动枚举`，
两条路**从不比对**。后果是下面这些都会**静默通过**：

| 情况 | 旧行为 | 新行为 |
|---|---|---|
| 漏一条道（如漏 `(2,2)`） | 静默通过，少一行 | **报错** |
| 漏两条 | 静默通过 | **报错** |
| 多一条宇称禁止的道（如 `(1,1)`） | 静默通过 | **报错** |
| `(l,s)` 标错（如 `phif223` 写成 `(4,3)`） | 静默通过 | **报错** |
| 顺序不同、集合相同 | 通过 | 通过（顺序承重，由 TOML 决定） |
| 完全不写 `[[channel]]` | 自动枚举 | 自动枚举 |

道表是 tensor cache 的**行序**，错了不会崩，只会让拟合结果错位 ——
所以这条校验是整份 spec 里性价比最高的一条。

```
$ python -m agent.tensor check phif2.toml
  道（5 条，显式）：
    phif201          (l,s)=(0,1)
    ...
  道表完整性：与宇称+三角枚举一致（5 条）
```

**只要子集**（例如先只做 `f0` 族两条）时，显式关掉：

```toml
[chain]
check_channels = false
```

此时 `check` 会打印 `道表完整性：**已用 check_channels=false 关闭**（枚举应为 N 条）`，
提醒你枚举本应有几条。parity 不全时无法枚举，交叉检查自动跳过。

## 9d. 跨算法一致性检验（`verify` / `run` 后自动跑）

`agent.tensor` 有两条**独立实现**的路线：

| 路线 | 内核 | 产物 |
|---|---|---|
| cartesian | `kernels/covariant.py`（Chung 协变张量） | `<chain>_kk.npy` |
| helicity | 同一份 `T` 乘 `EPS_A` | `<chain>_hel.npy` |
| omega | `kernels/omega.py`（级联角 + CG 装配） | `Omega/K/M/g/norm` |

它们必须给出同一个物理量，只差**逐道的复常数** `c_i`：

```
W_i^{cov}(e,lam) = c_i * g_i * K_i(e) * (M_i . Omega)_{e,lam}      逐事例成立
```

`c_i` 是逐道的整体归一化（耦合 + 归一化 + 相位，由拟合吸收），所以**不能**
要求 `c_i = 1`；能要求的是 `c_i` **与事例无关**。

```bash
# 只看已有产物（不需要四动量）
python -m agent.tensor verify phif2.toml \
    --cartesian <cartesian 目录> --omega <omega 目录>

# run 之后自动跑（同时出了 omega 与 cartesian/helicity 时）
python -m agent.tensor run phif2.toml --momenta <npz> --out <dir>
```

输出：

```
跨算法一致性（covariant/EPS_A  vs  Omega）
  phif201    OK   c=  -14.605935+0.000000j  rel=9.672e-13  分量=5984
  phif221    OK   c=   27.541215-0.000000j  rel=3.055e-10  分量=5978
  ...
  最大相对残差 7.525e-10（阈值 1e-06）-> PASS
```

### 能抓到什么（实测的负向测试）

| 注入的错误 | 结果 |
|---|---|
| 两条道的张量互换 | **抓到**（rel 0.89） |
| `T` 的某个分量整体取反（EPS_A 约定写错） | **抓到**（rel 1.05） |
| `Omega` 的事例轴错位 | **抓到**（rel 1.01） |
| `K` 的事例轴错位 | **抓到**（rel 1.08） |
| 隔事例缩放（K/Omega/T） | **抓到**（rel 0.1–0.23） |
| 单条道随事例漂移 | **抓到**（rel 0.18–0.36） |
| **逐道**常数偏移（K、g、单道整体） | 通过 —— 按定义被 `c_i` 吸收，**不是** bug |

### 它抓到的真实问题

第一次在**新生成**的产物上跑，`phif221/222/223/243` 残差停在 ~3e-4，
`phif201` 却是 1e-12。原因是两条路线的 **psi(2S) 口径不同**：

* covariant 侧：静态近似 `(0,0,0,psi_mass)`，取自 `[chain].psi_mass`；
* omega 侧：末态四动量之和。

真实样本的 psi 能量是 `3.685782956`，而 spec 里默认写 `3.686`
（相对差 5.9e-05）—— 正是这个差造成 1e-4 的残差。把 `psi_mass` 改成真实值后，
五条道全部回到 ~1e-10。**这个错误在旧流程里是静默的**（数值"看着像样"，
只有 `phif201` 恰好不敏感），现在会被明确拦下并给出提示。

`crosscheck.py` 里也修了一个隐患：`_precompute` 原本强制 `dtype=float64`，
读 `<chain>_hel.npy`（已经是复数 W）时会把虚部悄悄丢掉 —— 现已保留 dtype，
并用 `already_rotated=True` 区分"要不要再乘一次 `EPS_A`"。

## 10. 深度限制按表象分级（2026-10-19 修正）

`spec.validate` 原本无条件拒绝级联深度 > 2 的树，理由是「runtime 只算
`D0*D1*D2`」。但实测表明**这条限制只对 omega 内核成立**：

```
$ grep -c "chain\|nodes" agent/tensor/kernels/covariant.py
0
```

协变张量的道函数是**按道名硬编码的公式**，完全不读树结构；只有
`kernels/omega.py` 的 `chain_angles` / `D_lambda` 才逐节点读树。

现在改为**按 `[[basis]]` 分级**：

* 只要这份 spec 出了 `kernel = "omega"` 且深度 > 2 → **报错**（维持原行为）；
* 只出 `cartesian` / `helicity` → **放行**（深度不限）。

这与「`J<=4` / `l<=4` 是 omega 专属约束」是同一类问题：能力边界挂在**表象**上，
不该挂在 spec 上。`spec["depth"]` 现在总是被记录，供调用方查询。

判据（`agent/tensor/tests/test_ex3_channels.py::test_depth_grading`）：
三层树 `ψ→ρ′π, ρ′→φπ, φ→K⁺K⁻` 在 cartesian/helicity 下读入成功，
追加一个 `[[basis]] name="omega"` 后被拒。

## 11. 例 3 的 ρ′/b₁ 道（`chains/rhopi` + `chains/b1pi`）

文档 `analyses/kk_dis/document/ψ衰变到介子的分波分析协变张量形式.md` 374–416 行
的 10 个振幅，逐式移植自 `Tensor/FUNC.py::{_u_rho_*, _u_SS/SD/DS/DD_*}`（与
其余内核一样走 `_backend.xp`，jax / numpy 两条后端都可用）。

| 族 | 道 | 文档 | 完备标签 |
|---|---|---|---|
| ρ′ | `u_rho_1` `u_rho_2` | Eq.(46) | 只有一个独立振幅；`_1`/`_2` 是旁观者 π 的取法 |
| b₁ | `u_SS_{1,2}` `u_SD_{1,2}` `u_DS_{1,2}` `u_DD_{1,2}` | Eq.(47)–(50) | 顶层波 × 内层波 = SS/SD/DS/DD，各两种旁观者 |

三条容易踩的约定：

1. **`_1`/`_2` 在两个族里含义相反**（照抄参考实现的命名）：
   ρ′ 族 `_1` 的旁观者是粒子 3（π⁺），b₁ 族 `_1` 的旁观者是粒子 4（π⁻）。
2. **`(l,s)` 不是 b₁ 族的完备标签**：8 条道只映射到 2 组 `(l,s)`，所以
   `[chain].check_channels = false`，且**不能**用 omega 表象。
3. **签名统一为 `(psi, phi, f2, Kp, Km, Pip, Pim)`**，即使 ρ′/b₁ 族忽略
   `phi` / `f2` —— 这样道表可以混装两个族的道（见 `emitter.py::raw_tensors`）。

验收（两条独立路径）：

```bash
# 与 jax 原版逐式对拍（24 个对象：7 原语 + 7 旧道 + 10 新道，rel < 1e-12）
TENSOR_ROOT=<...>/tensor_example python agent/tensor/tests/test_covariant_vs_jax.py

# 纯 numpy 自洽性（8 项，永远能跑）：发现/导入/有限性/旁观者对调/独立性/深度分级/emit
python agent/tensor/tests/test_ex3_channels.py
# 端到端：合成四动量 -> 生成脚本 -> 产物 vs 内核 + W = EPS_A·T
analyses/kk_dis/.venv/bin/python analyses/tensor_examples/check_ex3.py --n 500
```

分析目录见 [`analyses/tensor_examples/README.md`](../analyses/tensor_examples/README.md)。

## 12. 已知边界（不在本次范围）

* `pipegen/{bundle,emitter,slice_bundle_data}.py` 的既有缺陷未修：
  `emitter.py:905-906` 的 `TASK_ROOT` 差一级（`exit=127`）；
  `bundle.py:902` 与 `emitter.py:743` 硬编码 specgen 路径；
  `bundle.py` 的 `shutil.rmtree(wd)` 会连带删掉 `gen/manifest_fit.toml`。
* `run/base_functions.py:445` 的 `shard_data_distributed` **没有**整除回退，
  5865 事例在 2 GPU 上会 `IndivisibleError`（修法：用 `spec = P()` 复制）。
* 文档漂移：`analyses/kk_dis/document/OPERATIONS.md §2.2` 原写"8 个 stage"、
  `documentation/dsh-client-plugin-design.md` 原写"8 个 stage"，实际是 10 个
  —— **已在本次修正**。
* `Tensor/` 目前只在被 gitignore 的 `task/` 树里；要真正跨分析复用需纳入版本
  控制，且出处/许可待确认（`4_三条路线/audit_provenance.md`）。
