# kk_dis：LLM 驱动的分布式分波分析拟合代码生成

`analyses/kk_dis` 是 LLMPWA 中一套**已跑通**的端到端示例：从共振态 TOML 配置出发，经多阶段流水线（Python handler + LLM + 交互式 Agent）自动生成 **JAX 事件级数据并行 + SciPy Newton-CG** 的完整拟合代码，并可在多机多卡 Docker 环境中执行。

配置入口：

```text
analyses/kk_dis/llm_config_fit.toml
```

运行产物（生成代码）：

```text
analyses/kk_dis/run/base_functions.py
analyses/kk_dis/run/likelihood_function.py
analyses/kk_dis/run/fit_script.py
```

---

## 1. 项目定位

分波分析（Partial Wave Analysis, PWA）拟合代码通常包含：

- 共振传播子与振幅构造
- 真实数据 / MC 数据加载与归一化
- 负对数似然（NLL）与约束项
- 高维参数优化与误差估计

手工编写上述代码成本高、易出错、难复用。LLMPWA 的思路是：

1. **物理模型**用 `resonances_config.toml` 声明式描述；
2. **通用数值与模板**（dplex、BW/Flatté、日志、路径）从模板复用；
3. **分析相关逻辑**（数据 shape、似然公式、拟合主程序）由 LLM/Agent 按阶段生成；
4. 最终得到可直接跑的分布式拟合入口 `run/fit_script.py`。

本目录 `kk_dis` 对应 **KK 道分布式拟合**场景（`phif0_*` / `phif2_*` 等共振），是当前「生成拟合脚本」流程的完成态参考实现。

---

## 2. 总体架构

```text
resonances_config.toml
        │
        ▼
┌───────────────────────────────────────────────────────┐
│  agent Engine 驱动 llm_config_fit.toml 各 stage       │
│  kind = python | llm | agent                          │
└───────────────────────────────────────────────────────┘
        │
        ├─► run/free_params.toml          自由参数表
        ├─► gen/fragments/*               中间片段
        ├─► run/base_functions.py         公共运行时
        ├─► run/likelihood_function.py    似然模块
        └─► run/fit_script.py             分布式拟合入口
                    │
                    ▼
        多节点 Docker / JAX Mesh (event 轴)
        SciPy Newton-CG + JIT NLL / Grad / HVP
                    │
                    ▼
        output/fit/{values,errors,free_params_fitted.toml}
```

### 计算架构要点

参数维数相对较小（本例约 **65** 维），计算瓶颈在事件级振幅与求和。因此并行轴是 **事件（event）** 而不是参数：

| 层级 | 做法 |
|------|------|
| 数据 | 各 GPU/进程持有 event shard：`data_*` / `mc_*` / `truth_*` |
| 参数 | 小向量复制到所有设备（`NamedSharding(mesh, P())`） |
| 似然 | 本地贡献 + 跨设备 all-reduce（XLA / `psum`）得到全局 NLL |
| 优化 | 每进程跑**同一** SciPy Newton-CG 循环，collective 顺序一致 |
| I/O | 仅 `process_id == 0` 写日志与结果 |

Newton-CG 需要 `fun` / `jac` / `hessp`。分布式下：

- `jit_likelihood`：全局 NLL 标量  
- `jit_grad`：约 65 维梯度  
- `jit_hvp`：HVP（通常 `jvp(grad)`），通信量与参数维同阶  

---

## 3. 生成流水线（`llm_config_fit.toml`）

引擎：`python -m agent.cli --workdir analyses/kk_dis --config llm_config_fit.toml`  
阶段按配置顺序执行；中间结果登记在 `gen/manifest_fit.toml`，下游可用 `<<stages.xxx.yyy>>` 引用。

### 3.1 `config_strip`（python）

- **输入**：`resonances_config.toml`
- **作用**：剥离固定/自由参数，整理 sbc/amp 列表
- **输出**：
  - `run/free_params.toml`
  - `gen/fragments/stripped_config.toml`
  - `gen/fragments/free_params_range.toml`（含高斯约束参考区间）
  - `all_sbc` / `all_amp` / `all_sbc_amp`（direct，供后续 stage）

### 3.2 `inspect_data_shapes`（agent，可缓存）

交互式检查 `data/` 下 npy 的 shape/dtype，并生成三个预处理函数片段：

| 输出 | 含义 |
|------|------|
| `gen/fragments/data_shapes_report.txt` | shape 报告 |
| `gen/fragments/load_data_func.py` | `load_data()` |
| `gen/fragments/normalize_data_func.py` | `normalize_data(data)` |
| `gen/fragments/shard_data_distributed_func.py` | `shard_data_distributed(data, mesh)` |

约定概要：

- 加载 `data/real_data/{var}.npy`、`data/mc_truth/{var}.npy`
- truth-MC 默认截断 `n_truth_mc = 150000`（可按可用长度 clamp）
- 分片：1D 事件 → `P("event")`；3D 振幅 → `P(None, "event", None)`

### 3.3 `make_initial_args` / `save_results`（python）

由 `free_params.toml` 生成：

- `gen/fragments/make_initial_args.py` → `make_initial_args()`
- `gen/fragments/save_results.py` → `save_result(...)`（拟合结果写回 free_params 风格 TOML）

### 3.4 `assemble_help_functions`（python）

拼接模板与上述片段，产出统一运行时：

```text
run/base_functions.py
```

包含（典型）：

- 路径 / `chdir` / 日志
- `dplex_*` 复数 einsum 工具
- `BW` / `flatte980` / `flatte1270` 等物理函数
- `make_initial_args` / `save_result`
- `load_data` / `normalize_data` / `shard_data_distributed`

**规则**：后续 LLM 产物应 `from base_functions import ...`，不重复实现这些符号。

### 3.5 `classification`（llm）

根据 stripped 配置对传播子/振幅分类，输出 JSON：

```text
gen/fragments/classification.json
```

例如本分析中的分组：`BW_flatte980`、`BW_BW`、`BW_flatte1270` 等，供似然生成时按类构造参数与传播子。

### 3.6 `generate_likelihood`（agent，需人工审阅）

- 通过 `ask_user` 索取 **likelihood 的 LaTeX/公式说明**（data/MC 项、相干求和、高斯约束等）
- 生成 `run/likelihood_function.py`
- 做冒烟测试，日志写入 `run/likelihood_test_result.txt`
- `require_approval = true`：可 reject 回灌反馈后重生成

期望 API（以实际生成代码为准）：

- `combined_likelihood` 或 `make_distributed_likelihood(data_size)`
- 可与 `jvp(grad(...))` 配合做 HVP

本仓库一次成功冒烟示例（子集数据）：

```text
n_args = 65
combined_likelihood ≈ 401.26
PASS
```

### 3.7 `generate_fit_script`（agent，需人工审阅）

- 只**组装**主程序，复用 `base_functions` 与 `likelihood_function`
- 生成 `run/fit_script.py`
- 轻量语法/import 检查即可，**不在生成阶段跑完整多节点 minimize**

主流程骨架：

1. `init_distributed()`（`JAX_COORDINATOR_ADDRESS` / `JAX_NUM_PROCESSES` / `JAX_PROCESS_ID`）
2. `build_mesh()` → `Mesh(..., axis_names=("event",))`
3. `load_data` → `normalize_data` → `shard_data_distributed`
4. JIT：`nll` / `grad` / `hvp`
5. 全进程 SciPy `minimize(..., method="Newton-CG", hessp=...)`
6. HVP 列式构造 Hessian → 对角误差
7. 仅 rank0 写出：
   - `output/fit/fit_result_values.npy`
   - `output/fit/fit_result_errors.npy`
   - `output/fit/free_params_fitted.toml`

---

## 4. 目录结构

```text
analyses/kk_dis/
├── llm_config_fit.toml          # 生成流水线定义（本流程入口）
├── resonances_config.toml       # 共振与振幅物理配置
├── node_config.toml             # 多节点 Docker 节点表
├── README.md                    # 本文档
├── config/                      # 日志等运行配置
├── data/
│   ├── real_data/               # 真实数据 npy
│   ├── mc_truth/                # MC / truth（拟合用）
│   ├── draw_data/ · draw_mc/    # 绘图相关（可选）
│   └── weight/
├── gen/
│   ├── prompts/                 # 各 stage 的 system/task 提示词
│   ├── handlers/                # python stage 处理器
│   ├── templates/               # fit/shared/physics 等代码模板
│   ├── fragments/               # 中间片段与分类结果
│   ├── llm_logs/                # agent transcript / 可读日志
│   └── manifest_fit.toml        # 阶段产物清单
├── run/
│   ├── free_params.toml
│   ├── base_functions.py        # 组装后的公共库
│   ├── likelihood_function.py   # 似然
│   ├── fit_script.py            # 拟合主入口
│   └── likelihood_test_result.txt
├── logs/
└── output/fit/                  # 拟合结果
```

仓库级相关组件：

| 路径 | 说明 |
|------|------|
| `agent/` | 通用引擎：`cli` / `engine` / stage runner / agent tools |
| `docker/run_fit_multinode.sh` | 同步 workdir、多节点起容器跑 `run/fit_script.py` |
| `docker/design.md` | 事件并行 Newton-CG / HVP 设计说明 |
| `docker/run_mulit_node.md` | 多节点环境变量示例 |

---

## 5. 如何使用

### 5.1 生成代码

在仓库根目录（需配置好 LLM API，如 `EASYTRANS_API_KEY`）：

```bash
# 静态检查
python -m agent.cli --workdir analyses/kk_dis --config llm_config_fit.toml --check-only

# 跑完整流水线
python -m agent.cli --workdir analyses/kk_dis --config llm_config_fit.toml

# 只重跑某一 stage
python -m agent.cli --workdir analyses/kk_dis --config llm_config_fit.toml --stage generate_fit_script
```

说明：

- `inspect_data_shapes` 默认 `cache = true`，配置/数据未变时可复用。
- `generate_likelihood` / `generate_fit_script` 会进入交互与人工审阅；拒绝后 Agent 可据反馈修改再 `task.finish`。

### 5.2 单机试跑

在分析目录（或依赖 `base_functions` 已 `chdir` 到 workdir）：

```bash
cd analyses/kk_dis
# 视环境设置 GPU / CPU
python run/fit_script.py
```

单进程时 `JAX_NUM_PROCESSES` 默认为 1，不强制 `jax.distributed.initialize`。

### 5.3 多节点 Docker 拟合

节点与容器配置见 `node_config.toml`（header IP/PORT、worker 列表、`container.script = "run/fit_script.py"` 等）。

```bash
# 仓库根目录
./docker/run_fit_multinode.sh --workdir analyses/kk_dis
# 或显式指定
./docker/run_fit_multinode.sh --workdir analyses/kk_dis --config analyses/kk_dis/node_config.toml
```

脚本会：

1. 将 workdir rsync 到各节点并挂载为容器工作区；
2. 注入 `JAX_COORDINATOR_ADDRESS` / `JAX_NUM_PROCESSES` / `JAX_PROCESS_ID` 及 NCCL 相关环境；
3. 各节点执行同一 `run/fit_script.py`；
4. 将 header 上的 `output` 同步回本地 workdir。

环境变量约定与 `fit_script.init_distributed()` 一致。

---

## 6. 关键产物与契约

### 运行时模块契约

| 模块 | 职责 | 被谁使用 |
|------|------|----------|
| `run/base_functions.py` | 工具、物理、数据、初值、存盘 | likelihood、fit_script |
| `run/likelihood_function.py` | NLL / 分布式 likelihood 工厂、参数抽取、约束 | fit_script |
| `run/fit_script.py` | 分布式初始化、Mesh、优化循环、结果写出 | Docker / 用户启动 |

### 拟合输出

```text
output/fit/fit_result_values.npy      # 拟合参数
output/fit/fit_result_errors.npy      # √diag(H⁻¹)
output/fit/free_params_fitted.toml    # 与 free_params 同结构的结果 TOML
```

### 物理配置侧

- 共振定义：`resonances_config.toml`（传播子类型 BW / flatte*、质量宽度、耦合 const/theta、`fixed`/`range`）
- 高斯约束参考：`gen/fragments/free_params_range.toml`（生成 likelihood 时使用）
- 似然数学说明可参考：`gen/fragments/combined_likelihood_math.md`（若已提供）

---

## 7. 设计原则与边界

1. **配置驱动**：改共振列表 / 固定自由参数 → 重跑相关 stage，而非手改整文件。
2. **模板 + 生成分离**：数值原语与路径日志走模板；分析相关公式与 glue 走 LLM/Agent。
3. **显式数据参数**：分布式下 NLL 将 `jax_data` 作为参数传入，避免在 `jit` 闭包中捕获不可寻址 sharded 数组。
4. **全进程同步优化循环**：保证 collective 顺序一致；写文件只在 chief。
5. **生成期不做重训练**：Agent 仅允许语法/import/小规模冒烟，完整拟合放在 Docker 多节点阶段。
6. **审阅门禁**：关键路径（likelihood、fit_script）`require_approval = true`，便于物理与工程双检。

---

## 8. 与仓库其他文档的关系

| 文档 | 关系 |
|------|------|
| 根目录 `README.md` | LLMPWA 总览（偏早期生成器说明） |
| `documentation/README_CN.md` | PWACG/安装与 JAX/ROOT 环境 |
| `docker/design.md` | 本流水线采用的事件并行 + Newton-CG/HVP 设计原文 |
| `docker/run_mulit_node.md` | 多机 Docker / NCCL 运行备忘 |
| 本文 `analyses/kk_dis/README.md` | **kk_dis 拟合脚本生成流程的完成态说明** |

---

## 9. 小结

`llm_config_fit.toml` 定义了一条从 **共振配置 → 参数剥离 → 数据预处理 → 公共运行时 → 分类 → 似然 → 分布式拟合主程序** 的完整生成链。`analyses/kk_dis` 展示了该链在 KK 道分析上的落地结果：约 65 维自由参数、JAX event-sharding、SciPy Newton-CG，以及可对接多节点 Docker 的 `run/fit_script.py`。

新分析可复制本目录结构，替换 `resonances_config.toml` 与 `data/`，按需调整 `gen/prompts/`，再执行同一引擎配置即可复用流程。
