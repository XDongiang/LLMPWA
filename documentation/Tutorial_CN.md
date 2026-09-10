# LLMPWA 使用教程

本教程以 `analyses/kk_dis` 为例，说明如何配置环境、运行 `agent/` 代码生成流水线，
以及运行生成的分布式拟合代码。更完整的 stage 说明与运行时契约见
[`analyses/kk_dis/README.md`](../analyses/kk_dis/README.md)。

## 1. 环境准备

### 1.1 生成器依赖

代码**生成器**只依赖三个第三方包：

```bash
pip install -r requirements.txt
```

即 `openai`、`python-dotenv`、`toml`。

### 1.2 运行拟合代码的依赖（可选）

生成代码后要**运行拟合**还需要物理/数值栈，安装 `pyproject.toml` 的 `runtime` extra：

```bash
pip install -e ".[runtime]"
```

并配套 CUDA 与 JAX GPU 环境（`jax[cuda...]`），后续步骤会用到。

### 1.3 配置 LLM API

```bash
export EASYTRANS_API_KEY="your_api_key_here"
# 可选：默认即可
export EASYTRANS_BASE_URL="https://api.easytransnote.com/v1"
export EASYTRANS_MODEL="gemini-2.5-pro"
```

也可以把这些写进仓库根目录的 `.env`（`agent/` 会通过 `load_dotenv()` 读取）。

## 2. 运行代码生成流水线

在仓库根目录执行：

```bash
# 静态检查（不调用 LLM）
python -m agent.cli --workdir analyses/kk_dis --config llm_config_fit.toml --check-only

# 跑完整流水线
python -m agent.cli --workdir analyses/kk_dis --config llm_config_fit.toml

# 只重跑某个 stage
python -m agent.cli --workdir analyses/kk_dis --config llm_config_fit.toml --stage generate_fit_script
```

流水线阶段一览（按 `llm_config_fit.toml` 顺序）：

| stage | kind | 产物 |
|-------|------|------|
| `config_strip` | python | `run/free_params.toml`、`gen/fragments/stripped_config.toml` 等 |
| `inspect_data_shapes` | agent | 数据 shape 报告与 `load_data`/`normalize_data`/`shard_data_distributed` 片段 |
| `make_initial_args` / `save_results` | python | 初值与存盘辅助函数 |
| `assemble_help_functions` | python | `run/base_functions.py` |
| `classification` | llm | `gen/fragments/classification.json` |
| `generate_likelihood` | agent | `run/likelihood_function.py`（需人工审阅） |
| `generate_fit_script` | agent | `run/fit_script.py`（需人工审阅） |

说明：

- `inspect_data_shapes` 默认 `cache = true`，配置/数据未变时可复用。
- `generate_likelihood` / `generate_fit_script` 会进入交互与人工审阅；拒绝后 Agent 可据
  反馈修改后再 `task.finish`。

## 3. 运行生成的拟合（单机）

```bash
cd analyses/kk_dis
python run/fit_script.py
```

单进程时 `JAX_NUM_PROCESSES` 默认为 1，不强制 `jax.distributed.initialize`。

## 4. 多节点 Docker 拟合

节点与容器配置见 `node_config.toml`（header IP/PORT、worker 列表、
`container.script = "run/fit_script.py"` 等）。

```bash
# 仓库根目录
./docker/run_fit_multinode.sh --workdir analyses/kk_dis
# 或显式指定
./docker/run_fit_multinode.sh --workdir analyses/kk_dis --config analyses/kk_dis/node_config.toml
```

脚本会：

1. 将 workdir rsync 到各节点并挂载为容器工作区；
2. 注入 `JAX_COORDINATOR_ADDRESS` / `JAX_NUM_PROCESSES` / `JAX_PROCESS_ID` 及 NCCL 环境；
3. 各节点执行同一 `run/fit_script.py`；
4. 将 header 上的 `output` 同步回本地 workdir。

## 5. 查看流水线状态与可视化

```bash
python agent/pipeline_state.py -w analyses/kk_dis -c llm_config_fit.toml -o gen/pipeline_state.json
python agent/pipeline_viz.py -w analyses/kk_dis -c llm_config_fit.toml -o gen/pipeline_report
```

详见 [pipeline_state_exporter.md](pipeline_state_exporter.md) 与
[pipeline_visualization_plan.md](pipeline_visualization_plan.md)。

## 6. 新建一个分析

复制 `analyses/kk_dis`，替换 `resonances_config.toml` 与 `data/`，按需调整
`gen/prompts/`，再执行同样的引擎命令即可复用整套流程。
