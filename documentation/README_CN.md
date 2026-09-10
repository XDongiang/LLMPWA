# LLMPWA — 大模型驱动的分波分析代码生成器

LLMPWA 通过 LLM/Agent 流水线，把声明式的共振态配置转换为分波分析（PWA）拟合代码。
端到端参考实现是 [`analyses/kk_dis`](../analyses/kk_dis/README.md)：从
`resonances_config.toml` 出发，产出 **JAX 事件级数据并行 + SciPy Newton-CG** 的分布式拟合入口。

## 工作原理

1. 用 TOML 声明共振与振幅（`resonances_config.toml`）。
2. `agent/` 引擎按 `llm_config_*.toml` 定义的 stage 顺序执行
   （`kind = python | llm | agent`），每个 stage 的输出登记进 manifest，
   下游可用 `<<stages.xxx.yyy>>` 引用。
3. 可复用的数值原语/模板走模板；分析相关逻辑（数据 shape、似然公式、拟合主程序）
   由 LLM/Agent 分阶段生成。
4. 最终得到可直接运行的 `run/fit_script.py`，可对接多节点 Docker 集群。

## 环境要求

- Python 3.8+（推荐 3.11+，可在有 `tomllib` 时优先使用标准库）
- EasyTrans 的 LLM API Key（`EASYTRANS_API_KEY`）

### 安装生成器依赖

```bash
pip install -r requirements.txt
```

只安装 `openai`、`python-dotenv`、`toml` —— 这三个是生成器本身所需的全部第三方包。
**运行**生成的拟合代码还需要安装 `pyproject.toml` 中 `runtime` extra 对应的物理/数值栈
（JAX、NumPy、SciPy 等），并配置好 CUDA/JAX 环境。

### 配置 API

```bash
export EASYTRANS_API_KEY="your_api_key_here"
# 可选覆盖
export EASYTRANS_BASE_URL="https://api.easytransnote.com/v1"
export EASYTRANS_MODEL="gemini-2.5-pro"
```

## 快速开始（以 kk_dis 为例）

### 生成代码

```bash
# 静态检查
python -m agent.cli --workdir analyses/kk_dis --config llm_config_fit.toml --check-only

# 跑完整流水线
python -m agent.cli --workdir analyses/kk_dis --config llm_config_fit.toml

# 只重跑某一 stage
python -m agent.cli --workdir analyses/kk_dis --config llm_config_fit.toml --stage generate_fit_script
```

流水线的 stage 列表与运行时契约见
[`analyses/kk_dis/README.md`](../analyses/kk_dis/README.md)。

### 运行生成的拟合（单机）

```bash
cd analyses/kk_dis
python run/fit_script.py
```

### 多节点 Docker 拟合

```bash
./docker/run_fit_multinode.sh --workdir analyses/kk_dis
```

## 仓库结构

```
agent/          通用生成引擎（cli、engine、stage runner、agent tools）
analyses/       各分析目录：配置、prompts、handlers 与生成代码
  kk_dis/       端到端参考实现
docker/         多节点 JAX Docker 拟合工具
documentation/  指南（流水线状态、可视化、dsh 工作台、教程）
```

## 文档索引

- [analyses/kk_dis/README.md](../analyses/kk_dis/README.md) — 规范生成流程
- [Tutorial_CN.md](Tutorial_CN.md) / [Tutorial_EN.md](Tutorial_EN.md)
- [docker/design.md](../docker/design.md) — 事件并行 Newton-CG/HVP 设计
- [docker/run_mulit_node.md](../docker/run_mulit_node.md) — 多机 Docker / NCCL 备忘

### 流水线状态与可视化

- [流水线状态导出器](pipeline_state_exporter.md)：`agent/pipeline_state.py` 将
  `resonances_config.toml` 汇总为 `gen/pipeline_state.json` 快照。
- [可视化方案(路线A)](pipeline_visualization_plan.md)：纯报告（Markdown/Mermaid /
  自包含 HTML）生成器。
- [DSH 工作台设计稿(路线B)](dsh-client-plugin-design.md)：原生 DeepSeek Harness
  client-plugin —— 左侧 Sidebar 按钮打开整页工作台，预览 analysis 的流水线 DAG。
- [为 dsh 写整页插件实战指南](dsh-page-plugin-guide.md)：把本次工作整理成的可复用
  cookbook —— 建包、slot 接入、四个 props share、三处注册、构建/测试/运行与踩坑清单。

## 贡献

欢迎任何形式的贡献，包括但不限于新功能、代码修复、文档改进等。请通过 Pull Requests
或 Issues 分享您的想法。

## 许可证

此项目采用 [MIT 许可证](../LICENSE)。

## 联系方式

- GitHub Issues：https://github.com/caihao/PWACG/issues
