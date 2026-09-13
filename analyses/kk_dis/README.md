# kk_dis 工作目录操作规范（Agent 约束）

> 本文件是 **agent（含 LLMPWA 代码生成器与人工驱动的拟合 agent）如何操作 `analyses/kk_dis` 工作目录的约束**。
> 违反以下规范的操作都应避免；有疑问先读本文件与 `document/` 下的手册。
> （项目架构说明见 `document/README_architecture.md`，本文件专注操作约束。）

---

## 1. 目录结构总览

```text
analyses/kk_dis/
├── README.md                 ← 本规范（约束 agent 操作）
├── llm_config_fit.toml       ← 代码生成流水线配置（8 个 stage）
├── resonances_config.toml    ← 共振态/振幅物理配置（**单一来源，见 §4**）
├── node_config.toml          ← 多节点 Docker 节点表（保留，仅参考）
├── run_pipeline.sh           ← 生成器入口脚本（no_proxy 修正 + venv + agent.cli）
├── config/                   ← 运行配置（logconfig_fit.json）
├── data/                     ← 物理数据（real_data/mc_truth/draw/weight），**只读，勿改**
├── document/                 ← 参考文档与操作手册（见 §2）
├── tmp/                      ← 临时文件（同步分片、临时日志等；**可随时清理**）
├── task/                     ← 拟合任务记录（见 §3）
├── gen/                      ← 生成器中间产物（prompts/handlers/templates/fragments/llm_logs）
├── run/                      ← 生成/手写的可执行代码（见 §4、§5）
├── output/                   ← 拟合输出（fit/ 与 multistart/ 报告）
└── .dsh/  .venv/  logs/      ← 工具链与运行时（勿手改）
```

---

## 2. document/ —— 参考文档与操作手册

存放**长期有效**的文档，按主题分类，命名清晰：

| 文件 | 内容 |
|---|---|
| `document/OPERATIONS.md` | **操作手册**：端到端流程（环境→生成→同步→拟合→监控） |
| `document/ENV_NOTES.md` | 环境依赖清单与坑（本地 venv / 远程 kk_fit） |
| `document/combined_likelihood_math.md` | 似然函数数学说明（**生成似然的唯一公式来源**） |
| `document/random_initial_perturbation.md` | 随机初值扰动算法（多起点拟合用） |
| `document/README_architecture.md` | 项目架构说明（原 README 存档） |

**规则**：
- 新增/修改文档放这里，不散落根目录。
- 公式类（似然、扰动算法）改动会直接影响生成器的 `[ref]` 引用，改完要同步 `llm_config_fit.toml` 的 ref 路径。

---

## 3. task/ —— 拟合任务记录

每个拟合任务一个文件夹，命名 `<日期>_<工作内容>`（如 `20260912_run100随机初值拟合_bench双卡对比`）。
内部分 5 个子目录：

```text
task/<日期>_<工作内容>/
├── 1_运行代码/    ← 本次运行用的完整代码快照（fit_script.py / likelihood_function.py / 多起点 runner 等）
├── 2_工作记录/    ← status.md：做什么、结果摘要、遇到的问题与修复、关键命令、产物位置
├── 3_报告/        ← 完成后输出的报告 md（结果分析、结论、对比表）
├── 4_图片/        ← 相关图片（若有：plot、nvidia-smi 截图等）
└── 5_生成器日志/  ← 代码生成器的详细日志（llm_logs/）、提示词（prompts/）、配置（llm_config_fit.toml、resonances_config.toml）
```

**规则**：
- **远程是同构结构**：`~/LLMPWA/kk_dis/task/<日期>_<内容>/`，拟合输出（summary/work 产物）**留在远程**，
  需要展示/归档的按需 `remote_pull` 到本地。
- 每次新拟合前：先查 `task/` 看是否已有类似任务（复用代码/避免重跑）。
- 每次任务结束：`2_工作记录/status.md` 必须更新（做了什么、结果、坑）。
- 大产物（work 目录、summary）不复制，记录路径即可。
- `tmp/` 是唯一可随便清理的地方；task/ 与 document/ 要长期保留。

---

## 4. 代码生成流程约束（关键）

### 4.1 能改 vs 不能改

**可以修改**（运行时适配，直接改代码即可，不用重跑生成器）：
- `run/fit_script.py` 的 **main 函数**：分布式初始化、Mesh、数据加载/归一化、shard、
  JIT nll/grad/hvp、SciPy minimize、Hessian/误差、结果保存——为适配拟合流程可自由改
  （如改优化器、加约束、改收敛判据、改输出格式）。
- `run` 下的工具/runner：如 `multi_start_fit.py`（多起点调度）、`bench_gpu_modes.py`（A/B）。

**不可直接修改**（由共振态配置/似然公式决定，必须改提示词→重跑生成器）：
- **共振态分组**：`resonances_config.toml` 里的共振列表、传播子类型（BW/Flatté）、
  耦合结构——改它 = 改物理模型 → 要重跑 `config_strip` → `classification` → **重新生成 likelihood**。
- **似然函数**：`run/likelihood_function.py` 的提取/组合/约束逻辑——它由
  `gen/prompts/generate_likelihood_*` 提示词 + `document/combined_likelihood_math.md` 公式生成。
  **若手改，下次重跑生成器会被覆盖丢改动**。

**规则一句话**：改共振态分组或似然公式 → **必须改提示词，重跑代码生成器**；
只适配拟合流程（主函数）→ 直接改 `run/fit_script.py`。

### 4.2 修改提示词的流程

1. 改公式：更新 `document/combined_likelihood_math.md`（或提示词里的公式描述）。
2. 改共振态：更新 `resonances_config.toml`（共振/传播子/耦合）。
3. 改提示词：编辑 `gen/prompts/generate_likelihood_{system,task}.txt`、
   `generate_fit_script_*` 等。
4. 重跑生成器：
   ```bash
   ./run_pipeline.sh                        # 全流程
   ./run_pipeline.sh --stage generate_likelihood   # 只重跑似然
   ```
5. 生成后检查 `run/likelihood_function.py`、`run/likelihood_test_result.txt`（冒烟 PASS）。
6. 同步变更文件到远程 `~/LLMPWA/kk_dis/`。

### 4.3 共振态配置只读语义

`resonances_config.toml` 是物理模型的**单一来源**。生成器 stage `config_strip` 会把它
拆成 `run/free_params.toml`（自由参数+range）与 `gen/fragments/*`。**不要绕过生成器手改
`run/free_params.toml` 的物理内容**（拟合初值例外——`multi_start_fit.py` 用临时副本做扰动）。

---

## 5. 运行代码的边界

- `run/base_functions.py`：模板拼接产物（工具/物理/数据/初值/存盘），由 `assemble_help_functions`
  stage 生成。少手改，除非是通用 bug 修复（`setup_logging` mkdir 这类防御）。
- `run/likelihood_function.py`：见 §4.1，**生成器产物，勿手改物理逻辑**。
- `run/fit_script.py`：主函数可改（§4.1）。若改动会被后续重跑生成器覆盖（`generate_fit_script`
  stage 会重写它），把适配性的改动挪到**独立模块**（如 `run/fit_adapt.py`）或记入
  `task/<日期>/2_工作记录/status.md`，重跑后重新应用。

---

## 6. 数据与产物约束

- `data/` **只读**：真实数据/MC/权重，任何拟合不得改写。
- `output/` 可写：`output/fit/`（单次结果）、`output/multistart/`（多起点 summary+报告）。
- 报告（md）写完后：副本归档到 `task/<日期>_<内容>/3_报告/`。
- 远程产物留在远程，本地只保留报告/小文件。

---

## 7. 环境

- 本地生成器：`analyses/kk_dis/.venv`（openai==2.54.0 / jax cpu）。跑生成器用 `run_pipeline.sh`。
- 远程拟合：`~/miniconda3/envs/kk_fit`（jax 0.10.2 cuda12）。跑拟合用
  `~/miniconda3/envs/kk_fit/bin/python run/fit_script.py`。
- 详细信息：`document/ENV_NOTES.md`、`document/OPERATIONS.md`。

---

## 8. Agent 操作速查（TL;DR）

1. **先查 document/ 手册**，再动手。
2. **查 task/ 历史**，避免重复劳动。
3. 拟合跑批 → 用 `run/multi_start_fit.py`（`--resume` 断点续跑）。
4. 改拟合主函数 → 直接改 `run/fit_script.py`。
5. 改共振态分组/似然 → **改提示词 + 重跑生成器**，不手改生成产物。
6. 新任务 → 建 `task/<日期>_<内容>/`，写 `2_工作记录/status.md`。
7. 结果 → 报告归档 `3_报告/`；远程产物留远程。
8. 临时文件 → `tmp/`（可清理）。
