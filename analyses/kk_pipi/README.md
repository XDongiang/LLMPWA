# kk_pipi 工作目录操作规范（Agent 约束）

> 本文件是 **agent（含 LLMPWA 代码生成器与人工驱动的拟合 agent）如何操作 `analyses/kk_pipi` 工作目录的约束**。
> 违反以下规范的操作都应避免；有疑问先读本文件与 `document/` 下的手册。
> （项目架构说明见 `document/README_architecture.md`，本文件专注操作约束。）

---

## 1. 目录结构总览

```text
analyses/kk_pipi/
├── README.md                 ← 本规范（约束 agent 操作）
├── llm_config_combine.toml   ← 代码生成流水线配置（kk+ππ 联合拟合，**规范入口**）
├── resonances_config_kk.toml ← KK̄ 道共振态/振幅物理配置
├── resonances_config_pipi.toml ← ππ 道共振态/振幅物理配置
├── resonances_config_ctrl.toml ← 跨通道共享参数绑定（**联合拟合核心**）
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

> **与 kk_dis 的差别**：kk_dis 是**单道**分析，用 `llm_config_fit.toml` + 单个
> `resonances_config.toml`；kk_pipi 是 **KK̄ + ππ 双道联合拟合**，用
> `llm_config_combine.toml` + 三个 `resonances_config_*.toml`。

---

## 2. document/ —— 参考文档与操作手册

存放**长期有效**的文档，按主题分类，命名清晰：

| 文件 | 内容 |
|---|---|
| `document/OPERATIONS.md` | **操作手册**：端到端流程（环境→生成→同步→拟合→监控） |
| `document/ENV_NOTES.md` | 环境依赖清单与坑 |
| `document/combined_likelihood_math.md` | 联合似然函数数学说明（**生成似然的唯一公式来源**） |
| `document/random_initial_perturbation.md` | 随机初值扰动算法（多起点拟合用） |
| `document/README_architecture.md` | 项目架构说明 |

**规则**：
- 新增/修改文档放这里，不散落根目录。
- 公式类（似然、扰动算法）改动会直接影响生成器的 `[ref]` 引用，改完要同步 `llm_config_combine.toml` 的 ref 路径。

---

## 3. task/ —— 拟合任务记录

每个拟合任务一个文件夹，命名 `<日期>_<工作内容>`。
内部分 5 个子目录：

```text
task/<日期>_<工作内容>/
├── status.json        ← 任务状态机（机器可读，前端快速解析，见下）
├── 1_运行代码/    ← 本次运行用的完整代码快照（fit_script.py / likelihood 等）
├── 2_工作记录/    ← status.md：做什么、结果摘要、遇到的问题与修复、关键命令、产物位置
├── 3_报告/        ← 完成后输出的报告 md（结果分析、结论、对比表）
├── 4_图片/        ← 相关图片（若有：plot、nvidia-smi 截图等）
└── 5_生成器日志/  ← 代码生成器的详细日志（llm_logs/）、提示词（prompts/）、配置
```

**规则**：
- **远程是同构结构**：`~/LLMPWA/kk_pipi/task/<日期>_<内容>/`，拟合输出（summary/work 产物）**留在远程**，
  需要展示/归档的按需 `remote_pull` 到本地。
- 每次新拟合前：先查 `task/` 看是否已有类似任务（复用代码/避免重跑）。
- 每次任务**创建时**：写 `status.json`（`status = "running"`）；**结束时**更新为 `"completed"`（或 `"failed"`），
  填入结果指标与产物路径。`2_工作记录/status.md` 是人类可读的详细版，两者**同步维护**。
- 大产物（work 目录、summary）不复制，记录路径即可。
- `tmp/` 是唯一可随便清理的地方；task/ 与 document/ 要长期保留。

### 3.1 status.json 字段约定（供前端解析）

```json
{
  "task_id": "<日期>_<工作内容>",       // 必填，唯一标识
  "title": "简短标题",
  "status": "running | completed | failed | paused",  // 必填，任务状态机
  "date_start": "ISO8601",              // 开始时间（可空）
  "date_end": "ISO8601",                // 结束时间（未完成时为 null）
  "environment": { "host": "", "gpu": "", "python_env": "" },
  "task_type": "fit_combine | fit_multistart | fit_single | bench | ...",
  "summary": {                          // 结果指标（completed 后填写）
    "n_starts": 0, "n_success": 0, "success_rate": 0.0,
    "best_nll": 0.0, "best_start": 0, "near_global_frac": 0.0,
    "wall_clock_h": 0.0
  },
  "artifacts": {
    "code_dir": "1_运行代码/",
    "records_dir": "2_工作记录/",
    "reports_dir": "3_报告/",
    "images_dir": "4_图片/",
    "generator_logs_dir": "5_生成器日志/",
    "remote_work_dir": "~/LLMPWA/kk_pipi/task/<id>/...",
    "summary_files": ["...", "..."]
  },
  "reports": ["3_报告/xxx.md"],
  "issues_fixed": ["...", "..."],
  "notes": "可选，引用相关文档"
}
```

- `status` 取值约束：`running` / `completed` / `failed` / `paused`（前端按此渲染颜色/进度）。
- `summary` 内的字段键名固定，前端可按 `best_nll` / `success_rate` 等直接取数。
- 未知/附加字段允许扩展，但**不要改变已有键的语义**（前端依赖）。
- 每次更新 status.json 时，HTML 特殊字符（`<>&"`）需保持 JSON 转义合法（UTF-8）。

---

## 4. 代码生成流程约束（关键）

### 4.1 能改 vs 不能改

**可以修改**（运行时适配，直接改代码即可，不用重跑生成器）：
- `run/fit_script.py` 的 **main 函数**：分布式初始化、Mesh、数据加载/归一化、shard、
  JIT nll/grad/hvp、SciPy minimize、Hessian/误差、结果保存——为适配拟合流程可自由改。
- `run` 下的工具/runner。

**不可直接修改**（由共振态配置/似然公式决定，必须改提示词→重跑生成器）：
- **共振态分组**：`resonances_config_*.toml` 里的共振列表、传播子类型（BW/Flatté）、
  耦合结构——改它 = 改物理模型 → 要重跑对应 stage。
- **似然函数**：联合似然由 `gen/prompts/*` 提示词 + `document/combined_likelihood_math.md` 公式生成。
  **若手改，下次重跑生成器会被覆盖丢改动**。
- **跨通道绑定**：`resonances_config_ctrl.toml` 决定哪些共振质量/宽度在两个通道间共享。
  改它 = 改共享结构，必须重跑 `combine_check` 与 `generate_args`。

**规则一句话**：改共振态分组/似然公式/共享绑定 → **必须改提示词，重跑代码生成器**；
只适配拟合流程（主函数）→ 直接改 `run/fit_script.py`。

### 4.2 修改提示词的流程

1. 改公式：更新 `document/combined_likelihood_math.md`（或提示词里的公式描述）。
2. 改共振态：更新 `resonances_config_*.toml`（共振/传播子/耦合/绑定）。
3. 改提示词：编辑 `gen/prompts/*`。
4. 重跑生成器：
   ```bash
   ./run_pipeline.sh                        # 全流程
   ./run_pipeline.sh --stage generate_args  # 只重跑某一 stage
   ```
5. 生成后检查 `run/fit_script.py` 与冒烟输出。
6. 同步变更文件到远程 `~/LLMPWA/kk_pipi/`。

### 4.3 共振态配置只读语义

`resonances_config_*.toml` 是物理模型的**单一来源**。生成器 stage（`config_strip_kk` /
`config_strip_pipi`）会把它拆成 `gen/fragments/free_params_*.toml` 等。**不要绕过生成器手改
`run/free_params*.toml` 的物理内容**。

---

## 5. 运行代码的边界

- `run/fit_script.py`：主函数可改（§4.1）。若改动会被后续重跑生成器覆盖（`assemble_final_code`
  stage 会重写它），把适配性的改动挪到**独立模块**或记入 `task/<日期>/2_工作记录/status.md`，
  重跑后重新应用。
- **生成器产物勿手改物理逻辑**（似然、共振计算、参数抽取）。

---

## 6. 数据与产物约束

- `data/` **只读**：真实数据/MC/权重，任何拟合不得改写。
- `output/` 可写：`output/fit/`（单次结果）、`output/multistart/`（多起点 summary+报告）。
- 报告（md）写完后：副本归档到 `task/<日期>_<内容>/3_报告/`。
- 远程产物留在远程，本地只保留报告/小文件。

---

## 7. 环境

- 本地生成器：`analyses/kk_pipi/.venv`。跑生成器用 `run_pipeline.sh`。
- 远程拟合：`~/miniconda3/envs/kk_fit`。详细信息：`document/ENV_NOTES.md`、`document/OPERATIONS.md`。

---

## 8. Agent 操作速查（TL;DR）

1. **先查 document/ 手册**，再动手。
2. **查 task/ 历史**，避免重复劳动。
3. 拟合跑批 → 用多起点 runner（若已生成）。
4. 改拟合主函数 → 直接改 `run/fit_script.py`。
5. 改共振态分组/似然/绑定 → **改提示词 + 重跑生成器**，不手改生成产物。
6. 新任务 → 建 `task/<日期>_<内容>/`，写 `status.json`（status=running）+ `2_工作记录/status.md`。
7. 结果 → 报告归档 `3_报告/`；更新 `status.json`（status=completed + 结果指标）；远程产物留远程。
8. 临时文件 → `tmp/`（可清理）。
