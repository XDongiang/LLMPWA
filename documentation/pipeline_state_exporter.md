# LLMPWA 流水线状态导出器（pipeline_state.py）

> 路线 B（DSH 原生 client-plugin 可视化面板）的**公共数据层**。
> LLMPWA 侧把流水线的"静态结构 + 动态状态"抽成一份 JSON 快照，
> DeepSeek Harness 的 host 插件直接消费它，**避免在 TypeScript 里重写 LLMPWA 的 TOML 解析逻辑**。

## 为什么需要它

DSH client-plugin 面板要展示 stage DAG、每 stage 状态、产物代码、共振态配置。
这些信息的"真相源"是几个 TOML / 文件：

| 信息 | 来源文件 |
|---|---|
| stage 依赖图 | `llm_config_fit.toml` 的 `[stages.*]`（含 `<<...>>` 引用、`foreach`、`input/output`） |
| 每 stage 运行状态 | `gen/manifest{suffix}.toml`（prompt_hash、是否缓存命中、输出路径） |
| 产物代码 | `gen/fragments/*.py`、`gen/templates/*.py`、`gen/llm_logs/*.md`、`run/*.py` |
| 参数 / 共振态配置 | `resonances_config.toml`（或 `[ref].resonances_config*` 指向的文件） |

为了避免在 DSH 侧（TS）重复实现 TOML 引用解析与参数提取，本模块把这些逻辑
统一放在 LLMPWA（Python）里，输出一份**可被 JSON 直接消费**的快照。

## 用法

```bash
# 生成到 gen/pipeline_state.json
python agent/pipeline_state.py -w analyses/kk_new -c llm_config_fit.toml -o gen/pipeline_state.json

# 打印到 stdout（host 侧可以直接管道/调用）
python agent/pipeline_state.py -w analyses/kk_new -c llm_config_fit.toml
```

作为模块（供 DSH host 插件调用）：

```python
import sys; sys.path.insert(0, "<LLMPWA>/agent")
from pipeline_state import build_pipeline_state
snapshot = build_pipeline_state(workdir="analyses/kk_new", config_name="llm_config_fit.toml")
```

> **只读依赖标准库 `tomllib`（Python ≥ 3.11）**，无需第三方 `toml` 包。
> 每次调用按当前磁盘状态重算 —— host 侧可轮询文件 mtime 或 watch 目录实现"实时/增量刷新"。

## 输出 JSON 结构

```jsonc
{
  "schema_version": 1,
  "config_file": "llm_config_fit.toml",
  "config_path": "llm_config_fit.toml",
  "generated_at": "2026-...",
  "workdir": "/abs/path/to/analyses/kk_new",
  "stage_order": ["config_strip", "classification", ...],  // 定义顺序

  // ① stage DAG 依赖图：每 stage 的依赖、foreach、声明
  "stages": [
    {
      "name": "classification",
      "kind": "llm",                 // python | llm | agent
      "order": 3,
      "refs": ["stages.config_strip.stripped_config", ...], // 引用的原始表达式
      "dependencies": ["config_strip"],                    // 命中上游 stage 名
      "deps_all": false,            // 是否出现 stages[*]（依赖所有）
      "foreach": {"type": "ref", "source": "...", "values": null} | null,
      "handlers": ["gen/handlers/xxx.py"],
      "prompt": {"type": "file", "path": "...", "chars": null},
      "output_type": "json",
      "manifest": { ... }            // 该 stage 的 manifest 状态（见下）
    }
  ],

  // ② 每 stage 运行状态
  "manifest": {
    "path": "gen/manifest_fit.toml",
    "present": true,
    "stages": {
      "config_strip": {
        "present": true,
        "status": "cached",          // cached | ran | missing
        "prompt_hash": "...",
        "outputs": { "free_params": {"type":"file","path":"...","exists":true} }
      }
    }
  },

  // ③ 产物代码文件清单（相对路径，按需再读内容）
  "artifacts": {
    "fragments":  ["gen/fragments/xxx.py", ...],
    "templates":  ["gen/templates/fit.py", ...],
    "prompts":    ["gen/prompts/xxx.txt", ...],
    "llm_logs":   ["gen/llm_logs/xxx.md", ...],
    "run_scripts":["run/fit_script.py", ...]
  },

  // ④ 参数 / 共振态配置
  "ref": { "code_template": [...], "resonances_config": {...} },
  "resonances": { ... },             // 内联或 ref 指向的 resonances_config 内容
  "params": [                         // 扁平化的参数字典列表
    { "path": "phif0_980.propagators.A_propagator.mass",
      "kind": "resonance:phif0_980",
      "value": 1.02, "fixed": true, "range": null, "error": null }
  ],
  "draw": {...},
  "fit":  {...}
}
```

## 覆盖的引用语法（DAG 边提取）

从 `prompt` / `human_prompt` / `system_prompt` / `foreach` / `input` / `output`
中提取 `<<...>>` 引用与裸 dotpath，识别指向的 stage：

| 语法 | 依赖结果 |
|---|---|
| `<<stages.config_strip.stripped_config>>` | `dependencies += [config_strip]` |
| `<<stages.resonance_calculation[*].all>>` | `dependencies += [resonance_calculation]` |
| `<<stages[*].all>>` | `deps_all = true` |
| `<<ref.code_template.xxx>>` / `<<key>>` | 跳过（非 stage） |
| `input.list = ["stages[*].all", ...]` | 同上（裸 dotpath 也扫描） |

## 验证情况

- `analyses/kk_new`（12 stage，内联 resonances）：DAG、参数（81）、产物扫描均正确。
- `analyses/kk_dis`（8 stage，含 `agent` kind）：DAG 正确。
- `analyses/kk_pipi`（20 stage，多份 `resonances_config_*`）：合并进 `resonances._sources`，参数（234）。
- 有 `manifest` 时的动态状态（cached/ran/missing）已验证。

> 注：`manifest` / `gen/fragments` / `gen/llm_logs` 是本地产物（gitignore），
> 流水线运行后才会出现；此时 `status` 会从 `missing` 变为 `ran`/`cached`，
> 从而驱动面板的"实时/增量刷新"。

## 可视化报告（路线 A）

`agent/pipeline_viz.py` 直接消费本模块的 `build_pipeline_state()` 输出，渲染
**Markdown + Mermaid** 或**自包含 HTML** 报告，可在 DSH 对话里作为 artifact 展示：

```bash
# Markdown + Mermaid（默认，适合在渲染器里直接看 DAG）
python agent/pipeline_viz.py -w analyses/kk_dis -c llm_config_fit.toml -o gen/pipeline_report.md

# 自包含 HTML（内嵌 CSS，Mermaid CDN + 纯文本 DAG 兜底 + 可折叠详情）
python agent/pipeline_viz.py -w analyses/kk_dis -c llm_config_fit.toml -f html -o gen/pipeline_report.html
```

> `pipeline_report.*` 与 `pipeline_state.json` 均为可再生成产物，已加入 `.gitignore`。

## 测试

`agent/test_pipeline_state.py` 覆盖引用解析、依赖提取、参数扁平化、多源
resonance 合并，以及三个真实 analyses 的构建；`agent/test_pipeline_viz.py`
覆盖 DAG 边、Markdown/HTML 渲染与真实 analyses 报告生成。
