# LLMPWA 重构计划

## 背景

当前 `agent/llm_code_generator.py` 是 ~700 行单体文件，集中承担了：配置解析、提示词构建、
LLM 调用+缓存、代码组装、以及 fit/draw/plot 三种 mode 的完整流程。

存在的主要问题：
- 缓存 key = prompt 的 SHA256，配置里改个参数初始值就 cache miss（即使生成代码结构不变）
- 没有用户侧干预入口，无法对单个 stage 追加定制需求
- 出错只能看 print，没有自动测试/修复/报警能力
- 三种 mode 逻辑全挤在一个文件，靠方法名前缀区分

---

## 目标 1：参数与结构分离（缓存友好）

**优先级：最高** — 这是后续所有改动的基础。

**现状**：`resonances_config.toml` 中的 `value` 直接进入 prompt，改初始值就 cache miss。

**目标**：LLM 只接收变量名和结构信息（哪些参数 fixed、类型、顺序），不接收具体数值。
数值在代码生成后，由运行时填充到 `args_list`。

**具体做法**：
- 新增 `agent/config_parser.py`，纯 Python 解析器，职责：
  - 从 TOML 提取所有 `fixed=false` 的参数，输出有序参数清单（parameter_manifest）
  - 清单分两部分：schema（名字+类型+range，给 LLM 看）和 values（具体数值，给运行时用）
- LLM prompt 中只传参数名列表，如 `["mass_f0_980", "g_kk_f0_980", ...]`
- 生成的代码用 `args[i]` 或命名解包，数值由 manifest 在运行时注入
- **效果**：改 `value = 0.98` → `0.99` 不触发 cache miss；改 `fixed = true → false` 才会

**验收标准**：
- [ ] 修改 TOML 中某参数的 value，重新运行 generator，命中缓存
- [ ] 修改 TOML 中某参数的 fixed 状态，重新运行 generator，正确 cache miss 并重新生成

---

## 目标 2：Mode 拆分到不同文件

**优先级：高** — 纯重构不改行为，先降低后续改动的心智负担。

**现状**：fit/draw/plot 三种 mode 逻辑全在 `llm_code_generator.py`，靠 `_prompt_*` /
`generate_*` / `assemble_*` 方法名前缀区分。

**目标**：按职责拆文件。

**建议结构**：
```
agent/
├── generator_base.py      # 基类：config 解析、cache、LLM 调用、工具集
├── generator_fit.py       # fit mode pipeline
├── generator_draw.py      # draw mode pipeline
├── generator_plot.py      # plot mode pipeline
├── config_parser.py       # 非 LLM 配置解析器（目标 1）
├── stage_agent.py         # StageAgent 基类（目标 4）
├── llm_code_generator.py  # 入口：argparse → 分发到对应 generator
└── ...（easytrans_client / common_template / code_compressor 保持不变）
```

**具体做法**：
- 把通用能力（`_load_config` / `_load_cache` / `_save_cache` / `_generate` / `_annotate`）
  提到 `generator_base.py` 的基类
- fit/draw/plot 各自的 pipeline 方法移到对应文件，继承基类
- `llm_code_generator.py` 退化为薄入口，按 `--mode` 实例化对应类

**验收标准**：
- [ ] 三种 mode 生成结果与重构前逐字节一致（用现有缓存对比）

---

## 目标 3：中间过程缓存改用 TOML

**优先级：中**

**现状**：Stage 1 分析结果是 LLM 直接输出的 JSON，后续 stage 依赖它；缓存均为 JSON。

**目标**：用 TOML 作为人类可读可编辑的中间缓存格式。

**缓存内容**：
- Stage 1 的分类结果（propagator_classification、amplitude_classification）
- 各 stage 生成的函数签名（函数名、输入参数、输出）

**具体做法**：
- Stage 1 输出改为 "LLM 输出 → 解析校验 → 写入 TOML"
- 后续 stage 直接读 TOML 字段，prompt 中写明需读取的字段名，不再动态拼接"上一步结果"
- 该 TOML 允许用户手动修正（如 LLM 分类错误时人工纠正）

**验收标准**：
- [ ] 中间缓存为 TOML，字段清晰、可手动编辑
- [ ] 手动修改分类 TOML 后，后续 stage 按修改后的内容生成

---

## 目标 4：Human prompt 注入（Stage 可定制化）

**优先级：中**

**现状**：每个 stage 只有系统提示词，无用户干预入口。

**目标**：保持分 stage 的线性流程不变，每个 stage 在系统提示词外，可加入 human 部分
（来自 `llm_config_fit.toml`），对特定流程做定制。

**配置示例**：
```toml
[stages.likelihood_function]
human_prompt = "在似然函数中对 mass 参数加上 boundary penalty，约束 mass ∈ [0.9, 1.1]"
```

**具体做法**：
- 每个 stage 的 prompt 组装为 `system + human`，human 部分按 stage 名从配置读取
- human prompt 参与 cache key 计算（改定制需求 → 重新生成）
- 未配置 human prompt 时，行为与现在完全一致

**验收标准**：
- [ ] 不配置 human prompt 时命中旧缓存（行为不变）
- [ ] 为某 stage 配置 human prompt 后，该 stage 重新生成且约束生效

---

## 目标 5：Agent 化每个 Stage

**优先级：最后** — 最大改动，依赖前面基础设施就绪。

**现状**：线性调用，出错只能看 print，无自动测试/修复能力。

**目标**：每个 stage 成为一个 agent，拥有工具集与自动重试能力。

**工具集**：
| Tool | 用途 |
|------|------|
| `code_write` | 写入生成的代码片段 |
| `shell` | 运行写入的代码做语法/逻辑测试 |
| `read` | 读取 config、reference、template 参考 |
| `alarm` | 按情况报警，要求人工接管或直接中断 |

**具体做法**：
- 定义 `StageAgent` 基类，封装 tool 调用协议
- 每个 stage 继承/实例化，配置自己的 system prompt + tools
- agent 自行重试（如生成代码语法检查不过则自动修正）
- `alarm` 触发条件可配置（如重试 N 次仍失败）
- 引入 `logging` 记录：API 请求（prompt hash、token 用量、耗时）、工具调用（tool 名、输入输出）

**验收标准**：
- [ ] 故意制造生成错误，agent 能通过 shell 测试发现并自动修正
- [ ] 超过重试上限时触发 alarm，暂停等待人工
- [ ] 日志完整记录 API 请求与工具调用

---

## 实施顺序

```
目标 1（参数/结构分离）  → 缓存基础，命中率大幅提升
  ↓
目标 2（Mode 拆文件）    → 纯重构，降低后续心智负担
  ↓
目标 3（中间缓存 TOML）  → Stage 1 输出标准化
  ↓
目标 4（Human prompt）   → 在清晰结构上加注入点
  ↓
目标 5（Agent 化）       → 最大改动放最后
```

## 已确认决策

- **配置分离**：`llm_config_fit.toml`（定制提示词）与 `resonances_config.toml`（物理配置）分开管理。
  标准分析流程不需要额外提示词，`llm_config_fit.toml` 仅在有定制需求时才创建。
- **Agent shell 测试**：需要实际运行到 JAX jit 编译。语法层面基本不出错，
  主要错误点是爱因斯坦求和的指标（einsum indices），只有 jit 编译时才暴露。
  → shell 工具需要能跑一个轻量 jit 编译测试（小 batch mock 数据），而非仅 `python -c import`。
- **中间缓存结构：方案 B**。manifest.toml + 代码片段独立 .py 文件。
  代码片段不允许人类编辑（永远由 LLM 重新生成），人类只编辑 manifest 中的结构信息。
  Stage 1 分类一变，下游全部重新生成（正确性优先，不做精确依赖追踪）。

## 缓存结构设计

```
cache/
├── manifest.toml          # 结构信息 + 每个 stage 的元数据，人类可读可编辑（仅结构部分）
├── fragments/
│   ├── data_load.py       # 生成的代码片段，独立 .py 文件（机器生成，不手改）
│   ├── resonance_BW_BW.py
│   ├── resonance_BW_flatte980.py
│   └── ...
└── prompt_log/            # 可选：保留原始 prompt 用于调试
    └── *.txt
```

**manifest.toml 示例**：
```toml
# Stage 1 输出：人类可修正
[classification]
propagator_groups = ["BW_BW", "BW_flatte980", "BW_flatte1270"]

[classification.BW_BW]
resonances = ["phif2_1525", "phif2_2150", "phif2_2340"]
[classification.BW_flatte980]
resonances = ["phif0_980"]

# 各 stage 元数据
[stages.data_load]
input_schema_hash = "b7e1..."
fragment = "fragments/data_load.py"
functions = ["load_real_data", "load_mc_data"]

[stages.resonance_calculation.BW_flatte980]
input_schema_hash = "a3f2..."
fragment = "fragments/resonance_BW_flatte980.py"
functions = ["calculate_BW_flatte980", "component_BW_flatte980"]
```

**缓存命中机制**：
- 每个 stage 的 `input_schema_hash` = hash(上游 manifest 内容 + 本 stage 的 prompt 模板 + human prompt)
- 运行时对比当前计算出的 anifest 中记录的 hash，一致则直接读 fragment 文件
- 不一致则调用 LLM 重新生成，覆盖 fragment 文件并更新 manifest
