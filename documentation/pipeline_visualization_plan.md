# LLMPWA 流程可视化 — 方案对比（设计稿）

> 目标：把 LLMPWA 的多阶段代码生成流水线在 DeepSeek Harness Web GUI 上可视化。
> 本文只做方案对比与设计，**不含实现代码**。
>
> **进度**：路线 A 的公共数据层（`agent/pipeline_state.py`）与报告生成器
> （`agent/pipeline_viz.py`）已落地，见 [pipeline_state_exporter.md](pipeline_state_exporter.md)。
> 路线 B 仍受「本机无 DSH 源码」制约，尚未开始。

---

## 0. 可视化对象与数据来源

流水线的"可可视化信息"有四类，**全部来自现有文件，无需改动运行逻辑即可拿到**：

| 信息 | 来源 |
|---|---|
| **Stage 依赖图** | `llm_config_fit.toml` 的 `[stages.*]` 段（静态定义每个 stage、其 `kind`、`foreach`、`<<...>>` 引用关系） |
| **每 stage 运行状态** | `manifest.toml`（每个 stage 的 `prompt_hash`、是否缓存命中、输出字段路径） |
| **产物代码** | `gen/fragments/*.py`、`run/fit_script.py`、`gen/templates/*.py` |
| **参数/共振态配置** | `resonances_config.toml`（共振态、传播子类型、质量/宽度/const/theta 及 fixed/range/error） |

> 关键点：**依赖图是静态的**（配置里写死），**运行状态是动态的**（每次 run 后 manifest 更新）。
> 因此"实时"只影响运行状态与产物这两类数据的刷新方式。

---

## 1. 三条路线的核心差异

### 路线 A — 流水线自产报告，在 GUI 里看（不动 DSH）

**思路**：给 LLMPWA 加一个 `visualize` / `report` 入口（Python 脚本），解析上面的四类文件，生成一份自包含的可视化报告，作为 agent 产出物在 DSH 对话/工作区里展示。

- 生成内容：stage DAG 结构 + 每个 stage 的状态、输入/输出、代码、LLM 日志、共振态配置。
- 渲染载体（三选一，均已被 GUI 消息渲染器兜底支持）：
  1. **HTML 报告**（内嵌 CSS/JS，用 Mermaid 或自绘 DAG）
  2. **Markdown + Mermaid 图**（`graph TD` 之类）
  3. **结构化 JSON + 一个通用渲染器**

**实时性**：流水线运行期间，agent 在每完成一个 stage 后重新生成/追加报告即可。可以是"每个 stage 输出一份快照"，或由一个后端长任务持续更新同一份 JSON、前端轮询/刷新。

**代价**：只改 LLMPWA 的 Python；DSH 完全不动。**低**。

**局限**：图是"报告式"的（静态图 + 可点开），不是实时内嵌面板；交互性取决于渲染载体。

---

### 路线 B — 原生 DSH client plugin 面板（完全内嵌、实时）

**思路**：写一个 `@deepseek-ai/dsh-client-ui-*` 插件，通过 DSH 的 **slot 机制**（如 `sidebar.settings`、`sidebar.footer.action`、workspace 相关 slot，或新增自定义 slot）把面板注入 GUI。

- 前端 React 组件渲染 DAG 图（React Flow / D3）+ 状态 + 代码查看器。
- 后端需要一个 DSH cordis 插件提供数据：读 LLMPWA 的 `manifest.toml` / `llm_config.toml` / `gen/`，并通过 **SSE（`text/event-stream` 路由）或 WebSocket**（`webserver.register` / `registerUpgrade`，DSH 均支持）把增量状态推给前端。

**代价**：**高**。需要：
1. clone `deepseek-ai/deepseek-harness` 源码仓库；
2. `pnpm install` + `pnpm run build` / `tsdown` 编译出 `lib/client.js`；
3. 把插件注册进 profile 的 `cordis.patch.yml`；
4. 开发期用 `pnpm run dev:web` + HMR。
   > **当前机器限制**：本机只有 DSH 的已发布 npm 包（`/home/iso/.npm/_npx/...`），**没有源码仓库**（全盘搜索无 `apps/web` / `packages`）。所以没有干净的地方 build 客户插件 —— 这是路线 B 最大的现实阻碍。

**代价**：**高**，且受源码缺失制约。

---

### 路线 C — 独立 Web Viewer（单独进程/端口）

**思路**：单独起一个可视化服务（React/Vite 或 D3 静态页），专用端口，渲染 stage DAG，长轮询/SSE 从 LLMPWA 拉状态。

**代价**：**中**。

**局限**：会另起一个 HTTP 服务；而当前 DSH Web GUI 用的是固定 URL（`http://127.0.0.1:3080`，注入 `window.__DSH_BOOT__`，且 Vite entry 不是独立应用）。**DSH 明确不建议另起替换服务器**，且外部端口需要额外信任配置（`--trusted-host`）。这也意味着 C 的"界面"与 DSH 主界面是割裂的跳转关系，不是内嵌。

---

## 2. 功能覆盖矩阵

| 需求 | A（报告） | B（原生插件） | C（独立Viewer） |
|---|---|---|---|
| Stage DAG 依赖图 | ✅（Mermaid/HTML） | ✅（React Flow，原生交互） | ✅（React Flow） |
| 每 stage 运行状态 | ✅（快照/增量） | ✅（实时 SSE/WS 推送） | ✅（轮询/SSE） |
| 产物代码浏览 | ✅（点开查看片段） | ✅（内嵌查看器） | ✅（内嵌查看器） |
| 参数/共振态配置 | ✅（表格/树） | ✅（树/表格） | ✅（树/表格） |
| 内嵌进 DSH 主界面 | ⚠️ 作为对话 artifact | ✅ 完全内嵌面板 | ❌ 独立页面/跳转 |
| 实时增量刷新 | ⚠️ 后端逐阶段更新+刷新 | ✅ SSE/WS 实时 | ✅ 轮询/SSE |
| 无需改 DSH | ✅ | ❌ 需源码+build | ⚠️ 需额外端口 |

---

## 3. 实时性如何落地（三条路线统一的技术点）

LLMPWA 本身是**同步的 Python 进程**（`engine.run()` 里逐 stage 执行）。要实现"实时/增量"，需要把执行过程暴露为可观测事件。两种做法：

1. **事件钩子（改 LLMPWA）**：在 `StageRunner.execute()` 处插入回调，每完成一个 stage 就 (a) 更新一份运行状态 JSON，或 (b) 通过 SSE/WebSocket 推一条 stage 完成事件。
2. **文件轮询（不改 LLMPWA）**：监控 `manifest.toml` / `gen/` 的修改时间与内容哈希变化，前端轮询或后端 watch。

> 事件钩子最干净，且这份"运行状态 JSON"正是路线 B 前端要消费的同一份数据。**建议所有路线都先做事件钩子 + 一份标准化的状态 JSON 数据模型**，它是一鱼多吃的关键资产。

---

## 4. 代价 / 风险 / 里程碑

### 路线 A
- **里程碑**：① 写 `visualize` 入口解析四类文件 → ② 定义运行状态 JSON schema → ③ 生成 HTML/Mermaid 报告 → ④ 每阶段更新报告 / 增量刷新.
- **风险**：GUI 渲染器对复杂 HTML/交互的支持有限；"实时"依赖后端每阶段更新 + 前端手动/自动刷新。
- **工期**：≈ 数天（视交互深度）。**不碰 DSH**。

### 路线 B
- **里程碑**：① clone 源码 + 搭 pnpm workspace → ② 写后端 cordis 插件暴露 `/api/pipeline` + SSE → ③ 写 client plugin（React + React Flow）注入 slot → ④ `pnpm dev:web` HMR 迭代 → ⑤ 打包验证。
- **风险**：**本机缺源码仓库**；需要熟悉 DSH 的 cordis / slot / client-modules 整套机制；包版本与 DSH 版本必须对齐（`0.1.2-rc.1`）。
- **工期**：≈ 数周。改动面最大。

### 路线 C
- **里程碑**：① 建独立前端工程（React Flow）→ ② 后端读 LLMPWA 状态并推 SSE → ③ 配置信任端口 → ④ 跳转集成。
- **风险**：另起服务 + 端口信任；界面与 DSH 割裂；"内嵌"打折扣。
- **工期**：≈ 1~2 周。

---

## 5. 推荐结论

**分两步走，先 A 后 B：**

1. **当前阶段（最短路径、零 DSH 改动）走 A**：把"依赖图 + 状态 + 代码 + 配置"抽成**标准化的运行状态 JSON 数据模型**（`agent/pipeline_state.py`，这是所有路线的公共底座），并生成一份可点开的 HTML/Mermaid 报告（`agent/pipeline_viz.py`），在 DSH 对话里作为 artifact 展示。

   > **已完成**：`pipeline_state.py` + `pipeline_viz.py` + 两份文档 + smoke 测试均已落地。

2. **数据模型稳定后，视需要演进到 B**：把同一份 JSON 数据 + 一套前端图组件，挪进一个原生 DSH client plugin，注入 slot + SSE/WS 实时推送。此时 A 阶段积累的解析逻辑与 schema 可直接复用，B 只补"UI 面板 + 实时通道 + 打包"。

> 不建议先跳 B：本机没有 DSH 源码仓库，build 客户插件无从谈起，投入产出比最差。
> 也不建议单独优先 C：多一个端口 + 界面割裂，且与"在 DSH 界面上"的目标有偏差。

---

## 6. 待确认问题

1. LLMPWA 这段流水线 **之后是否要长期对外用、多人用**？若是，B（原生面板）的长期价值最大，A 只是过渡。若只是个人/内部调试，A 足够。
2. 运行 LLMPWA 的主机与 DSH GUI 是否同一台？（同一台则 A 的"读文件"最直接；跨机则需要经过 DSH agent 或网络。）
3. 是否接受为 B **clone 源码 + 搭 pnpm 环境**？这是 B 的唯一现实前置。
