# LLMPWA 流水线可视化 — 路线 B（原生 DSH client-plugin）设计稿（v2）

> 前置：本机已有 deepseek-harness 源码仓库（`~/openclaw/deepseek-harness`），
> 替代了旧稿 `pipeline_visualization_plan.md` 中「路线 B 受本机无源码制约」的结论。
> 运行端口约定：**`3081`**（覆盖默认 3080）。
> v2 变更：面板形态由「右侧 Sidebar tab」改为「**左侧 Sidebar 按钮 → 打开整页工作台**」。

> **实现状态：已完成（可运行），分析面板已按「顶部分页 + Agent 抽屉」改版。**
> `packages/client/ui-llmpwa-pipeline` 已建包，注册进 `cordis.patch.yml` /
> `tsconfig.client.json` / `web-app/package.json` 三处；`FooterButton`（`sidebar.footer.action`）
> 与 `Workbench`（`shell.overlay`）均已实现并通过 103 个单测（每文件 100% 覆盖，含分支）。
> 本机 `kk_dis` 已生成 `gen/pipeline_state.json`（42KB，8 stage），供实机预览。
> 面板上方是「DAG / 文档」两个分页 Tab；文档页列出该 analysis 的参考文件并预览文本。
> 蓝色「打开 Agent」按钮弹出右侧抽屉，在**该 analysis 的工作目录**（
> `<workspace>/LLMPWA/analyses/<analysis>`）启动一个 dsh agent 会话，就绪后跳回主对话区查看。
> 本稿保留为「设计 + 落地说明」；后续迭代再扩展参数/产物浏览。

---

## 0. 目标与范围

在 DeepSeek Harness Web GUI 里新增一个**原生 client-plugin**，呈现为：
- **左侧 Sidebar 脚部一个按钮**（`LLMPWA 工作台`）。
- 点击后，**右侧/主区域打开一整页工作台**，里面先做两件事：
  1. **选择 analysis**（下拉列出 `LLMPWA/analyses/` 下子目录）。
  2. 选中后**预览该 analysis 的流水线 DAG**（stage 依赖图 + 状态）。

本期范围**只做 DAG 预览**；参数/产物/详情浏览留到后续迭代。

**数据真相源不变**：`agent/pipeline_state.py` 产出的 `gen/pipeline_state.json`
（路线 A / B 公共底座，已实现并测过）。面板只读消费这份 JSON 快照并渲染。

---

## 1. 两个关键接入点（DSH slot 机制）

### 1.1 左侧按钮：`sidebar.footer.action`

`ui-layout` 在左侧 column 的 `sidebar` slot 里声明了子 slot `sidebar.footer.action`
（`kind: 'list'`），`ui-cordis` 已用它注册脚部按钮（`cordis-panel`），是现成模板。
注册一个 `id: 'llmpwa-panel'` 的条目即可。

### 1.2 整页工作台：`shell.overlay`

`ui-layout` 声明了 frame-wide 的 `shell.overlay`（`kind: 'list'`），AppFrame 在
`data-shell-overlay` 层渲染它。它是一个**覆盖全帧的浮层**，默认点击穿透，条目可自行
`position: fixed; inset: 0` 铺满并 opt-in 指针事件 —— 正好用来承载「整页工作台」。

> 选这两个 slot 而非 `conversation` / `sidebar`：后两者是 `single` 且已被 occupant
> 占据，注册会整体替换其内容并销毁其声明的子 slot，绝不可用于「加一个入口」。

---

## 2. 总体架构（数据流）

```text
LLMPWA/agent/pipeline_state.py  (Python, 已实现)
        │  生成
        ▼
LLMPWA/analyses/<workdir>/gen/pipeline_state.json   ← 真相源（工作区文件）
        │  ① remote.workspaceFiles.read / list
        ▼
DSH 工作台 client-plugin（本设计新增）
        │  ② 解析 + 求值（纯函数）
        ▼
纯 UI 呈现组件（analysis 选择器 + DAG 预览）
```

1. **读数据**：工作台经 `@deepseek-ai/dsh-api-workspace-files` 的 `workspaceFiles`
   Remote 命名空间读取。analysis 列表用 `list`（列 `LLMPWA/analyses/`），选中项的
   快照用 `read`。LLMPWA 位于 DSH 工作区根之下（`LLMPWA/analyses/*`），走工作区相对
   路径，无需新增后端端点。
2. **解析 | 求值**：`pipeline_state.json` 是 `pipeline_state.py` 输出的 JSON，含
   `stages[]`（含依赖/kind）、`manifest`、`params[]`、`artifacts`。浏览器端直接消费。
3. **呈现**：分成「数据求值（纯函数）+ UI 组件」两层。

---

## 3. 包结构与职责

```
packages/client/ui-llmpwa-pipeline/
├── package.json               # @deepseek-ai/dsh-client-ui-llmpwa-pipeline
├── tsconfig.json              # extends tsconfig.base.client.json
├── tsdown.config.ts           # clientBundle(...)（同 ui-sidebar-files）
├── src/
│   ├── index.ts               # host 半：无 host 树，apply() 空
│   ├── client/
│   │   ├── index.ts           # apply(ctx)：注册字典 + sidebar.footer.action 按钮 + shell.overlay 工作台
│   │   ├── FooterButton.tsx   # 左侧按钮组件（纯，four-share props）
│   │   ├── Workbench.tsx      # 整页工作台：顶部分页（DAG/文档）+ 右侧 Agent 抽屉，position:fixed 铺满
│   │   ├── face.ts            # inject face：list/read → store actions 写回；openAgent 建会话；处理 abort
│   │   ├── store.ts           # createWorkbenchStore()：open、selected、快照、references、view、agent phase
│   │   ├── load.ts            # 纯函数：list 列出 analyses；read 读 JSON → 解析
│   │   ├── presenters.ts      # 纯函数：stages→DAG 节点/边/分层；manifest→状态色
│   │   ├── Dag.tsx            # DAG 呈现组件（纯，无副作用）
│   │   ├── locales.ts         # typed 字典（zh/en），i18n copy
│   │   └── *.module.css
│   └── css-modules.d.ts
└── tests/
    ├── presenters.client.spec.ts   # DAG 边/节点/状态/参数纯函数
    ├── load.client.spec.ts         # list/read/解析/缺失/损坏（scripted remote）
    ├── store.client.spec.ts        # store actions（open/list/select/snapshot phase）
    ├── face.client.spec.ts         # face：写回 store + abort 抑制
    ├── apply.client.spec.ts        # 注册两个 slot + 字典 + 共享 store/face + 卸载返还原
    ├── dag.client.spec.tsx         # DAG 组件渲染/边/状态色/防御分支
    ├── footer.client.spec.tsx      # 按钮点击切换 store.open
    └── workbench.client.spec.tsx   # jsdom：空态/列表/选中/快照 phase/DAG/错误
```

### 3.1 依赖注入

browser `inject`（cordis 服务名）：

```ts
export const inject = ['slots', 'locale', 'sessions', 'remote', 'remote.workspaceFiles']
```

- 需要 `@deepseek-ai/dsh-api-workspace-files`（提供 `remote.workspaceFiles`）、
  `@deepseek-ai/dsh-api-remotes`（提供 `remote`）、
  `@deepseek-ai/dsh-api-session-controller`（提供 `sessions`，供 Agent 抽屉建会话）、
  `@deepseek-ai/dsh-client-locale`、`@deepseek-ai/dsh-client-ui-slots`、
  `@deepseek-ai/dsh-client-store`。
- 这两个目标 slot（`sidebar.footer.action` / `shell.overlay`）的**声明方**是 ui-sidebar /
  ui-layout；跨包只 `import type` 拉声明，运行时用 `ctx.slots.inject`（它等待声明完成、可 HMR）。
- 跨包值导入一律不做；不 runtime-import 其它 feature plugin（红线）。DAG 不放 React Flow，
  手写布局，避免引入重依赖与 module-graph 复杂度。

### 3.2 注册（`src/client/index.ts` 关键段）

```ts
export function apply(ctx: ClientContext): void {
  ctx.effect(() => ctx.locale.register(NS, { zh, en }), 'ui-llmpwa-pipeline: dictionaries')

  const store = createWorkbenchStore()
  const face = workbenchFace(ctx.remote)

  // ① 左侧脚部按钮（读 store.open，点击切换）
  ctx.effect(() => ctx.slots.inject('sidebar.footer.action', () => ctx.slots.register(
    { name: 'sidebar.footer.action', id: ENTRY_ID, locale: NS, store, inject: face },
    FooterButton,
  )), 'ui-llmpwa-pipeline: workbench foot action')

  // ② 整页工作台（shell.overlay，覆盖全帧；与按钮共用一个 store + face）
  ctx.effect(() => ctx.slots.inject('shell.overlay', () => ctx.slots.register(
    { name: 'shell.overlay', id: ENTRY_ID, locale: NS, store, inject: face },
    Workbench,
  )), 'ui-llmpwa-pipeline: workbench overlay')
}
```

> 两个 slot 均 `scope: 'root'`，所以二者共享一个 store 实例；`workbenchFace(remote)`
> 返回 `(actions) => WorkbenchInjected`，root-scoped 注册时收到 baked `actions`。
> 组件把两个 Remote 读（列表 + 快照）交给 face 执行，face 经 store 的 `actions.*`
> 写回结果。

---

## 4. 数据访问

用 `remote.workspaceFiles`：

- **列 analyses**：`list({ path: 'LLMPWA/analyses' })` → 子目录名列表。
- **读快照**：`read({ path: 'LLMPWA/analyses/<dir>/gen/pipeline_state.json', offset, limit })`
  → 文本 `JSON.parse`。

要点：
- 路径工作区相对；默认 analysis：如果有记忆值用记忆值，否则首个目录。
- 刷新：工作台内「重新读取」按钮；面板不触发生成快照（快照由 LLMPWA 侧产出）。
- 快照缺失 / JSON 损坏：工作台展示可读错误（区分「未生成快照」与「解析失败」）。
  `gen/pipeline_state.json` 是产物，流水线/导出器跑过才会出现。
  空态给出生成命令提示（`python LLMPWA/agent/pipeline_state.py -w analyses/<dir> ...`）。

---

## 5. 展示模型（纯函数层 `presenters.ts`）

不依赖 React / DSH，输出可直接渲染的展示模型：

| 输入 | 输出 |
|---|---|
| `stages[]` | `{ nodes: [{id,label,kind,status}], edges: [{from,to}], layers: {...} }` |
| `manifest` | 每 stage 的 `cached / ran / missing` 状态与色标 |
| `params[]` | （本期可选）分组表行 |
| `artifacts` | （本期可选）按类别分组清单 |

- **DAG**：手写分层布局。按依赖做拓扑分层（同层无依赖边），纵向排布；节点间连线画
  贝塞尔/正交折线。不引入 React Flow。
- 状态色：`cached`(绿) / `ran`(蓝) / `missing`(红)。
- 首版尽显：每 node 显示 `name` + `kind`（python/llm/agent）+ 状态点；`foreach` 标注、
  依赖箭头。

---

## 6. 注册进 web profile

三处必有（missing any 会在不同点失败）：
1. **`packages/bundle/web-app/cordis.patch.yml`** browser roster 追加：

   ```yaml
   - id: ui-llmpwa-pipeline
     name: '@deepseek-ai/dsh-client-ui-llmpwa-pipeline'
   ```

2. **`tsconfig.client.json`** aggregate 的 `references` 加该包。
3. **`packages/bundle/web-app/package.json`** dependency 加 `"@deepseek-ai/dsh-client-ui-llmpwa-pipeline": "workspace:^"`。

> `discoverPluginDirs()` 按 `dsh.client.platform === 'web'` 自动发现，`dev:web` 会 watch 它。

---

## 7. i18n

所有界面文字走 typed 字典 + `t`，声明 `LocaleNamespaceMap` 扩充，注册到 `ctx.locale`。
`verify-client-ui-i18n` 会拒绝硬编码文案。

---

## 8. 测试与快照

遵循 client 包规范（每文件 100% 覆盖、component spec 用 `// @vitest-environment jsdom`、
注册表贡献证明 disposal）：

1. `presenters.client.spec.ts`：DAG 分层/边/状态纯函数单测（含错误输入）。
2. `load.client.spec.ts`：scripted `remote.workspaceFiles.list/read`，断言 analyses 列表、
   JSON 解析、缺失/损坏错误分支。
3. `workbench.real.spec.tsx`：Boot test-only `cordis.yml`，经 Loader + app 挂载，scripted
   workspace-files 提供快照，断言用户可见输出（按钮出现、打开后渲染 DAG stage 名/状态）。
   HMR-safety 测试：dispose 后条目移除。
4. `test:web` 快照：对工作台做记录式回放快照（`test:web:built`）。

> 测试统一用 **scripted workspace-files** 注入固定快照，不依赖真实 LLMPWA 数据，保证确定性。

---

## 9. 运行与验证（端口 3081）

```bash
export PNPM_HOME=$PWD/.pnpm-home
export PATH="/home/iso/.local/share/pnpm/bin:$PNPM_HOME:$PATH"
pnpm install && pnpm run build      # dev:web 要求先前有完整构建

# LLMPWA 侧产出快照（示例）
python LLMPWA/agent/pipeline_state.py -w LLMPWA/analyses/kk_dis -c llm_config_fit.toml -o gen/pipeline_state.json

pnpm run dev:web                    # watch 客户端 bundle
pnpm exec dsh --profile web --port 3081
# 浏览器打开 http://127.0.0.1:3081
```

验证点：左侧脚部出现 `LLMPWA 工作台` 按钮；点击后整页工作台打开；下拉可选
`kk_dis` 等 analysis；选中后渲染出 8 个 stage 的 DAG 与状态；快照缺失时展示可读错误。

---

## 10. 风险 / 开放问题

| 风险 | 影响 | 应对 |
|---|---|---|
| `pnpm install`/`build` 规模大、首次耗时长 | 前置成本高 | 一次完整构建后 `dev:web` 增量 |
| pnpm global store 只读 | 首次 corepack 下载失败 | `PNPM_HOME` 指向工作区本地 |
| 快照是产物，跑之前不存在 | 工作台空态 | 空态提示 + 生成命令；不自动触发 |
| `shell.overlay` 点击默认穿透 | 工作台无法交互 | 条目自身 `position:fixed; inset:0` + opt-in pointer events |
| 手写 DAG 布局在大型图可能拥挤 | 可读性 | 先做「分层 + 顺序展开」，大图允许横向滚动 |
| workspace-files `read` 对超长文件分页 | 需处理分页 | 快照通常 <1MB，一页读完；超限分页拼接 |
| 面板只读，无「生成快照」入口 | 需手动生成 | 列入后续（agent 侧 command/tool 触发导出） |

## 11. 里程碑（完成态）

1. ✅ 建包 `ui-llmpwa-pipeline` + 三处注册（cordis.patch / tsconfig refs / web-app dep）。
2. ✅ 纯函数层（presenters/load）+ store + face + 单测。
3. ✅ `FooterButton` + `Workbench` 组件 + i18n + `store`，接两个 slot（共用一个 store/face）。
4. ✅ 分析面板改版：预览区顶部加「DAG / 文档」分页；「文档」页列出该 analysis 的
   参考文件并预览文本；蓝色「打开 Agent」按钮弹出右侧抽屉，在 analysis 目录启动 dsh
   agent 会话（`sessions.create({ cwd })`），就绪后跳回主对话区查看。
5. ✅ 组件/注册/卸载测试（103 个单测，每文件 100% 覆盖，含分支）；`test:web` 快照留待后续。
6. ✅ `pnpm run build`（host+client）通过；`lib/client.js` 已产出。
7. ⏳ 实机验证（`dsh --profile web --port 3081` 读真实 `kk_dis` 快照）作为现场验收；
   本机 `kk_dis/gen/pipeline_state.json` 已生成（8 stage，42KB）供预览。
8. ✅ 回写本设计稿为「设计 + 落地说明」，更新 `documentation/README_CN.md` 关联。

> 与设计稿 §3.2 的差异：两 slot 均 `scope: 'root'` 且共用同一 store 与 inject face；
> 列表读与快照读都经 `face.ts` 异步执行并写回 store，组件不直接 await。
> 读会话选择：`workspaceFiles` 按 session 的 `header.cwd`（sandbox policy）解析工作区根，
> 若当前对话是嵌套的 agent 会话（面板「打开 Agent」会切到 `…/analyses/<dir>` cwd），
> `LLMPWA/analyses` 就解析不到。因此面板用一个**属于 LLMPWA workspace 的读会话**
> （`useWorkspaces(...).items[].sessionIds` 取一个）来做读，而非裸用 `useSessions.current`；
> 无可用读会话时显示 `noWorkspace` 提示。
> 已知约束：抽屉内不内嵌整段对话渲染（`conversation` 仅随 shell 当前 session 声明），
> 改版首版采用「创建并打开 session 后跳回主对话区」的手持交付；若要在抽屉内渲染任意
> session 的对话，需另为 ui-conversation 新增 `conversation.embedded` 类 slot（跨包 PR）。
