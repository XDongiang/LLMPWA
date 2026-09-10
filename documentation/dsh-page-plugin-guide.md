# 为 dsh 写「整页」插件：实战指南（以 LLMPWA 工作台为例）

> 本文把上面实现 LLMPWA 工作台的经验整理成一份**可复用的 cookbook**：如何给
> DeepSeek Harness Web GUI（dsh web）写一个原生 client-plugin，并在界面里加一个
> **入口按钮 + 整页工作台**。文中所有代码/命令都来自已跑通的真实现
> （`packages/client/ui-llmpwa-pipeline`，源码见 deepseek-harness 仓库）。
> 设计说明见 [dsh-client-plugin-design.md](dsh-client-plugin-design.md)；本文更偏向
> 「怎么写」，那一篇更偏向「为什么这样设计」。

---

## 0. 一句话概括

dsh 的 Web UI 是**插件化**的：客户端 UI 以 Cordis client-plugin 形式存在，通过
**slot 机制**往既有的界面容器里「加一块」。要做一个「整页」功能，关键是选对两个
slot：

1. **入口按钮** → 注册进 `sidebar.footer.action`（左侧 sidebar 脚部，`list`，root 作用域）。
2. **整页内容** → 注册进 `shell.overlay`（覆盖全帧的浮层，`list`，root 作用域）。

两个 slot 都是 `kind: 'list'`，所以是**追加**（additive）而非替换；且都是
`scope: 'root'`（不随 session 重建），非常适合一个「页面级」的独立工作台。

> ⚠️ 千万不要注册进 `conversation` / `sidebar` / `rightbar` —— 这些是
> `kind: 'single'` 且已被 occupant 占据，注册会**整体替换**其表面并销毁其声明的
> 子 slot，是用来「换掉整个区域」，不是「加一个入口」。

---

## 1. 前置认知：slot 与注入

### 1.1 slot 声明在哪

- `shell.overlay` 由 `packages/client/ui-layout` 在 `AppFrame` 里声明并渲染，位于
  `data-shell-overlay` 层（`position: fixed; inset: 0`，默认 `pointer-events: none`，
  条目自身可 opt-in `pointer-events: auto`）。
- `sidebar.footer.action` 由 `packages/client/ui-sidebar` 声明。已有 `ui-cordis`
  用它放
  `cordis-panel`，是现成模板。

### 1.2 怎么注册（关键 API）

一个插件只能通过 `ctx.slots.register({ name, children?, store?, inject? }, Component)`
加入 UI。**注册是副作用**，必须包在 `ctx.effect()` 里；注册进别人声明的 slot 要用
`ctx.slots.inject(name, () => ctx.slots.register(...))`——它会**等待该 slot 被声明**、
在被替换后重跑、并在插件卸载时移除（HMR 安全）。

```ts
ctx.effect(() => ctx.slots.inject(slotName, () => ctx.slots.register({
  name: slotName,
  id: 'my-entry',        // list slot 需要唯一 id
  locale: NS,            // 本插件文案命名空间
  store,                 // 可选：跨注册共享的状态
  inject: face,          // 可选：业务 face（含异步读）
}, MyComponent)))
```

---

## 2. 建包骨架

照抄 `packages/client/ui-llmpwa-pipeline`（一个 UI 功能 = 一个插件包）：

```
packages/client/ui-llmpwa-pipeline/
├── package.json          # @deepseek-ai/dsh-client-ui-llmpwa-pipeline
├── tsconfig.json         # extends tsconfig.base.client.json + references
├── tsdown.config.ts      # clientBundle('...', ['lib/types/index.js'])
├── src/
│   ├── index.ts          # host 半：export function apply(){}（无 host 贡献）
│   ├── css-modules.d.ts  # CSS Modules 类型
│   └── client/
│       ├── index.ts      # 客户端半：export const inject / export function apply
│       ├── FooterButton.tsx   # 左侧按钮（接 sidebar.footer.action）
│       ├── Workbench.tsx      # 整页工作台（接 shell.overlay）
│       ├── face.ts       # 业务 face：异步读 → store.actions 写回
│       ├── store.ts      # createWorkbenchStore() 工厂
│       ├── load.ts       # 纯函数：workspaceFiles.list/read + 解析
│       ├── presenters.ts # 纯函数：stages → DAG 展示模型
│       ├── Dag.tsx       # 纯 SVG DAG 组件
│       ├── locales.ts    # typed 字典（zh/en）
│       └── *.module.css
└── tests/                # vitest 单测（jsdom 组件/纯函数/apply）
```

**package.json 关键字段**：

```jsonc
{
  "name": "@deepseek-ai/dsh-client-ui-my-feature",
  "exports": { ".": {...}, "./client": {...}, "./src/*": "./src/*", "./package.json": "./package.json" },
  "dsh": {
    "client": {
      "inject": ["@deepseek-ai/dsh-api-remotes", "@deepseek-ai/dsh-api-workspace-files"],
      "platform": "web"
    }
  },
  "scripts": { "bundle": "tsdown", "watch": "tsdown --watch" }
}
```

`dsh.client` 的 `inject` 只列**包名依赖边**（preflight/HMR 展示用），不等同于 Cordis 服务
注入；真正注入的是 `src/client/index.ts` 里的 `export const inject = [...]`（服务名）。

---

## 3. 组件 props = 四个 share（不要去手写）

组件的 props 是一个「派生交集」，永远用类型推导，不要自己重写：

```ts
type MyProps =
  PropsRuntime<'shell.overlay'>          // 运行时 share：owner props + 全局标准 hooks（useSessions/useWorkspaces）
  & PropsStore<WorkbenchStore>           // store share：useStore(selector) + actions
  & InjectFace<WorkbenchInjected>        // inject face：业务数据/回调
  & PropsLocale<'llmpwa'>                // locale share：t
```

### 3.1 状态 store

跨注册共享/需要跨 remount 保留的状态，用 `defineStore` 的**工厂**：

```ts
export function createWorkbenchStore(): EngineStoreHandle<WorkbenchState, WorkbenchActions> {
  return defineStore({ init: () => ({ open: false, ... }), actions: { opened, closed, ... } })
}
```

- 在 `apply` 里**创建一次**，把同一个 handle 声明到多个注册上（本例两个 slot 共用）。
- 组件读：`const state = props.useStore(s => s)`（**必须传 selector**，这是
  `SnapshotSelectorHook`，传 `(s) => s` 取全量）。
- 组件写：`props.actions.opened()` 等（actions 是 bake 好的写集）。

> ⚠️ `useStore()` / `useSessions()` 这些 hook 都要求一个 selector 参数，`useStore()` 不带参
> 会报 "Expected 1-2 arguments, but got 0"。

### 3.2 业务 face（异步读）

组件**不 await**。异步读写在一个 `inject` 工厂里做，收到 store 的 `actions`，完成后
写回：

```ts
// root-scope + 声明了 store → 工厂签名是 (actions) => Face
export function workbenchFace(remote, sessions): (actions: BoundActions<WorkbenchStore>) => WorkbenchInjected {
  return (actions) => {
    return {
      listAnalyses: (sessionId, signal) => {
        if (signal.aborted) return
        actions.listing()
        void listAnalyses(remote, sessionId, signal).then(analyses => {
          if (signal.aborted) return
          actions.listed(analyses)
        })
      },
      // loadSnapshot(...) / listReferences(...) / loadReference(...) 同理
      // 建会话：sessions.create({ cwd }) → sessions.open(id) / 失败 actions.agentFailed(error)
      openAgent: (workspacePath, analysis) => {
        actions.agentOpening()
        void sessions.create({ cwd: `${workspacePath}/LLMPWA/analyses/${analysis}` }).then(id => {
          sessions.open(id)
          actions.agentReady(id)
        }).catch(err => actions.agentFailed({ kind: 'unexpected', message: String(err) }))
      },
    }
  }
}
```

要点：inject face 里可以拿到 `apply` 闭包的 `ctx`/`remote`；**决不能在组件里 new 订阅
或直接碰 ctx**——组件只通过四个 props share 拿数据。

---

## 4. 数据访问：workspace-files Remote

整页插件常用「读工作区文件」。经 `@deepseek-ai/dsh-api-workspace-files` 的
`remote.workspaceFiles` 命名空间：

- `list(sessionId, path, signal)` → 列工作区目录（返回 `entries`，目录的 `type: 'directory'`）。
- `read(sessionId, path, { offset, limit }, signal)` → 分页读文本。

两个坑（我们实际踩到）：

1. **`limit` 有上限（默认 `maxLines`，当前 5000）**；传更大的会被 Host **直接拒绝**。
   所以**不要传 `limit`**（用 Host 默认），并自己**分页**：读到 `eof === false` 就
   `offset += lines` 继续，直到 `eof` 为真。
2. 路径是**工作区相对**的（相对 session 的 workspace root），如
   `LLMPWA/analyses/kk_dis/gen/pipeline_state.json`。

正确分页写法见 `packages/client/ui-llmpwa-pipeline/src/client/load.ts` 的 `loadSnapshot`。
纯函数层（`presenters.ts`）与 React/DSH 无关，直接消费 JSON。

---

## 5. 三处注册（缺一不可）

新插件要进入 `dsh --profile web` 必须有三处登记（缺任何一处会在不同时机/层面失败）：

1. **`packages/bundle/web-app/cordis.patch.yml`** 的 `dsh.client` browser roster 加一行：

   ```yaml
   - id: ui-my-feature
     name: '@deepseek-ai/dsh-client-ui-my-feature'
   ```

2. **`tsconfig.client.json`** 的 `references` 加 `{ "path": "./packages/client/ui-my-feature" }`。
   （还要在 **`tsconfig.base.json`** 加两条 `paths` 映射：
   `"@deepseek-ai/dsh-client-ui-my-feature": ["./packages/client/ui-my-feature/src"]`
   和 `/client` → `.../src/client`。）

3. **`packages/bundle/web-app/package.json`** 加 dependency：
   `"@deepseek-ai/dsh-client-ui-my-feature": "workspace:^"`。

---

## 6. i18n

所有用户可见文案放进 typed 字典，组件经 `t` 取：

```ts
declare module '@deepseek-ai/dsh-client-ui-slots' {
  interface LocaleNamespaceMap { llmpwa: import('./locales.ts').LlmpwaKey }
}
// apply 里注册：ctx.effect(() => ctx.locale.register(NS, { zh, en }), ...)
// 组件：props.t('panel.title')
```

`verify-client-ui-i18n` 会拒绝硬编码文案，所以别在 JSX 里写死字符串（代号/代码 token 除外）。

---

## 7. 构建、测试、运行

### 7.1 构建（先完整 build 一次，dev 才会增量）

```bash
export PNPM_HOME=$PWD/.pnpm-home
export PATH="/home/iso/.local/share/pnpm/bin:$PNPM_HOME:$PATH"
pnpm install
pnpm run build          # host + client 两遍；会生成 /remote *.d.ts 与各包 lib/
pnpm run build:lib:host   # 只 host 遍（含 typert/remote d.ts 生成）
pnpm run build:lib:client # 只 client 遍（tsc + tsdown 全量）
```

单包校验：

```bash
pnpm --filter @deepseek-ai/dsh-client-ui-llmpwa-pipeline run bundle   # 产出 lib/client.js
# 或
node node_modules/.bin/tsc -b packages/client/ui-llmpwa-pipeline   # 类型检查
```

### 7.2 测试

- 遵循 client 包规范：**每文件 100% 覆盖**（`packages/*/*/src`）。
- 组件 spec 用 `// @vitest-environment jsdom` 首行。
- 组件测试**直接喂 props**（真实 store 实例 + `useStore`/`useSessions` 的 selector stub，
  actions 用 store 实例），不引入 render 引擎。
- `apply` 测试用真实 Cordis `Context`，slots/locale/remote 用 recorder，断言注册内容 +
  dispose 后全部移除（HMR 安全）。
- 注意 `SessionId` 是**品牌类型**，测试里要 `'s1' as SessionId`。

```bash
pnpm --filter @deepseek-ai/dsh-client-ui-llmpwa-pipeline test   # 若有
# 或直接
node_modules/.bin/vitest run packages/client/ui-llmpwa-pipeline
node_modules/.bin/vitest run packages/client/ui-llmpwa-pipeline --coverage   # 看覆盖
```

### 7.3 运行（两个进程）

`dev:web` 是 **watch-build**（只负责在源码变更时重打 bundle），**不是服务器**，也不会
退出。真正跑 GUI 的是 `dsh web`。

```bash
# 终1：watch-build（保持运行；Ctrl+C 即停并杀掉子 watcher）
pnpm run dev:web

# 终2：真正的 web GUI 服务器
pnpm dsh --profile web --port 3081 --no-open
# 打开 http://127.0.0.1:3081
```

> ⚠️ `pnpm exec dsh` 会报 `Command "dsh" not found` —— `dsh` 是根 `package.json` 的
> **script**，不是已安装的 bin，要用 `pnpm dsh ...`。

---

## 8. 踩坑清单（来自本篇实现）

1. **`workspaceFiles.read` 的 `limit` 上限 5000**：不要传大于上限的 `limit`，会报
   `limit must be at most 5000`；改为不传 `limit` + 自己分页。
2. **`useStore` / `useSessions` 要传 selector**：`useStore(s => s)`，不能 `useStore()`。
3. **root-scope inject 工厂只收到 `actions`**（声明了 store 时）。只有 `scope: 'session'`
   才会收到 `(sessionId, actions)`；`session-maybe` 收到 `(sessionId | undefined, actions | undefined)`。
4. **`noUncheckedIndexedAccess` 开启**：`arr[i]` / `map.get(k)` 返回 `| undefined`，要
   `?? 默认值` 或断言；否则 tsc 报 `string | undefined` 不可赋给 `string`。
5. **跨包值导入禁止**（bundle purity gate）：跨插件只 `import type`，行为走 Cordis 服务/
   slot；共享运行时只能放 `client/store`、`ui-primitives` 等窄稳定层。baseline 外部化
   （react / cordis / `dsh-client-store` / `ui-slots` / `ui-primitives`）自动生效。
6. **CSS 用 `--dsw-alias-*`**：规范 token 是 `--dsw-alias-bg-base`、`--dsw-alias-label-primary`、
   `--dsw-alias-border-inverted` 等；不存在裸 `--dsw-*`。
7. **别注册进 `conversation`/`sidebar`/`rightbar`**：single slot 会被整体替换。
8. **组件永远看不到 ctx**：所有数据/回调走四个 props share；业务异步读放 inject face。
9. **`SessionId` 是品牌类型**：跨函数传 session 时要用 `as SessionId`。
10. **`dev:web` 与 `dsh web` 是两个进程**；`dev:web` 要求先有一次完整 `pnpm run build`，
    否则某阶段会静默 no-op，出现「改了没反应」。
11. **新增一个 Cordis 服务注入（如 `sessions`）要四处登记**：`src/client/index.ts` 的
    `export const inject` 数组加服务名；`package.json` 的 `dsh.client.inject` 加包名；
    `devDependencies` 加 `"pkg": "workspace:^"`；`tsconfig.json` 的 `references` 加该包
    client leaf（如 `../../api/session-controller/tsconfig.client.json`）。
12. **root-scope 标准 hook `useWorkspaces` 不是默认可见**：它由
    `@deepseek-ai/dsh-client-ui-workspace` 的类型扩充（`GlobalStandardProps`）提供，需在
    客户端入口 `import type {} from '@deepseek-ai/dsh-client-ui-workspace/client'` 才能解析；
    会话的工作区根路径从这里取（`items[].path`，选包含当前 session 的那个，否则回退
    `items[0]`）。`tsconfig.json` 也要 reference 该包。
13. **启动一个「工作目录在指定目录」的 agent 会话**：`sessions.create({ cwd })`，Host 把
    `cwd` 解析为绝对路径并 `mkdir -p`；拿到 sessionId 后 `sessions.open(id)` 切到该会话。
    cwd 由「工作区根路径 + `LLMPWA/analyses/<analysis>`」拼成。
14. **组件测试里要让「点击改 store → 界面变化」生效**：`useStore` stub 需用
    `useSyncExternalStore(instance.subscribe, () => sel(instance.getSnapshot()))`（slot 运行时
    的 observableHook 就是这么绑的）；纯 `sel(instance.getSnapshot())` 不响应后续 mutation，
    点击后 DOM 不会更新。
15. **抽屉内不内嵌整段对话渲染**：`conversation` 相关 slot 只随 shell 的当前 session 声明，
    当前无「按任意 session 渲染对话」的扩展点；要内嵌需给 ui-conversation 加
    `conversation.embedded` 类 slot（跨包改动）。改版首版采用「创建并打开 session 后跳回
    主对话区」的手持交付。
16. **`workspaceFiles` 按 session 的工作目录解析，不是「当前对话 session」就能读到 LLMPWA**：
    工作区根取自 `session.header.cwd`（sandbox policy）。若当前对话是一个嵌套 agent 会话
    （如从本面板「打开 Agent」创建的 `…/LLMPWA/analyses/<dir>` 会话），其 cwd 是 analysis
    目录，`LLMPWA/analyses` 就解析不到。要让工具始终读到 LLMPWA 工作区，应改用一个属于
    **LLMPWA workspace** 的读会话（从 `useWorkspaces(...).items[].sessionIds` 取一个），
    而不是裸用 `useSessions(...).current`。无可用读会话时给出 `noWorkspace` 提示，而不是
    误显示「没有 analysis」。

---

## 9. 模板速查：最小「按钮 + 整页」注册

```ts
// src/client/index.ts
import type { Context } from '@deepseek-ai/cordis'
import type {} from '@deepseek-ai/dsh-api-session-controller/client'   // 解析 ctx.sessions 类型
import { FooterButton } from './FooterButton.tsx'
import { Workbench } from './Workbench.tsx'
import { workbenchFace } from './face.ts'
import { createWorkbenchStore } from './store.ts'
import { en, zh } from './locales.ts'

const NS = 'llmpwa'
const ENTRY_ID = 'llmpwa-workbench'

// sessions：Cordis 服务名（供 Agent 抽屉建会话）；remote*：读工作区文件
export const inject = ['slots', 'locale', 'sessions', 'remote', 'remote.workspaceFiles']

export function apply(ctx: Context): void {
  ctx.effect(() => ctx.locale.register(NS, { zh, en }), 'feature: dictionaries')

  const store = createWorkbenchStore()
  const face = workbenchFace(ctx.remote, ctx.sessions)

  ctx.effect(() => ctx.slots.inject('sidebar.footer.action', () => ctx.slots.register(
    { name: 'sidebar.footer.action', id: ENTRY_ID, locale: NS, store, inject: face },
    FooterButton,
  )), 'feature: entry button')

  ctx.effect(() => ctx.slots.inject('shell.overlay', () => ctx.slots.register(
    { name: 'shell.overlay', id: ENTRY_ID, locale: NS, store, inject: face },
    Workbench,
  )), 'feature: full page')
}
```

`Workbench` 组件：`if (!state.open) return null`，打开时铺满整帧并读 store 的
selected/snapshot 渲染；两个 Remote 读交给 `face`，face 经 `actions.*` 写回。

---

## 10. 参考

- 设计稿/原理：[dsh-client-plugin-design.md](dsh-client-plugin-design.md)
- DSH 官方 Web Client 架构：deepseek-harness `docs/subsystems/web-client.md`
- Slots 参考：`docs/subsystems/slots.md`
- 客户端包规范：`packages/client/AGENTS.md`（slot 纪律、四 props share、i18n、覆盖、依赖声明）
- 完整示例源码：`packages/client/ui-llmpwa-pipeline/`
