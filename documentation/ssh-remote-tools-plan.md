# LLMPWA 远程 SSH 工具组（方案一）实施计划

> 状态：待评审。架构分工：**DSH = 管理层**（工作台 `packages/client/ui-llmpwa-pipeline` 在每个 analysis 工作目录启动 dsh agent 会话，负责该 analysis 的管理，需要本地 + 远程双重执行能力）；**LLMPWA Python 代码生成 agent（`LLMPWA/agent/`）保持固定本地流程不动**（读本地文件、跑语法测试）。

## 1. 目标

- 本地世界保持默认：现有 `dsh-tool-fs`（read/write/edit）、`dsh-tool-bash` + 本地 provider 原样工作。
- 另加一组**独立的远程 SSH 工具**（`remote_exec` / `remote_read` / `remote_write` / `remote_edit` + 互传 `remote_push` / `remote_pull`），让每个 analysis 的 dsh agent 能把代码/命令执行到远程服务器（不同 analysis 可指向不同设备），并能双向搬运文件：**推送脚本到远程、拉取实验结果数据回本地**。
- 每个 analysis 的远程配置放在 `LLMPWA/analyses/<analysis>/.dsh/` 下，工具按调用方 session 的工作目录自动找到对应配置。

## 2. 落点与包结构（决策点 D1）

推荐在 DSH 仓库内新建包组 `packages/ssh/`，与 `ui-llmpwa-pipeline` 同仓库的先例一致：

| 包 | 角色 | ctx 键 | 内容 |
|---|---|---|---|
| `packages/ssh/ssh` | Service Definition | `ctx.remote` | 抽象 `RemoteExecutor`：连接解析、run 命令、read/write/edit 原语、typed 错误 |
| `packages/ssh/ssh-ssh2` | Provider | 注册 `ctx.remote` | ssh2 实现：认证、连接复用、取消、host key 校验、环境 scrub |
| `packages/ssh/tool-ssh` | Consumers | `ctx.tools` | `remote_exec` / `remote_read` / `remote_write` / `remote_edit` / `remote_push` / `remote_pull` 六个工具 |

备选：作为独立插件包放在 LLMPWA 仓库侧（见决策点 D1）。

## 3. 工具契约（模型可见面）

全部用 `defineTool` 注册，输出为结构化 JSON（`output.schema`）+ 原生文本渲染：

| 工具 | 参数 | 返回 |
|---|---|---|
| `remote_exec` | `command`, `workdir?`, `timeout_ms?` | `{ exit_code, stdout, stderr, cmd }`（非零退出 = 成功结果中的 `exit_code`，不是 isError） |
| `remote_read` | `path`, `offset?`, `limit?` | `{ path, lines: [{number, text}], total_lines, truncated? }`（文本窗口，行号输出） |
| `remote_write` | `path`, `content` | `{ path, created: bool }`（临时文件 + rename 原子落地） |
| `remote_edit` | `path`, `old_string`, `new_string`, `replace_all?` | `{ path, occurrences }`（literal 替换，唯一匹配要求） |
| `remote_push` | `local_path`, `remote_path`, `overwrite?` | `{ direction: 'push', local_path, remote_path, bytes }`（本地 → 远程，上传脚本） |
| `remote_pull` | `remote_path`, `local_path`, `overwrite?` | `{ direction: 'pull', local_path, remote_path, bytes }`（远程 → 本地，拉取实验结果） |

互传工具走 ssh2 自带 **SFTP 通道**（与 exec 共用同一连接），不依赖远端 `scp` 是否安装：

设计要点：

- **本地世界保持默认**：目录/文件能力走本地工具（tool-fs/tool-bash），远程只在「上服务器执行」与「跨机搬运」语义下使用，schema 独立命名空间。
- `remote_exec` 的 `workdir` 默认取该 analysis 配置的 `remoteRoot`；`remote_read/write/edit` 的相对路径基于远程 `remoteRoot` 解析，拒绝 `..` 越界（沿用 `session-cwd.ts` 的检查思路）。
- **互传方向与路径语义**：`remote_push` 的 `local_path` 相对本机会话 cwd（分析目录）解析，`remote_path` 相对远程 `remoteRoot` 解析；`remote_pull` 反之。两端都拒绝 `..` 越界。默认不支持覆盖已有文件（`overwrite: true` 显式开启），避免模型误覆盖远端实验结果或本地产物。
- **互传可靠性**：单文件对流式写入；远端侧 `remote_push` 先传临时文件再 `rename`（同构于 fs-e2b 原子发布），`remote_pull` 落本地同样先临时文件后原子改名；`exec.signal` 取消时中断传输并清理半成品临时文件。目录互传（递归）列为本期之外，用 tar+`remote_exec` 作为临时手段（决策点 D6）。
- 远程结果格式与本地结果对齐（行号、`[exit code: N]` 类标记），模型不需要学两套语法。
- UI 展示：`presentCall`/`presentResult` 复用 `terminal` / `diff` / `read` 卡片。

## 4. Per-analysis 配置与凭据

```
LLMPWA/analyses/<analysis>/.dsh/
├── config.yml                  # 远程主机配置（可提交，只含引用）
│     host, port, user, remoteRoot, timeoutMs?, knownHosts?/fingerprint?
│     maxTransferBytes?         # 单次互传字节上限（默认如 256 MiB）
│     keyPath: ./.dsh/secrets/id_ed25519        # key 文件引用（0600）
│     passwordRef: LLPWA_<analysis>_SSH_PASSWORD # 密码引用（走 credentials）
└── secrets/                    # 0700，加入 .gitignore
    └── id_ed25519              # 0600，该 analysis 专用 key
```

- **key 文件**：每个 analysis 一把独立 key，路径进配置、内容不进；passphrase 走 ssh-agent（`SSH_AUTH_SOCK`）。
- **password**：走 DSH 的 `dsh-credentials` seam（`credentialRef`，环境变量优先、store 次之、值永不出现在配置/日志/模型面）；如要每 analysis 完全隔离，可用独立 store 文件 + `isolate` realm。
- 工具在 `execute` 时按 `exec.agent.session.header.cwd` 找到 `<analysis>/.dsh/config.yml`（与 `session-cwd.ts` 同法），解析后**只在 provider 内部**持有凭据，绝不进入模型输入 / tool args / session log。

## 5. Provider 关键实现点（ssh-ssh2）

- 连接按 `(analysisKey, host, user)` 缓存复用，idle 超时关闭；`exec.signal` 传播（连接中、命令中、远端 cleanup 三处检查）。
- host key 校验：默认严格（`config.yml` 的 `knownHosts`/`fingerprint`），缺省拒绝连接而非静默接受。
- 命令执行：远端 `bash -c` 包装 + 显式 env（只透传请求的 env，scrub 凭据形状变量）；stdout/stderr 有界收集 + 溢出 spill。
- 互传：ssh2 的 `sftp()` 子通道；单文件流式读写 + 传输字节上限（`config.yml` 的 `maxTransferBytes`，超出拒绝而非静默截断）；传输失败映射为 typed 错误（远端空间不足 / 权限拒绝 / 超出上限），并保证不留半成品文件。
- 取消/清理：`exec.signal` → 关闭 channel + 远端进程组 kill（沿用 subprocess-e2b 的 SIGTERM → SIGKILL teardown 阶梯思路）。
- 错误映射：连接失败 / host key 不匹配 / 超时 / 权限拒绝 → typed 错误（稳定代码，模型不猜）。

## 6. 测试与验证

- **unit**：参数校验（非空 command、路径越界拒绝、`old_string` 唯一性）、渲染输出、错误映射、配置解析（缺配置 loud-fail）、互传参数（方向、覆盖语义、`maxTransferBytes` 上限）。
- **e2e（真实链路）**：本地 Docker 起一个 `sshd` 容器，跑通 `remote_exec → remote_push（脚本）→ remote_exec（跑脚本）→ remote_pull（拉产物）→ remote_read/write/edit` 全链路；无凭据/无 docker 时 self-skip。
- 模型可见输出用快照锁定（若进 DSH 仓库，按 testing policy 补 snapshot）。
- 集成验证：工作台里对 `kk_dis` 开一个 agent 会话，配 `.dsh/config.yml` 连测试 sshd，跑一条 `remote_exec` 验证真实会话链路。

## 7. 实施步骤

1. **包骨架 + Service Definition**：`packages/ssh/ssh` — `RemoteExecutor` 接口、`Config` schema、typed 错误码。
2. **ssh2 Provider**：认证（keyPath / agent / passwordRef）、连接复用、取消、host key 校验、输出 framing。
3. **`remote_exec` 工具**：注册、参数校验、装配、渲染、`[exit code: N]` 标记。
4. **`remote_read` / `remote_write` / `remote_edit` 工具**：远程原语 + 渲染 + diff card。
5. **`remote_push` / `remote_pull` 工具**：SFTP 单文件流式互传、临时文件原子发布、`maxTransferBytes` 上限、覆盖语义。
6. **Per-analysis `.dsh/` 接线**：session cwd → config 解析、secrets 权限/gitignore、credentials ref 解析。
7. **测试**：unit + docker e2e + 快照。
8. **文档**：包 README（沿用仓库文档规范）+ LLMPWA 侧接入说明（如何给一个 analysis 配远程主机）。

## 8. 决策点（待拍板）

| 编号 | 决策 | 推荐 |
|---|---|---|
| D1 | 包位置：DSH 仓库 `packages/ssh/` vs LLMPWA 侧独立插件包 | `packages/ssh/`（与 `ui-llmpwa-pipeline` 同仓库先例） |
| D2 | 工具命名：`remote_exec`/`remote_read`/`remote_write`/`remote_edit` vs `ssh_*` | `remote_*` |
| D3 | host 绑定：`config.yml` 绑定（模型不可覆盖 host）vs 工具参数可覆盖 | `config.yml` 绑定 |
| D4 | 会话复用：按 analysis 复用 SSH 连接 vs 每次新建 | 复用（避免每调用一次握手） |
| D5 | 互传默认覆盖：`overwrite` 默认 false（防误覆盖远端结果/本地产物）vs 默认 true（省一步参数） | 默认 false，显式开启 |
| D6 | 目录互传：本期仅单文件（目录用 tar+`remote_exec` 临时解决）vs 本期就做递归目录 | 本期仅单文件 |

本计划不涉及改动 `agent-loop`、不新增 session 事件、不触碰 `LLMPWA/agent/`（Python 代码生成 agent）。
