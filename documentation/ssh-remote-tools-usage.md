# LLMPWA 远程 SSH 工具接入说明（方案一）

> 前提：deepseek-harness 仓库已包含 `packages/ssh/` 远程执行工具组（`dsh-ssh` 服务定义 + `dsh-ssh-ssh2` 提供者 + `dsh-tool-ssh` 六个 `remote_*` 工具）。
> LLMPWA 代码生成 agent（`LLMPWA/agent/`）保持固定本地流程不动；本能力服务于每个 analysis 的 dsh 管理 agent 会话。

## 1. 接线总览

```text
dsh agent 会话 (工作台 analyses/<analysis> 启动)
   │  execute
   ▼
remote_exec / remote_read / remote_write / remote_edit
remote_push / remote_pull        (dsh-tool-ssh, 模型可见)
   │  读取
   ▼
analyses/<analysis>/.dsh/config.yml   ← 每 analysis 的远程主机/凭据引用
   │  调用
   ▼
ctx.remote (dsh-ssh seam)
   │
   ▼
dsh-ssh-ssh2 (ssh2 提供者)  ── SSH/SFTP ──▶ 远程服务器
```

- **本地世界不变**：`read`/`write`/`edit`/`bash` 等本地工具照常可用，`remote_*` 是独立的远程工具组。
- **每个 analysis 独立配置**：不同 analysis 可在 `.dsh/config.yml` 里指向不同主机/用户/远程根目录，对应不同计算设备。

## 2. 组合（composition）

在运行该 analysis agent 的 dsh 组合中挂载两个插件（例如补进工作台使用的 `cordis.patch.yml` 或对应 profile/preset）：

```yaml
- name: '@deepseek-ai/dsh-ssh-ssh2'
- name: '@deepseek-ai/dsh-tool-ssh'
```

工具注册前提：`ctx.remote` 已由提供者注册，`ctx.tools`/`ctx.systemPrompt` 已就绪。

## 3. 每个 analysis 的 `.dsh/config.yml`

在 `LLMPWA/analyses/<analysis>/.dsh/config.yml` 增加 `remote:` 块：

```yaml
remote:
  host: compute-1
  port: 22
  user: phys
  remoteRoot: /home/phys/workspace
  hostKeyFingerprint: <SHA256 base64，主机钥指纹>
  timeoutMs: 60000            # 可选，默认由工具插件配置
  maxTransferBytes: 268435456  # 可选，默认 256 MiB
  auth:
    kind: key                # key | password | agent
    keyPath: ./.dsh/secrets/id_ed25519   # 相对 analysis 目录；缺省即此路径
    # passwordRef: LLPWA_<ANALYSIS>_SSH_PASSWORD   # 或走凭据引用
```

- **hostKeyFingerprint**：必填（严格模式）。用 `ssh-keyscan compute-1` 或首次手动 `ssh` 获取服务器主机钥，取 SHA256 base64 值。
- **key 文件**：`.dsh/secrets/id_ed25519`（权限 0600，目录 0700，加入 `.gitignore`）。每 analysis 一把独立 key。
- **密码**：`passwordRef` 命名一个凭据（环境变量或 `dsh-credentials` store），值永不出现在工具参数、结果或会话日志。

## 4. 工具用法

| 工具 | 用途 | 示例 |
|---|---|---|
| `remote_exec` | 远程执行 shell 命令 | `{ "command": "python fit.py --gpu", "workdir": "run" }` |
| `remote_read` | 读取远程文本文件（带行号） | `{ "path": "run/likelihood_function.py", "limit": 120 }` |
| `remote_write` | 原子创建/覆盖远程文件 | `{ "path": "run/params.toml", "content": "..." }` |
| `remote_edit` | 远程精确文本替换 | `{ "path": "run/fit.py", "old_string": "a", "new_string": "b" }` |
| `remote_push` | 本地 → 远程（推送脚本） | `{ "local_path": "generated/fit_script.py", "remote_path": "run/fit_script.py" }` |
| `remote_pull` | 远程 → 本地（回收结果） | `{ "remote_path": "run/results/likelihood_test_result.txt", "local_path": "result_repo/...", "overwrite": true }` |

约定：远端相对路径基于 `remoteRoot` 解析；`..` 越界与覆盖（默认拒绝，除非 `overwrite: true`）都会被拒绝并给出稳定错误。

## 5. 典型工作流

1. 本地代码生成 agent 产出脚本到 `<analysis>/generated/`（本地工具）。
2. `remote_push` 把脚本推到服务器 `run/`。
3. `remote_exec` 在服务器上运行（按需指定 `workdir`、`timeout_ms`）。
4. `remote_pull` 把实验结果收回 `result_repo/`（`overwrite: true` 覆盖同名旧结果）。
5. `remote_read`/`remote_write`/`remote_edit` 直接管理服务器上的输入输出文件。

## 6. 已知限制

- 互传为单文件；批量目录用 `remote_exec` + `tar`。
- 远端主机只能来自 `.dsh/config.yml`（无逐调用覆盖，属有意策略）。
- 无远程目录列举工具；列目录用 `remote_exec` 的 `ls`。
