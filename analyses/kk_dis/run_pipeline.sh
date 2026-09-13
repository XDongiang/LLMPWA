#!/usr/bin/env bash
# Non-interactive kk_dis pipeline runner.
# Fixes the httpx/no_proxy bracket-IPv6 bug without touching the shell profile.
set -uo pipefail
export no_proxy='localhost,127.0.0.1,.localdomain,::1'
export NO_PROXY='localhost,127.0.0.1,.localdomain,::1'
cd "$(dirname "$0")/../.."   # LLMPWA repo root (where .env lives)
exec /home/iso/openclaw/deepseek-harness/LLMPWA/analyses/kk_dis/.venv/bin/python -m agent.cli \
  --workdir analyses/kk_dis --config llm_config_fit.toml "$@"
