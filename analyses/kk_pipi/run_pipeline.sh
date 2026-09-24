#!/usr/bin/env bash
# Non-interactive kk_pipi combine pipeline runner.
# Fixes the httpx/no_proxy bracket-IPv6 bug without touching the shell profile.
set -uo pipefail
export no_proxy='localhost,127.0.0.1,.localdomain,::1'
export NO_PROXY='localhost,127.0.0.1,.localdomain,::1'
cd "$(dirname "$0")/../.."   # LLMPWA repo root (where .env lives)
exec /home/iso/openclaw/deepseek-harness/LLMPWA/analyses/kk_pipi/.venv/bin/python -m agent.cli \
  --workdir analyses/kk_pipi --config llm_config_combine.toml "$@"
