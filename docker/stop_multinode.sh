#!/usr/bin/env bash
# 停止所有节点(header + worker)上运行的对应容器
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-"${SCRIPT_DIR}/node_config_example.toml"}"

# ---------- 用 Python 解析 TOML，输出 shell 安全的赋值语句 ----------
_TOML_OUT="$(python3 - "$CONFIG" <<'PYEOF'
import sys, json

try:
    import tomllib
    with open(sys.argv[1], "rb") as f:
        cfg = tomllib.load(f)
except ModuleNotFoundError:
    import toml
    cfg = toml.load(sys.argv[1])

header    = cfg.get("header", {})
container = cfg.get("container", {})
workers   = cfg.get("worker", [])

def q(v):
    return "'" + str(v).replace("'", "'\\''") + "'"

print(f'HEADER_IP={q(header.get("IP", ""))}')
print(f'HEADER_USER={q(header.get("Username", "root"))}')
print(f'CONTAINER_NAME={q(container.get("container", "kkfit"))}')
print(f'WORKERS_JSON={q(json.dumps(workers))}')
PYEOF
)"
eval "$_TOML_OUT"

NUM_WORKERS=$(python3 -c "import json,sys; print(len(json.loads(sys.argv[1])))" "$WORKERS_JSON")

echo "============================================"
echo " Config      : $CONFIG"
echo " Container   : $CONTAINER_NAME"
echo " Header      : ${HEADER_USER}@${HEADER_IP}"
echo " Workers     : ${NUM_WORKERS}"
echo "============================================"

stop_on_host() {
    local user="$1" ip="$2"
    echo "[${user}@${ip}] stopping container ${CONTAINER_NAME} (if running)"
    ssh "${user}@${ip}" "docker stop '${CONTAINER_NAME}' 2>/dev/null || true" &
}

for i in $(seq 0 $((NUM_WORKERS - 1))); do
    ip=$(python3   -c "import json,sys; print(json.loads(sys.argv[1])[${i}]['IP'])"                      "$WORKERS_JSON")
    user=$(python3 -c "import json,sys; w=json.loads(sys.argv[1]); print(w[${i}].get('Username','root'))" "$WORKERS_JSON")
    stop_on_host "$user" "$ip"
done

stop_on_host "$HEADER_USER" "$HEADER_IP"

wait
echo "All containers stopped (or were not running)."
