#!/usr/bin/env bash
# 使用 node_config_example.toml 在 docker 上运行 fit_script_distributed.py
# kk_test 目录会被 rsync 同步到每个 worker，并挂载到 docker 容器的 /workspace/kk_test
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-"${SCRIPT_DIR}/node_config_example.toml"}"
KK_TEST_DIR="${SCRIPT_DIR}/kk_test"

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
print(f'HEADER_PORT={q(header.get("PORT", "12345"))}')
print(f'HEADER_IB={q(header.get("IB_IF", ""))}')
print(f'HEADER_USER={q(header.get("Username", "root"))}')
print(f'IMAGE={q(container.get("image", "jax:latest"))}')
print(f'WORKERS_JSON={q(json.dumps(workers))}')

envs = container.get("env", [])
print(f'EXTRA_ENV_COUNT={len(envs)}')
for i, e in enumerate(envs):
    print(f'EXTRA_ENV_{i}={q(e)}')
PYEOF
)"
eval "$_TOML_OUT"

# 重建 EXTRA_ENVS 数组
EXTRA_ENV_FLAGS=""
for i in $(seq 0 $((EXTRA_ENV_COUNT - 1))); do
    varname="EXTRA_ENV_${i}"
    EXTRA_ENV_FLAGS+=" -e ${!varname}"
done

NUM_WORKERS=$(python3 -c "import json,sys; print(len(json.loads(sys.argv[1])))" "$WORKERS_JSON")
NUM_PROCESSES=$((NUM_WORKERS + 1))
COORDINATOR="${HEADER_IP}:${HEADER_PORT}"

echo "============================================"
echo " Config      : $CONFIG"
echo " Image       : $IMAGE"
echo " Header      : ${HEADER_USER}@${HEADER_IP}:${HEADER_PORT} (process 0)"
echo " Workers     : ${NUM_WORKERS}"
echo " Total procs : $NUM_PROCESSES"
echo " kk_test dir : $KK_TEST_DIR"
echo "============================================"

# ---------- 构建 docker run 命令字符串 ----------
build_docker_cmd() {
    local process_id="$1" ib_if="$2" remote_kk_test="$3"
    echo "docker run --rm --gpus all --privileged --network=host" \
         "-e JAX_COORDINATOR_ADDRESS=${COORDINATOR}" \
         "-e JAX_NUM_PROCESSES=${NUM_PROCESSES}" \
         "-e JAX_PROCESS_ID=${process_id}" \
         "-e NCCL_IB_HCA=${ib_if}" \
         ${EXTRA_ENV_FLAGS} \
         "-v ${remote_kk_test}:/workspace/kk_test" \
         "${IMAGE} python3 /workspace/kk_test/run/fit_script_distributed.py"
}

# ---------- worker 节点通过 SSH 执行（后台） ----------
run_on_worker() {
    local idx="$1"
    local ip ib user
    ip=$(python3   -c "import json,sys; print(json.loads(sys.argv[1])[${idx}]['IP'])"                        "$WORKERS_JSON")
    ib=$(python3   -c "import json,sys; w=json.loads(sys.argv[1]); print(w[${idx}].get('IB_IF','${HEADER_IB}'))" "$WORKERS_JSON")
    user=$(python3 -c "import json,sys; w=json.loads(sys.argv[1]); print(w[${idx}].get('Username','root'))"   "$WORKERS_JSON")
    local process_id=$((idx + 1))
    local remote_kk_test="/tmp/kk_test"

    echo "[worker ${process_id}] rsync kk_test -> ${user}@${ip}:${remote_kk_test}"
    rsync -az --delete "${KK_TEST_DIR}/" "${user}@${ip}:${remote_kk_test}/"

    local cmd
    cmd=$(build_docker_cmd "$process_id" "$ib" "$remote_kk_test")
    echo "[worker ${process_id}] ssh ${user}@${ip}"
    ssh "${user}@${ip}" "$cmd" &
}

for i in $(seq 0 $((NUM_WORKERS - 1))); do
    run_on_worker "$i"
done

# ---------- 本地启动 header（process 0，前台） ----------
HEADER_CMD=$(build_docker_cmd 0 "${HEADER_IB}" "${KK_TEST_DIR}")
echo "[header] Running locally as process 0 ..."
echo "$HEADER_CMD"
eval "$HEADER_CMD"

wait
echo "All processes finished."
