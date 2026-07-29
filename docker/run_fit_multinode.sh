#!/usr/bin/env bash
# 在多节点 docker 上运行 fit 脚本。
#
# 用法:
#   ./docker/run_fit_multinode.sh --workdir analyses/kk_dis
#   ./docker/run_fit_multinode.sh --workdir analyses/kk_dis --config analyses/kk_dis/node_config.toml
#   ./docker/run_fit_multinode.sh --workdir analyses/kk_dis analyses/kk_dis/node_config.toml
#
# workdir 会被 rsync 同步到每个节点（含 header），并挂载到容器 /workspace/work。
# 运行完成后，header 节点的 output 目录会拷贝回本地 workdir/output。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

WORKDIR=""
CONFIG=""

usage() {
    cat <<'EOF'
Usage:
  ./docker/run_fit_multinode.sh --workdir <dir> [--config <node_config.toml>]
  ./docker/run_fit_multinode.sh --workdir <dir> [node_config.toml]

Options:
  --workdir DIR   分析工作目录（会同步到各节点并挂载到容器）
  --config FILE   节点配置 TOML；默认优先使用 <workdir>/node_config.toml，
                  否则使用 docker/node_config_example.toml
  -h, --help      显示帮助
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --workdir)
            [[ $# -ge 2 ]] || { echo "error: --workdir requires a path" >&2; exit 1; }
            WORKDIR="$2"
            shift 2
            ;;
        --config)
            [[ $# -ge 2 ]] || { echo "error: --config requires a path" >&2; exit 1; }
            CONFIG="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo "error: unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
        *)
            # 兼容旧用法：位置参数作为 config
            if [[ -z "$CONFIG" ]]; then
                CONFIG="$1"
            else
                echo "error: unexpected argument: $1" >&2
                usage >&2
                exit 1
            fi
            shift
            ;;
    esac
done

if [[ -z "$WORKDIR" ]]; then
    # 兼容旧默认：docker/kk_test
    WORKDIR="${SCRIPT_DIR}/kk_test"
fi

# 相对路径相对当前工作目录解析；解析失败时再尝试相对仓库根
resolve_path() {
    local p="$1"
    local dir base absdir
    if [[ "$p" = /* ]]; then
        printf '%s\n' "$p"
        return
    fi
    dir="$(dirname "$p")"
    base="$(basename "$p")"
    if [[ -e "$p" || -d "$dir" ]]; then
        absdir="$(cd "$dir" && pwd)"
        printf '%s\n' "${absdir}/${base}"
        return
    fi
    if [[ -e "${REPO_ROOT}/$p" ]]; then
        absdir="$(cd "$(dirname "${REPO_ROOT}/$p")" && pwd)"
        printf '%s\n' "${absdir}/${base}"
        return
    fi
    # 仍返回绝对化路径，便于后续报错
    printf '%s\n' "$(pwd)/$p"
}

WORKDIR="$(resolve_path "$WORKDIR")"
if [[ ! -d "$WORKDIR" ]]; then
    echo "error: workdir does not exist: $WORKDIR" >&2
    exit 1
fi

if [[ -z "$CONFIG" ]]; then
    if [[ -f "${WORKDIR}/node_config.toml" ]]; then
        CONFIG="${WORKDIR}/node_config.toml"
    else
        CONFIG="${SCRIPT_DIR}/node_config_example.toml"
    fi
else
    CONFIG="$(resolve_path "$CONFIG")"
fi

if [[ ! -f "$CONFIG" ]]; then
    echo "error: config not found: $CONFIG" >&2
    exit 1
fi

WORKDIR_BASENAME="$(basename "$WORKDIR")"
REMOTE_WORKDIR="/tmp/${WORKDIR_BASENAME}"

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
print(f'CONTAINER_NAME={q(container.get("container", "kkfit"))}')
print(f'SCRIPT_REL={q(container.get("script", "run/fit_script.py"))}')
print(f'WORKERS_JSON={q(json.dumps(workers))}')

envs = container.get("env", [])
print(f'EXTRA_ENV_COUNT={len(envs)}')
for i, e in enumerate(envs):
    print(f'EXTRA_ENV_{i}={q(e)}')
PYEOF
)"
eval "$_TOML_OUT"

# 重建 EXTRA_ENVS
EXTRA_ENV_FLAGS=""
if [[ "${EXTRA_ENV_COUNT}" -gt 0 ]]; then
    for i in $(seq 0 $((EXTRA_ENV_COUNT - 1))); do
        varname="EXTRA_ENV_${i}"
        EXTRA_ENV_FLAGS+=" -e ${!varname}"
    done
fi

NUM_WORKERS=$(python3 -c "import json,sys; print(len(json.loads(sys.argv[1])))" "$WORKERS_JSON")
NUM_PROCESSES=$((NUM_WORKERS + 1))
COORDINATOR="${HEADER_IP}:${HEADER_PORT}"
CONTAINER_SCRIPT="/workspace/work/${SCRIPT_REL}"

if [[ ! -f "${WORKDIR}/${SCRIPT_REL}" ]]; then
    echo "error: script not found in workdir: ${WORKDIR}/${SCRIPT_REL}" >&2
    exit 1
fi

echo "============================================"
echo " Config      : $CONFIG"
echo " Workdir     : $WORKDIR"
echo " Image       : $IMAGE"
echo " Container   : $CONTAINER_NAME"
echo " Script      : $SCRIPT_REL"
echo " Header      : ${HEADER_USER}@${HEADER_IP}:${HEADER_PORT} (process 0)"
echo " Workers     : ${NUM_WORKERS}"
echo " Total procs : $NUM_PROCESSES"
echo " Remote dir  : $REMOTE_WORKDIR"
echo "============================================"

# ---------- 构建 docker run 命令字符串 ----------
build_docker_cmd() {
    local process_id="$1" ib_if="$2" remote_workdir="$3"
    echo "docker run --rm --gpus all --privileged --network=host" \
         "--name ${CONTAINER_NAME}" \
         "-e JAX_COORDINATOR_ADDRESS=${COORDINATOR}" \
         "-e JAX_NUM_PROCESSES=${NUM_PROCESSES}" \
         "-e JAX_PROCESS_ID=${process_id}" \
         "-e NCCL_IB_HCA=${ib_if}" \
         ${EXTRA_ENV_FLAGS} \
         "-v ${remote_workdir}:/workspace/work" \
         "-w /workspace/work" \
         "${IMAGE} python3 ${CONTAINER_SCRIPT}"
}

# ---------- worker 节点通过 SSH 执行（后台） ----------
run_on_worker() {
    local idx="$1"
    local ip ib user
    ip=$(python3   -c "import json,sys; print(json.loads(sys.argv[1])[${idx}]['IP'])"                        "$WORKERS_JSON")
    ib=$(python3   -c "import json,sys; w=json.loads(sys.argv[1]); print(w[${idx}].get('IB_IF','${HEADER_IB}'))" "$WORKERS_JSON")
    user=$(python3 -c "import json,sys; w=json.loads(sys.argv[1]); print(w[${idx}].get('Username','root'))"   "$WORKERS_JSON")
    local process_id=$((idx + 1))

    echo "[worker ${process_id}] rsync workdir -> ${user}@${ip}:${REMOTE_WORKDIR}"
    rsync -az --delete "${WORKDIR}/" "${user}@${ip}:${REMOTE_WORKDIR}/"

    local cmd
    cmd=$(build_docker_cmd "$process_id" "$ib" "$REMOTE_WORKDIR")
    echo "[worker ${process_id}] ssh ${user}@${ip}"
    ssh "${user}@${ip}" "$cmd" &
}

for i in $(seq 0 $((NUM_WORKERS - 1))); do
    run_on_worker "$i"
done

# ---------- header 节点（process 0） ----------
echo "[header] rsync workdir -> ${HEADER_USER}@${HEADER_IP}:${REMOTE_WORKDIR}"
rsync -az --delete "${WORKDIR}/" "${HEADER_USER}@${HEADER_IP}:${REMOTE_WORKDIR}/"

HEADER_CMD=$(build_docker_cmd 0 "${HEADER_IB}" "${REMOTE_WORKDIR}")
echo "[header] ssh ${HEADER_USER}@${HEADER_IP} (process 0)"
echo "$HEADER_CMD"
ssh "${HEADER_USER}@${HEADER_IP}" "$HEADER_CMD" &

wait
echo "All processes finished."

# ---------- 拷贝 header 节点的 output 回本地 ----------
echo "[header] rsync output <- ${HEADER_USER}@${HEADER_IP}:${REMOTE_WORKDIR}/output/"
mkdir -p "${WORKDIR}/output"
rsync -az "${HEADER_USER}@${HEADER_IP}:${REMOTE_WORKDIR}/output/" "${WORKDIR}/output/"
echo "Output copied to ${WORKDIR}/output"
