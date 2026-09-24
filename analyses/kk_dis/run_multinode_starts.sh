#!/usr/bin/env bash
# kk_dis 跨节点(HEP1 header + HEP3 worker) 多起点随机初值稳定性测试驱动。
#
# 用法（在 HEP1 上）:
#   bash run_multinode_starts.sh --starts 50 --seed 2024 --tag ms50 --timeout-min 40
#
# 对每个起点:
#   1. python 生成扰动初值 -> workdir/ms_x0/start_N.npy
#   2. 从 node_config.toml 模板生成 node_config_ms_N.toml（env 注入
#      KKDIS_X0_NPY=/workspace/work/ms_x0/start_N.npy, KKDIS_OUTDIR=ms_<tag>/start_N）
#   3. 后台采样 HEP1 GPU0 与 HEP3 GPU0 的 utilization（5s 间隔）
#   4. 调用 run_fit_multinode.sh（rsync workdir + 启动两机容器 + wait）
#   5. 解析 Fit finished 行 -> success/NLL/iterations；统计 GPU util 均值与墙钟
# 汇总: workdir/output/ms_<tag>/summary.toml
set -uo pipefail

WORKDIR="$HOME/LLMPWA/kk_dis"
CFG_TEMPLATE="$WORKDIR/node_config.toml"
# HEP1 宿主 python3 无 numpy；x0/config/summary 均用 docker 内 /opt/env/bin/python
PY="docker run --rm -i -v ${WORKDIR}:/workspace/work -w /workspace/work jax-fit:latest /opt/env/bin/python"
STARTS=50
SEED=2024
TAG=""
TIMEOUT_MIN=40
GPU_SAMPLE_INTERVAL=5
SSH="ssh -F /dev/null -o BatchMode=yes -o ConnectTimeout=25 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/tmp/dsh_probe_known_hosts"
WORKER_IP="192.168.200.171"

usage() {
    echo "usage: $0 --starts N --seed S --tag T [--timeout-min M]"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --starts) STARTS="$2"; shift 2;;
        --seed) SEED="$2"; shift 2;;
        --tag) TAG="$2"; shift 2;;
        --timeout-min) TIMEOUT_MIN="$2"; shift 2;;
        *) usage;;
    esac
done
[[ -n "$TAG" ]] || usage

cd "$WORKDIR" || exit 1
mkdir -p "ms_x0" "output/ms_${TAG}"

RESULTS="output/ms_${TAG}/summary.toml"
: > "$RESULTS"
echo "# kk_dis cross-node multistart (HEP1 header + HEP3 worker, 1Gbps ethernet)" > "$RESULTS"
echo "tag = \"$TAG\"" >> "$RESULTS"
echo "seed = $SEED" >> "$RESULTS"
echo "starts = $STARTS" >> "$RESULTS"
echo "[[results]]" > /tmp/ms_results_toml.tmp
: > /tmp/ms_results_toml.tmp

# ---------- 1. 生成 50 个扰动初值 ----------
# 容器的 cwd=/workspace/work（挂载自 $WORKDIR），用相对路径读写
$PY - "$STARTS" "$SEED" <<'PYEOF'
import sys, toml, numpy as onp
starts, seed = int(sys.argv[1]), int(sys.argv[2])

def load_free_params():
    with open("run/free_params.toml", "r") as f:
        data = toml.load(f)["data"]
    data = sorted(data, key=lambda d: d["arg_index"])
    return onp.array([d["value"] for d in data], dtype=onp.float64)

def find_const_theta_indices(paths):
    const_idx = [i for i, p in enumerate(paths) if ".Amplitude.const" in p]
    theta_idx = [i for i, p in enumerate(paths) if ".Amplitude.theta" in p]
    return onp.array(const_idx, dtype=int), onp.array(theta_idx, dtype=int)

def perturb_args(base_args, const_idx, theta_idx, seed, disturb=50.0):
    rng = onp.random.RandomState(seed)
    args = onp.array(base_args, dtype=float).copy()
    if const_idx.size > 0:
        theta = 2 * onp.pi * rng.rand(theta_idx.shape[0])
        args[theta_idx] = 0.1 * onp.cos(theta)
        args[const_idx] = 0.1 * onp.sin(theta)
    args = args * ((rng.rand(args.shape[0]) - 0.5) / disturb + 1.0)
    return args

values = load_free_params()
data = toml.load("run/free_params.toml")["data"]
paths = [d["path"] for d in sorted(data, key=lambda d: d["arg_index"])]
const_idx, theta_idx = find_const_theta_indices(paths)
for s in range(starts):
    if s == 0:
        x0 = values
    else:
        x0 = perturb_args(values, const_idx, theta_idx, seed + s * 1000)
    onp.save(f"ms_x0/start_{s}.npy", x0)
print(f"generated {starts} x0 files", flush=True)
PYEOF
echo "x0 generation done"

# ---------- 2. 逐起点运行 ----------
for s in $(seq 0 $((STARTS - 1))); do
    echo "=== start ${s}/${STARTS} (seed $((SEED + s * 1000))) ==="

    # 生成该起点的 node_config（env 注入 x0 与 outdir）；相对路径（容器 cwd=/workspace/work）
    $PY - "node_config.toml" "node_config_ms_${s}.toml" "$s" "$TAG" <<'PYEOF'
import sys, tomllib
src, dst, s, tag = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
with open(src, "rb") as f:
    cfg = tomllib.load(f)
env = list(cfg.get("container", {}).get("env", []))
env.append(f"KKDIS_X0_NPY=/workspace/work/ms_x0/start_{s}.npy")
env.append(f"KKDIS_OUTDIR=ms_{tag}/start_{s}")
env.append("KKDIS_MAXITER=400")
cfg["container"]["env"] = env
with open(dst, "w") as f:
    import toml as _t
    _t.dump(cfg, f)
print("wrote", dst, flush=True)
PYEOF

    # 后台采样两机 GPU utilization
    nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits \
        -i 0 -lms 5000 > /tmp/ms_gpu_hep1_${s}.txt 2>/dev/null &
    GPUPID_H1=$!
    $SSH hyx@${WORKER_IP} \
        "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0 -lms 5000 > /tmp/ms_gpu_hep3_${s}.txt 2>/dev/null" 2>/dev/null &
    GPUPID_H3=$!

    T0=$(date +%s)
    # 调用 run_fit_multinode.sh（内部 rsync + 两机 docker + wait）
    bash run_fit_multinode.sh --workdir "$WORKDIR" --config "$WORKDIR/node_config_ms_${s}.toml" \
        > /tmp/ms_run_${s}.log 2>&1
    RC=$?
    T1=$(date +%s)
    ELAPSED=$((T1 - T0))

    kill "$GPUPID_H1" >/dev/null 2>&1
    $SSH hyx@${WORKER_IP} "pkill -f 'ms_gpu_hep3_${s}' 2>/dev/null" >/dev/null 2>&1

    # 解析 Fit finished
    FIN=$(grep -aE "Fit finished: success=" /tmp/ms_run_${s}.log | tail -1)
    SUCCESS="false"
    NLL="null"
    ITERS="null"
    if [[ -n "$FIN" ]]; then
        SUCCESS=$(echo "$FIN" | sed -n 's/.*success=\([A-Za-z]*\).*/\1/p')
        NLL=$(echo "$FIN" | sed -n 's/.*likelihood=\([-\.[:digit:]]*[eE]*-*[[:digit:]]*\).*/\1/p')
        ITERS=$(echo "$FIN" | sed -n 's/.*iterations=\([[:digit:]]*\).*/\1/p')
    fi

    # GPU util 均值（HEP1 本地 + HEP3 ssh）
    GPUH1=$(awk '{s+=$1;n++} END{if(n>0) printf "%.1f", s/n; else print "null"}' /tmp/ms_gpu_hep1_${s}.txt 2>/dev/null)
    GPUH3=$(ssh -F /dev/null -o BatchMode=yes -o ConnectTimeout=25 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/tmp/dsh_probe_known_hosts hyx@${WORKER_IP} "awk '{s+=\$1;n++} END{if(n>0) printf \"%.1f\", s/n; else print \"null\"}' /tmp/ms_gpu_hep3_${s}.txt 2>/dev/null" 2>/dev/null)

    echo "    rc=${RC} ok=${SUCCESS} nll=${NLL} iter=${ITERS} ${ELAPSED}s gpu_h1=${GPUH1}% gpu_h3=${GPUH3}%"
    $PY - "output/ms_${TAG}/summary.toml" "$s" "$SEED" "$RC" "$SUCCESS" "$NLL" "$ITERS" "$ELAPSED" "$GPUH1" "$GPUH3" <<'PYEOF'
import sys, toml
summary, s, seed, rc, ok, nll, iters, elapsed, g1, g3 = (
    sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5],
    sys.argv[6], sys.argv[7], sys.argv[8], sys.argv[9], sys.argv[10])
prev = toml.load(summary) if open(summary).read().strip() else {}
results = prev.get("results", [])
results.append({
    "start": s, "seed": seed + s * 1000, "returncode": rc,
    "success": ok == "True", "nll": float(nll) if nll not in ("null", "") else None,
    "iterations": int(iters) if iters not in ("null", "") else None,
    "elapsed_s": float(elapsed), "gpu_util_h1": g1, "gpu_util_h3": g3,
})
prev["results"] = results
with open(summary, "w") as f:
    toml.dump(prev, f)
PYEOF
done

# ---------- 汇总 ----------
$PY - "output/ms_${TAG}/summary.toml" <<'PYEOF'
import sys, statistics, toml
summary = sys.argv[1]
data = toml.load(summary)
res = data.get("results", [])
ok = [r for r in res if r.get("success")]
print("\n========== SUMMARY ==========")
print(f"starts: {len(res)} | success: {len(ok)} ({100.0*len(ok)/max(len(res),1):.1f}%)")
if ok:
    nlls = [r["nll"] for r in ok if r.get("nll") is not None]
    if nlls:
        print(f"nll: min={min(nlls):.6f} max={max(nlls):.6f} mean={statistics.mean(nlls):.6f}")
    times = [r["elapsed_s"] for r in ok]
    print(f"wall: mean={statistics.mean(times):.1f}s min={min(times):.1f}s max={max(times):.1f}s")
    g1 = [float(r["gpu_util_h1"]) for r in ok if r.get("gpu_util_h1") not in (None, "null", "")]
    g3 = [float(r["gpu_util_h3"]) for r in ok if r.get("gpu_util_h3") not in (None, "null", "")]
    if g1 and g3:
        print(f"gpu util h1(coord): mean={statistics.mean(g1):.1f}% | h3(worker): mean={statistics.mean(g3):.1f}%")
PYEOF
