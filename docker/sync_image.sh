#!/usr/bin/env bash
# Sync docker image from header to all workers defined in node_config.toml

set -euo pipefail

CONFIG="${1:-$(dirname "$0")/../analyses/kk_new/node_config.toml}"

if [[ ! -f "$CONFIG" ]]; then
    echo "Usage: $0 [node_config.toml]"
    echo "Config file not found: $CONFIG"
    exit 1
fi

# Parse toml helpers (pure bash, no deps)
parse_value() {
    local key="$1"
    grep -E "^${key}\s*=" "$CONFIG" | head -1 | sed 's/.*=\s*"\(.*\)"/\1/' | tr -d ' '
}

# Read header info
HEADER_IP=$(awk '/^\[header\]/{f=1} f && /^IP/{gsub(/.*= *"|".*/, ""); print; exit}' "$CONFIG")
HEADER_USER=$(awk '/^\[header\]/{f=1} f && /^Username/{gsub(/.*= *"|".*/, ""); print; exit}' "$CONFIG")

# Read container image
IMAGE=$(awk '/^\[container\]/{f=1} f && /^image/{gsub(/.*= *"|".*/, ""); print; exit}' "$CONFIG")

# Read all worker IPs and usernames
mapfile -t WORKER_IPS < <(awk '/^\[\[worker\]\]/{f=1; next} f && /^IP/{gsub(/.*= *"|".*/, ""); print; f=0}' "$CONFIG")
mapfile -t WORKER_USERS < <(awk '/^\[\[worker\]\]/{f=1; next} f && /^Username/{gsub(/.*= *"|".*/, ""); print; f=0}' "$CONFIG")

if [[ -z "$HEADER_IP" || -z "$IMAGE" ]]; then
    echo "Failed to parse header IP or image from $CONFIG"
    exit 1
fi

echo "Header  : ${HEADER_USER}@${HEADER_IP}"
echo "Image   : ${IMAGE}"
echo "Workers : ${#WORKER_IPS[@]}"
echo ""

# Save image on header, pipe via ssh to load on worker
sync_to_worker() {
    local idx="$1"
    local worker_ip="${WORKER_IPS[$idx]}"
    local worker_user="${WORKER_USERS[$idx]:-nvidia}"

    echo "[$(date +%H:%M:%S)] Syncing to ${worker_user}@${worker_ip} ..."

    # docker save on header | ssh pipe | docker load on worker
    ssh "${HEADER_USER}@${HEADER_IP}" "docker save '${IMAGE}'" \
        | ssh "${worker_user}@${worker_ip}" "docker load"

    echo "[$(date +%H:%M:%S)] Done: ${worker_ip}"
}

FAILED=0
for i in "${!WORKER_IPS[@]}"; do
    sync_to_worker "$i" || { echo "ERROR: failed for ${WORKER_IPS[$i]}"; FAILED=$((FAILED+1)); }
done

if [[ $FAILED -gt 0 ]]; then
    echo ""
    echo "Sync completed with $FAILED error(s)."
    exit 1
else
    echo ""
    echo "All workers synced successfully."
fi
