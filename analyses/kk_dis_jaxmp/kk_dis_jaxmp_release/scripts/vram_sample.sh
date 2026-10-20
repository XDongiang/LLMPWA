#!/bin/bash
# 显存采样器：后台按固定间隔记录每张卡的 used/total 显存。
# 用法: vram_sample.sh <输出csv> <间隔秒> <总时长秒>
# 输出: 每行 "elapsed_s,index,mem_used_mib,mem_total_mib,util_pct"
OUT="$1"; INTERVAL="${2:-5}"; DURATION="${3:-86400}"
START=$(date +%s)
: > "$OUT"
while :; do
    NOW=$(date +%s)
    EL=$((NOW - START))
    [ "$EL" -ge "$DURATION" ] && break
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader,nounits 2>/dev/null | \
    while IFS=',' read -r idx used total util; do
        echo "$EL,$(echo $idx|tr -d ' '),$(echo $used|tr -d ' '),$(echo $total|tr -d ' '),$(echo $util|tr -d ' ')" >> "$OUT"
    done
    sleep "$INTERVAL"
done
