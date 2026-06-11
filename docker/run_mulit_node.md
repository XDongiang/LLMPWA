## 多节点jax运行设置，我们使用docker进行部署

Header 节点
```
docker run --rm --privileged --network=host \
  -e NCCL_IGNORE_CPU_AFFINITY=1 \
  -e NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0 \
  -e NCCL_IB_GID_INDEX=3 \
  -e JAX_COORDINATOR_ADDRESS=192.168.1.10:1234 \
  -e JAX_NUM_PROCESSES=2 \
  -e JAX_PROCESS_ID=0 \
  jax-nccl:latest python3 your_script.py
```

Worker 节点（192.168.1.11）：
```
docker run --rm --privileged --network=host \
  -e NCCL_IGNORE_CPU_AFFINITY=1 \
  -e NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0 \
  -e NCCL_IB_GID_INDEX=3 \
  -e JAX_COORDINATOR_ADDRESS=192.168.1.10:1234 \
  -e JAX_NUM_PROCESSES=2 \
  -e JAX_PROCESS_ID=1 \
  jax-nccl:latest python3 your_script.py
```

## 多节点jax

node_config.toml --- ./run.sh --- docker run python fit_script.py