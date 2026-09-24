# kk_pipi：LLM 驱动的双道（KK̄ + ππ）联合分波分析拟合代码生成

`analyses/kk_pipi` 是 LLMPWA 中一套**双道联合拟合**示例：KK̄ 道与 ππ 道各自拥有共振态配置，
通过 `resonances_config_ctrl.toml` 绑定跨通道共享参数（如某些共振的质量/宽度），
再经多阶段流水线生成 **JAX 事件级数据并行 + SciPy Newton-CG** 的联合拟合代码。

配置入口：

```text
analyses/kk_pipi/llm_config_combine.toml
```

运行产物（生成代码）：

```text
analyses/kk_pipi/run/fit_script.py
```

---

## 1. 项目定位

与单道分析（如 kk_dis）不同，kk_pipi 处理 **同一物理过程的两条衰变道**（KK̄ 与 ππ），
其中部分共振态在两个通道共享（例如 `phif0_980`、`phif0_1710` 的质量/宽度在两通道取值相同）。
联合拟合的关键是：

1. **双道各自声明共振态**：`resonances_config_kk.toml`、`resonances_config_pipi.toml`。
2. **共享参数绑定**：`resonances_config_ctrl.toml` 规定哪些共振参数在两通道间共享（绑定到哪个通道取值）。
3. **统一参数空间**：`generate_args` stage 把两道的自由参数合成为**一个联合参数向量**，
   并为每道生成索引映射，使 `total_args` 同时驱动两道的似然。
4. **联合似然**：总似然 = KK̄ 道似然 + ππ 道似然（各含自己的 data/MC 项与约束）。

---

## 2. 总体架构

```text
resonances_config_kk.toml  +  resonances_config_pipi.toml  +  resonances_config_ctrl.toml
        │                              │                              │
        ▼                              ▼                              ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  agent Engine 驱动 llm_config_combine.toml 各 stage (python | llm | agent)     │
└──────────────────────────────────────────────────────────────────────────────┘
        │
        ├─► combine_check        校验共享参数在两配置均存在
        ├─► config_strip_kk/pipi 各自剥离自由参数 → run/free_params_*.toml
        ├─► generate_args        合成联合参数 total_args + 各道索引映射
        ├─► split_args           生成 split_args(total_args) 代码
        ├─► classification_*/     各道传播子分类
        ├─► likelihood_*         各道似然构造
        └─► assemble_final_code  组装 run/fit_script.py
                    │
                    ▼
        多节点 Docker / JAX Mesh (event 轴)
        SciPy Newton-CG + JIT NLL / Grad / HVP
                    │
                    ▼
        output/fit/{values,errors,free_params_fitted.toml}
```

### 联合拟合计算架构要点

- **事件轴并行**：各 GPU/进程持有各自道的 event shard。
- **联合参数向量**：`total_args`（小向量）复制到所有设备。
- **总似然**：`total_likelihood = combined_likelihood_kk + combined_likelihood_pipi`。
- **优化**：每进程跑**同一** SciPy Newton-CG 循环，collective 顺序一致。
- **I/O**：仅 `process_id == 0` 写日志与结果。

---

## 3. 生成流水线（`llm_config_combine.toml`）

引擎：`python -m agent.cli --workdir analyses/kk_pipi --config llm_config_combine.toml`
阶段按配置顺序执行；中间结果用 `<<stages.xxx.yyy>>` 引用。

| stage | kind | 产物 |
|-------|------|------|
| `combine_check` | python | 校验共享参数 |
| `config_strip_kk` / `config_strip_pipi` | python | `run/free_params_*.toml` |
| `generate_args` | python | `run/free_params_total.toml` + 索引映射 |
| `split_args` | python | `gen/fragments/split_args.py` |
| `make_initial_args` / `save_results` | python | 初值/存盘辅助函数 |
| `classification_kk` / `classification_pipi` | llm | 各道传播子分类 |
| `data_load` | llm | 数据加载函数 |
| `resonance_calculation` | llm | 共振态振幅计算 |
| `extract_parameters_*` | llm | 参数抽取 |
| `step_function_*` / `likelihood_function_*` | llm | 各道似然 |
| `run_load_data` / `main_section` | llm | 加载与主函数 |
| `assemble_final_code` | python | `run/fit_script.py` |

---

## 4. 目录结构

```text
analyses/kk_pipi/
├── llm_config_combine.toml     # 生成流水线定义（本流程入口）
├── resonances_config_kk.toml   # KK̄ 道共振与振幅物理配置
├── resonances_config_pipi.toml # ππ 道共振与振幅物理配置
├── resonances_config_ctrl.toml # 跨通道共享参数绑定
├── node_config.toml            # 多节点 Docker 节点表
├── README.md                   # 本规范（agent 约束）
├── config/                     # 日志等运行配置
├── data/                       # real_data/mc_truth/draw/weight（只读）
├── gen/                        # prompts / handlers / templates / fragments / llm_logs
├── run/                        # 生成/手写可执行代码
├── output/fit/                 # 拟合结果
├── task/                       # 任务记录
└── .dsh/ .venv/ logs/ tmp/ document/
```

---

## 5. 如何使用

### 5.1 生成代码

```bash
# 静态检查
python -m agent.cli --workdir analyses/kk_pipi --config llm_config_combine.toml --check-only

# 跑完整流水线
python -m agent.cli --workdir analyses/kk_pipi --config llm_config_combine.toml

# 只重跑某一 stage
python -m agent.cli --workdir analyses/kk_pipi --config llm_config_combine.toml --stage generate_args
```

### 5.2 单机试跑

```bash
cd analyses/kk_pipi
python run/fit_script.py
```

### 5.3 多节点 Docker 拟合

```bash
./docker/run_fit_multinode.sh --workdir analyses/kk_pipi
```

---

## 6. 与仓库其他文档的关系

| 文档 | 关系 |
|------|------|
| 根目录 `README.md` | LLMPWA 总览 |
| `documentation/README_CN.md` | 中文项目说明与环境安装 |
| `docker/design.md` | 事件并行 + Newton-CG/HVP 设计原文 |
| 本文 `analyses/kk_pipi/README.md` | **本分析的操作约束标准** |
