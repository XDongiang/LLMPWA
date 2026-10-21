### 这是一份**对照 spec**，不是生产链

`ex5_kst2_paper` 与 [`ex5_kst2`](ex5_kst2.md) 共用同一条链目录
（`agent/tensor/chains/kstpi/`）与同一套拓扑，**唯一区别是道名**：

| 道名 | 取哪个文件 | 形式 | A1 |
|---|---|---|---|
| `u_PP_1` / `u_PP_2` / `u_FP_1` / `u_FP_2` | `u_*.py` | **修正版（生产）** | ✅ $\sim10^{-14}$ |
| `u_PP_1_paper` / … | `u_*_paper.py` | 历史形式（`FUNC.py` 逐式移植） | ❌ $10^{-2}\sim10^{-1}$ |

`emitter` 按 **`channel.name`** 去 `chains/<chain_name>/<channel.name>.py`
取函数体，所以"生成论文形式"**不需要改任何代码** —— 只要把 spec 里的
道名写成 `*_paper`。本 spec 把两种形式放进**同一份产物**，便于逐道对照。

### 为什么这份 spec 的 A1 必然失败

4 条 `*_paper` 道**就是缺陷本身**（内层无迹化用错参考系）。
所以：

* `agent/tensor/tests/test_handwritten_complete.py` 把它列入
  `A1_SKIP_SPECS`（整份跳过 A1，只验 L1/L2/L3）；
* `analyses/tensor_examples/probes/verify_all_specs.py` 也跳过它。

**生产请用 `ex5_kst2`**（`u_PP_*`/`u_FP_*` 已改为修正版）。

### 生成与对照

```bash
cd <LLMPWA>

# 生成两种形式共存的产物
python -m agent.tensor check analyses/tensor_examples/specs/ex5_kst2_paper.toml
python -m agent.tensor emit  analyses/tensor_examples/specs/ex5_kst2_paper.toml

# 三形式对拍：历史形式（应失败）vs 修正版（应到机器精度）
python analyses/tensor_examples/probes/probe_ex5_paper_form.py
```

### 机制与公式

完整推导、公式对照、代码逐行对照见
[`agent/tensor/chains/kstpi/CORRECTION.md`](../../../agent/tensor/chains/kstpi/CORRECTION.md)。

### 相关

* [链 `ex5_kst2`](ex5_kst2.md) —— 生产链（修正版）
* [链 `ex5_kst0`](ex5_kst0.md) —— 同一移植的 $J=0$ 部分
* [链索引](README.md)
