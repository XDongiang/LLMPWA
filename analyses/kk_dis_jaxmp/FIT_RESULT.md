# 完整拟合结果（4 × V100，事例轴真分片）

> 2026-10-19，作业 **3728767**，`gpu` 分区单节点 4 卡，耗时 **1 分 34 秒**。

## 结果

```json
{
    "success": true,
    "status": 0,
    "nll": -3089.00736641229,
    "nit": 55,
    "message": "Optimization terminated successfully."
}
```

**收敛成功**：Newton-CG 迭代 55 次，`xtol=1e-6` 下正常终止（不是迭代上限）。

## 运行配置

| 项 | 值 |
|---|---|
| 数据 | `n_data=23108`（截断）、`n_mc=925000`、`n_truth=150000` |
| 进程/设备 | 4 进程 × 1 卡 = 4 设备 |
| 分片 | 每卡 **5777** 事例（23108 / 4），**真分片非复制** |
| `KK_DATA_CAP` | `23108` |
| `KKDIS_MAXITER` | `0`（不限） |
| `KKDIS_XTOL` | `1e-6` |
| 输出 | `output/fit_full/` |

### 关于 23108

原始实验数据是 **23109** 事例，不能被 4 整除（23109 = 4×5777 + 1）。
按 README 说明，不能整除时该数组会退化为**复制到每张卡**（结果仍正确，
只是不省显存）。为让分片真正生效，加了 `KK_DATA_CAP` 选项取前 23108 个。

已用作业 3728764 直接检查过每卡的 shard 形状确认：

```
data_phi_kk  global=(23108,)  shards=[(5777,)]   <- 每卡 5777，真分片
```

> 代价：丢掉最后 1 个事例。相对 23109 的统计影响可忽略（~4e-5），
> 换来的是每卡显存从"整份副本"降到 1/4。

## 收敛轨迹

| 迭代 | nll |
|---|---|
| 1 | 282.0547 |
| 5 | -1930.85 |
| 10 | -2868.21 |
| 20 | -3070.47 |
| 30 | -3077.43 |
| 40 | -3085.09 |
| 50 | -3088.83 |
| **55** | **-3089.0074** |

单调下降，后期步长收敛，符合正常 Newton-CG 行为。完整日志见
[fit_full_run.log](results/fit_full/fit_full_run.log)。

## 结果健全性检查

| 检查 | 结果 |
|---|---|
| 92 个自由参数全为有限值 | ✅ |
| 92 个误差全为有限值且 > 0 | ✅ |
| 完整参数向量 219 槽 | ✅ |
| 误差量级 | 5.7e-4 ~ 2.4e-2（中位 2.0e-3） |
| 参数值范围 | -0.0620 ~ 0.2534 |
| 6 个输出文件齐全 | ✅ |

前 8 个自由参数（值 ± 误差）：

```
[ 0]     0.047077  +- 0.002134
[ 1]     0.081600  +- 0.002298
[ 2]    -0.015870  +- 0.007495
[ 3]    -0.016242  +- 0.006379
[ 4]    -0.008764  +- 0.004728
[ 5]    -0.016414  +- 0.007890
[ 6]    -0.009343  +- 0.007658
[ 7]     0.004007  +- 0.005631
```

## 产物

| 文件 | 内容 |
|---|---|
| [result_info.json](results/fit_full/result_info.json) | success / status / nll / nit / message |
| [fit_result_values.npy](results/fit_full/fit_result_values.npy) | 92 个自由参数拟合值 |
| [fit_result_errors.npy](results/fit_full/fit_result_errors.npy) | 92 个误差（Hessian 对角开方） |
| [fit_result_kk.npz](results/fit_full/fit_result_kk.npz) | `fvalue=` / `ferror=` |
| [fit_result_full_args.npy](results/fit_full/fit_result_full_args.npy) | 写回 219 槽的完整参数向量 |
| [pwa_info_kk.json](results/fit_full/pwa_info_kk.json) | 道/参数分类 + 自由参数值 |

远程完整目录：`/scratch/caihao1/hyx/kk_dis_jaxmp/output/fit_full/`

## 复现命令

```bash
cd /scratch/caihao1/hyx/kk_dis_jaxmp
sbatch run/fit_full.sbatch
```

`run/fit_full.sbatch` 已固定 `KK_DATA_CAP=23108` / `KKDIS_OUTDIR=fit_full`。
