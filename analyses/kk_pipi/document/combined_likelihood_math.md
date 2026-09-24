# 双道（KK̄ + ππ）联合似然：从 `combined_likelihood` 到 `calculate/component` 的数学过程

> 本文是 kk_pipi 联合拟合似然的**唯一公式来源**，经 `llm_config_combine.toml` 注入各似然相关 stage。
> 与 kk_dis（单道）的差别：存在**两条道**，各自有自己的 data/MC 项，共享一个联合参数向量 `total_args`。

## 1. 总 likelihood

代码中：

```python
total_likelihood(args, jax_data)
    = combined_likelihood_kk(args_kk, jax_data_kk)
      + combined_likelihood_pipi(args_pipi, jax_data_pipi)
```

数学上写为：

$$
\mathcal{L}_{\rm total}(\theta)
=
\mathcal{L}_{\rm comb}^{kk}(\theta_{kk})
+
\mathcal{L}_{\rm comb}^{pipi}(\theta_{pipi})
$$

其中：

- \(\theta\)：所有拟合参数（`total_args`，两道的并集）；
- \(\theta_{kk}\) / \(\theta_{pipi}\)：各道参数子集（由 `split_args` 从 `total_args` 切出）；
- \(\mathcal{L}_{\rm comb}^{X}\)：道 \(X\) 的负对数似然。

## 2. 每道的 data 部分

对道 \(X \in \{kk, pipi\}\)，其 data 负对数似然：

$$
\mathcal{L}_{\rm data}^{X}(\theta_X)
=
-
\sum_{e\in{\rm data}_X}
\log I_e^X(\theta_X)
+
C^X_{\rm penalty}(\theta_X)
$$

其中事件强度：

$$
I_e^X(\theta_X)
=
\sum_k
\left|
\mathcal{A}_{e k}^X(\theta_X)
\right|^2
$$

约束项来自 `data_step_function_X`：

$$
C^X_{\rm penalty}(\theta_X)
=
\lambda^X
\left(
f_{\rm total}^X - f_{\rm target}^X
\right)^2
+
\sum_r
\frac{
\left(x_r-y_{r,0}\right)^2
}{
2\sigma_r^2
}
$$

（每道有自己的 `total_frac` / `lambda_tfc`，见 `resonances_config_*.toml` 的 `[fit]`。）

## 3. 每道的 MC 归一化项

$$
\mathcal{N}_{\rm MC}^X(\theta_X)
=
\frac{1}{N_{\rm MC}^X}
\sum_{e\in{\rm MC}_X}
\sum_k
\left|
\mathcal{A}^{\rm MC}_{e k}{}^X(\theta_X)
\right|^2
$$

因此每道总目标函数：

$$
\boxed{
\mathcal{L}_{\rm comb}^X(\theta_X)
=
-
\sum_{e\in{\rm data}_X}
\log I_e^X(\theta_X)
+
C^X_{\rm penalty}(\theta_X)
+
N_{\rm data}^X
\log
\mathcal{N}_{\rm MC}^X(\theta_X)
}
$$

## 4. `calculate_*` 的通用振幅结构

设输入的角分布/动力学张量为：

$$
T_{i e k}
$$

其中 \(i\)：振幅基/helicity 分量；\(e\)：事件；\(k\)：末态张量分量。

复耦合常数由 `Amplitude_param_const` 与 `Amplitude_param_theta` 构造：

$$
c_{\ell i}
=
a_{\ell i}
+
i\phi_{\ell i}
$$

先做耦合加权：

$$
B_{\ell e k}
=
\sum_i
T_{i e k}
c_{\ell i}
$$

再乘传播子 \(P_{\ell e}\)。多子态 \(\ell\) 相干求和：

$$
\mathcal{A}_{e k}
=
\sum_\ell
B_{\ell e k}
P_{\ell e}
$$

`component_*` 不对 \(\ell\) 求和，保留每个分量（用于 fraction 计算）。

## 5. 传播子

（与 kk_dis 相同，见 `document/combined_likelihood_math.md` 的 §5——BW / BW_flatte980 / BW_BW / BW_flatte1270。
双方各自使用自己的 Sbc：KK̄ 道用 `phi_kk` / `f_kk`，ππ 道用 `phi_pipi` / `f_pipi`。）

## 6. 跨通道共享参数

`resonances_config_ctrl.toml` 规定某些共振的质量/宽度在两通道间**共享**：
被绑定的参数（`binding.resonances.<name>.param[*]` 的 `usevalue`）在两道的 `free_params` 中
引用**同一个**联合参数槽位，从而在一次拟合中同时约束两道的该参数。
`generate_args` stage 据此合成联合参数向量并生成索引映射。

## 7. 分布式求和含义

事件轴被 shard 到多个设备/进程。公式中的 \(\sum_e\) 和 \(\frac{1}{N}\sum_e\) 分别对应
`jnp.sum` 和 `jnp.mean`，JAX/XLA 自动插入跨设备 all-reduce，数学上仍等价于全数据集求和/平均。
