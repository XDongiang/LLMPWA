"""
多节点数据并行 PWA 拟合脚本（somax Newton-CG 版本，随机初值多次重启扫描）

架构：与 fit_script_distributed_somax.py 完全一致的分布式 event-shard + somax
Newton-CG 优化，唯一区别是：
- 每次试验先对初始参数做随机偏移（const/theta 重新采样到半径 0.1 的圆上，
  再对全部参数施加一个统一的乘性抖动），然后从该偏移点完整跑一次 Newton-CG
  优化直至收敛
- 记录每次试验的初始/最终 likelihood、迭代次数、是否收敛、耗时
- 所有进程使用同一个 seed（base_seed + trial_idx）生成偏移，因此各进程算出的
  偏移值完全一致，不需要额外通信
- 结果保存到 output/fit/random_restart_trace.npz；likelihood 最优的一组参数
  额外保存为 output/fit/free_params_fitted_best_restart.toml
"""
import json
import logging
import logging.config
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import time
import numpy as onp
from functools import partial

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, jvp, device_put
from jax import config
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

import somax

import toml
import sys

foo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(foo_path)
sys.path.append(foo_path)


# =============================================================================
# 分布式初始化
# =============================================================================

def init_distributed():
    """通过环境变量初始化 JAX 多节点通信"""
    coordinator = os.environ.get("JAX_COORDINATOR_ADDRESS", "")
    num_processes = int(os.environ.get("JAX_NUM_PROCESSES", 1))
    process_id = int(os.environ.get("JAX_PROCESS_ID", 0))

    if num_processes > 1:
        jax.distributed.initialize(
            coordinator_address=coordinator,
            num_processes=num_processes,
            process_id=process_id,
        )

    return num_processes, process_id


def build_mesh():
    """构建设备网格，所有全局设备组成一个 'event' 轴"""
    devices = jax.devices()
    mesh = Mesh(devices, axis_names=("event",))
    return mesh


# =============================================================================
# dplex 复数运算工具（实/虚部纵向叠加，axis=0）
# =============================================================================

def dplex_deinsum(subscript, aa, bb):
    real = jnp.einsum(subscript, aa[0], bb[0]) - jnp.einsum(subscript, aa[1], bb[1])
    imag = jnp.einsum(subscript, aa[0], bb[1]) + jnp.einsum(subscript, aa[1], bb[0])
    return jnp.stack([real, imag], axis=0)


def dplex_deinsum_ord(subscript, aa, bb):
    real = jnp.einsum(subscript, aa, bb[0])
    imag = jnp.einsum(subscript, aa, bb[1])
    return jnp.stack([real, imag], axis=0)


def dplex_dabs(aa):
    return aa[0]**2 + aa[1]**2


def dplex_dtomine(aa):
    return jnp.stack([jnp.real(aa), jnp.imag(aa)], axis=0)


def dplex_dconstruct(aa, bb):
    return jnp.stack([aa, bb], axis=0)


def dplex_ddivide(a, bb):
    real = a * bb[0] / dplex_dabs(bb)
    imag = -a * bb[1] / dplex_dabs(bb)
    return jnp.stack([real, imag], axis=0)


# =============================================================================
# 共振形状函数（与单节点版本完全一致）
# =============================================================================

def BW(m_, w_, Sbc):
    l = (Sbc.shape)[0]
    temp = dplex_dconstruct(m_*m_ - Sbc, -m_*w_*jnp.ones(l))
    return dplex_ddivide(1.0, temp)


def BW_relativity(m_, w_, Sbc):
    gamma = jnp.sqrt(m_*m_*(m_*m_+w_*w_))
    k = jnp.sqrt(2*jnp.sqrt(2)*m_*jnp.abs(w_)*gamma/jnp.pi/jnp.sqrt(m_*m_+gamma))
    l = (Sbc.shape)[0]
    temp = dplex_dconstruct(m_*m_ - Sbc, -m_*w_*jnp.ones(l))
    return dplex_ddivide(k, temp)


def flatte980(m_, g_pipi, rg, Sbc):
    g_kk = rg * g_pipi
    m_k = 0.493677
    m_pi = 0.13957061
    rho_kk = jnp.sqrt(jnp.abs(1 - 4*m_k*m_k / Sbc))
    rho_pipi = jnp.sqrt(jnp.abs(1 - 4*m_pi*m_pi / Sbc))
    tmp_A = dplex_dconstruct(m_**2 - Sbc, -1*(g_pipi*rho_pipi + g_kk*rho_kk))
    return dplex_ddivide(1.0, tmp_A)


def flatte1270(m_, w_, Sbc):
    rm = m_ * m_
    gr = m_ * w_
    q2r = 0.25 * rm - 0.0194792
    b2r = q2r * (q2r + 0.1825) + 0.033306
    g11270 = gr * b2r / jnp.power(q2r, 2.5)
    q2 = 0.25 * Sbc - 0.0194792
    b2 = q2 * (q2 + 0.1825) + 0.033306
    g1 = g11270 * jnp.power(q2, 2.5) / b2
    tmp = dplex_dconstruct(Sbc - rm, g1)
    return dplex_ddivide(gr, tmp)


def flatte500(m_, b1, b2, b3, b4, b5, Sbc):
    m2 = m_*m_
    rp = 0.139556995
    mpi2d2 = 0.009739946882
    cro1 = jnp.sqrt(jnp.abs((Sbc-(2*rp)**2)*Sbc))/Sbc
    cro2 = jnp.sqrt(jnp.abs((m2-(2*rp)**2)*m2))/m2
    pip1 = jnp.sqrt(jnp.abs(1.0 - 0.3116765584/Sbc))/(1.0+jnp.exp(9.8-3.5*Sbc))
    pip2 = jnp.sqrt(jnp.abs(1.0 - 0.3116765584/m2))/(1.0+jnp.exp(9.8-3.5*m2))
    cgam1 = m_*(b1+b2*Sbc)*(Sbc-mpi2d2)/(m2-mpi2d2)*jnp.exp(-(Sbc-m2)/b3)*cro1/cro2
    cgam2 = m_*b4*pip1/pip2
    tmp = dplex_dconstruct(m2-Sbc, -b5*(cgam1+cgam2))
    return dplex_ddivide(1.0, tmp)


# =============================================================================
# 振幅组合函数（与单节点版本完全一致）
# =============================================================================

def calculate_BW_flatte980(A_mass, A_width, phi_kk, B_mass, B_g_kk, B_rg, f_kk, Amplitude_param_AMP, Amplitude_param_const, Amplitude_param_theta):
    A_propagator = BW(A_mass, A_width, phi_kk)
    B_propagator = flatte980(B_mass, B_g_kk, B_rg, f_kk)
    propagator_combined = dplex_deinsum("j, j->j", A_propagator, B_propagator)
    const_ph = dplex_dconstruct(Amplitude_param_const, Amplitude_param_theta)
    result = dplex_deinsum_ord("ijk,li->ljk", Amplitude_param_AMP, const_ph)
    result = dplex_deinsum("ljk,j->jk", result, propagator_combined)
    return result


def component_BW_flatte980(A_mass, A_width, phi_kk, B_mass, B_g_kk, B_rg, f_kk, Amplitude_param_AMP, Amplitude_param_const, Amplitude_param_theta):
    A_propagator = BW(A_mass, A_width, phi_kk)
    B_propagator = flatte980(B_mass, B_g_kk, B_rg, f_kk)
    propagator_combined = dplex_deinsum("j, j->j", A_propagator, B_propagator)
    const_ph = dplex_dconstruct(Amplitude_param_const, Amplitude_param_theta)
    result = dplex_deinsum_ord("ijk,li->ljk", Amplitude_param_AMP, const_ph)
    result = dplex_deinsum("ljk,j->ljk", result, propagator_combined)
    return result


def calculate_BW_BW(A_mass, A_width, phi_kk, B_mass, B_width, f_kk, Amplitude_param_AMP, Amplitude_param_const, Amplitude_param_theta):
    A_propagator = BW(A_mass, A_width, phi_kk)
    B_propagator = jnp.moveaxis(
        vmap(partial(BW, Sbc=f_kk))(B_mass, B_width), 1, 0
    )
    propagator_combined = dplex_deinsum("j, ij->ij", A_propagator, B_propagator)
    const_ph = dplex_dconstruct(Amplitude_param_const, Amplitude_param_theta)
    result = dplex_deinsum_ord("ijk,li->ljk", Amplitude_param_AMP, const_ph)
    result = dplex_deinsum("ljk,lj->jk", result, propagator_combined)
    return result


def component_BW_BW(A_mass, A_width, phi_kk, B_mass, B_width, f_kk, Amplitude_param_AMP, Amplitude_param_const, Amplitude_param_theta):
    A_propagator = BW(A_mass, A_width, phi_kk)
    B_propagator = jnp.moveaxis(
        vmap(partial(BW, Sbc=f_kk))(B_mass, B_width), 1, 0
    )
    propagator_combined = dplex_deinsum("j, ij->ij", A_propagator, B_propagator)
    const_ph = dplex_dconstruct(Amplitude_param_const, Amplitude_param_theta)
    result = dplex_deinsum_ord("ijk,li->ljk", Amplitude_param_AMP, const_ph)
    result = dplex_deinsum("ljk,lj->ljk", result, propagator_combined)
    return result


def calculate_BW_flatte1270(A_mass, A_width, phi_kk, B_mass, B_width, f_kk, Amplitude_param_AMP, Amplitude_param_const, Amplitude_param_theta):
    A_propagator = BW(A_mass, A_width, phi_kk)
    B_propagator = flatte1270(B_mass, B_width, f_kk)
    propagator_combined = dplex_deinsum("j, j->j", A_propagator, B_propagator)
    const_ph = dplex_dconstruct(Amplitude_param_const, Amplitude_param_theta)
    result = dplex_deinsum_ord("ijk,li->ljk", Amplitude_param_AMP, const_ph)
    result = dplex_deinsum("ljk,j->jk", result, propagator_combined)
    return result


def component_BW_flatte1270(A_mass, A_width, phi_kk, B_mass, B_width, f_kk, Amplitude_param_AMP, Amplitude_param_const, Amplitude_param_theta):
    A_propagator = BW(A_mass, A_width, phi_kk)
    B_propagator = flatte1270(B_mass, B_width, f_kk)
    propagator_combined = dplex_deinsum("j, j->j", A_propagator, B_propagator)
    const_ph = dplex_dconstruct(Amplitude_param_const, Amplitude_param_theta)
    result = dplex_deinsum_ord("ijk,li->ljk", Amplitude_param_AMP, const_ph)
    result = dplex_deinsum("ljk,j->ljk", result, propagator_combined)
    return result


# =============================================================================
# 参数提取（与单节点一致）
# =============================================================================

def extract_parameters(args):
    return {
        'phi_mass': jnp.array([1.02]),
        'phi_width': jnp.array([0.004]),
        'phif0_kk_BW_flatte980_mass': jnp.array([args[0]]),
        'phif0_kk_BW_flatte980_g_kk': jnp.array([args[1]]),
        'phif0_kk_BW_flatte980_rg': jnp.array([args[2]]),
        'phif0_kk_BW_flatte980_const': jnp.array([0.1, args[3]]).reshape(-1, 2),
        'phif0_kk_BW_flatte980_theta': jnp.array([0.1, args[4]]).reshape(-1, 2),
        'phif0_kk_BW_BW_mass': jnp.array([args[5], args[59]]),
        'phif0_kk_BW_BW_width': jnp.array([args[6], args[60]]),
        'phif0_kk_BW_BW_const': jnp.array([args[7], args[8], args[61], args[62]]).reshape(-1, 2),
        'phif0_kk_BW_BW_theta': jnp.array([args[9], args[10], args[63], args[64]]).reshape(-1, 2),
        'phif2_kk_BW_flatte1270_mass': jnp.array([args[11]]),
        'phif2_kk_BW_flatte1270_width': jnp.array([args[12]]),
        'phif2_kk_BW_flatte1270_const': jnp.array([args[13], args[14], args[15], args[16], args[17]]).reshape(-1, 5),
        'phif2_kk_BW_flatte1270_theta': jnp.array([args[18], args[19], args[20], args[21], args[22]]).reshape(-1, 5),
        'phif2_kk_BW_BW_mass': jnp.array([args[23], args[35], args[47]]),
        'phif2_kk_BW_BW_width': jnp.array([args[24], args[36], args[48]]),
        'phif2_kk_BW_BW_const': jnp.array([args[25], args[26], args[27], args[28], args[29], args[37], args[38], args[39], args[40], args[41], args[49], args[50], args[51], args[52], args[53]]).reshape(-1, 5),
        'phif2_kk_BW_BW_theta': jnp.array([args[30], args[31], args[32], args[33], args[34], args[42], args[43], args[44], args[45], args[46], args[54], args[55], args[56], args[57], args[58]]).reshape(-1, 5),
    }


# =============================================================================
# 配置 I/O（与单节点一致）
# =============================================================================

def setup_logging():
    with open("config/logconfig_fit.json", "r") as config_file:
        LOGGING_CONFIG = json.load(config_file)
        logging.config.dictConfig(LOGGING_CONFIG)
    return logging.getLogger("fit")


def make_initial_args():
    _data = toml.load("run/free_params.toml")["data"]
    _data.sort(key=lambda x: x["arg_index"])
    args = [d["value"] for d in _data]
    ranges = [d.get("range", [-onp.inf, onp.inf]) for d in _data]
    errors = [d["error"] for d in _data]
    paths = [d["path"] for d in _data]
    return onp.array(args), onp.array(ranges), onp.array(errors), paths


def save_result(args, errors, path="output/free_params_fitted.toml"):
    _data = toml.load("run/free_params.toml")["data"]
    _data.sort(key=lambda x: x["arg_index"])
    for i, d in enumerate(_data):
        d["value"] = float(args[i])
        d["error"] = float(errors[i])
    with open(path, "w") as f:
        toml.dump({"data": _data}, f)


# =============================================================================
# 数据加载 & 归一化（与单节点一致）
# =============================================================================

def load_data():
    n_repeat = 1
    data = {}
    data['data_phi_kk'] = onp.tile(onp.load("data/real_data/phi_kk.npy"), n_repeat)
    data['data_f_kk'] = onp.tile(onp.load("data/real_data/f_kk.npy"), n_repeat)
    data['data_phif0_kk'] = onp.tile(onp.load("data/real_data/phif0_kk.npy"), (1, n_repeat, 1))
    data['data_phif2_kk'] = onp.tile(onp.load("data/real_data/phif2_kk.npy"), (1, n_repeat, 1))

    data['mc_phi_kk'] = onp.tile(onp.load("data/mc_truth/phi_kk.npy"), n_repeat)
    data['mc_f_kk'] = onp.tile(onp.load("data/mc_truth/f_kk.npy"), n_repeat)
    data['mc_phif0_kk'] = onp.tile(onp.load("data/mc_truth/phif0_kk.npy"), (1, n_repeat, 1))
    data['mc_phif2_kk'] = onp.tile(onp.load("data/mc_truth/phif2_kk.npy"), (1, n_repeat, 1))

    data['truth_phi_kk'] = data['mc_phi_kk'][0:150000 * n_repeat]
    data['truth_f_kk'] = data['mc_f_kk'][0:150000 * n_repeat]
    data['truth_phif0_kk'] = data['mc_phif0_kk'][:, 0:150000 * n_repeat]
    data['truth_phif2_kk'] = data['mc_phif2_kk'][:, 0:150000 * n_repeat]

    return data


def normalize_data(data):
    regular_phif0_kk = 1. / onp.average(
        onp.sqrt(onp.sum(onp.asarray(data['mc_phif0_kk'])**2, axis=2)), axis=1
    )
    regular_phif2_kk = 1. / onp.average(
        onp.sqrt(onp.sum(onp.asarray(data['mc_phif2_kk'])**2, axis=2)), axis=1
    )

    data['data_phif0_kk'] = onp.einsum("jkl,j->jkl", data['data_phif0_kk'], regular_phif0_kk)
    data['mc_phif0_kk'] = onp.einsum("jkl,j->jkl", data['mc_phif0_kk'], regular_phif0_kk)
    data['truth_phif0_kk'] = onp.einsum("jkl,j->jkl", data['truth_phif0_kk'], regular_phif0_kk)
    data['data_phif2_kk'] = onp.einsum("jkl,j->jkl", data['data_phif2_kk'], regular_phif2_kk)
    data['mc_phif2_kk'] = onp.einsum("jkl,j->jkl", data['mc_phif2_kk'], regular_phif2_kk)
    data['truth_phif2_kk'] = onp.einsum("jkl,j->jkl", data['truth_phif2_kk'], regular_phif2_kk)

    return data


# =============================================================================
# 分布式数据分片（核心区别：按事件轴切分到多设备）
# =============================================================================

def shard_data_distributed(data, mesh):
    """
    将数据按事件维切片到所有设备上。
    - phi_kk / f_kk: shape (N_events,) → 按 event 轴切
    - phif0_kk / phif2_kk: shape (channels, N_events, components) → 仅按 event 轴(axis=1)切
    参数 args 很小（65维），全局复制（不在此处理）。
    """
    event_1d = NamedSharding(mesh, P("event"))
    event_axis1 = NamedSharding(mesh, P(None, "event"))
    event_axis1_3d = NamedSharding(mesh, P(None, "event", None))

    jax_data = {}

    # 1D 事件数组：phi_kk, f_kk — shape (N_events,)
    for prefix in ('data', 'mc', 'truth'):
        for name in ('phi_kk', 'f_kk'):
            key = f'{prefix}_{name}'
            arr = jnp.array(data[key])
            jax_data[key] = jax.device_put(arr, event_1d)

    # 多维事件数组：phif0_kk, phif2_kk
    for prefix in ('data', 'mc', 'truth'):
        for name in ('phif0_kk', 'phif2_kk'):
            key = f'{prefix}_{name}'
            arr = jnp.array(data[key])
            if arr.ndim == 3:
                jax_data[key] = jax.device_put(arr, event_axis1_3d)
            else:
                jax_data[key] = jax.device_put(arr, event_axis1)

    return jax_data


def data_step_function(total_frac, args):
    step_value = jnp.power(total_frac - total_frac_kk, 2.0) * constraint_strength
    step_value += jnp.power(0.98 - args[0], 2.0) / jnp.power(10.0, 2.0) / 2.0
    step_value += jnp.power(1.704 - args[5], 2.0) / jnp.power(1.0, 2.0) / 2.0
    step_value += jnp.power(0.123 - args[6], 2.0) / jnp.power(1.0, 2.0) / 2.0
    step_value += jnp.power(1.2755 - args[11], 2.0) / jnp.power(1.0, 2.0) / 2.0
    step_value += jnp.power(0.1867 - args[12], 2.0) / jnp.power(1.0, 2.0) / 2.0
    step_value += jnp.power(1.517 - args[23], 2.0) / jnp.power(1.0, 2.0) / 2.0
    step_value += jnp.power(0.086 - args[24], 2.0) / jnp.power(1.0, 2.0) / 2.0
    step_value += jnp.power(2.157 - args[35], 2.0) / jnp.power(1.0, 2.0) / 2.0
    step_value += jnp.power(0.152 - args[36], 2.0) / jnp.power(1.0, 2.0) / 2.0
    step_value += jnp.power(2.345 - args[47], 2.0) / jnp.power(0.01, 2.0) / 2.0
    step_value += jnp.power(0.322 - args[48], 2.0) / jnp.power(1.0, 2.0) / 2.0
    step_value += jnp.power(2.47 - args[59], 2.0) / jnp.power(0.007, 2.0) / 2.0
    step_value += jnp.power(0.075 - args[60], 2.0) / jnp.power(0.011, 2.0) / 2.0
    return step_value


def data_likelihood_kk(args, jax_data):
    params = extract_parameters(args)

    d_phi_kk = jax_data['data_phi_kk']
    d_f_kk = jax_data['data_f_kk']
    d_phif0_kk = jax_data['data_phif0_kk']
    d_phif2_kk = jax_data['data_phif2_kk']

    t_phi_kk = jax_data['truth_phi_kk']
    t_f_kk = jax_data['truth_f_kk']
    t_phif0_kk = jax_data['truth_phif0_kk']
    t_phif2_kk = jax_data['truth_phif2_kk']

    data_phif0_kk_BW_flatte980 = calculate_BW_flatte980(
        params['phi_mass'], params['phi_width'], d_phi_kk,
        params['phif0_kk_BW_flatte980_mass'], params['phif0_kk_BW_flatte980_g_kk'],
        params['phif0_kk_BW_flatte980_rg'], d_f_kk,
        d_phif0_kk, params['phif0_kk_BW_flatte980_const'], params['phif0_kk_BW_flatte980_theta']
        )
    data_phif0_kk_BW_BW = calculate_BW_BW(
        params['phi_mass'], params['phi_width'], d_phi_kk,
        params['phif0_kk_BW_BW_mass'], params['phif0_kk_BW_BW_width'], d_f_kk,
        d_phif0_kk, params['phif0_kk_BW_BW_const'], params['phif0_kk_BW_BW_theta']
        )
    data_phif2_kk_BW_flatte1270 = calculate_BW_flatte1270(
        params['phi_mass'], params['phi_width'], d_phi_kk,
        params['phif2_kk_BW_flatte1270_mass'], params['phif2_kk_BW_flatte1270_width'], d_f_kk,
        d_phif2_kk, params['phif2_kk_BW_flatte1270_const'], params['phif2_kk_BW_flatte1270_theta']
        )
    data_phif2_kk_BW_BW = calculate_BW_BW(
        params['phi_mass'], params['phi_width'], d_phi_kk,
        params['phif2_kk_BW_BW_mass'], params['phif2_kk_BW_BW_width'], d_f_kk,
        d_phif2_kk, params['phif2_kk_BW_BW_const'], params['phif2_kk_BW_BW_theta']
        )

    comp_f0_flatte980 = component_BW_flatte980(
        params['phi_mass'], params['phi_width'], t_phi_kk,
        params['phif0_kk_BW_flatte980_mass'], params['phif0_kk_BW_flatte980_g_kk'],
        params['phif0_kk_BW_flatte980_rg'], t_f_kk,
        t_phif0_kk, params['phif0_kk_BW_flatte980_const'], params['phif0_kk_BW_flatte980_theta']
        )
    comp_f0_BW = component_BW_BW(
        params['phi_mass'], params['phi_width'], t_phi_kk,
        params['phif0_kk_BW_BW_mass'], params['phif0_kk_BW_BW_width'], t_f_kk,
        t_phif0_kk, params['phif0_kk_BW_BW_const'], params['phif0_kk_BW_BW_theta']
        )
    comp_f2_flatte1270 = component_BW_flatte1270(
        params['phi_mass'], params['phi_width'], t_phi_kk,
        params['phif2_kk_BW_flatte1270_mass'], params['phif2_kk_BW_flatte1270_width'], t_f_kk,
        t_phif2_kk, params['phif2_kk_BW_flatte1270_const'], params['phif2_kk_BW_flatte1270_theta']
        )
    comp_f2_BW = component_BW_BW(
        params['phi_mass'], params['phi_width'], t_phi_kk,
        params['phif2_kk_BW_BW_mass'], params['phif2_kk_BW_BW_width'], t_f_kk,
        t_phif2_kk, params['phif2_kk_BW_BW_const'], params['phif2_kk_BW_BW_theta']
        )

    sum_frac = jnp.sum(dplex_dabs(
            jnp.einsum("mljk->mjk", comp_f0_flatte980) +
            jnp.einsum("mljk->mjk", comp_f0_BW) +
            jnp.einsum("mljk->mjk", comp_f2_flatte1270) +
            jnp.einsum("mljk->mjk", comp_f2_BW)
        ))
    frac_f0_flatte980 = jnp.sum(jnp.einsum("ljk->l", dplex_dabs(comp_f0_flatte980)) / sum_frac)
    frac_f0_BW = jnp.sum(jnp.einsum("ljk->l", dplex_dabs(comp_f0_BW)) / sum_frac)
    frac_f2_flatte1270 = jnp.sum(jnp.einsum("ljk->l", dplex_dabs(comp_f2_flatte1270)) / sum_frac)
    frac_f2_BW = jnp.sum(jnp.einsum("ljk->l", dplex_dabs(comp_f2_BW)) / sum_frac)
    total_frac = frac_f0_flatte980 + frac_f0_BW + frac_f2_flatte1270 + frac_f2_BW

    step_function = data_step_function(total_frac, args)

    total_amplitude = (data_phif0_kk_BW_flatte980 + data_phif0_kk_BW_BW +
                           data_phif2_kk_BW_flatte1270 + data_phif2_kk_BW_BW)
    likelihood = -jnp.sum(jnp.log(jnp.sum(dplex_dabs(total_amplitude), axis=1))) + step_function
    return likelihood


def mc_likelihood_kk(args, jax_data):
    params = extract_parameters(args)

    m_phi_kk = jax_data['mc_phi_kk']
    m_f_kk = jax_data['mc_f_kk']
    m_phif0_kk = jax_data['mc_phif0_kk']
    m_phif2_kk = jax_data['mc_phif2_kk']

    total_mc = calculate_BW_flatte980(
            params['phi_mass'], params['phi_width'], m_phi_kk,
            params['phif0_kk_BW_flatte980_mass'], params['phif0_kk_BW_flatte980_g_kk'],
            params['phif0_kk_BW_flatte980_rg'], m_f_kk,
            m_phif0_kk, params['phif0_kk_BW_flatte980_const'], params['phif0_kk_BW_flatte980_theta']
        )
    total_mc = total_mc + calculate_BW_BW(
            params['phi_mass'], params['phi_width'], m_phi_kk,
            params['phif0_kk_BW_BW_mass'], params['phif0_kk_BW_BW_width'], m_f_kk,
            m_phif0_kk, params['phif0_kk_BW_BW_const'], params['phif0_kk_BW_BW_theta']
        )
    total_mc = total_mc + calculate_BW_flatte1270(
            params['phi_mass'], params['phi_width'], m_phi_kk,
            params['phif2_kk_BW_flatte1270_mass'], params['phif2_kk_BW_flatte1270_width'], m_f_kk,
            m_phif2_kk, params['phif2_kk_BW_flatte1270_const'], params['phif2_kk_BW_flatte1270_theta']
        )
    total_mc = total_mc + calculate_BW_BW(
            params['phi_mass'], params['phi_width'], m_phi_kk,
            params['phif2_kk_BW_BW_mass'], params['phif2_kk_BW_BW_width'], m_f_kk,
            m_phif2_kk, params['phif2_kk_BW_BW_const'], params['phif2_kk_BW_BW_theta']
        )
    return jnp.mean(jnp.sum(dplex_dabs(total_mc), axis=1))


def make_distributed_likelihood(data_size):
    """
    构建分布式 combined_likelihood 函数。
    combined_likelihood(args, jax_data) 接收全局复制的参数向量和分片数据 dict，
    符合 somax 的 LossFn 协议：loss_fn(params, batch) -> scalar
    """

    def combined_likelihood(args, jax_data):
        return data_likelihood_kk(args, jax_data) + data_size * jnp.log(mc_likelihood_kk(args, jax_data))

    return combined_likelihood


# =============================================================================
# 随机初值扰动
#
# 参照旧版单节点脚本的抖动逻辑：
#   theta = 2*pi*rand()
#   theta_val = 0.1 * cos(theta); const_val = 0.1 * sin(theta)   （落在半径 0.1 的圆上）
#   args_float *= (rand() - 0.5) / disturb + 1.0                 （disturb=50 → 整体乘性抖动 ±1%）
# 这里改为对每次试验使用独立的 numpy RandomState(seed)，seed 由所有分布式进程
# 用同一个 base_seed + trial_idx 算出，因此各进程生成完全一致的偏移，无需通信。
# =============================================================================

def find_const_theta_indices(paths):
    """从 free_params.toml 的 path 字段中找出所有 Amplitude.constN / Amplitude.thetaN 的 arg_index"""
    const_idx = [i for i, p in enumerate(paths) if ".Amplitude.const" in p]
    theta_idx = [i for i, p in enumerate(paths) if ".Amplitude.theta" in p]
    return onp.array(const_idx, dtype=int), onp.array(theta_idx, dtype=int)


def perturb_args(base_args, const_idx, theta_idx, seed, disturb=50.0):
    """
    对初始参数做随机偏移，返回新的参数数组（不修改 base_args）。
    - const/theta：重新采样到半径 0.1 的圆上（幅值固定为 0.1，相位随机）
    - 全部参数：额外乘以一个统一的随机乘性抖动 (rand()-0.5)/disturb + 1.0
    """
    rng = onp.random.RandomState(seed)
    args = onp.array(base_args, dtype=float).copy()

    if const_idx.size > 0:
        theta = 2 * onp.pi * rng.rand(theta_idx.shape[0])
        args[theta_idx] = 0.1 * onp.cos(theta)
        args[const_idx] = 0.1 * onp.sin(theta)

    args = args * ((rng.rand(args.shape[0]) - 0.5) / disturb + 1.0)
    return args


# =============================================================================
# Main：分布式 somax Newton-CG 拟合，随机初值多次重启扫描
# =============================================================================

def main():
    num_processes, process_id = init_distributed()
    is_chief = (process_id == 0)

    config.update("jax_enable_x64", True)

    logger = setup_logging()
    if is_chief:
        logger.info("开始分布式 somax Newton-CG 随机初值重启扫描")
        logger.info(f"进程数: {num_processes}, 全局设备数: {jax.device_count()}")

    global constraint_strength, total_frac_kk
    constraint_strength = 1000.0
    total_frac_kk = 1.1

    mesh = build_mesh()

    data = load_data()
    data = normalize_data(data)
    data_size = len(data['data_phi_kk'])

    base_args, _, _, paths = make_initial_args()
    const_idx, theta_idx = find_const_theta_indices(paths)

    with jax.set_mesh(mesh):
        jax_data = shard_data_distributed(data, mesh)

        combined_likelihood = make_distributed_likelihood(data_size)

        def hvp_combined_likelihood(x, v, jax_data):
            zero_data = jax.tree.map(jnp.zeros_like, jax_data)
            return jvp(grad(combined_likelihood), (x, jax_data), (v, zero_data))[1]

        replicated = NamedSharding(mesh, P())
        jit_likelihood = jit(combined_likelihood)
        jit_hvp = jit(hvp_combined_likelihood)

        args_size = base_args.shape[0]

        # somax Newton-CG 优化器，构建一次即可复用于每次随机重启试验
        method = somax.make(
            "newton_cg",
            loss_fn=combined_likelihood,
            lam_policy="trust_region",
            lam_kwargs={"max_lam": 1e9, "min_lam": 100, "dec": 0.97, "inc": 1.5},
            tol=1e-4,
            maxiter=args_size * 5,
            warm_start=True,
            stabilise_every=1,
            learning_rate=1,
            record_cg_stats=True,
        )

        @jax.jit
        def opt_step(params, state, rng, jax_data):
            return method.step(params, jax_data, state, rng)

        avextol = 1e-8
        xtol = args_size * avextol
        max_steps = 1000

        n_trials = 10
        base_seed = 20260710

        trace_trial = []
        trace_seed = []
        trace_init_args = []
        trace_init_likelihood = []
        trace_final_args = []
        trace_final_likelihood = []
        trace_n_iter = []
        trace_converged = []
        trace_elapsed = []

        best_final_loss = onp.inf
        best_final_args = None

        if is_chief:
            logger.info(f"计划运行 {n_trials} 次随机初值重启试验，每次跑满 Newton-CG 至收敛")

        for trial in range(n_trials):
            trial_seed = base_seed + trial

            perturbed = perturb_args(base_args, const_idx, theta_idx, trial_seed)
            params = jax.device_put(jnp.asarray(perturbed), replicated)

            init_loss = float(jit_likelihood(params, jax_data))

            # 在每次试验开始优化前，用 H 计算所需的 lam0
            H0 = onp.zeros((args_size, args_size))
            for i in range(args_size):
                v = onp.zeros(args_size)
                v[i] = 1.0
                H0[:, i] = onp.asarray(jit_hvp(params, jnp.asarray(v), jax_data))

            w = onp.linalg.eigvalsh((H0 + H0.T) / 2)
            lam0 = float(- w[0] * 2 + 1)
            if is_chief:
                logger.info(f"Trial {trial}: H eig min={w[0]}, max={w[-1]}, lam={lam0}")

            method.damping = method.damping.replace(lam0=lam0)
            state = method.init(params)
            rng = jax.random.PRNGKey(trial_seed)

            update_l1norm = onp.finfo(float).max
            n_iter = 0
            converged = False

            trial_start = time.time()

            for i in range(max_steps):
                rng, subkey = jax.random.split(rng)
                new_params, state, info = opt_step(params, state, subkey, jax_data)
                update_l1norm = float(jnp.linalg.norm(new_params - params, ord=1))


                if onp.isnan(info.get("cg_resid", float("nan"))):
                    break

                if update_l1norm <= xtol:
                    converged = True
                    break

                params = new_params
                n_iter = i + 1

                if i % 10 == 0 or i == max_steps - 1:
                    cur_loss = jit_likelihood(params, jax_data)
                    if is_chief:
                        logger.info(
                            f"Trial {trial}: iter {i}, "
                            f"cg_resid={info['cg_resid']:.6f}, "
                            f"update_l1norm={update_l1norm:.6e}, "
                            f"lam_used={info['lam_used']:.6e}, "
                            f"loss={cur_loss:.6f}"
                        )

            trial_elapsed = time.time() - trial_start
            final_loss = float(jit_likelihood(params, jax_data))
            final_args = onp.asarray(params)

            # final_loss / final_args 由 all-reduce 得到，所有进程结果一致；
            # 必须在所有进程上都更新 best_final_*，否则后续 Hessian 计算
            # （collective 操作）会在各进程间使用不同的 best_x，破坏同步。
            if final_loss < best_final_loss:
                best_final_loss = final_loss
                best_final_args = final_args

            if is_chief:
                trace_trial.append(trial)
                trace_seed.append(trial_seed)
                trace_init_args.append(perturbed)
                trace_init_likelihood.append(init_loss)
                trace_final_args.append(final_args)
                trace_final_likelihood.append(final_loss)
                trace_n_iter.append(n_iter)
                trace_converged.append(converged)
                trace_elapsed.append(trial_elapsed)

                logger.info(
                    f"Trial {trial}: init_likelihood={init_loss:.6f}, "
                    f"final_likelihood={final_loss:.6f}, n_iter={n_iter}, "
                    f"converged={converged}, elapsed={trial_elapsed:.2f}s"
                )

        if is_chief:
            logger.info("=" * 50)
            logger.info(f"随机初值重启扫描完成，共 {n_trials} 次试验")
            logger.info(f"最优 likelihood: {best_final_loss:.6f}")
            logger.info("=" * 50)

        # 只对最优结果计算误差（逐列构造 Hessian，所有进程一致执行 collective）
        if is_chief:
            logger.info("计算最优结果的参数误差（Hessian逆矩阵）...")
        best_x = jnp.asarray(
            best_final_args if best_final_args is not None else onp.asarray(base_args)
        )
        best_x = jax.device_put(best_x, replicated)
        hessian_matrix = onp.zeros([args_size, args_size])
        for i in range(args_size):
            v = onp.zeros(args_size)
            v[i] = 1.0
            hessian_matrix[:, i] = onp.array(jit_hvp(best_x, jnp.asarray(v), jax_data))
        ferror = onp.sqrt(onp.diag(onp.linalg.inv(hessian_matrix)))
        if is_chief:
            logger.info("误差计算完成")

        # 只有 chief 进程写结果，避免多进程竞争写同一文件
        if is_chief:
            os.makedirs("output/fit", exist_ok=True)

            best_x_np = onp.asarray(best_x)
            onp.save("output/fit/random_restart_best_values.npy", best_x_np)
            onp.save("output/fit/random_restart_best_errors.npy", ferror)
            save_result(best_x_np, ferror, "output/fit/free_params_fitted_best_restart.toml")
            logger.info("最优结果已保存至 output/fit/free_params_fitted_best_restart.toml")

            onp.savez(
                "output/fit/random_restart_trace.npz",
                trial=onp.asarray(trace_trial),
                seed=onp.asarray(trace_seed),
                init_args=onp.asarray(trace_init_args),
                init_likelihood=onp.asarray(trace_init_likelihood),
                final_args=onp.asarray(trace_final_args),
                final_likelihood=onp.asarray(trace_final_likelihood),
                n_iter=onp.asarray(trace_n_iter),
                converged=onp.asarray(trace_converged),
                elapsed=onp.asarray(trace_elapsed),
                best_final_loss=onp.asarray(best_final_loss),
            )
            logger.info("随机重启扫描记录已保存至 output/fit/random_restart_trace.npz")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
