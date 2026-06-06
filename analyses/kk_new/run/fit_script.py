import json
import logging
import logging.config
import os
# must set before importing jax
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import time
import numpy as onp
from functools import partial
from scipy.optimize import minimize

import jax.numpy as np
from jax import device_put, grad, jit, vmap, jvp
from jax import config




import sys
foo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(foo_path)
sys.path.append(foo_path)




def setup_logging():
    """Setup logging configuration"""
    with open("config/logconfig_fit.json", "r") as config_file:
        LOGGING_CONFIG = json.load(config_file)
        logging.config.dictConfig(LOGGING_CONFIG)
    return logging.getLogger("fit")




def dplex_deinsum(subscript, aa, bb):
    real = np.einsum(subscript, aa[0], bb[0]) - np.einsum(subscript, aa[1], bb[1])
    imag = np.einsum(subscript, aa[0], bb[1]) + np.einsum(subscript, aa[1], bb[0])
    return np.stack([real, imag], axis=0)

def dplex_deinsum_ord(subscript, aa, bb):
    real = np.einsum(subscript, aa, bb[0])
    imag = np.einsum(subscript, aa, bb[1])
    return np.stack([real, imag], axis=0)

def dplex_dabs(aa):
    return aa[0]**2 + aa[1]**2 # 因为是纵向叠加所以aa[0]是纵向上第一个数组

def dplex_dtomine(aa):
    return np.stack([np.real(aa), np.imag(aa)], axis=0) # 组合而成的数组 axis=0 的第一个数组为实部，第二个数组为虚部

def dplex_dconstruct(aa, bb):
    return np.stack([aa, bb], axis=0) # 纵向叠加数组 第一维为实部，第二维为虚部

def dplex_ddivide(a, bb):
    real = a * bb[0] / dplex_dabs(bb)
    imag = -a * bb[1] / dplex_dabs(bb)
    return np.stack([real, imag], axis=0)



# Physics calculation functions (resonance shape functions)
# Provided by the user for each analysis. Copy this file to your analysis
# directory as physics_functions.py and keep only the functions you need.
# ==============================================================================




def BW(m_, w_, Sbc):
    l = (Sbc.shape)[0]
    temp = dplex_dconstruct(m_*m_ - Sbc, -m_*w_*np.ones(l))
    return dplex_ddivide(1.0, temp)


def BW_relativity(m_, w_, Sbc):
    gamma = np.sqrt(m_*m_*(m_*m_+w_*w_))
    k = np.sqrt(2*np.sqrt(2)*m_*np.abs(w_)*gamma/np.pi/np.sqrt(m_*m_+gamma))
    l = (Sbc.shape)[0]
    temp = dplex_dconstruct(m_*m_ - Sbc, -m_*w_*np.ones(l))
    return dplex_ddivide(k, temp)


def flatte980(m_, g_pipi, rg, Sbc):
    g_kk = rg * g_pipi
    m_k = 0.493677
    m_pi = 0.13957061
    rho_kk = np.sqrt(np.abs(1 - 4*m_k*m_k / Sbc))
    rho_pipi = np.sqrt(np.abs(1 - 4*m_pi*m_pi / Sbc))
    tmp_A = dplex_dconstruct(m_**2 - Sbc, -1*(g_pipi*rho_pipi + g_kk*rho_kk))
    return dplex_ddivide(1.0, tmp_A)


def flatte1270(m_, w_, Sbc):
    rm = m_ * m_
    gr = m_ * w_
    q2r = 0.25 * rm - 0.0194792
    b2r = q2r * (q2r + 0.1825) + 0.033306
    g11270 = gr * b2r / np.power(q2r, 2.5)
    q2 = 0.25 * Sbc - 0.0194792
    b2 = q2 * (q2 + 0.1825) + 0.033306
    g1 = g11270 * np.power(q2, 2.5) / b2
    tmp = dplex_dconstruct(Sbc - rm, g1)
    return dplex_ddivide(gr, tmp)


def flatte500(m_, b1, b2, b3, b4, b5, Sbc):
    m2 = m_*m_
    rp = 0.139556995
    mpi2d2 = 0.009739946882
    cro1 = np.sqrt(np.abs((Sbc-(2*rp)**2)*Sbc))/Sbc
    cro2 = np.sqrt(np.abs((m2-(2*rp)**2)*m2))/m2
    pip1 = np.sqrt(np.abs(1.0 - 0.3116765584/Sbc))/(1.0+np.exp(9.8-3.5*Sbc))
    pip2 = np.sqrt(np.abs(1.0 - 0.3116765584/m2))/(1.0+np.exp(9.8-3.5*m2))
    cgam1 = m_*(b1+b2*Sbc)*(Sbc-mpi2d2)/(m2-mpi2d2)*np.exp(-(Sbc-m2)/b3)*cro1/cro2
    cgam2 = m_*b4*pip1/pip2
    tmp = dplex_dconstruct(m2-Sbc, -b5*(cgam1+cgam2))
    return dplex_ddivide(1.0, tmp)


def calculate_BW_BW(A_mass, A_width, phi_kk, B_mass, B_width, f_kk, Amplitude_param_AMP, Amplitude_param_const, Amplitude_param_theta):
    A_propagator = BW(A_mass, A_width, phi_kk)
    B_propagator = np.moveaxis(
        vmap(partial(BW, Sbc=f_kk))(B_mass, B_width), 1, 0
    )
    propagator_combined = dplex_deinsum("j, ij->ij", A_propagator, B_propagator)
    const_ph = dplex_dconstruct(Amplitude_param_const, Amplitude_param_theta)
    result = dplex_deinsum_ord("ijk,li->ljk", Amplitude_param_AMP, const_ph)
    result = dplex_deinsum("ljk,lj->jk", result, propagator_combined)
    return result

def component_BW_BW(A_mass, A_width, phi_kk, B_mass, B_width, f_kk, Amplitude_param_AMP, Amplitude_param_const, Amplitude_param_theta):
    A_propagator = BW(A_mass, A_width, phi_kk)
    B_propagator = np.moveaxis(
        vmap(partial(BW, Sbc=f_kk))(B_mass, B_width), 1, 0
    )
    propagator_combined = dplex_deinsum("j, ij->ij", A_propagator, B_propagator)
    const_ph = dplex_dconstruct(Amplitude_param_const, Amplitude_param_theta)
    result = dplex_deinsum_ord("ijk,li->ljk", Amplitude_param_AMP, const_ph)
    result = dplex_deinsum("ljk,lj->ljk", result, propagator_combined)
    return result

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


def make_initial_args():
    _data = toml.load("run/free_params.toml")["data"]
    _data.sort(key=lambda x: x["arg_index"])

    args = [d["value"] for d in _data]
    ranges = [d.get("range", [-onp.inf, onp.inf]) for d in _data]
    errors = [d["error"] for d in _data]

    return onp.array(args), onp.array(ranges), onp.array(errors)



def save_result(args, errors, path="run/free_params.toml"):
    _data = toml.load("run/free_params.toml")["data"]
    _data.sort(key=lambda x: x["arg_index"])

    for i, d in enumerate(_data):
        d["value"] = float(args[i])
        d["error"] = float(errors[i])

    with open(path, "w") as f:
        toml.dump({"data": _data}, f)


def load_data():
    data = {}

    # Load real data
    data['data_phi_kk'] = onp.load("data/real_data/phi_kk.npy")
    data['data_f_kk'] = onp.load("data/real_data/f_kk.npy")
    data['data_kst2_124_kk'] = onp.load("data/real_data/kst2_124_kk.npy")
    data['data_kst2_123_kk'] = onp.load("data/real_data/kst2_123_kk.npy")
    data['data_phif0_kk'] = onp.load("data/real_data/phif0_kk.npy")
    data['data_phif2_kk'] = onp.load("data/real_data/phif2_kk.npy")
    data['data_u_kst2_r_kk'] = onp.load("data/real_data/u_kst2_r_kk.npy")
    data['data_u_kst2_l_kk'] = onp.load("data/real_data/u_kst2_l_kk.npy")

    # Load MC data
    data['mc_phi_kk'] = onp.load("data/mc_truth/phi_kk.npy")
    data['mc_f_kk'] = onp.load("data/mc_truth/f_kk.npy")
    data['mc_kst2_124_kk'] = onp.load("data/mc_truth/kst2_124_kk.npy")
    data['mc_kst2_123_kk'] = onp.load("data/mc_truth/kst2_123_kk.npy")
    data['mc_phif0_kk'] = onp.load("data/mc_truth/phif0_kk.npy")
    data['mc_phif2_kk'] = onp.load("data/mc_truth/phif2_kk.npy")
    data['mc_u_kst2_r_kk'] = onp.load("data/mc_truth/u_kst2_r_kk.npy")
    data['mc_u_kst2_l_kk'] = onp.load("data/mc_truth/u_kst2_l_kk.npy")

    # MC truth data (for constraints)
    data['truth_phi_kk'] = data['mc_phi_kk'][0:150000]
    data['truth_f_kk'] = data['mc_f_kk'][0:150000]
    data['truth_kst2_124_kk'] = data['mc_kst2_124_kk'][0:150000]
    data['truth_kst2_123_kk'] = data['mc_kst2_123_kk'][0:150000]
    data['truth_phif0_kk'] = data['mc_phif0_kk'][:, 0:150000]
    data['truth_phif2_kk'] = data['mc_phif2_kk'][:, 0:150000]
    data['truth_u_kst2_r_kk'] = data['mc_u_kst2_r_kk'][:, 0:150000]
    data['truth_u_kst2_l_kk'] = data['mc_u_kst2_l_kk'][:, 0:150000]

    # Weight data
    try:
        data['wt_data_kk'] = onp.load("data/weight/weight_kk.npy")
    except FileNotFoundError:
        data['wt_data_kk'] = onp.ones_like(data['data_phi_kk'])

    return data

def normalize_data(data):
    # 计算归一化因子
    regular_phif0_kk = 1. / onp.average(
        onp.sqrt(onp.sum(onp.asarray(data['mc_phif0_kk'])**2, axis=2)), axis=1
    )
    regular_phif2_kk = 1. / onp.average(
        onp.sqrt(onp.sum(onp.asarray(data['mc_phif2_kk'])**2, axis=2)), axis=1
    )
    regular_u_kst2_r_kk = 1. / onp.average(
        onp.sqrt(onp.sum(onp.asarray(data['mc_u_kst2_r_kk'])**2, axis=2)), axis=1
    )
    regular_u_kst2_l_kk = 1. / onp.average(
        onp.sqrt(onp.sum(onp.asarray(data['mc_u_kst2_l_kk'])**2, axis=2)), axis=1
    )

    # 应用归一化
    data['data_phif0_kk'] = onp.einsum("jkl,j->jkl", data['data_phif0_kk'], regular_phif0_kk)
    data['mc_phif0_kk'] = onp.einsum("jkl,j->jkl", data['mc_phif0_kk'], regular_phif0_kk)
    data['truth_phif0_kk'] = onp.einsum("jkl,j->jkl", data['truth_phif0_kk'], regular_phif0_kk)
    data['data_phif2_kk'] = onp.einsum("jkl,j->jkl", data['data_phif2_kk'], regular_phif2_kk)
    data['mc_phif2_kk'] = onp.einsum("jkl,j->jkl", data['mc_phif2_kk'], regular_phif2_kk)
    data['truth_phif2_kk'] = onp.einsum("jkl,j->jkl", data['truth_phif2_kk'], regular_phif2_kk)
    data['data_u_kst2_r_kk'] = onp.einsum("jkl,j->jkl", data['data_u_kst2_r_kk'], regular_u_kst2_r_kk)
    data['mc_u_kst2_r_kk'] = onp.einsum("jkl,j->jkl", data['mc_u_kst2_r_kk'], regular_u_kst2_r_kk)
    data['truth_u_kst2_r_kk'] = onp.einsum("jkl,j->jkl", data['truth_u_kst2_r_kk'], regular_u_kst2_r_kk)
    data['data_u_kst2_l_kk'] = onp.einsum("jkl,j->jkl", data['data_u_kst2_l_kk'], regular_u_kst2_l_kk)
    data['mc_u_kst2_l_kk'] = onp.einsum("jkl,j->jkl", data['mc_u_kst2_l_kk'], regular_u_kst2_l_kk)
    data['truth_u_kst2_l_kk'] = onp.einsum("jkl,j->jkl", data['truth_u_kst2_l_kk'], regular_u_kst2_l_kk)

    return data

def prepare_data_for_jax(data, device=None):
    jax_data = {}
    for key, value in data.items():
        jax_data[key] = device_put(np.array(value), device=device)
    return jax_data

def extract_parameters(args):
    return {
        'phi_mass': onp.array([1.02]),
        'phi_width': onp.array([0.004]),
        'u_kst2_r_kk_BW_BW_mass': onp.array([args[65], args[71], args[77]]),
        'u_kst2_r_kk_BW_BW_width': onp.array([args[66], args[72], args[78]]),
        'u_kst2_r_kk_BW_BW_const': onp.array([args[67], args[68], args[73], args[74], args[79], args[80]]).reshape(-1, 2),
        'u_kst2_r_kk_BW_BW_theta': onp.array([args[69], args[70], args[75], args[76], args[81], args[82]]).reshape(-1, 2),
        'u_kst2_l_kk_BW_BW_mass': onp.array([args[65], args[71], args[77]]),
        'u_kst2_l_kk_BW_BW_width': onp.array([args[66], args[72], args[78]]),
        'u_kst2_l_kk_BW_BW_const': onp.array([args[67], args[68], args[73], args[74], args[79], args[80]]).reshape(-1, 2),
        'u_kst2_l_kk_BW_BW_theta': onp.array([args[69], args[70], args[75], args[76], args[81], args[82]]).reshape(-1, 2),
        'phif0_kk_BW_flatte980_mass': onp.array([args[0]]),
        'phif0_kk_BW_flatte980_g_kk': onp.array([args[1]]),
        'phif0_kk_BW_flatte980_rg': onp.array([args[2]]),
        'phif0_kk_BW_flatte980_const': onp.array([0.1, args[3]]).reshape(-1, 2),
        'phif0_kk_BW_flatte980_theta': onp.array([0.1, args[4]]).reshape(-1, 2),
        'phif0_kk_BW_BW_mass': onp.array([args[5], args[59]]),
        'phif0_kk_BW_BW_width': onp.array([args[6], args[60]]),
        'phif0_kk_BW_BW_const': onp.array([args[7], args[8], args[61], args[62]]).reshape(-1, 2),
        'phif0_kk_BW_BW_theta': onp.array([args[9], args[10], args[63], args[64]]).reshape(-1, 2),
        'phif2_kk_BW_flatte1270_mass': onp.array([args[11]]),
        'phif2_kk_BW_flatte1270_width': onp.array([args[12]]),
        'phif2_kk_BW_flatte1270_const': onp.array([args[13], args[14], args[15], args[16], args[17]]).reshape(-1, 5),
        'phif2_kk_BW_flatte1270_theta': onp.array([args[18], args[19], args[20], args[21], args[22]]).reshape(-1, 5),
        'phif2_kk_BW_BW_mass': onp.array([args[23], args[35], args[47]]),
        'phif2_kk_BW_BW_width': onp.array([args[24], args[36], args[48]]),
        'phif2_kk_BW_BW_const': onp.array([args[25], args[26], args[27], args[28], args[29], args[37], args[38], args[39], args[40], args[41], args[49], args[50], args[51], args[52], args[53]]).reshape(-1, 5),
        'phif2_kk_BW_BW_theta': onp.array([args[30], args[31], args[32], args[33], args[34], args[42], args[43], args[44], args[45], args[46], args[54], args[55], args[56], args[57], args[58]]).reshape(-1, 5),
    }

def data_likelihood_kk(args):
    params = extract_parameters(args)
    data_u_kst2_r_kk_BW_BW = calculate_BW_BW(
        params['phi_mass'], params['phi_width'], data_phi_kk,
        params['u_kst2_r_kk_BW_BW_mass'], params['u_kst2_r_kk_BW_BW_width'], data_f_kk,
        data_u_kst2_r_kk, params['u_kst2_r_kk_BW_BW_const'], params['u_kst2_r_kk_BW_BW_theta']
    )
    data_u_kst2_l_kk_BW_BW = calculate_BW_BW(
        params['phi_mass'], params['phi_width'], data_phi_kk,
        params['u_kst2_l_kk_BW_BW_mass'], params['u_kst2_l_kk_BW_BW_width'], data_f_kk,
        data_u_kst2_l_kk, params['u_kst2_l_kk_BW_BW_const'], params['u_kst2_l_kk_BW_BW_theta']
    )
    data_phif0_kk_BW_flatte980 = calculate_BW_flatte980(
        params['phi_mass'], params['phi_width'], data_phi_kk,
        params['phif0_kk_BW_flatte980_mass'], params['phif0_kk_BW_flatte980_g_kk'], params['phif0_kk_BW_flatte980_rg'], data_f_kk,
        data_phif0_kk, params['phif0_kk_BW_flatte980_const'], params['phif0_kk_BW_flatte980_theta']
    )
    data_phif0_kk_BW_BW = calculate_BW_BW(
        params['phi_mass'], params['phi_width'], data_phi_kk,
        params['phif0_kk_BW_BW_mass'], params['phif0_kk_BW_BW_width'], data_f_kk,
        data_phif0_kk, params['phif0_kk_BW_BW_const'], params['phif0_kk_BW_BW_theta']
    )
    data_phif2_kk_BW_flatte1270 = calculate_BW_flatte1270(
        params['phi_mass'], params['phi_width'], data_phi_kk,
        params['phif2_kk_BW_flatte1270_mass'], params['phif2_kk_BW_flatte1270_width'], data_f_kk,
        data_phif2_kk, params['phif2_kk_BW_flatte1270_const'], params['phif2_kk_BW_flatte1270_theta']
    )
    data_phif2_kk_BW_BW = calculate_BW_BW(
        params['phi_mass'], params['phi_width'], data_phi_kk,
        params['phif2_kk_BW_BW_mass'], params['phif2_kk_BW_BW_width'], data_f_kk,
        data_phif2_kk, params['phif2_kk_BW_BW_const'], params['phif2_kk_BW_BW_theta']
    )
    component_data_u_kst2_r_kk_BW_BW = component_BW_BW(
        params['phi_mass'], params['phi_width'], truth_phi_kk,
        params['u_kst2_r_kk_BW_BW_mass'], params['u_kst2_r_kk_BW_BW_width'], truth_f_kk,
        truth_u_kst2_r_kk, params['u_kst2_r_kk_BW_BW_const'], params['u_kst2_r_kk_BW_BW_theta']
    )
    component_data_u_kst2_l_kk_BW_BW = component_BW_BW(
        params['phi_mass'], params['phi_width'], truth_phi_kk,
        params['u_kst2_l_kk_BW_BW_mass'], params['u_kst2_l_kk_BW_BW_width'], truth_f_kk,
        truth_u_kst2_l_kk, params['u_kst2_l_kk_BW_BW_const'], params['u_kst2_l_kk_BW_BW_theta']
    )
    component_data_phif0_kk_BW_flatte980 = component_BW_flatte980(
        params['phi_mass'], params['phi_width'], truth_phi_kk,
        params['phif0_kk_BW_flatte980_mass'], params['phif0_kk_BW_flatte980_g_kk'], params['phif0_kk_BW_flatte980_rg'], truth_f_kk,
        truth_phif0_kk, params['phif0_kk_BW_flatte980_const'], params['phif0_kk_BW_flatte980_theta']
    )
    component_data_phif0_kk_BW_BW = component_BW_BW(
        params['phi_mass'], params['phi_width'], truth_phi_kk,
        params['phif0_kk_BW_BW_mass'], params['phif0_kk_BW_BW_width'], truth_f_kk,
        truth_phif0_kk, params['phif0_kk_BW_BW_const'], params['phif0_kk_BW_BW_theta']
    )
    component_data_phif2_kk_BW_flatte1270 = component_BW_flatte1270(
        params['phi_mass'], params['phi_width'], truth_phi_kk,
        params['phif2_kk_BW_flatte1270_mass'], params['phif2_kk_BW_flatte1270_width'], truth_f_kk,
        truth_phif2_kk, params['phif2_kk_BW_flatte1270_const'], params['phif2_kk_BW_flatte1270_theta']
    )
    component_data_phif2_kk_BW_BW = component_BW_BW(
        params['phi_mass'], params['phi_width'], truth_phi_kk,
        params['phif2_kk_BW_BW_mass'], params['phif2_kk_BW_BW_width'], truth_f_kk,
        truth_phif2_kk, params['phif2_kk_BW_BW_const'], params['phif2_kk_BW_BW_theta']
    )
    sum_frac = np.sum(dplex_dabs(
        np.einsum("mljk->mjk", component_data_u_kst2_r_kk_BW_BW) +
        np.einsum("mljk->mjk", component_data_u_kst2_l_kk_BW_BW) +
        np.einsum("mljk->mjk", component_data_phif0_kk_BW_flatte980) +
        np.einsum("mljk->mjk", component_data_phif0_kk_BW_BW) +
        np.einsum("mljk->mjk", component_data_phif2_kk_BW_flatte1270) +
        np.einsum("mljk->mjk", component_data_phif2_kk_BW_BW)
    ))
    frac_u_kst2_r_BW = np.sum(np.einsum("ljk->l", dplex_dabs(component_data_u_kst2_r_kk_BW_BW)) / sum_frac)
    frac_u_kst2_l_BW = np.sum(np.einsum("ljk->l", dplex_dabs(component_data_u_kst2_l_kk_BW_BW)) / sum_frac)
    frac_f0_flatte = np.sum(np.einsum("ljk->l", dplex_dabs(component_data_phif0_kk_BW_flatte980)) / sum_frac)
    frac_f0_BW = np.sum(np.einsum("ljk->l", dplex_dabs(component_data_phif0_kk_BW_BW)) / sum_frac)
    frac_f2_flatte = np.sum(np.einsum("ljk->l", dplex_dabs(component_data_phif2_kk_BW_flatte1270)) / sum_frac)
    frac_f2_BW = np.sum(np.einsum("ljk->l", dplex_dabs(component_data_phif2_kk_BW_BW)) / sum_frac)
    total_frac = frac_u_kst2_r_BW + frac_u_kst2_l_BW + frac_f0_flatte + frac_f0_BW + frac_f2_flatte + frac_f2_BW
    step_function = data_step_function(total_frac, args)
    total_amplitude = data_u_kst2_r_kk_BW_BW
    total_amplitude = total_amplitude + data_u_kst2_l_kk_BW_BW
    total_amplitude = total_amplitude + data_phif0_kk_BW_flatte980
    total_amplitude = total_amplitude + data_phif0_kk_BW_BW
    total_amplitude = total_amplitude + data_phif2_kk_BW_flatte1270
    total_amplitude = total_amplitude + data_phif2_kk_BW_BW
    likelihood = -np.sum(np.log(np.sum(dplex_dabs(total_amplitude), axis=1))) + step_function
    return likelihood

def mc_likelihood_kk(args):
    params = extract_parameters(args)
    total_mc = calculate_BW_BW(
        params['phi_mass'], params['phi_width'], mc_phi_kk,
        params['u_kst2_r_kk_BW_BW_mass'], params['u_kst2_r_kk_BW_BW_width'], mc_f_kk,
        mc_u_kst2_r_kk, params['u_kst2_r_kk_BW_BW_const'], params['u_kst2_r_kk_BW_BW_theta']
    )
    total_mc = total_mc + calculate_BW_BW(
        params['phi_mass'], params['phi_width'], mc_phi_kk,
        params['u_kst2_l_kk_BW_BW_mass'], params['u_kst2_l_kk_BW_BW_width'], mc_f_kk,
        mc_u_kst2_l_kk, params['u_kst2_l_kk_BW_BW_const'], params['u_kst2_l_kk_BW_BW_theta']
    )
    total_mc = total_mc + calculate_BW_flatte980(
        params['phi_mass'], params['phi_width'], mc_phi_kk,
        params['phif0_kk_BW_flatte980_mass'], params['phif0_kk_BW_flatte980_g_kk'], params['phif0_kk_BW_flatte980_rg'], mc_f_kk,
        mc_phif0_kk, params['phif0_kk_BW_flatte980_const'], params['phif0_kk_BW_flatte980_theta']
    )
    total_mc = total_mc + calculate_BW_BW(
        params['phi_mass'], params['phi_width'], mc_phi_kk,
        params['phif0_kk_BW_BW_mass'], params['phif0_kk_BW_BW_width'], mc_f_kk,
        mc_phif0_kk, params['phif0_kk_BW_BW_const'], params['phif0_kk_BW_BW_theta']
    )
    total_mc = total_mc + calculate_BW_flatte1270(
        params['phi_mass'], params['phi_width'], mc_phi_kk,
        params['phif2_kk_BW_flatte1270_mass'], params['phif2_kk_BW_flatte1270_width'], mc_f_kk,
        mc_phif2_kk, params['phif2_kk_BW_flatte1270_const'], params['phif2_kk_BW_flatte1270_theta']
    )
    total_mc = total_mc + calculate_BW_BW(
        params['phi_mass'], params['phi_width'], mc_phi_kk,
        params['phif2_kk_BW_BW_mass'], params['phif2_kk_BW_BW_width'], mc_f_kk,
        mc_phif2_kk, params['phif2_kk_BW_BW_const'], params['phif2_kk_BW_BW_theta']
    )
    return np.mean(np.sum(dplex_dabs(total_mc), axis=1))

data = load_data()
data = normalize_data(data)
jax_data = prepare_data_for_jax(data)

data_phi_kk = jax_data['data_phi_kk']
data_f_kk = jax_data['data_f_kk']
data_kst2_124_kk = jax_data['data_kst2_124_kk']
data_kst2_123_kk = jax_data['data_kst2_123_kk']
data_phif0_kk = jax_data['data_phif0_kk']
data_phif2_kk = jax_data['data_phif2_kk']
data_u_kst2_r_kk = jax_data['data_u_kst2_r_kk']
data_u_kst2_l_kk = jax_data['data_u_kst2_l_kk']

mc_phi_kk = jax_data['mc_phi_kk']
mc_f_kk = jax_data['mc_f_kk']
mc_kst2_124_kk = jax_data['mc_kst2_124_kk']
mc_kst2_123_kk = jax_data['mc_kst2_123_kk']
mc_phif0_kk = jax_data['mc_phif0_kk']
mc_phif2_kk = jax_data['mc_phif2_kk']
mc_u_kst2_r_kk = jax_data['mc_u_kst2_r_kk']
mc_u_kst2_l_kk = jax_data['mc_u_kst2_l_kk']

truth_phi_kk = jax_data['truth_phi_kk']
truth_f_kk = jax_data['truth_f_kk']
truth_kst2_124_kk = jax_data['truth_kst2_124_kk']
truth_kst2_123_kk = jax_data['truth_kst2_123_kk']
truth_phif0_kk = jax_data['truth_phif0_kk']
truth_phif2_kk = jax_data['truth_phif2_kk']
truth_u_kst2_r_kk = jax_data['truth_u_kst2_r_kk']
truth_u_kst2_l_kk = jax_data['truth_u_kst2_l_kk']

wt_data_kk = jax_data['wt_data_kk']

data_size = len(data_phi_kk)

if __name__  == "__main__":
    """使用Newton-CG方法和HVP的拟合函数"""
    import toml

    logger = setup_logging()
    logger.info("开始HVP优化版PWA拟合（Newton-CG方法）")

    config.update("jax_enable_x64", True)

    constraint_strength = 1000.0
    total_frac_kk = 1.1

    args_list, ranges, initial_errors = make_initial_args()

    def combined_likelihood(args):
        return data_likelihood_kk(args) + data_size * np.log(mc_likelihood_kk(args))

    def hvp_combined_likelihood(x, v):
        return jvp(grad(combined_likelihood), (x,), (v,))[1]

    logger.info("编译JAX函数（HVP版本）...")
    jit_likelihood = jit(combined_likelihood)
    jit_grad = jit(grad(combined_likelihood))
    jit_hvp = jit(hvp_combined_likelihood)

    test_result = jit_likelihood(args_list)
    logger.info(f"初始似然值: {test_result}")

    test_vector = onp.ones_like(args_list)
    test_hvp = jit_hvp(args_list, test_vector)
    logger.info(f"HVP测试完成，结果形状: {test_hvp.shape}")

    def my_callback(x):
        current_likelihood = jit_likelihood(x)
        logger.info(f"当前似然值: {current_likelihood}")

    def hessp(x, p):
        return onp.array(jit_hvp(x, p))

    logger.info("开始优化（Newton-CG + HVP）...")
    start_time = time.time()

    result = minimize(
        fun=lambda x: float(jit_likelihood(x)),
        x0=args_list,
        jac=lambda x: onp.array(jit_grad(x)),
        hessp=hessp,
        method="Newton-CG",
        callback=my_callback,
        options={"disp": False, "xtol": 1e-8}
    )

    end_time = time.time()

    logger.info("="*50)
    logger.info(f"HVP优化完成!")
    logger.info(f"成功: {result.success}")
    logger.info(f"最终似然值: {result.fun}")
    logger.info(f"迭代次数: {result.nit}")
    logger.info(f"函数调用次数: {result.nfev}")
    logger.info(f"梯度调用次数: {result.njev}")
    logger.info(f"Hessian调用次数: {result.nhev}")
    logger.info(f"优化时间: {end_time - start_time:.2f} 秒")
    logger.info(f"优化信息: {result.message}")
    logger.info("="*50)

    logger.info("计算参数误差（Hessian逆矩阵）...")
    args_size = args_list.shape[0]
    hessian_matrix = onp.zeros([args_size, args_size])
    for i in range(args_size):
        v = onp.zeros(args_size)
        v[i] = 1.0
        hessian_matrix[:, i] = onp.array(jit_hvp(result.x, v))
    ferror = onp.sqrt(onp.diag(onp.linalg.inv(hessian_matrix)))
    logger.info(f"误差计算完成")

    os.makedirs("output/fit", exist_ok=True)
    onp.save("output/fit/fit_result_values.npy", result.x)
    onp.save("output/fit/fit_result_errors.npy", ferror)
    logger.info("参数已保存至 output/fit/fit_result_values.npy")

    save_result(result.x, ferror, "output/fit/free_params_fit.toml")
    logger.info("配置已保存至 output/fit/free_params_fit.toml")

def data_step_function(total_frac, args):
    ranges = {
        0: (0.98, 10.0),
        5: (1.704, 1.0),
        6: (0.123, 1.0),
        11: (1.2755, 1.0),
        12: (0.1867, 1.0),
        23: (1.517, 1.0),
        24: (0.086, 1.0),
        35: (2.157, 1.0),
        36: (0.152, 1.0),
        47: (2.345, 0.01),
        48: (0.322, 1.0),
        59: (2.47, 0.007),
        60: (0.075, 0.011),
        65: (2.1, 10.0),
        66: (0.1, 10.0),
        71: (1.819, 10.0),
        72: (0.264, 10.0),
        77: (2.247, 10.0),
        78: (0.18, 10.0),
    }
    step_value = np.power(total_frac - total_frac_kk, 2.0) * constraint_strength
    step_value += sum(np.power(center - args[i], 2.0) / np.power(width, 2.0) / 2.0 for i, (center, width) in ranges.items())
    return step_value

