# Auto-generated draw weight script by LLMResonanceGenerator — do not edit manually
import copy
import json
import logging
import logging.config
import os
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


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ==============================================================================
# SECTION: PHYSICS_FUNCTIONS
# Physics calculation functions (resonance shape functions)
# Provided by the user for each analysis. Copy this file to your analysis
# directory as physics_functions.py and keep only the functions you need.
# ==============================================================================

import jax.numpy as np


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


def load_data():
    data = {}

    # Load real data
    data['data_phi_kk'] = onp.load("data/real_data/phi_kk.npy")
    data['data_f_kk'] = onp.load("data/real_data/f_kk.npy")
    data['data_phif0_kk'] = onp.load("data/real_data/phif0_kk.npy")
    data['data_phif2_kk'] = onp.load("data/real_data/phif2_kk.npy")

    # Load MC data
    data['mc_phi_kk'] = onp.load("data/mc_truth/phi_kk.npy")
    data['mc_f_kk'] = onp.load("data/mc_truth/f_kk.npy")
    data['mc_phif0_kk'] = onp.load("data/mc_truth/phif0_kk.npy")
    data['mc_phif2_kk'] = onp.load("data/mc_truth/phif2_kk.npy")

    # MC truth data (for constraints)
    data['truth_phi_kk'] = data['mc_phi_kk'][0:150000]
    data['truth_f_kk'] = data['mc_f_kk'][0:150000]
    data['truth_phif0_kk'] = data['mc_phif0_kk'][:, 0:150000]
    data['truth_phif2_kk'] = data['mc_phif2_kk'][:, 0:150000]

    # Extra Sbc data (1D kinematic variables, no normalization needed)
    data['data_b123_kk'] = onp.load("data/real_data/b123_kk.npy")
    data['mc_b123_kk'] = onp.load("data/mc_truth/b123_kk.npy")
    data['truth_b123_kk'] = data['mc_b123_kk'][0:150000]
    data['data_b124_kk'] = onp.load("data/real_data/b124_kk.npy")
    data['mc_b124_kk'] = onp.load("data/mc_truth/b124_kk.npy")
    data['truth_b124_kk'] = data['mc_b124_kk'][0:150000]

    return data

def normalize_data(data):
    # 计算归一化因子
    regular_phif0_kk = 1. / onp.average(
        onp.sqrt(onp.sum(onp.asarray(data['mc_phif0_kk'])**2, axis=2)), axis=1
    )
    regular_phif2_kk = 1. / onp.average(
        onp.sqrt(onp.sum(onp.asarray(data['mc_phif2_kk'])**2, axis=2)), axis=1
    )

    # 应用归一化
    data['data_phif0_kk'] = onp.einsum("jkl,j->jkl", data['data_phif0_kk'], regular_phif0_kk)
    data['mc_phif0_kk'] = onp.einsum("jkl,j->jkl", data['mc_phif0_kk'], regular_phif0_kk)
    data['truth_phif0_kk'] = onp.einsum("jkl,j->jkl", data['truth_phif0_kk'], regular_phif0_kk)
    data['data_phif2_kk'] = onp.einsum("jkl,j->jkl", data['data_phif2_kk'], regular_phif2_kk)
    data['mc_phif2_kk'] = onp.einsum("jkl,j->jkl", data['mc_phif2_kk'], regular_phif2_kk)
    data['truth_phif2_kk'] = onp.einsum("jkl,j->jkl", data['truth_phif2_kk'], regular_phif2_kk)

    return data

def prepare_data_for_jax(data, device=None):
    jax_data = {}
    for key, value in data.items():
        jax_data[key] = device_put(np.array(value), device=device)
    return jax_data

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

args_list = onp.array([
    0.9794812115574156, 0.10678616326827592, 8.570187550432664,
    0.1065182971468388, 0.03807025376236671, 1.6761965590304995,
    0.16270071108440043, 0.02201375198386279, 0.007433204877151768,
    0.008302300118288414, -0.018544178045626723, 1.2896149318644679,
    0.1959796900380022, -0.013533706721054747, -0.01650921272412254,
    0.015339403283337268, 0.029798824827026338, 0.020982478502465995,
    0.05458389002821978, 0.016032334798079934, 0.017179622009723918,
    -0.050143087437086675, -0.008718872479379924, 1.5222602842746435,
    2.1619576785269476, 2.547889297662712, 0.08576525399315078,
    0.15906049251159413, 0.324001266488012, -0.005345214125595505,
    0.0031770703345810553, -0.0036743677603351056, 0.004813366936315499,
    0.010000673114238258, 0.004643720949518616, -0.0009682564746806725,
    -0.0029804674908844213, 0.008315390493289363, 0.0005819695034846763,
    -0.15266341362240637, -0.05030214288962155, -0.01856511044577769,
    0.0908719088071242, 0.029544599934778714, -0.010529063356530807,
    0.003910028530572422, 0.0031787953173277725, 0.0333658928584713,
    -0.006053516058970728, 0.00545992748088964, -0.012498776031677225,
    0.002196563552197887, -0.016814307435916394, -0.007232946300343621,
    0.06277085590848677, -0.0031366601030189344, 0.13756083806604205,
    -0.0033564526519642953, 0.06849361819185686
])

def extract_parameters(args):
    """提取所有物理参数的公共函数"""
    return {
        'phi_mass': np.array([1.02]),
        'phi_width': np.array([0.004]),
        'f980_mass': np.array([args[0]]),
        'f980_g_kk': np.array([args[1]]),
        'f980_rg': np.array([args[2]]),
        'f980_const': np.array([0.1, args[3]]).reshape(-1, 2),
        'f980_theta': np.array([0.1, args[4]]).reshape(-1, 2),
        'f0_mass': np.array([args[5]]),
        'f0_width': np.array([args[6]]),
        'f0_const': np.array([args[7], args[8]]).reshape(-1, 2),
        'f0_theta': np.array([args[9], args[10]]).reshape(-1, 2),
        'f1270_mass': np.array([args[11]]),
        'f1270_width': np.array([args[12]]),
        'f1270_const': np.array([args[13], args[14], args[15], args[16], args[17]]).reshape(-1, 5),
        'f1270_theta': np.array([args[18], args[19], args[20], args[21], args[22]]).reshape(-1, 5),
        'f2_mass': np.array([args[23], args[24], args[25]]),
        'f2_width': np.array([args[26], args[27], args[28]]),
        'f2_const': np.array([args[29], args[30], args[31], args[32], args[33],
                               args[34], args[35], args[36], args[37], args[38],
                               args[39], args[40], args[41], args[42], args[43]]).reshape(-1, 5),
        'f2_theta': np.array([args[44], args[45], args[46], args[47], args[48],
                               args[49], args[50], args[51], args[52], args[53],
                               args[54], args[55], args[56], args[57], args[58]]).reshape(-1, 5)
    }

def build_config(args, errors=None):
    def err(i):
        return 0.0 if errors is None else errors[i]

    return {
        'resonances': {
            'phif0_980': {
                'propagators': {
                    'A_propagator': {
                        'propagator_type': 'BW',
                        'mass': {'value': 1.02, 'fixed': True, 'error': 0.0},
                        'width': {'value': 0.004, 'fixed': True, 'error': 0.0},
                        'Sbc': 'phi_kk'
                    },
                    'B_propagator': {
                        'propagator_type': 'flatte980',
                        'mass': {'value': args[0], 'range': [0.98, 10.0], 'fixed': False, 'error': err(0)},
                        'g_kk': {'value': args[1], 'fixed': False, 'error': err(1)},
                        'rg': {'value': args[2], 'fixed': False, 'error': err(2)},
                        'Sbc': 'f_kk'
                    }
                },
                'Amplitude': {
                    'AMP': 'phif0_kk',
                    'const1': {'value': 0.1, 'fixed': True, 'error': 0.0},
                    'const2': {'value': args[3], 'fixed': False, 'error': err(3)},
                    'theta1': {'value': 0.1, 'fixed': True, 'error': 0.0},
                    'theta2': {'value': args[4], 'fixed': False, 'error': err(4)}
                }
            },
            'phif0_1710': {
                'propagators': {
                    'A_propagator': {
                        'propagator_type': 'BW',
                        'mass': {'value': 1.02, 'fixed': True, 'error': 0.0},
                        'width': {'value': 0.004, 'fixed': True, 'error': 0.0},
                        'Sbc': 'phi_kk'
                    },
                    'B_propagator': {
                        'propagator_type': 'BW',
                        'mass': {'value': args[5], 'range': [1.704, 1.0], 'fixed': False, 'error': err(5)},
                        'width': {'value': args[6], 'range': [0.123, 1.0], 'fixed': False, 'error': err(6)},
                        'Sbc': 'f_kk'
                    }
                },
                'Amplitude': {
                    'AMP': 'phif0_kk',
                    'const1': {'value': args[7], 'fixed': False, 'error': err(7)},
                    'const2': {'value': args[8], 'fixed': False, 'error': err(8)},
                    'theta1': {'value': args[9], 'fixed': False, 'error': err(9)},
                    'theta2': {'value': args[10], 'fixed': False, 'error': err(10)}
                }
            },
            'phif2_1270': {
                'propagators': {
                    'A_propagator': {
                        'propagator_type': 'BW',
                        'mass': {'value': 1.02, 'fixed': True, 'error': 0.0},
                        'width': {'value': 0.004, 'fixed': True, 'error': 0.0},
                        'Sbc': 'phi_kk'
                    },
                    'B_propagator': {
                        'propagator_type': 'flatte1270',
                        'mass': {'value': args[11], 'range': [1.2755, 1.0], 'fixed': False, 'error': err(11)},
                        'width': {'value': args[12], 'range': [0.1867, 1.0], 'fixed': False, 'error': err(12)},
                        'Sbc': 'f_kk'
                    }
                },
                'Amplitude': {
                    'AMP': 'phif2_kk',
                    'const1': {'value': args[13], 'fixed': False, 'error': err(13)},
                    'const2': {'value': args[14], 'fixed': False, 'error': err(14)},
                    'const3': {'value': args[15], 'fixed': False, 'error': err(15)},
                    'const4': {'value': args[16], 'fixed': False, 'error': err(16)},
                    'const5': {'value': args[17], 'fixed': False, 'error': err(17)},
                    'theta1': {'value': args[18], 'fixed': False, 'error': err(18)},
                    'theta2': {'value': args[19], 'fixed': False, 'error': err(19)},
                    'theta3': {'value': args[20], 'fixed': False, 'error': err(20)},
                    'theta4': {'value': args[21], 'fixed': False, 'error': err(21)},
                    'theta5': {'value': args[22], 'fixed': False, 'error': err(22)}
                }
            },
            'phif2_1525': {
                'propagators': {
                    'A_propagator': {
                        'propagator_type': 'BW',
                        'mass': {'value': 1.02, 'fixed': True, 'error': 0.0},
                        'width': {'value': 0.004, 'fixed': True, 'error': 0.0},
                        'Sbc': 'phi_kk'
                    },
                    'B_propagator': {
                        'propagator_type': 'BW',
                        'mass': {'value': args[23], 'range': [1.517, 1.0], 'fixed': False, 'error': err(23)},
                        'width': {'value': args[26], 'range': [0.086, 1.0], 'fixed': False, 'error': err(26)},
                        'Sbc': 'f_kk'
                    }
                },
                'Amplitude': {
                    'AMP': 'phif2_kk',
                    'const1': {'value': args[29], 'fixed': False, 'error': err(29)},
                    'const2': {'value': args[30], 'fixed': False, 'error': err(30)},
                    'const3': {'value': args[31], 'fixed': False, 'error': err(31)},
                    'const4': {'value': args[32], 'fixed': False, 'error': err(32)},
                    'const5': {'value': args[33], 'fixed': False, 'error': err(33)},
                    'theta1': {'value': args[44], 'fixed': False, 'error': err(44)},
                    'theta2': {'value': args[45], 'fixed': False, 'error': err(45)},
                    'theta3': {'value': args[46], 'fixed': False, 'error': err(46)},
                    'theta4': {'value': args[47], 'fixed': False, 'error': err(47)},
                    'theta5': {'value': args[48], 'fixed': False, 'error': err(48)}
                }
            },
            'phif2_2150': {
                'propagators': {
                    'A_propagator': {
                        'propagator_type': 'BW',
                        'mass': {'value': 1.02, 'fixed': True, 'error': 0.0},
                        'width': {'value': 0.004, 'fixed': True, 'error': 0.0},
                        'Sbc': 'phi_kk'
                    },
                    'B_propagator': {
                        'propagator_type': 'BW',
                        'mass': {'value': args[24], 'range': [2.157, 1.0], 'fixed': False, 'error': err(24)},
                        'width': {'value': args[27], 'range': [0.152, 1.0], 'fixed': False, 'error': err(27)},
                        'Sbc': 'f_kk'
                    }
                },
                'Amplitude': {
                    'AMP': 'phif2_kk',
                    'const1': {'value': args[34], 'fixed': False, 'error': err(34)},
                    'const2': {'value': args[35], 'fixed': False, 'error': err(35)},
                    'const3': {'value': args[36], 'fixed': False, 'error': err(36)},
                    'const4': {'value': args[37], 'fixed': False, 'error': err(37)},
                    'const5': {'value': args[38], 'fixed': False, 'error': err(38)},
                    'theta1': {'value': args[49], 'fixed': False, 'error': err(49)},
                    'theta2': {'value': args[50], 'fixed': False, 'error': err(50)},
                    'theta3': {'value': args[51], 'fixed': False, 'error': err(51)},
                    'theta4': {'value': args[52], 'fixed': False, 'error': err(52)},
                    'theta5': {'value': args[53], 'fixed': False, 'error': err(53)}
                }
            },
            'phif2_2340': {
                'propagators': {
                    'A_propagator': {
                        'propagator_type': 'BW',
                        'mass': {'value': 1.02, 'fixed': True, 'error': 0.0},
                        'width': {'value': 0.004, 'fixed': True, 'error': 0.0},
                        'Sbc': 'phi_kk'
                    },
                    'B_propagator': {
                        'propagator_type': 'BW',
                        'mass': {'value': args[25], 'range': [2.345, 0.01], 'fixed': False, 'error': err(25)},
                        'width': {'value': args[28], 'range': [0.322, 1.0], 'fixed': False, 'error': err(28)},
                        'Sbc': 'f_kk'
                    }
                },
                'Amplitude': {
                    'AMP': 'phif2_kk',
                    'const1': {'value': args[39], 'fixed': False, 'error': err(39)},
                    'const2': {'value': args[40], 'fixed': False, 'error': err(40)},
                    'const3': {'value': args[41], 'fixed': False, 'error': err(41)},
                    'const4': {'value': args[42], 'fixed': False, 'error': err(42)},
                    'const5': {'value': args[43], 'fixed': False, 'error': err(43)},
                    'theta1': {'value': args[54], 'fixed': False, 'error': err(54)},
                    'theta2': {'value': args[55], 'fixed': False, 'error': err(55)},
                    'theta3': {'value': args[56], 'fixed': False, 'error': err(56)},
                    'theta4': {'value': args[57], 'fixed': False, 'error': err(57)},
                    'theta5': {'value': args[58], 'fixed': False, 'error': err(58)}
                }
            }
        },
        'draw': {
            'extra_sbc': ['b123_kk', 'b124_kk']
        }
    }

def weight_kk(args):
    params = extract_parameters(args)
    comp_phif0_kk_BW_flatte980 = component_BW_flatte980(
        params['phi_mass'], params['phi_width'], data_phi_kk,
        params['f980_mass'], params['f980_g_kk'], params['f980_rg'], data_f_kk,
        data_phif0_kk, params['f980_const'], params['f980_theta']
    )
    comp_phif0_kk_BW_BW = component_BW_BW(
        params['phi_mass'], params['phi_width'], data_phi_kk,
        params['f0_mass'], params['f0_width'], data_f_kk,
        data_phif0_kk, params['f0_const'], params['f0_theta']
    )
    comp_phif2_kk_BW_flatte1270 = component_BW_flatte1270(
        params['phi_mass'], params['phi_width'], data_phi_kk,
        params['f1270_mass'], params['f1270_width'], data_f_kk,
        data_phif2_kk, params['f1270_const'], params['f1270_theta']
    )
    comp_phif2_kk_BW_BW = component_BW_BW(
        params['phi_mass'], params['phi_width'], data_phi_kk,
        params['f2_mass'], params['f2_width'], data_f_kk,
        data_phif2_kk, params['f2_const'], params['f2_theta']
    )
    total_wt = np.sum(dplex_dabs(
        np.einsum("mljk->mjk", comp_phif0_kk_BW_flatte980) +
        np.einsum("mljk->mjk", comp_phif0_kk_BW_BW) +
        np.einsum("mljk->mjk", comp_phif2_kk_BW_flatte1270) +
        np.einsum("mljk->mjk", comp_phif2_kk_BW_BW)
    ), axis=1)
    wt_list = [
        total_wt,
        np.einsum("ljk->lj", dplex_dabs(comp_phif0_kk_BW_flatte980)),
        np.einsum("ljk->lj", dplex_dabs(comp_phif0_kk_BW_BW)),
        np.einsum("ljk->lj", dplex_dabs(comp_phif2_kk_BW_flatte1270)),
        np.einsum("ljk->lj", dplex_dabs(comp_phif2_kk_BW_BW)),
    ]
    return wt_list

def weight_truth_kk(args):
    params = extract_parameters(args)
    comp_phif0_kk_BW_flatte980 = component_BW_flatte980(
        params['phi_mass'], params['phi_width'], truth_phi_kk,
        params['f980_mass'], params['f980_g_kk'], params['f980_rg'], truth_f_kk,
        truth_phif0_kk, params['f980_const'], params['f980_theta']
    )
    comp_phif0_kk_BW_BW = component_BW_BW(
        params['phi_mass'], params['phi_width'], truth_phi_kk,
        params['f0_mass'], params['f0_width'], truth_f_kk,
        truth_phif0_kk, params['f0_const'], params['f0_theta']
    )
    comp_phif2_kk_BW_flatte1270 = component_BW_flatte1270(
        params['phi_mass'], params['phi_width'], truth_phi_kk,
        params['f1270_mass'], params['f1270_width'], truth_f_kk,
        truth_phif2_kk, params['f1270_const'], params['f1270_theta']
    )
    comp_phif2_kk_BW_BW = component_BW_BW(
        params['phi_mass'], params['phi_width'], truth_phi_kk,
        params['f2_mass'], params['f2_width'], truth_f_kk,
        truth_phif2_kk, params['f2_const'], params['f2_theta']
    )
    total_wt = np.sum(dplex_dabs(
        np.einsum("mljk->mjk", comp_phif0_kk_BW_flatte980) +
        np.einsum("mljk->mjk", comp_phif0_kk_BW_BW) +
        np.einsum("mljk->mjk", comp_phif2_kk_BW_flatte1270) +
        np.einsum("mljk->mjk", comp_phif2_kk_BW_BW)
    ), axis=1)
    wt_list = [
        total_wt,
        np.einsum("ljk->lj", dplex_dabs(comp_phif0_kk_BW_flatte980)),
        np.einsum("ljk->lj", dplex_dabs(comp_phif0_kk_BW_BW)),
        np.einsum("ljk->lj", dplex_dabs(comp_phif2_kk_BW_flatte1270)),
        np.einsum("ljk->lj", dplex_dabs(comp_phif2_kk_BW_BW)),
    ]
    return wt_list

def run_weight(args_list, mode="pass"):
    args = np.array(args_list)
    jit_weight = jit(weight_kk)
    jit_weight_truth = jit(weight_truth_kk)
    if mode == "pass":
        wt_list = jit_weight(args)
    elif mode == "truth":
        wt_list = jit_weight_truth(args)
    sum_wt = onp.sum(wt_list[0])
    total_weight = {"all_mods_wt": wt_list[0]}
    total_fit_frac = 0.0
    resonance_names = ["phif0_kk_BW_flatte980", "phif0_kk_BW_BW", "phif2_kk_BW_flatte1270", "phif2_kk_BW_BW"]
    for i, name in enumerate(resonance_names):
        wt = wt_list[i + 1]
        for j in range(wt.shape[0]):
            key = f"{name}_{j}"
            total_weight[key] = wt[j]
            frac = onp.sum(wt[j]) / sum_wt
            print(f"{key} frac: {frac}")
            total_fit_frac += frac
    total_weight["fit_value"] = onp.array(args_list)
    total_weight["sum_wt"] = sum_wt
    print(f"total fit fraction: {total_fit_frac}")
    os.makedirs("output/draw", exist_ok=True)
    if mode == "pass":
        onp.savez("output/draw/weight.npz", **total_weight)
    elif mode == "truth":
        onp.savez("output/draw/weight_truth.npz", **total_weight)
    return total_fit_frac

data = load_data()
data = normalize_data(data)
jax_data = prepare_data_for_jax(data)

data_phi_kk = jax_data['data_phi_kk']
data_f_kk = jax_data['data_f_kk']
data_phif0_kk = jax_data['data_phif0_kk']
data_phif2_kk = jax_data['data_phif2_kk']
data_b123_kk = jax_data['data_b123_kk']
data_b124_kk = jax_data['data_b124_kk']

mc_phi_kk = jax_data['mc_phi_kk']
mc_f_kk = jax_data['mc_f_kk']
mc_phif0_kk = jax_data['mc_phif0_kk']
mc_phif2_kk = jax_data['mc_phif2_kk']
mc_b123_kk = jax_data['mc_b123_kk']
mc_b124_kk = jax_data['mc_b124_kk']

truth_phi_kk = jax_data['truth_phi_kk']
truth_f_kk = jax_data['truth_f_kk']
truth_phif0_kk = jax_data['truth_phif0_kk']
truth_phif2_kk = jax_data['truth_phif2_kk']
truth_b123_kk = jax_data['truth_b123_kk']
truth_b124_kk = jax_data['truth_b124_kk']

if __name__ == "__main__":
    config.update("jax_enable_x64", True)

    args_list = onp.load("output/fit/fit_result_values.npy")

    print("计算数据权重 (mode=pass)...")
    run_weight(args_list, mode="pass")

    print("计算 truth MC 权重 (mode=truth)...")
    run_weight(args_list, mode="truth")

    print("权重计算完成，结果已保存至 output/draw/")