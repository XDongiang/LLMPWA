
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
from jax import device_put, grad, jit, vmap, jvp, config
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
    return aa[0]**2 + aa[1]**2

def dplex_dtomine(aa):
    return np.stack([np.real(aa), np.imag(aa)], axis=0)

def dplex_dconstruct(aa, bb):
    return np.stack([aa, bb], axis=0)

def dplex_ddivide(a, bb):
    real = a * bb[0] / dplex_dabs(bb)
    imag = -a * bb[1] / dplex_dabs(bb)
    return np.stack([real, imag], axis=0)

def BW(m_,w_,Sbc):
    l = (Sbc.shape)[0]
    temp = dplex_dconstruct(m_*m_ - Sbc,  -m_*w_*np.ones(l))
    return dplex_ddivide(1.0, temp)

def BW_relativity(m_,w_,Sbc):
    gamma=np.sqrt(m_*m_*(m_*m_+w_*w_))
    k = np.sqrt(2*np.sqrt(2)*m_*np.abs(w_)*gamma/np.pi/np.sqrt(m_*m_+gamma))
    l = (Sbc.shape)[0]
    temp = dplex_dconstruct(m_*m_ - Sbc,  -m_*w_*np.ones(l))
    return dplex_ddivide(k, temp)

def flatte980(m_,g_pipi,rg,Sbc):
    g_kk = rg * g_pipi
    m_k = 0.493677
    m_pi = 0.13957061
    rho_kk = np.sqrt(np.abs(1 - 4*m_k*m_k / Sbc))
    rho_pipi = np.sqrt(np.abs(1 - 4*m_pi*m_pi / Sbc))
    tmp_A = dplex_dconstruct(m_**2 - Sbc, -1*(g_pipi*rho_pipi + g_kk*rho_kk))
    return dplex_ddivide(1.0, tmp_A)

def flatte1270(m_,w_,Sbc):
    rm = m_ * m_
    gr = m_ * w_
    q2r = 0.25 * rm - 0.0194792
    b2r = q2r * (q2r + 0.1825) + 0.033306
    g11270 = gr * b2r / np.power(q2r,2.5)
    q2 = 0.25 * Sbc - 0.0194792
    b2 = q2 * (q2 + 0.1825) + 0.033306
    g1 = g11270 * np.power(q2,2.5) / b2
    tmp = dplex_dconstruct(Sbc - rm, g1)
    return dplex_ddivide(gr, tmp)

def flatte500(m_,b1,b2,b3,b4,b5,Sbc):
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

    data['data_phi_kk'] = onp.load("data/real_data/phi_kk.npy")
    data['data_f_kk'] = onp.load("data/real_data/f_kk.npy")
    data['data_phif0_kk'] = onp.load("data/real_data/phif0_kk.npy")
    data['data_phif2_kk'] = onp.load("data/real_data/phif2_kk.npy")

    data['mc_phi_kk'] = onp.load("data/mc_truth/phi_kk.npy")
    data['mc_f_kk'] = onp.load("data/mc_truth/f_kk.npy")
    data['mc_phif0_kk'] = onp.load("data/mc_truth/phif0_kk.npy")
    data['mc_phif2_kk'] = onp.load("data/mc_truth/phif2_kk.npy")

    data['truth_phi_kk'] = data['mc_phi_kk'][0:150000]
    data['truth_f_kk'] = data['mc_f_kk'][0:150000]
    data['truth_phif0_kk'] = data['mc_phif0_kk'][:, 0:150000]
    data['truth_phif2_kk'] = data['mc_phif2_kk'][:, 0:150000]

    try:
        data['wt_data_kk'] = onp.load("data/weight/weight_kk.npy")
    except FileNotFoundError:
        data['wt_data_kk'] = onp.ones_like(data['data_phi_kk'])

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

    def param(value, fixed, error=0.0, range_value=None):
        result = {'value': value, 'fixed': fixed}
        if range_value is not None:
            result['range'] = range_value
        result['error'] = error
        return result

    return {
        'resonances': {
            'phif0_980': {
                'propagators': {
                    'A_propagator': {
                        'propagator_type': 'BW',
                        'mass': param(1.02, True),
                        'width': param(0.004, True),
                        'Sbc': 'phi_kk'
                    },
                    'B_propagator': {
                        'propagator_type': 'flatte980',
                        'mass': param(args[0], False, err(0), [0.98, 10.0]),
                        'g_kk': param(args[1], False, err(1)),
                        'rg': param(args[2], False, err(2)),
                        'Sbc': 'f_kk'
                    }
                },
                'Amplitude': {
                    'AMP': 'phif0_kk',
                    'const1': param(0.1, True),
                    'const2': param(args[3], False, err(3)),
                    'theta1': param(0.1, True),
                    'theta2': param(args[4], False, err(4))
                }
            },
            'phif0_1710': {
                'propagators': {
                    'A_propagator': {
                        'propagator_type': 'BW',
                        'mass': param(1.02, True),
                        'width': param(0.004, True),
                        'Sbc': 'phi_kk'
                    },
                    'B_propagator': {
                        'propagator_type': 'BW',
                        'mass': param(args[5], False, err(5), [1.704, 1.0]),
                        'width': param(args[6], False, err(6), [0.123, 1.0]),
                        'Sbc': 'f_kk'
                    }
                },
                'Amplitude': {
                    'AMP': 'phif0_kk',
                    'const1': param(args[7], False, err(7)),
                    'const2': param(args[8], False, err(8)),
                    'theta1': param(args[9], False, err(9)),
                    'theta2': param(args[10], False, err(10))
                }
            },
            'phif2_1270': {
                'propagators': {
                    'A_propagator': {
                        'propagator_type': 'BW',
                        'mass': param(1.02, True),
                        'width': param(0.004, True),
                        'Sbc': 'phi_kk'
                    },
                    'B_propagator': {
                        'propagator_type': 'flatte1270',
                        'mass': param(args[11], False, err(11), [1.2755, 1.0]),
                        'width': param(args[12], False, err(12), [0.1867, 1.0]),
                        'Sbc': 'f_kk'
                    }
                },
                'Amplitude': {
                    'AMP': 'phif2_kk',
                    'const1': param(args[13], False, err(13)),
                    'const2': param(args[14], False, err(14)),
                    'const3': param(args[15], False, err(15)),
                    'const4': param(args[16], False, err(16)),
                    'const5': param(args[17], False, err(17)),
                    'theta1': param(args[18], False, err(18)),
                    'theta2': param(args[19], False, err(19)),
                    'theta3': param(args[20], False, err(20)),
                    'theta4': param(args[21], False, err(21)),
                    'theta5': param(args[22], False, err(22))
                }
            },
            'phif2_1525': {
                'propagators': {
                    'A_propagator': {
                        'propagator_type': 'BW',
                        'mass': param(1.02, True),
                        'width': param(0.004, True),
                        'Sbc': 'phi_kk'
                    },
                    'B_propagator': {
                        'propagator_type': 'BW',
                        'mass': param(args[23], False, err(23), [1.517, 1.0]),
                        'width': param(args[26], False, err(26), [0.086, 1.0]),
                        'Sbc': 'f_kk'
                    }
                },
                'Amplitude': {
                    'AMP': 'phif2_kk',
                    'const1': param(args[29], False, err(29)),
                    'const2': param(args[30], False, err(30)),
                    'const3': param(args[31], False, err(31)),
                    'const4': param(args[32], False, err(32)),
                    'const5': param(args[33], False, err(33)),
                    'theta1': param(args[44], False, err(44)),
                    'theta2': param(args[45], False, err(45)),
                    'theta3': param(args[46], False, err(46)),
                    'theta4': param(args[47], False, err(47)),
                    'theta5': param(args[48], False, err(48))
                }
            },
            'phif2_2150': {
                'propagators': {
                    'A_propagator': {
                        'propagator_type': 'BW',
                        'mass': param(1.02, True),
                        'width': param(0.004, True),
                        'Sbc': 'phi_kk'
                    },
                    'B_propagator': {
                        'propagator_type': 'BW',
                        'mass': param(args[24], False, err(24), [2.157, 1.0]),
                        'width': param(args[27], False, err(27), [0.152, 1.0]),
                        'Sbc': 'f_kk'
                    }
                },
                'Amplitude': {
                    'AMP': 'phif2_kk',
                    'const1': param(args[34], False, err(34)),
                    'const2': param(args[35], False, err(35)),
                    'const3': param(args[36], False, err(36)),
                    'const4': param(args[37], False, err(37)),
                    'const5': param(args[38], False, err(38)),
                    'theta1': param(args[49], False, err(49)),
                    'theta2': param(args[50], False, err(50)),
                    'theta3': param(args[51], False, err(51)),
                    'theta4': param(args[52], False, err(52)),
                    'theta5': param(args[53], False, err(53))
                }
            },
            'phif2_2340': {
                'propagators': {
                    'A_propagator': {
                        'propagator_type': 'BW',
                        'mass': param(1.02, True),
                        'width': param(0.004, True),
                        'Sbc': 'phi_kk'
                    },
                    'B_propagator': {
                        'propagator_type': 'BW',
                        'mass': param(args[25], False, err(25), [2.345, 0.01]),
                        'width': param(args[28], False, err(28), [0.322, 1.0]),
                        'Sbc': 'f_kk'
                    }
                },
                'Amplitude': {
                    'AMP': 'phif2_kk',
                    'const1': param(args[39], False, err(39)),
                    'const2': param(args[40], False, err(40)),
                    'const3': param(args[41], False, err(41)),
                    'const4': param(args[42], False, err(42)),
                    'const5': param(args[43], False, err(43)),
                    'theta1': param(args[54], False, err(54)),
                    'theta2': param(args[55], False, err(55)),
                    'theta3': param(args[56], False, err(56)),
                    'theta4': param(args[57], False, err(57)),
                    'theta5': param(args[58], False, err(58))
                }
            }
        }
    }

def data_likelihood_kk(args):
    params = extract_parameters(args)
    data_phif0_kk_BW_flatte980 = calculate_BW_flatte980(
        params['phi_mass'], params['phi_width'], data_phi_kk,
        params['f980_mass'], params['f980_g_kk'], params['f980_rg'], data_f_kk,
        data_phif0_kk, params['f980_const'], params['f980_theta']
    )
    data_phif0_kk_BW_BW = calculate_BW_BW(
        params['phi_mass'], params['phi_width'], data_phi_kk,
        params['f0_mass'], params['f0_width'], data_f_kk,
        data_phif0_kk, params['f0_const'], params['f0_theta']
    )
    data_phif2_kk_BW_flatte1270 = calculate_BW_flatte1270(
        params['phi_mass'], params['phi_width'], data_phi_kk,
        params['f1270_mass'], params['f1270_width'], data_f_kk,
        data_phif2_kk, params['f1270_const'], params['f1270_theta']
    )
    data_phif2_kk_BW_BW = calculate_BW_BW(
        params['phi_mass'], params['phi_width'], data_phi_kk,
        params['f2_mass'], params['f2_width'], data_f_kk,
        data_phif2_kk, params['f2_const'], params['f2_theta']
    )
    component_data_phif0_kk_BW_flatte980 = component_BW_flatte980(
        params['phi_mass'], params['phi_width'], truth_phi_kk,
        params['f980_mass'], params['f980_g_kk'], params['f980_rg'], truth_f_kk,
        truth_phif0_kk, params['f980_const'], params['f980_theta']
    )
    component_data_phif0_kk_BW_BW = component_BW_BW(
        params['phi_mass'], params['phi_width'], truth_phi_kk,
        params['f0_mass'], params['f0_width'], truth_f_kk,
        truth_phif0_kk, params['f0_const'], params['f0_theta']
    )
    component_data_phif2_kk_BW_flatte1270 = component_BW_flatte1270(
        params['phi_mass'], params['phi_width'], truth_phi_kk,
        params['f1270_mass'], params['f1270_width'], truth_f_kk,
        truth_phif2_kk, params['f1270_const'], params['f1270_theta']
    )
    component_data_phif2_kk_BW_BW = component_BW_BW(
        params['phi_mass'], params['phi_width'], truth_phi_kk,
        params['f2_mass'], params['f2_width'], truth_f_kk,
        truth_phif2_kk, params['f2_const'], params['f2_theta']
    )
    sum_frac = np.sum(dplex_dabs(
        np.einsum("mljk->mjk", component_data_phif0_kk_BW_flatte980) +
        np.einsum("mljk->mjk", component_data_phif0_kk_BW_BW) +
        np.einsum("mljk->mjk", component_data_phif2_kk_BW_flatte1270) +
        np.einsum("mljk->mjk", component_data_phif2_kk_BW_BW)
    ))
    frac_f980_flatte = np.sum(np.einsum("ljk->l", dplex_dabs(component_data_phif0_kk_BW_flatte980)) / sum_frac)
    frac_f0_BW = np.sum(np.einsum("ljk->l", dplex_dabs(component_data_phif0_kk_BW_BW)) / sum_frac)
    frac_f1270_flatte = np.sum(np.einsum("ljk->l", dplex_dabs(component_data_phif2_kk_BW_flatte1270)) / sum_frac)
    frac_f2_BW = np.sum(np.einsum("ljk->l", dplex_dabs(component_data_phif2_kk_BW_BW)) / sum_frac)
    total_frac = frac_f980_flatte + frac_f0_BW + frac_f1270_flatte + frac_f2_BW
    step_function = np.power(total_frac - 1.03, 2.0) * constraint_strength
    total_amplitude = data_phif0_kk_BW_flatte980
    total_amplitude = total_amplitude + data_phif0_kk_BW_BW
    total_amplitude = total_amplitude + data_phif2_kk_BW_flatte1270
    total_amplitude = total_amplitude + data_phif2_kk_BW_BW
    likelihood = -np.sum(np.log(np.sum(dplex_dabs(total_amplitude), axis=1))) + 10000.0 * step_function
    return likelihood

def mc_likelihood_kk(args):
    params = extract_parameters(args)
    total_mc = calculate_BW_flatte980(
        params['phi_mass'], params['phi_width'], mc_phi_kk,
        params['f980_mass'], params['f980_g_kk'], params['f980_rg'], mc_f_kk,
        mc_phif0_kk, params['f980_const'], params['f980_theta']
    )
    total_mc = total_mc + calculate_BW_BW(
        params['phi_mass'], params['phi_width'], mc_phi_kk,
        params['f0_mass'], params['f0_width'], mc_f_kk,
        mc_phif0_kk, params['f0_const'], params['f0_theta']
    )
    total_mc = total_mc + calculate_BW_flatte1270(
        params['phi_mass'], params['phi_width'], mc_phi_kk,
        params['f1270_mass'], params['f1270_width'], mc_f_kk,
        mc_phif2_kk, params['f1270_const'], params['f1270_theta']
    )
    total_mc = total_mc + calculate_BW_BW(
        params['phi_mass'], params['phi_width'], mc_phi_kk,
        params['f2_mass'], params['f2_width'], mc_f_kk,
        mc_phif2_kk, params['f2_const'], params['f2_theta']
    )
    return np.mean(np.sum(dplex_dabs(total_mc), axis=1))

def combined_likelihood(args):
    """组合似然函数: data_likelihood + datasize * log(mc_likelihood)"""

    data_lh = data_likelihood_kk(args)

    mc_lh = mc_likelihood_kk(args)

    combined_lh = data_lh + data_size * np.log(mc_lh)

    return combined_lh

def hvp_combined_likelihood(args, vector):
    """计算Hessian-vector product for combined likelihood"""
    return jvp(grad(combined_likelihood), [args], [vector])[1]

data = load_data()
data = normalize_data(data)
jax_data = prepare_data_for_jax(data)

data_phi_kk = jax_data['data_phi_kk']
data_f_kk = jax_data['data_f_kk']
data_phif0_kk = jax_data['data_phif0_kk']
data_phif2_kk = jax_data['data_phif2_kk']

mc_phi_kk = jax_data['mc_phi_kk']
mc_f_kk = jax_data['mc_f_kk']
mc_phif0_kk = jax_data['mc_phif0_kk']
mc_phif2_kk = jax_data['mc_phif2_kk']

truth_phi_kk = jax_data['truth_phi_kk']
truth_f_kk = jax_data['truth_f_kk']
truth_phif0_kk = jax_data['truth_phif0_kk']
truth_phif2_kk = jax_data['truth_phif2_kk']

wt_data_kk = jax_data['wt_data_kk']

data_size = len(data_phi_kk)

if __name__ == "__main__":
    logger = setup_logging()
    logger.info("开始HVP优化版PWA拟合（Newton-CG方法）")

    config.update("jax_enable_x64", True)

    constraint_strength = 0.0

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

    result_config = build_config(result.x, ferror)
    with open("output/fit/fit_result_config.json", "w", encoding="utf-8") as f:
        json.dump(result_config, f, indent=4)
    logger.info("配置已保存至 output/fit/fit_result_config.json")