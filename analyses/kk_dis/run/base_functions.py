import json
import logging
import logging.config
import os
import toml
# must set before importing jax
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import time
import numpy as onp
from functools import partial
from scipy.optimize import minimize

import jax.numpy as np
from jax import device_put, grad, jit, vmap, jvp
from jax import config
from jax.sharding import NamedSharding, PartitionSpec as P




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

# ---------------------------------------------------------------------------
# Hybrid helicity amplitude layer.
#
# Reference: task/20260929_螺旋度耦合振幅推导/3_报告/tensor_jax_helicity_rewrite.md
#            (P0-3 fixes the convention, P1-1/P1-3 are implemented here)
#
# EPS_A contracts the covariant tensor T(e, mu) with the psi(2S) polarisation
# vectors to give helicity amplitudes:
#
#     rows    = lambda = +1, 0, -1
#     columns = mu     = x, y, z          (metric diag(-1,-1,-1,1))
#
#     EPS_A = [[ 1/sqrt2, -i/sqrt2, 0],
#              [ 0,        0,       1],
#              [-1/sqrt2, -i/sqrt2, 0]]
#
# The row convention is load bearing: with the mirrored (1,+i) convention the
# per-event overlap with tf-pwa's HelicityDecay(+add_covariant_term) drops from
# 1.000000 to 0.62-0.67 on the reference 2000-event sample.  Do not transpose
# EPS_A: A^T_{mu,lambda} T_mu is a different object.
#
# Only the two transverse rows are reachable while the cached tensors store
# (x, y) only; the lambda = 0 slot is then EXACTLY zero.  Cache the z component
# too (P2-1) and the same code picks it up with no further change.
# ---------------------------------------------------------------------------
EPS_A = onp.array(
    [
        [1.0 / onp.sqrt(2.0), -1.0j / onp.sqrt(2.0), 0.0],
        [0.0, 0.0, 1.0],
        [-1.0 / onp.sqrt(2.0), -1.0j / onp.sqrt(2.0), 0.0],
    ],
    dtype=onp.complex128,
)

# Slots of the helicity axis that carry the transverse initial state.
#   k = 3  -> (lambda=+1, lambda=0, lambda=-1)   -> (0, 2)
#   k = 2  -> (lambda=+1, lambda=-1)             -> (0, 1)
HELICITY_TRANSVERSE = {2: (0, 1), 3: (0, 2)}

HELICITY_LAYER_ENV = "HELICITY_LAYER"
TENSOR_BASIS_ENV = "TENSOR_BASIS"


def helicity_layer_enabled():
    """Runtime switch, so old and new amplitude layers can be compared in one
    process (HELICITY_LAYER=1 selects the helicity layer)."""
    return os.environ.get(HELICITY_LAYER_ENV, "0") == "1"


def precomputed_omega_basis():
    """True when the phif2 cache is the CHANNEL-INDEPENDENT Omega + constants.

    Method 4 (2026-10-07).  On disk:

        Omega.npy  (N, 3, 3, 5) complex   D^{1*}_{lam_A, lam_phi-lam_f}
                                          x f^1_{|lam_phi|}(gamma_phi)
                                          x f^2_{|lam_f|}(gamma_f2)
                                          x a_phi[lam_phi] x a_f[lam_f]
        K.npy      (5, N)       float     per-channel node factors (Blatt-Weisskopf x q^l)
        M.npy      (5, 3, 5)    float     constant CG matrix (== tf-pwa's _get_cg_matrix)
        g.npy      (5,)         float     per-channel convention constant

    and `W_i[e,lam_A] = g_i K_i(e) sum_{a,b} M_i[lam_A,a,b] Omega[e,lam_A,a,b]`
    is assembled at load time.  The phif0 channels keep their plain cache.
    """
    return os.environ.get(TENSOR_BASIS_ENV, "cartesian") == "omega"


def precomputed_helicity_basis():
    """True when the cached arrays are ALREADY W_i(e,lambda).

    TENSOR_BASIS=helicity means the EPS_A rotation was folded into tensor
    generation, so the fit loop must not rotate again.  See report section 8.6
    for why this is mathematically valid but usually a pessimisation: the
    Cartesian cache is real, so `dplex_deinsum_ord` contracts it with 2 real
    einsums, while a rotated cache is complex and needs 4.

    TENSOR_BASIS=omega also lands here: method 4's cache is complex W_i as well,
    only assembled from the channel-independent Omega instead of stored whole.
    Everything downstream is therefore shared with method 3.
    """
    return os.environ.get(TENSOR_BASIS_ENV, "cartesian") in ("helicity", "omega")


def helicity_transverse(n_components):
    n = int(n_components)
    return HELICITY_TRANSVERSE.get(n, tuple(range(n)))


def to_helicity(total):
    """Cartesian -> helicity amplitudes, keeping the dplex (Re, Im) layout.

    `total` has shape (2, ..., k) with axis 0 = (Re, Im) and k = 2 or 3
    Cartesian components (x, y[, z]).  The result has shape (2, ..., 3) with
    the last axis ordered (lambda=+1, lambda=0, lambda=-1); when k == 2 the
    lambda=0 slot is exactly zero because z was never cached.

        a_{+1} = ( A_x - i A_y) / sqrt2
        a_{ 0} =   A_z
        a_{-1} = (-A_x - i A_y) / sqrt2

    Summing |a|^2 over lambda = +-1 reproduces |A_x|^2 + |A_y|^2 exactly (it is
    an algebraic identity, see report section 4), so this layer is numerically
    a no-op for the current likelihood.
    """
    re, im = total[0], total[1]
    rx, ry = re[..., 0], re[..., 1]
    ix, iy = im[..., 0], im[..., 1]
    r2 = 1.0 / np.sqrt(2.0)
    if re.shape[-1] >= 3:
        re0, im0 = re[..., 2], im[..., 2]
    else:
        re0 = np.zeros_like(rx)
        im0 = np.zeros_like(rx)
    return np.stack(
        [
            np.stack([(rx + iy) * r2, (ix - ry) * r2], axis=0),   # lambda = +1
            np.stack([re0, im0], axis=0),                         # lambda =  0
            np.stack([(iy - rx) * r2, (-ix - ry) * r2], axis=0),  # lambda = -1
        ],
        axis=-1,
    )


def apply_helicity(amplitude):
    """Rotate a dplex amplitude block to the helicity basis when needed.

    `to_helicity` is linear on the component axis and commutes with the
    couplings, the propagators and the sum over states, so it can be applied
    once to the TOTAL amplitude instead of once per coupled block.  Doing it
    here (rather than inside `_coupled_components`) cuts the rotation work by
    sum(l) / 1 = 7x for the current model, which is a measurable saving since
    the rotation is allocation-heavy.
    """
    if helicity_layer_enabled() and not precomputed_helicity_basis():
        return to_helicity(amplitude)
    return amplitude


# Slots of the Cartesian component axis that the transverse initial state
# reaches.  With k = 2 the cache already holds exactly these; with k = 3 the
# z component is cached for the helicity layer but must NOT enter this sum.
CARTESIAN_TRANSVERSE = (0, 1)


def intensity_from_amplitude(component):
    """|amplitude|^2 summed over the physical components, per event.

    Cartesian layer: |T_x|^2 + |T_y|^2 -- the two directions the transverse
        initial state reaches.  Written explicitly rather than "sum whatever is
        cached" so that a 3-component (P2-1) cache does not silently add the
        lambda = 0 strength, which no initial state can populate.
    Helicity layer : only lambda = +-1 (P1-3) -- the transverse initial state.
    Accepts the (2, ..., k) dplex layout and returns the trailing shape with the
    component axis contracted away.  `component` must already be in the basis
    selected by TENSOR_BASIS (use `apply_helicity`).
    """
    abs2 = dplex_dabs(component)
    if helicity_layer_enabled():
        return np.sum(abs2[..., helicity_transverse(abs2.shape[-1])], axis=-1)
    return np.sum(abs2[..., CARTESIAN_TRANSVERSE], axis=-1)


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



def make_initial_args():
    _data = toml.load("run/free_params.toml")["data"]
    _data.sort(key=lambda x: x["arg_index"])

    args = [d["value"] for d in _data]
    ranges = [d.get("range", [-onp.inf, onp.inf]) for d in _data]
    errors = [d["error"] for d in _data]

    return onp.array(args), onp.array(ranges), onp.array(errors)



def save_result(args, errors, path="output/free_params_fitted.toml"):
    _data = toml.load("run/free_params.toml")["data"]
    _data.sort(key=lambda x: x["arg_index"])

    for i, d in enumerate(_data):
        d["value"] = float(args[i])
        d["error"] = float(errors[i])

    with open(path, "w") as f:
        toml.dump({"data": _data}, f)


def _assemble_omega_cache(path):
    """Rebuild W_i(e,lambda) from the channel-independent Omega + constants.

    Method 4: `W_i[e,lam_A] = g_i K_i(e) sum_{a,b} M_i[lam_A,a,b] Omega[e,lam_A,a,b]`.
    This is exactly tf-pwa's `H . D`; it reproduces the method-3 cache to
    ~1e-14 (see 4_三条路线/method4_omega/README.md).

    Two families share this code, distinguished by the file name only:

        phif2*  ->  Omega.npy  (N,3,3,5)   K.npy  (5,N)  M.npy  (5,3,5)  g.npy  (5,)
        phif0*  ->  Omega0.npy (N,3,3,1)   K0.npy (2,N)  M0.npy (2,3,1)  g0.npy (2,)

    The J_R = 0 case keeps the trailing lambda_f axis of length 1, so the very
    same `einsum("iab,enab->ien")` works for both (MATH.md §2.5/§3.2).
    """
    d = os.path.dirname(path)
    f0 = os.path.basename(path).startswith("phif0")
    suf = "0" if f0 else ""
    Om = onp.load(os.path.join(d, "Omega%s.npy" % suf))   # (N,3,3,5) or (N,3,3,1)
    K = onp.load(os.path.join(d, "K%s.npy" % suf))        # (5,N) or (2,N)
    M = onp.load(os.path.join(d, "M%s.npy" % suf))        # (5,3,5) or (2,3,1)
    g = onp.load(os.path.join(d, "g%s.npy" % suf))        # (5,) or (2,)
    P = onp.einsum("iab,enab->ien", M, Om)                # (n_ch, N, 3)
    W = g[:, None, None] * K[:, :, None] * P
    return onp.stack([W.real, W.imag], axis=0)            # dplex (2, n_ch, N, 3)


def _load_cached_tensor(path):
    """Load one cached channel-tensor file, honouring TENSOR_BASIS.

    Cartesian: real `(n_ch, N, k)`, k = 2 or 3 Lorentz components (x, y[, z]).
    Helicity : complex `(n_ch, N, 3)` -> dplex `(2, n_ch, N, 3)`, axis 0 = (Re, Im).
    Omega    : BOTH phif0 and phif2 entries are ASSEMBLED from the
               channel-independent Omega + K + M + g (see above); every other
               entry is read as a plain helicity cache.

    Mixing the two silently would give a wrong but plausible-looking answer, so
    the mismatch is a hard error rather than a coercion.
    """
    base = os.path.basename(path)
    if precomputed_omega_basis() and (base.startswith("phif2") or base.startswith("phif0")):
        return _assemble_omega_cache(path)
    arr = onp.load(path)
    if precomputed_helicity_basis():
        if not onp.iscomplexobj(arr):
            raise ValueError(
                "%s holds real data but TENSOR_BASIS=helicity; regenerate it with "
                "task/.../4_三条路线/method3_helicity_native/gen_tensor.py" % path
            )
        return onp.stack([arr.real, arr.imag], axis=0)
    if onp.iscomplexobj(arr):
        raise ValueError(
            "%s holds complex data but TENSOR_BASIS=%s; set TENSOR_BASIS=helicity "
            "(method 3) or point at the Cartesian cache"
            % (path, os.environ.get(TENSOR_BASIS_ENV, "cartesian"))
        )
    return arr


def _tile_cached_tensor(arr, n_repeat, n_truth_mc):
    """Repeat events `n_repeat` times and slice the truth-MC prefix."""
    if arr.ndim == 4:                      # dplex (2, n_ch, N, k)
        rep = onp.tile(arr, (1, 1, n_repeat, 1))
    elif arr.ndim == 3:                    # real (n_ch, N, k)
        rep = onp.tile(arr, (1, n_repeat, 1))
    else:
        raise ValueError("unsupported cached-tensor rank %d" % arr.ndim)
    return rep, rep[..., 0:n_truth_mc * n_repeat, :]


def load_data():
    n_repeat = int(os.environ.get("N_REPEAT", "3"))
    n_truth_mc = 150000  # intentional truth-MC cap; this is not n_data
    # KK_DATA_ROOT lets a regenerated dataset (e.g. the 3-component P2-1 caches
    # under data/tensor3/) be used without touching the production arrays.  The
    # tensor component count is picked up automatically: k=2 -> lambda=+-1 only,
    # k=3 -> the helicity layer also sees lambda=0.
    root = os.environ.get("KK_DATA_ROOT", "data")
    real = os.path.join(root, "real_data")
    mc = os.path.join(root, "mc_truth")
    data = {}

    data['data_phi_kk'] = onp.tile(onp.load(os.path.join(real, 'phi_kk.npy')), n_repeat)
    data['mc_phi_kk'] = onp.tile(onp.load(os.path.join(mc, 'phi_kk.npy')), n_repeat)
    data['truth_phi_kk'] = data['mc_phi_kk'][0:n_truth_mc * n_repeat]

    data['data_f_kk'] = onp.tile(onp.load(os.path.join(real, 'f_kk.npy')), n_repeat)
    data['mc_f_kk'] = onp.tile(onp.load(os.path.join(mc, 'f_kk.npy')), n_repeat)
    data['truth_f_kk'] = data['mc_f_kk'][0:n_truth_mc * n_repeat]

    for var in ('phif0_kk', 'phif2_kk'):
        data['data_' + var], _ = _tile_cached_tensor(
            _load_cached_tensor(os.path.join(real, var + '.npy')), n_repeat, n_truth_mc)
        data['mc_' + var], data['truth_' + var] = _tile_cached_tensor(
            _load_cached_tensor(os.path.join(mc, var + '.npy')), n_repeat, n_truth_mc)

    data['data_b123_kk'] = onp.tile(onp.load(os.path.join(real, 'b123_kk.npy')), n_repeat)
    data['mc_b123_kk'] = onp.tile(onp.load(os.path.join(mc, 'b123_kk.npy')), n_repeat)
    data['truth_b123_kk'] = data['mc_b123_kk'][0:n_truth_mc * n_repeat]

    data['data_b124_kk'] = onp.tile(onp.load(os.path.join(real, 'b124_kk.npy')), n_repeat)
    data['mc_b124_kk'] = onp.tile(onp.load(os.path.join(mc, 'b124_kk.npy')), n_repeat)
    data['truth_b124_kk'] = data['mc_b124_kk'][0:n_truth_mc * n_repeat]

    return data


def _cached_tensor_magnitude(arr):
    """|tensor| per (channel, event); basis aware.

    Cartesian `(n_ch, N, k)` -> sqrt(sum_k T_k^2)
    dplex     `(2, n_ch, N, k)` -> sqrt(sum_k Re^2 + Im^2)
    """
    if arr.ndim == 4:
        return onp.sqrt(onp.sum(arr[0] ** 2 + arr[1] ** 2, axis=-1))
    return onp.sqrt(onp.sum(arr ** 2, axis=-1))


def normalize_data(data):
    for var in ('phif0_kk', 'phif2_kk'):
        mc_arr = data['mc_' + var]
        regular = 1.0 / onp.mean(_cached_tensor_magnitude(mc_arr), axis=1)
        for prefix in ('data_', 'mc_', 'truth_'):
            key = prefix + var
            arr = data[key]
            if arr.ndim == 4:
                # dplex (2, n_ch, N, k): the channel axis is 1, not 0
                data[key] = onp.einsum('c,ecnk->ecnk', regular, arr)
            else:
                data[key] = onp.einsum('c,cek->cek', regular, arr)
    return data


def shard_data_distributed(data, mesh):
    jax_data = {}
    for key, value in data.items():
        array = np.asarray(value)
        if array.ndim == 1:
            spec = P('event')
        elif array.ndim == 3:
            spec = P(None, 'event', None)
        elif array.ndim == 4:
            # dplex helicity cache (2, n_ch, N, k): only the event axis is sharded
            spec = P(None, None, 'event', None)
        else:
            raise ValueError('Unsupported data rank for ' + key + ': ' + str(array.ndim))
        jax_data[key] = device_put(array, NamedSharding(mesh, spec))
    return jax_data


