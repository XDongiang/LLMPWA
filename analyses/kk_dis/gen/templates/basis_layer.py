# SECTION: BASIS_LAYER
# ==============================================================================
# 表象层（basis layer）—— 由框架 `agent/tensor/basis.py` 规范定义
#
# 为什么这个文件存在
# ------------------
# 本段（EPS_A / apply_helicity / intensity_from_amplitude / _assemble_omega_cache
# 等，共 ~255 行）此前**不在任何模板里**：重放 `assemble_help_functions`
# 只能得到 206 行 / 18 个符号，而 `run/likelihood_function.py:4-20` 直接 import
# 其中 5 个（EPS_A, apply_helicity, helicity_layer_enabled,
# intensity_from_amplitude, precomputed_helicity_basis）。
# 任何一次全量重跑生成器都会让 likelihood ImportError。
#
# 现在这一层是**受版本控制的模板**，由分析内 agent 实例化；框架侧的定义见
# `agent/tensor/basis.py`（4 个表象钩子 + 承重约定）。
#
# 四个表象（`TENSOR_BASIS` 环境变量，**调用时**读取，所以一份 run/ 可以服务多条路线）
# ------------------------------------------------------------------------------
#   cartesian        预计算实数 T_i(e,mu)，拟合 I = |T_x|^2 + |T_y|^2
#   helicity         预计算复数 W_i(e,lambda) = EPS_A . T_i，拟合不再旋转
#   omega            预计算与 (l,s) 无关的 Omega + K/M/g，加载时装配 W_i
#   （cartesian + HELICITY_LAYER=1）= 路线 2：cache 同 cartesian，拟合时旋转
#
# EPS_A 的取法承重（`4_三条路线/eps_a_provenance.md`）
# ---------------------------------------------------
# 行 = lambda(+1, 0, -1)，列 = mu(x, y, z)：
#     [[ 1/sqrt2, -i/sqrt2, 0],
#      [ 0,        0,       1],
#      [-1/sqrt2, -i/sqrt2, 0]]
# 把第三行取正号、或换成 (1,+i) 那一支，逐事例与 tf-pwa 的重叠度会从
# 1.000000 掉到 0.62–0.67。**不要"简化"这个矩阵。**
# ==============================================================================
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


# ==============================================================================
# Cache 装载层（同样此前不在任何模板里，重放生成器会丢）
# ==============================================================================
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


def _cached_tensor_magnitude(arr):
    """|tensor| per (channel, event); basis aware.

    Cartesian `(n_ch, N, k)` -> sqrt(sum_k T_k^2)
    dplex     `(2, n_ch, N, k)` -> sqrt(sum_k Re^2 + Im^2)
    """
    if arr.ndim == 4:
        return onp.sqrt(onp.sum(arr[0] ** 2 + arr[1] ** 2, axis=-1))
    return onp.sqrt(onp.sum(arr ** 2, axis=-1))
