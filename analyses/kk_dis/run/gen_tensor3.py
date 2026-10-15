#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P2-1: regenerate the cached covariant tensors KEEPING the z component.

`Tensor/CacheTensor.py::Calculate` ends with

    total_tensor.append(tensor[:, 0:2])          # CacheTensor.py:220

which throws away (z, t) and leaves the two Cartesian components the current
likelihood uses.  This driver keeps (x, y, z) -> (n_ch, N, 3), which is what the
hybrid helicity layer needs in order to expose lambda = 0 (report P2-1).

The (x, y) block must reproduce the existing `data/*/phif{0,2}_kk.npy` files;
`--check-against` verifies that.

Runs the tensor functions with jit + vmap over an event chunk on the GPU, so it
is ~3 orders of magnitude faster than the original per-slice eager loop.

Examples
--------
    # real data (10k events)
    python run/gen_tensor3.py --npz data/real_data/Momentum_kk.npz \
        --out data/tensor3/real_data --tag real

    # MC truth (1.2M events)
    python run/gen_tensor3.py --npz data/mc_truth/Momentum_kk.npz \
        --out data/tensor3/mc_truth --tag mc --batch 128
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import types

import numpy as onp

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
os.chdir(_root)
for _p in (_root, _here):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

TENSOR_DIR = os.path.join(
    _root, "task", "20260929_螺旋度耦合振幅推导", "1_运行代码", "tensor_example"
)

F0_NAMES = ["phif001", "phif021"]
F2_NAMES = ["phif201", "phif221", "phif222", "phif223", "phif243"]


def load_tensor_module(x64=True, psi_mass=None):
    """Build the channel functions, controlling `psi2s` dtype and mass.

    `CacheTensor.Calculate` does

        my_func = pwa_func(); my_func.fill_func()   # -> FUNCTION(), Modifiers()
        calc.Calculate(my_func, mycdl)              # -> config.update(x64, True)

    so its `FUNCTION.psi2s = jnp.array([0,0,0,3.686])` is created while x64 is
    still OFF, i.e. **float32**.  That is what the delivered caches were built
    with, and reproducing them bit-for-bit requires the same ordering.

    `x64=True` builds FUNCTION with x64 already on, so psi2s is float64.  That
    is the more accurate choice and the one to use for new datasets; measured
    difference against the float32 caches is ~2e-7 relative on every channel
    (e.g. phif222: 2.79e-7 vs 1.86e-16).  Do not mix the two in one dataset.

    `psi_mass` (opt-in, 2026-10-15) overrides the hard-coded `3.686` in
    `FUNC.py:15`.  This is the psi four-momentum convention switch documented in
    `method4_omega/psi_mass_residual.md` §6b: the real sample's actual
    `E_psi = 3.685782956` is a constant to 2e-10, while the tensor generator
    hard-codes the nominal 3.686.  Omega is built from the RECONSTRUCTED
    A = Kp+Km+Pip+Pim, so mixing the two leaves a ~1.8e-4 residual on every
    ell != 0 channel (real side only; the delivered MC is exactly 3.686).
    Passing `psi_mass=3.685782956` for the real sample removes it.

    ⚠️ `psi2s` must be set BEFORE `Modifiers()`: that call installs
    `jit(vmap(...))`, which bakes the constant into the jaxpr.  Assigning to
    `f.psi2s` afterwards does not recompile (see
    `method4_omega/checks/f0_psimass_causal.py`).
    """
    import jax

    jax.config.update("jax_enable_x64", bool(x64))
    # the vendored code still does `from jax.config import config`
    if "jax.config" not in sys.modules:
        m = types.ModuleType("jax.config")
        m.config = jax.config
        m.update = jax.config.update
        sys.modules["jax.config"] = m
    if TENSOR_DIR not in sys.path:
        sys.path.insert(0, TENSOR_DIR)
    from Tensor import FUNC as func

    f = func.FUNCTION()
    if psi_mass is not None:
        import jax.numpy as _jnp
        f.psi2s = _jnp.array([0.0, 0.0, 0.0, float(psi_mass)])
    f.Modifiers()
    jax.config.update("jax_enable_x64", True)
    return jax, f


def build_batched(jax, f, batch):
    """Return the channel callables.

    `FUNC.FUNCTION.Modifiers()` already installs `jit(vmap(f))` for every
    channel, so each one takes a leading event axis and must NOT be vmapped
    again here.
    """
    return {name: getattr(f, name) for name in F0_NAMES + F2_NAMES}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=0, help="0 = all remaining")
    ap.add_argument("--check-against", default="",
                    help="directory holding the existing 2-component tensors")
    ap.add_argument("--keep", type=int, default=3, choices=(2, 3))
    ap.add_argument("--dtype", default="float64", choices=("float32", "float64"),
                    help="psi2s precision: float32 reproduces CacheTensor.Calculate "
                         "bit-for-bit, float64 is the accurate choice for new sets")
    ap.add_argument("--basis", default="cartesian", choices=("cartesian", "helicity"),
                    help="cartesian: store T_i(e,mu) (real Lorentz components). "
                         "helicity: store W_i(e,lambda) = EPS_A . T_i (complex), i.e. "
                         "the EPS_A rotation is folded into PRE-COMPUTATION "
                         "(three-route deliverable, method 3)")
    ap.add_argument("--psi-mass", type=float, default=None,
                    help="覆盖 FUNC.py 硬编码的 psi 质量 3.686（四动量口径统一，见 "
                         "method4_omega/psi_mass_residual.md §6b）。真实样本请用 3.685782956；"
                         "不传则保持现状")
    args = ap.parse_args()

    jax, f = load_tensor_module(x64=(args.dtype == "float64"),
                                psi_mass=args.psi_mass)
    print("devices:", jax.devices())
    print("psi2s dtype = %s" % onp.asarray(f.psi2s).dtype)
    print("psi2s =", onp.asarray(f.psi2s))
    fn = build_batched(jax, f, args.batch)

    d = onp.load(args.npz)
    total = int(d["Kp"].shape[0])
    stop = total if args.n == 0 else min(total, args.start + args.n)
    n = stop - args.start
    print("%s: %d events (of %d), batch=%d, keep=%d"
          % (args.npz, n, total, args.batch, args.keep))

    Kp_all = onp.asarray(d["Kp"][args.start:stop], dtype=onp.float64)
    Km_all = onp.asarray(d["Km"][args.start:stop], dtype=onp.float64)
    Pip_all = onp.asarray(d["Pip"][args.start:stop], dtype=onp.float64)
    Pim_all = onp.asarray(d["Pim"][args.start:stop], dtype=onp.float64)
    del d

    out_f0 = onp.zeros((len(F0_NAMES), n, args.keep), dtype=onp.float64)
    out_f2 = onp.zeros((len(F2_NAMES), n, args.keep), dtype=onp.float64)

    # EPS_A: rows lambda = +1, 0, -1 ; columns mu = x, y, z
    EPS_A = onp.array(
        [[1.0 / onp.sqrt(2.0), -1.0j / onp.sqrt(2.0), 0.0],
         [0.0, 0.0, 1.0],
         [-1.0 / onp.sqrt(2.0), -1.0j / onp.sqrt(2.0), 0.0]],
        dtype=onp.complex128,
    )
    hel_f0 = onp.zeros((len(F0_NAMES), n, 3), dtype=onp.complex128)
    hel_f2 = onp.zeros((len(F2_NAMES), n, 3), dtype=onp.complex128)

    t0 = time.time()
    for beg in range(0, n, args.batch):
        end = min(beg + args.batch, n)
        Kp, Km = Kp_all[beg:end], Km_all[beg:end]
        Pip, Pim = Pip_all[beg:end], Pim_all[beg:end]
        phi, f2 = Kp + Km, Pip + Pim

        for i, name in enumerate(F0_NAMES):
            if name == "phif001":
                T = fn[name](phi, Kp, Km)
            else:
                T = fn[name](phi, f2, Kp, Km)
            T = onp.asarray(T, dtype=onp.float64)
            out_f0[i, beg:end] = T[..., :args.keep]
            if args.basis == "helicity":
                hel_f0[i, beg:end] = onp.einsum("lm,em->el", EPS_A, T[..., :3])

        for i, name in enumerate(F2_NAMES):
            T = fn[name](phi, f2, Kp, Km, Pip, Pim)
            T = onp.asarray(T, dtype=onp.float64)
            out_f2[i, beg:end] = T[..., :args.keep]
            if args.basis == "helicity":
                hel_f2[i, beg:end] = onp.einsum("lm,em->el", EPS_A, T[..., :3])

        if beg % (args.batch * 20) == 0 or end == n:
            done = end
            rate = done / max(time.time() - t0, 1e-9)
            eta = (n - done) / max(rate, 1e-9)
            print("  %7d/%d  %.1f ev/s  elapsed %.1fs  eta %.1fs"
                  % (done, n, rate, time.time() - t0, eta), flush=True)

    print("tensor computation finished in %.1fs" % (time.time() - t0))

    if args.check_against and args.dtype != "float32":
        print("  [check] NOTE: --dtype float64 against a float32-built reference "
              "will show ~2e-7 relative; that is the psi2s difference, not a bug")
        for tag, arr, names in (("phif0", out_f0, F0_NAMES),
                                ("phif2", out_f2, F2_NAMES)):
            ref_path = os.path.join(args.check_against, "%s_kk.npy" % tag)
            if not os.path.exists(ref_path):
                print("  [check] %s missing, skipped" % ref_path)
                continue
            ref = onp.load(ref_path, mmap_mode="r")
            got = arr[..., :2]
            ref_slice = onp.asarray(ref[:, args.start:stop, :])
            scale = max(float(onp.abs(ref_slice).max()), 1e-300)
            ad = float(onp.abs(got - ref_slice).max())
            print("  [check] %-6s max|d| vs %s = %.3e  rel = %.3e  %s"
                  % (tag, ref_path, ad, ad / scale,
                     "OK" if ad / scale < 1e-12 else "**MISMATCH**"))

    os.makedirs(args.out, exist_ok=True)
    onp.save(os.path.join(args.out, "phif0_kk.npy"), out_f0)
    onp.save(os.path.join(args.out, "phif2_kk.npy"), out_f2)
    print("saved %s (shapes %s, %s)" % (args.out, out_f0.shape, out_f2.shape))

    if args.basis == "helicity":
        # method 3: the cache IS the helicity amplitude W_i(e, lambda).
        # Verified identity (report section 4):
        #     |W_{+1}|^2 + |W_{-1}|^2  ==  |T_x|^2 + |T_y|^2   exactly.
        onp.save(os.path.join(args.out, "phif0_hel.npy"), hel_f0)
        onp.save(os.path.join(args.out, "phif2_hel.npy"), hel_f2)
        for tag, hel, cart, names in (("phif0", hel_f0, out_f0, F0_NAMES),
                                      ("phif2", hel_f2, out_f2, F2_NAMES)):
            lhs = onp.abs(hel[..., [0, 2]]) ** 2
            lhs = lhs.sum(-1)
            rhs = (cart[..., :2] ** 2).sum(-1)
            ad = float(onp.abs(lhs - rhs).max())
            scale = max(float(onp.abs(rhs).max()), 1e-300)
            print("  [identity] %s |W_+1|^2+|W_-1|^2 vs |T_x|^2+|T_y|^2 : "
                  "max|d| = %.3e  rel = %.3e  %s"
                  % (tag, ad, ad / scale, "OK" if ad / scale < 1e-12 else "**MISMATCH**"))
            for i, nm in enumerate(names):
                s = onp.abs(hel[i]) ** 2
                lam0 = float((s[:, 1] / onp.maximum(s.sum(-1), 1e-300)).mean())
                print("    %-8s helicity cache: mean |W_0|^2/|W|^2 = %.6f  "
                      "max|Im| = %.3e" % (nm, lam0, float(onp.abs(hel[i].imag).max())))

    z_share = {}
    for tag, arr, names in (("phif0", out_f0, F0_NAMES), ("phif2", out_f2, F2_NAMES)):
        for i, nm in enumerate(names):
            s = (arr[i] ** 2).sum(-1)
            with onp.errstate(divide="ignore", invalid="ignore"):
                frac = onp.where(s > 0, arr[i][..., 2] ** 2 / onp.maximum(s, 1e-300), 0.0)
            z_share[nm] = float(onp.median(frac))
            print("  %-8s median |T_z|^2/|T|^2 = %.4f   max|T_z| = %.4e"
                  % (nm, z_share[nm], float(onp.abs(arr[i][..., 2]).max())))
    with open(os.path.join(args.out, "z_power_share.txt"), "w") as fh:
        for k, v in z_share.items():
            fh.write("%s %.8f\n" % (k, v))


if __name__ == "__main__":
    main()
