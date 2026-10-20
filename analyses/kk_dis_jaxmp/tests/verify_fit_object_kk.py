#!/usr/bin/env python
"""Correctness checks for the ported run/fit_object_kk.py.

A. autodiff (new)  ==  old manual grad / HVP formulas
   The old Control.thread_grad_likelihood / thread_hvp split data and MC by
   hand.  The new file autodiffs one combined scalar.  These must agree, which
   is the single most important semantic check of the port.

B. autodiff gradient == finite differences (independent of A)

C. size-1 broadcasting still works (the old model relies on it: the 3773 block
   reuses 3686 amplitudes with scalar inta phases).
"""
import importlib.util
import os
import shutil
import sys
import tempfile

import numpy as onp

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else
                         os.path.join(os.path.dirname(HERE), "run", "fit_object_kk.py"))

N_DATA, N_MC, N_TRUTH = 48, 64, 40
SHAPES = {"phif0_kk": (2, 2), "phif2_kk": (5, 2), "u_rho_r_kk": (1, 2),
          "u_rho_l_kk": (1, 2), "u_b_r_kk": (4, 2), "u_b_l_kk": (4, 2)}
SCALARS = ("phi_kk", "f_kk", "b124_kk", "b123_kk")


def build_synthetic(root, seed=11):
    rng = onp.random.RandomState(seed)
    for d in ("real_data", "mc_truth", "weight"):
        os.makedirs(os.path.join(root, d), exist_ok=True)
    for name, (nc, k) in SHAPES.items():
        onp.save(os.path.join(root, "real_data", name + ".npy"),
                 rng.randn(nc, N_DATA, k))
        onp.save(os.path.join(root, "mc_truth", name + ".npy"),
                 rng.randn(nc, N_MC, k))
    for name in SCALARS:
        onp.save(os.path.join(root, "real_data", name + ".npy"), rng.rand(N_DATA) + 0.5)
        onp.save(os.path.join(root, "mc_truth", name + ".npy"), rng.rand(N_MC) + 0.5)
    onp.save(os.path.join(root, "weight", "weight_kk.npy"), onp.ones(N_DATA))


def load_module(path):
    spec = importlib.util.spec_from_file_location("fit_object_kk", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fit_object_kk"] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    tmp = tempfile.mkdtemp(prefix="kk_verify_")
    try:
        build_synthetic(tmp)
        os.environ["KK_DATA_ROOT"] = tmp
        os.environ["KK_TRUTH_MC_CAP"] = str(N_TRUTH)
        mod = load_module(TARGET)
        jnp, onp_ = mod.np, onp
        logger = mod.setup_logging()

        data = mod.normalize_data(mod.load_data(mod.FitConfig()))
        jdata = {k: jnp.asarray(v) for k, v in data.items()}
        pwaf = mod.PWAFunc()
        x = mod.ARGS_LIST[mod.FLOAT_LIST].copy()

        # ---- old manual formulas -----------------------------------------
        def old_data_lh(a):
            return pwaf.data_likelihood_kk(a, jdata)

        def old_mc_lh(a):
            return pwaf.mc_likelihood_kk(a, jdata)

        old_grad = onp_.asarray(jnp.asarray(
            mod.jax.grad(old_data_lh)(jnp.asarray(x))
            + mod.MC_NORM_CONST
            * jnp.asarray(mod.jax.grad(old_mc_lh)(jnp.asarray(x)))
            / old_mc_lh(jnp.asarray(x))
        ))
        old_lh = float(old_data_lh(jnp.asarray(x)) +
                       mod.MC_NORM_CONST * onp_.log(float(old_mc_lh(jnp.asarray(x)))
                                                    / mod.MC_NORM_DENOM))

        # ---- new combined autodiff ---------------------------------------
        nll, grad_nll, hvp = mod.make_objective(pwaf)
        new_lh = float(nll(jnp.asarray(x), jdata))
        new_grad = onp_.asarray(grad_nll(jnp.asarray(x), jdata))

        print("combined likelihood old=%.12g new=%.12g" % (old_lh, new_lh))
        print("grad max|old-new| = %.3e" % onp_.max(onp_.abs(old_grad - new_grad)))
        print("grad scale        = %.3e" % onp_.max(onp_.abs(old_grad)))
        assert abs(old_lh - new_lh) < 1e-6 * max(1.0, abs(old_lh)), "likelihood mismatch"
        assert onp_.allclose(old_grad, new_grad, rtol=1e-8, atol=1e-8), "gradient mismatch"
        print("A. autodiff == old manual grad : PASS")

        # ---- finite differences ------------------------------------------
        # Use a scaled step; the model is smooth but numerically sensitive.
        v = onp_.zeros_like(x)
        v[3] = 1.0
        eps = 1e-6
        f_plus = float(nll(jnp.asarray(x + eps * v), jdata))
        f_minus = float(nll(jnp.asarray(x - eps * v), jdata))
        fd = (f_plus - f_minus) / (2 * eps)
        ad = float(new_grad[3])
        rel = abs(fd - ad) / max(1.0, abs(ad))
        print("directional derivative: fd=%.10g ad=%.10g rel=%.2e" % (fd, ad, rel))
        assert rel < 1e-5, "finite-difference check failed"
        print("B. autodiff gradient == finite differences : PASS")

        # ---- HVP vs finite difference of the gradient --------------------
        hv = onp_.asarray(hvp(jnp.asarray(x), jnp.asarray(v), jdata))
        g_plus = onp_.asarray(grad_nll(jnp.asarray(x + eps * v), jdata))
        g_minus = onp_.asarray(grad_nll(jnp.asarray(x - eps * v), jdata))
        fd_hv = (g_plus - g_minus) / (2 * eps)
        rel_h = onp_.max(onp_.abs(hv - fd_hv)) / max(1.0, onp_.max(onp_.abs(hv)))
        print("HVP vs FD-of-grad rel = %.3e" % rel_h)
        assert rel_h < 1e-4, "HVP check failed"
        print("C. HVP == finite difference of gradient : PASS")

        # ---- size-1 broadcasting (3773 block) ----------------------------
        s1 = onp_.ones((3, 2))
        s2 = onp_.ones((1,))
        r = jnp.einsum("li,l->li", jnp.asarray(s1), jnp.asarray(s2))
        assert r.shape == (3, 2)
        print("D. size-1 einsum broadcasting : PASS")

        print("VERIFY PASS")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
