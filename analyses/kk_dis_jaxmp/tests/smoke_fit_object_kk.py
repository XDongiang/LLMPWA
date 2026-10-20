#!/usr/bin/env python
"""Smoke test for the ported run/fit_object_kk.py using SYNTHETIC data.

The real data/ is missing the u_rho_* / u_b_* tensors (and weight length does
not match), so this generates correctly-shaped random tensors, then exercises:
    load -> normalize -> shard -> nll / grad / hvp -> a few Newton-CG iterations
That validates the port mechanically without the production data.

Usage:
    python smoke_fit_object_kk.py [path/to/fit_object_kk.py]
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

N_DATA = 64
N_MC = 96
N_TRUTH = 48
SHAPES = {
    "phif0_kk": (2, 2),
    "phif2_kk": (5, 2),
    "u_rho_r_kk": (1, 2),
    "u_rho_l_kk": (1, 2),
    "u_b_r_kk": (4, 2),
    "u_b_l_kk": (4, 2),
}
SCALARS = ("phi_kk", "f_kk", "b124_kk", "b123_kk")


def build_synthetic(root, seed=7):
    rng = onp.random.RandomState(seed)
    real = os.path.join(root, "real_data")
    mct = os.path.join(root, "mc_truth")
    wdir = os.path.join(root, "weight")
    for d in (real, mct, wdir):
        os.makedirs(d, exist_ok=True)

    for name, (n_ch, k) in SHAPES.items():
        for d, n in ((real, N_DATA), (mct, N_MC)):
            onp.save(os.path.join(d, name + ".npy"),
                     rng.randn(n_ch, n, k).astype(onp.float64))
    for name in SCALARS:
        for d, n in ((real, N_DATA), (mct, N_MC)):
            onp.save(os.path.join(d, name + ".npy"),
                     rng.rand(n).astype(onp.float64) + 0.5)
    # weight matches the data length (production files currently do not)
    onp.save(os.path.join(wdir, "weight_kk.npy"),
             onp.ones(N_DATA, dtype=onp.float64))


def load_module(path):
    spec = importlib.util.spec_from_file_location("fit_object_kk", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fit_object_kk"] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    tmp = tempfile.mkdtemp(prefix="kk_smoke_")
    try:
        build_synthetic(tmp)
        os.environ["KK_DATA_ROOT"] = tmp
        os.environ["KK_TRUTH_MC_CAP"] = str(N_TRUTH)
        os.environ["KKDIS_MAXITER"] = "3"
        os.environ["KKDIS_XTOL"] = "1e-4"
        os.environ["KKDIS_OUTDIR"] = "smoke"

        # the module chdir()s to its own repo root; remember it to find config/logs
        mod = load_module(TARGET)
        print("loaded:", TARGET)
        print("ARGS_LIST", mod.ARGS_LIST.shape, "FLOAT_LIST", mod.FLOAT_LIST.shape)

        data = mod.normalize_data(mod.load_data(mod.FitConfig()))
        print("data keys:", len(data))
        for k in ("data_phi_kk", "data_phif0_kk_3686", "data_phif2_kk_3686",
                  "data_u_b_r_kk_3686", "truth_phi_kk", "wt_data_kk"):
            print("   ", k, onp.asarray(data[k]).shape)

        pwaf = mod.PWAFunc()
        args_float = mod.ARGS_LIST[mod.FLOAT_LIST]
        print("n_free =", args_float.size)

        # --- pure-numpy reference path (no mesh) ---------------------------
        d_lh = pwaf.data_likelihood_kk(mod.np.asarray(args_float), {
            k: mod.np.asarray(v) for k, v in data.items()})
        m_lh = pwaf.mc_likelihood_kk(mod.np.asarray(args_float), {
            k: mod.np.asarray(v) for k, v in data.items()})
        print("data_likelihood =", float(d_lh))
        print("mc_likelihood   =", float(m_lh))
        assert onp.isfinite(float(d_lh)), "data likelihood not finite"
        assert onp.isfinite(float(m_lh)), "mc likelihood not finite"
        assert float(m_lh) > 0, "mc likelihood must be positive (inside log)"

        # --- distributed driver path ---------------------------------------
        os.makedirs("logs", exist_ok=True)
        mod.setup_logging()
        fit = mod.DistributedFit(mod.FitConfig())
        rc = fit.setup().run()
        print("run() returned", rc)

        outdir = os.path.join("output", "smoke")
        for fn in ("fit_result_values.npy", "fit_result_errors.npy",
                   "fit_result_kk.npz", "fit_result_full_args.npy",
                   "pwa_info_kk.json", "result_info.json"):
            p = os.path.join(outdir, fn)
            print("   ", fn, "OK" if os.path.exists(p) else "MISSING")
            assert os.path.exists(p), p
        vals = onp.load(os.path.join(outdir, "fit_result_values.npy"))
        errs = onp.load(os.path.join(outdir, "fit_result_errors.npy"))
        full = onp.load(os.path.join(outdir, "fit_result_full_args.npy"))
        print("fitted", vals.shape, "errors", errs.shape, "full", full.shape)
        assert vals.shape == args_float.shape
        assert errs.shape == args_float.shape
        assert full.shape == mod.ARGS_LIST.shape
        assert onp.all(onp.isfinite(vals)), "fitted values not finite"
        onp.testing.assert_allclose(full[mod.FLOAT_LIST], vals)

        # the full 219-slot vector must carry the fitted free params
        print("SMOKE PASS")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
