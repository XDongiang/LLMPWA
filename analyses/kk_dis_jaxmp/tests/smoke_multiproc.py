#!/usr/bin/env python
"""Multi-process test: run the ported file with JAX_NUM_PROCESSES=2 on CPU.

This exercises the real native-JAX multiprocess path:
    jax.distributed.initialize -> mesh over 2 CPU "devices" -> sharded data
    -> chief-only scipy minimize -> worker serves broadcast requests.

Both processes must be launched with the SAME coordinator address.  We spawn
them from one driver so the test is self-contained.

Usage:
    python smoke_multiproc.py [path/to/fit_object_kk.py]
"""
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else
                         os.path.join(os.path.dirname(HERE), "run", "fit_object_kk.py"))
N_DATA, N_MC, N_TRUTH = 64, 96, 48
SHAPES = {"phif0_kk": (2, 2), "phif2_kk": (5, 2), "u_rho_r_kk": (1, 2),
          "u_rho_l_kk": (1, 2), "u_b_r_kk": (4, 2), "u_b_l_kk": (4, 2)}
SCALARS = ("phi_kk", "f_kk", "b124_kk", "b123_kk")

# Data must be divisible by the process count so sharding does not fall back.
assert N_DATA % 2 == 0 and N_MC % 2 == 0 and N_TRUTH % 2 == 0


def build_synthetic(root, seed=3):
    import numpy as onp
    rng = onp.random.RandomState(seed)
    for d in ("real_data", "mc_truth", "weight"):
        os.makedirs(os.path.join(root, d), exist_ok=True)
    for name, (nc, k) in SHAPES.items():
        onp.save(os.path.join(root, "real_data", name + ".npy"), rng.randn(nc, N_DATA, k))
        onp.save(os.path.join(root, "mc_truth", name + ".npy"), rng.randn(nc, N_MC, k))
    for name in SCALARS:
        onp.save(os.path.join(root, "real_data", name + ".npy"), rng.rand(N_DATA) + 0.5)
        onp.save(os.path.join(root, "mc_truth", name + ".npy"), rng.rand(N_MC) + 0.5)
    onp.save(os.path.join(root, "weight", "weight_kk.npy"), onp.ones(N_DATA))


def main():
    tmp = tempfile.mkdtemp(prefix="kk_mp_")
    port = 12411 + (os.getpid() % 400)
    coord = "127.0.0.1:%d" % port
    try:
        build_synthetic(tmp)
        env = dict(os.environ)
        env.update({
            "KK_DATA_ROOT": tmp,
            "KK_TRUTH_MC_CAP": str(N_TRUTH),
            "KKDIS_MAXITER": "2",
            "KKDIS_XTOL": "1e-4",
            "KKDIS_OUTDIR": "mp",
            "JAX_COORDINATOR_ADDRESS": coord,
            "JAX_NUM_PROCESSES": "2",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "JAX_PLATFORMS": "cpu",
        })
        procs = []
        for pid in (0, 1):
            e = dict(env)
            e["JAX_PROCESS_ID"] = str(pid)
            log = open(os.path.join(HERE, "mp_proc%d.log" % pid), "w")
            procs.append((subprocess.Popen([sys.executable, TARGET], env=e,
                                           cwd=os.path.dirname(os.path.dirname(TARGET)),
                                           stdout=log, stderr=subprocess.STDOUT), log))
        rcs = []
        for p, log in procs:
            rc = p.wait(timeout=1800)
            log.close()
            rcs.append(rc)
        print("exit codes:", rcs)
        for pid in (0, 1):
            print("---- proc%d log (tail) ----" % pid)
            with open(os.path.join(HERE, "mp_proc%d.log" % pid)) as fh:
                print("".join(fh.readlines()[-12:]))
        if rcs != [0, 0]:
            print("MULTIPROC FAIL")
            return 1
        # chief output must exist and be finite
        import numpy as onp
        outdir = os.path.join(os.path.dirname(os.path.dirname(TARGET)), "output", "mp")
        vals = onp.load(os.path.join(outdir, "fit_result_values.npy"))
        print("fitted", vals.shape, "finite:", bool(onp.all(onp.isfinite(vals))))
        assert onp.all(onp.isfinite(vals))
        print("MULTIPROC PASS")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
