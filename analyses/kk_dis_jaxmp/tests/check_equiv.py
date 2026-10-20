#!/usr/bin/env python
"""Cross-check: single-process vs 2-process NLL must agree.

Sharding must be a pure performance transformation -- the combined likelihood
on the full event set has to be numerically identical (to float tolerance)
whether it runs on 1 device or on 2 processes each holding half the events.

Run as:
    python check_equiv.py            # driver: spawns both configurations
"""
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)   # tests/ -> analysis root
TARGET = os.path.join(ROOT, "run", "fit_object_kk.py")

CHILD = r'''
import importlib.util, os, sys, json
import numpy as onp
spec = importlib.util.spec_from_file_location("fit_object_kk", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
sys.modules["fit_object_kk"] = mod
spec.loader.exec_module(mod)
mod.setup_logging()
nproc, pid = mod.init_distributed()   # must run before any jax array is created
data = mod.normalize_data(mod.load_data(mod.FitConfig()))
mesh = mod.build_mesh()
with mod._mesh_context(mesh):
    jd = mod.shard_data_distributed(data, mesh)
    nll, g, h = mod.make_objective(mod.PWAFunc())
    x = mod.ARGS_LIST[mod.FLOAT_LIST].copy()
    val = float(nll(mod.device_put(mod.np.asarray(x),
                                   mod.NamedSharding(mesh, mod.P(""))), jd))
    gr = onp.asarray(g(mod.device_put(mod.np.asarray(x),
                                      mod.NamedSharding(mesh, mod.P(""))), jd))
    # sharding evidence: report the spec actually used
    spec_ok = str(jd["data_phi_kk"].sharding.spec)
    # a cross-process array cannot be fetched with onp.asarray(); inspect the
    # process-local shard instead -- its size proves real partitioning.
    local = int(onp.asarray(jd["data_phi_kk"].addressable_shards[0].data).shape[0])
    print("RESULT " + json.dumps({
        "nll": val, "grad_norm": float(onp.linalg.norm(gr)),
        "grad0": float(gr[0]), "spec": spec_ok,
        "n_dev": int(mesh.devices.size),
        "local_events": local,
        "n_shards": len(jd["data_phi_kk"].addressable_shards),
    }))
'''


def main():
    import numpy as onp
    sys.path.insert(0, HERE)
    import smoke_multiproc as sm

    tmp = tempfile.mkdtemp(prefix="kk_equiv_")
    try:
        sm.build_synthetic(tmp, seed=5)
        child = os.path.join(HERE, "_equiv_child.py")
        with open(child, "w") as fh:
            fh.write(CHILD)

        base_env = dict(os.environ)
        base_env.update({
            "KK_DATA_ROOT": tmp,
            "KK_TRUTH_MC_CAP": str(sm.N_TRUTH),
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "JAX_PLATFORMS": "cpu",
        })

        def run(nproc, coord):
            env = dict(base_env)
            env["JAX_NUM_PROCESSES"] = str(nproc)
            env["JAX_COORDINATOR_ADDRESS"] = coord
            outs = []
            procs = []
            for pid in range(nproc):
                e = dict(env)
                e["JAX_PROCESS_ID"] = str(pid)
                p = subprocess.Popen([sys.executable, child, TARGET], env=e,
                                     cwd=ROOT, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True)
                procs.append(p)
            for p in procs:
                out, _ = p.communicate(timeout=1800)
                outs.append(out)
            results = []
            for o in outs:
                for line in o.splitlines():
                    if line.startswith("RESULT "):
                        import json
                        results.append(json.loads(line[len("RESULT "):]))
            return results, outs

        coord1 = "127.0.0.1:%d" % (13000 + os.getpid() % 300)
        r1, o1 = run(1, coord1)
        coord2 = "127.0.0.1:%d" % (13500 + os.getpid() % 300)
        r2, o2 = run(2, coord2)

        assert len(r1) == 1, "single-process run produced no RESULT"
        assert len(r2) == 2, "2-process run produced %d RESULT lines" % len(r2)
        a, b = r1[0], r2[0]
        print("1 proc :", a)
        print("2 proc :", b)
        dnll = abs(a["nll"] - b["nll"]) / max(1.0, abs(a["nll"]))
        dg = abs(a["grad0"] - b["grad0"]) / max(1.0, abs(a["grad0"]))
        print("relative NLL diff = %.3e" % dnll)
        print("relative grad[0] diff = %.3e" % dg)
        assert dnll < 1e-9, "NLL differs between 1 and 2 processes"
        assert dg < 1e-9, "gradient differs between 1 and 2 processes"
        assert b["n_dev"] == 2, "2-process run saw %d devices" % b["n_dev"]
        assert "'event'" in b["spec"], "event axis not sharded: %s" % b["spec"]
        # each rank must hold only its half of the 64 events
        assert b["local_events"] == sm.N_DATA // 2, (
            "rank 0 holds %d events, expected %d -- data was replicated"
            % (b["local_events"], sm.N_DATA // 2)
        )
        print("rank-0 local events = %d (of %d) -> real partitioning"
              % (b["local_events"], sm.N_DATA))
        print("EQUIV PASS: sharding is numerically transparent")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        p = os.path.join(HERE, "_equiv_child.py")
        if os.path.exists(p):
            os.remove(p)


if __name__ == "__main__":
    raise SystemExit(main())
