#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression test for the hybrid helicity amplitude layer.

Compares the two amplitude layers of `likelihood_function.py` on identical data
and identical parameters:

    HELICITY_LAYER=0   Cartesian   I = |A_x|^2 + |A_y|^2
    HELICITY_LAYER=1   helicity    I = |a_{+1}|^2 + |a_{-1}|^2

The two are related by the algebraic identity of report section 4, so the
per-event intensity, the NLL, its gradient and its HVP must agree to float64
round-off (they are NOT expected to be bit identical).

Usage:
    python run/test_helicity_layer.py              # smoke subset
    N_DATA=... N_MC=... N_TRUTH=... python run/test_helicity_layer.py
"""

import os
import sys
import time

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
os.chdir(_root)
for _p in (_root, _here):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as onp  # noqa: E402

from jax import config  # noqa: E402

config.update("jax_enable_x64", True)

from jax import grad, jvp  # noqa: E402

import base_functions as B  # noqa: E402
import likelihood_function as L  # noqa: E402


def reference_helicity(re, im, use_z):
    """Straight numpy complex reference for to_helicity()."""
    r2 = 1.0 / onp.sqrt(2.0)
    a = onp.asarray(re, dtype=onp.float64) + 1j * onp.asarray(im, dtype=onp.float64)
    ax, ay = a[..., 0], a[..., 1]
    out = [(-1j * 0 + (ax - 1j * ay)) * r2]
    out.append(a[..., 2] if use_z else onp.zeros_like(ax))
    out.append((-ax - 1j * ay) * r2)
    return onp.stack(out, axis=-1)


def check_eps_a():
    print("[0] EPS_A constant and to_helicity() against a complex reference")
    A = B.EPS_A
    print("    EPS_A =\n%s" % onp.array2string(A, precision=6))
    assert A.shape == (3, 3)
    expected = onp.array(
        [[1 / onp.sqrt(2), -1j / onp.sqrt(2), 0],
         [0, 0, 1],
         [-1 / onp.sqrt(2), -1j / onp.sqrt(2), 0]]
    )
    assert onp.allclose(A, expected), "EPS_A is not the P0-3 matrix"
    assert onp.allclose(A @ A.conj().T, onp.eye(3)), "EPS_A rows are not orthonormal"
    print("    matches P0-3 and A A^dagger = 1  (rows orthonormal)")

    rng = onp.random.default_rng(0)
    ok = True
    for k in (2, 3):
        re = rng.standard_normal((2, 3, 4, k))
        im = rng.standard_normal((2, 3, 4, k))
        got = onp.asarray(B.to_helicity(onp.stack([re, im], axis=0)))
        ref = reference_helicity(re, im, use_z=(k == 3))
        # got[0] = Re, got[1] = Im of the helicity amplitudes
        gotc = got[0] + 1j * got[1]
        d = onp.abs(gotc - ref).max()
        ok &= d < 1e-14
        print("    k=%d: max|to_helicity - A.T| = %.3e   (lambda=0 slot %s)"
              % (k, d, "= A_z" if k == 3 else "= 0 exactly: %s"
                 % bool(onp.all(onp.asarray(B.to_helicity(
                     onp.stack([re, im], axis=0)))[..., 1] == 0))))
    assert ok
    print("    -> OK\n")


def identity_check():
    """sum_{lam=+-1} |a_lam|^2 == |A_x|^2 + |A_y|^2, random complex A."""
    rng = onp.random.default_rng(1)
    re = rng.standard_normal((2, 5, 1000, 2))
    im = rng.standard_normal((2, 5, 1000, 2))
    t = onp.stack([re, im], axis=0)
    h = onp.asarray(B.to_helicity(t))
    lhs = h[0][..., 0] ** 2 + h[1][..., 0] ** 2 + h[0][..., 2] ** 2 + h[1][..., 2] ** 2
    rhs = re[..., 0] ** 2 + im[..., 0] ** 2 + re[..., 1] ** 2 + im[..., 1] ** 2
    rel = onp.abs(lhs - rhs).max() / onp.abs(rhs).max()
    print("[1] algebraic identity on random complex amplitudes: relative dev %.3e" % rel)
    assert rel < 1e-14
    print("    -> OK\n")


def evaluate(mode, args, data):
    os.environ["HELICITY_LAYER"] = mode
    t0 = time.time()
    per_event = onp.asarray(L.event_intensity(args, data, "data_"), dtype=onp.float64)
    nll = float(L.combined_likelihood(args, data))
    g = onp.asarray(grad(L.combined_likelihood, argnums=0)(args, data), dtype=onp.float64)
    v = onp.ones_like(args)
    hv = onp.asarray(
        jvp(lambda x: grad(L.combined_likelihood, argnums=0)(x, data), (args,), (v,))[1],
        dtype=onp.float64,
    )
    return dict(per_event=per_event, nll=nll, grad=g, hvp=hv, secs=time.time() - t0)


def compare(name, a, b):
    a = onp.asarray(a, dtype=onp.float64)
    b = onp.asarray(b, dtype=onp.float64)
    scale = max(float(onp.abs(b).max()), 1e-300)
    ad = float(onp.abs(a - b).max())
    rel = ad / scale
    print("    %-22s max|d| = %.3e   rel = %.3e   %s"
          % (name, ad, rel, "OK" if rel < 1e-13 else "**MISMATCH**"))
    return rel < 1e-13


def main():
    check_eps_a()
    identity_check()

    n_data = int(os.environ.get("N_DATA", "128"))
    n_mc = int(os.environ.get("N_MC", "512"))
    n_truth = int(os.environ.get("N_TRUTH", "256"))
    print("[2] smoke subset: n_data=%d n_mc=%d n_truth=%d" % (n_data, n_mc, n_truth))
    data = L._load_smoke_subset(n_data=n_data, n_mc=n_mc, n_truth=n_truth)
    args, _, _ = B.make_initial_args()
    print("    n_args = %d" % args.size)

    cart = evaluate("0", args, data)
    hel = evaluate("1", args, data)
    print("    cartesian layer: %.2fs   helicity layer: %.2fs"
          % (cart["secs"], hel["secs"]))

    print("\n[3] Cartesian vs helicity (must agree to float64 round-off)")
    ok = True
    for key, label in (("per_event", "per-event intensity"),
                       ("nll", "combined NLL"),
                       ("grad", "gradient (65)"),
                       ("hvp", "HVP (J v)")):
        if key == "nll":
            a, b = cart[key], hel[key]
            rel = abs(a - b) / max(abs(b), 1e-300)
            print("    %-22s %.12g vs %.12g   rel = %.3e   %s"
                  % (label, a, b, rel, "OK" if rel < 1e-13 else "**MISMATCH**"))
            ok &= rel < 1e-13
        else:
            ok &= compare(label, cart[key], hel[key])

    print("\n[4] intensity path: cartesian restored after the helicity run?")
    os.environ["HELICITY_LAYER"] = "0"
    again = onp.asarray(L.event_intensity(args, data, "data_"), dtype=onp.float64)
    same = float(onp.abs(again - cart["per_event"]).max())
    print("    max|I(switch off) - I(original)| = %.3e   %s"
          % (same, "OK" if same == 0.0 else "**DIFFERS**"))
    ok &= same == 0.0

    print("\n[5] summary: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
