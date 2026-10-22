#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**一个 case 的确定性构建**：spec -> 四动量 -> 协变/螺度张量 -> 参数表。

设计原则（与 `kk_dis/task/.../pipegen` 同一个思路）：能确定性做的事不要交给 LLM。

    1. 复制 spec（冻结输入，附 provenance）
    2. tf-pwa 造四动量（`common/make_momenta.py`，独立实现）
    3. `agent.tensor emit` -> 自足产物脚本 -> 跑出 cartesian / helicity_cascade 两份 cache
    4. 从 spec 的 `[[channel]]` 自动生成 `params/free_params.toml` 与
       `params/resonances_config.toml`（每个 (family, channel) 一对实参 =
       一个复耦合，随机初值，可冻结）
    5. 反解两份 cache 的逐道复常数（`--only calibrate`），写进 `tensors/index.toml`
    6. 写 `<case>/README.md`（这个 case 的自述，全部数字来自盘上产物）

**不写** `case.toml`：那是人手写的输入（族、道、冻结标记），生成器只读它。

一个 **case = 一个末态**。同一个末态下的多个共振族（例 1 的 ρ/ρ₃/ρ₅、例 6 的
f₀/f₂ 子分支）是**多份 spec、同一批四动量、同一个 likelihood 里相干求和** ——
这才是真实拟合的结构，所以目录按 case 分，不按 spec 分。

用法
----
    # 一个 case（读 cases/<name>/case.toml）
    python analyses/fit_examples/common/case.py build analyses/fit_examples/cases/psi_kkpipi

    # 只重建某一环
    ... build <case> --only momenta
    ... build <case> --only tensors
    ... build <case> --only params
    ... build <case> --only calibrate

⚠️ **cwd 必须是仓库根** `LLMPWA/`：产物脚本按相对路径读 `data/momenta.npz`，
`python -m agent.tensor` 也要求根目录在 sys.path 上。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as onp

try:
    import tomllib as _toml
except ModuleNotFoundError:                              # python < 3.11
    import tomli as _toml

_HERE = os.path.dirname(os.path.abspath(__file__))
_FIT = os.path.dirname(_HERE)                             # analyses/fit_examples
_ROOT = os.path.dirname(os.path.dirname(_FIT))            # LLMPWA/
TENSOR_EXAMPLES = os.path.join(_ROOT, "analyses", "tensor_examples")

# 表象 -> (emit 用的 [[basis]] 名, 产物子目录)
BASES = {"cartesian": "covariant", "helicity_cascade": "helicity_cascade"}


# ------------------------------------------------------------------ 读配置
def load_case(case_dir):
    path = os.path.join(case_dir, "case.toml")
    with open(path, "rb") as fh:
        case = _toml.load(fh)
    case.setdefault("family", [])
    for fam in case["family"]:
        fam["spec"] = _spec_name(fam["spec"])
    return case


def _spec_path(case_dir, spec):
    """spec 可以在 case 目录内，也可以直接在 tensor_examples/specs。

    接受带/不带 `.toml`（`load_case` 已把 `[[family]].spec` 归一成不带扩展名，
    但直接调用时也可能带）。
    """
    name = _spec_name(spec)
    for d in (os.path.join(case_dir, "specs"), os.path.join(case_dir, "."),
              os.path.join(TENSOR_EXAMPLES, "specs")):
        cand = os.path.join(d, name + ".toml")
        if os.path.exists(cand):
            return os.path.abspath(cand)
    raise SystemExit("找不到 spec %r（试过 %s）"
                     % (spec, [os.path.join(d, name + ".toml")
                               for d in (case_dir + "/specs", case_dir,
                                         TENSOR_EXAMPLES + "/specs")]))


def _spec_name(p):
    """去掉 `.toml` 的 spec 名 —— case.toml 里写不写扩展名都接受。

    `[[family]].spec` 是**人写的**，写成 `ex3_rhopi` 或 `ex3_rhopi.toml` 都
    自然；拼接处若不做归一，就会出现 `ex3_rhopi.toml.toml`（第一次就踩了）。
    """
    return os.path.splitext(os.path.basename(p))[0]


# ------------------------------------------------------------------ 1) spec
def stage_specs(case_dir, case, log):
    dst = os.path.join(case_dir, "specs")
    os.makedirs(dst, exist_ok=True)
    rows = []
    for fam in case["family"]:
        src = _spec_path(case_dir, fam["spec"]) if os.path.exists(
            os.path.join(case_dir, fam["spec"])) else _spec_path(case_dir, fam["spec"])
        name = _spec_name(src)
        target = os.path.join(dst, name + ".toml")
        if os.path.abspath(src) != os.path.abspath(target):
            shutil.copyfile(src, target)
        rel = os.path.relpath(src, _ROOT)
        rows.append({"family": fam["name"], "spec": name, "source": rel,
                     "sha1": _sha1(target)})
        log("  spec %-24s <- %s" % (name, rel))
    return rows


def _sha1(path):
    import hashlib
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()[:12]


# ------------------------------------------------------------------ 2) 四动量
def stage_momenta(case_dir, case, log, force=False):
    npz = os.path.join(case_dir, "data", "momenta.npz")
    if os.path.exists(npz) and not force:
        d = onp.load(npz)
        log("  四动量已存在 %s  %s" % (os.path.relpath(npz, case_dir),
                                      {k: d[k].shape for k in d.files}))
        return npz
    spec0 = os.path.join(case_dir, "specs", _spec_name(
        _spec_path(case_dir, case["family"][0]["spec"])) + ".toml")
    n = int(case["case"].get("n_events", 2000))
    seed = int(case["case"].get("seed", 0))
    cmd = [sys.executable, os.path.join(_HERE, "make_momenta.py"),
           "--spec", spec0, "--n", str(n), "--seed", str(seed), "--out", npz]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    log("  $ " + " ".join(cmd))
    log("  " + r.stdout.strip().replace("\n", "\n  "))
    if r.returncode != 0:
        err = r.stderr[-3000:]
        if "tf_pwa" in err or "No module named" in err:
            # 这是**环境**问题，不是配置问题。给出可以直接粘贴的命令，
            # 而不是让人对着 traceback 猜要挂哪个镜像。
            raise SystemExit(
                "四动量生成需要 tf_pwa，当前解释器里没有：\n%s\n\n"
                "用 tf-pwa docker 跑（无需 GPU）：\n"
                "  docker run --rm -v $PWD/fit_examples_phsp:/work \\n"
                "    -v $PWD/tmp/tf-pwa:/src:ro -e PYTHONPATH=/src \\n"
                "    -e TF_CPP_MIN_LOG_LEVEL=3 tf-pwa-check:latest \\n"
                "    bash -lc 'python /work/gen_phsp_tfpwa.py --jobs /work/jobs.json'\n"
                "然后把 npz 放到 <case>/data/momenta.npz；"
                "本仓库的 5 个 case 已经生成好了。\n"
                "（`common/gen_phsp_tfpwa.py` 是零依赖版，按 jobs.json 批量生成。）"
                % err)
    log("  %.1fs" % (time.time() - t0))
    return npz


# ------------------------------------------------------------------ 3) 张量
def stage_tensors(case_dir, case, log, force=False, n_max=None, env=None):
    npz = os.path.join(case_dir, "data", "momenta.npz")
    gen_root = os.path.join(case_dir, "gen")
    out_root = os.path.join(case_dir, "tensors")
    report = []
    for fam in case["family"]:
        # `--only tensors` 时 specs/ 可能是空的：先自愈一次（幂等），
        # 否则会以 "找不到 spec" 收场 —— 而真实原因只是"跳过了上一环"。
        spec = os.path.join(case_dir, "specs", fam["spec"] + ".toml")
        if not os.path.exists(spec):
            stage_specs(case_dir, case, log)
        if not os.path.exists(spec):
            raise SystemExit("spec 仍然不存在：%s" % spec)
        name = fam["spec"]
        gdir = os.path.join(gen_root, name)
        if force or not glob.glob(os.path.join(gdir, "gen_tensor_*.py")):
            r = subprocess.run([sys.executable, "-m", "agent.tensor", "emit", spec,
                                "--out-dir", gdir], cwd=_ROOT,
                               capture_output=True, text=True)
            if r.returncode != 0:
                raise SystemExit("emit 失败 %s\n%s" % (spec, r.stderr[-3000:]))
            log("  emit %-24s -> %s" % (name, os.path.relpath(gdir, case_dir)))
        for basis, suffix in BASES.items():
            odir = os.path.join(out_root, name, basis)
            os.makedirs(odir, exist_ok=True)
            script = os.path.join(gdir, "gen_tensor_%s_%s.py" % (name, suffix))
            if not os.path.exists(script):
                log("  !! %s 没有 %s 产物（该 spec 未声明这个 [[basis]]）"
                    % (name, basis))
                continue
            cmd = [sys.executable, script, "--momenta", npz, "--out", odir]
            if n_max:
                cmd += ["--n", str(n_max)]
            t0 = time.time()
            r = subprocess.run(cmd, capture_output=True, text=True,
                               env=dict(os.environ, **(env or {})))
            dt = time.time() - t0
            if r.returncode != 0:
                raise SystemExit("张量生成失败 %s\n%s" % (script, r.stderr[-3000:]))
            files = sorted(os.path.basename(f) for f in glob.glob(odir + "/*.npy"))
            arr = onp.load(os.path.join(odir, files[-1]))
            report.append({"family": fam["name"], "spec": name, "basis": basis,
                           "file": files[-1], "shape": list(arr.shape),
                           "dtype": str(arr.dtype), "seconds": round(dt, 2),
                           "finite": bool(onp.isfinite(arr).all()),
                           "max": float(onp.abs(arr).max())})
            log("  %-24s %-17s %-14s %-9s %5.2fs  max=%.3g"
                % (name, basis, str(tuple(arr.shape)),
                   str(arr.dtype).replace("128", ""), dt, float(onp.abs(arr).max())))
    with open(os.path.join(case_dir, "tensors", "report.json"), "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    return report


# ------------------------------------------------------------------ 4) 参数表
def _channel_names(spec):
    with open(spec, "rb") as fh:
        raw = _toml.load(fh)
    chans = raw.get("channel") or []
    if not chans:
        raise SystemExit("%s 没有显式 [[channel]]；参数表需要显式的道名" % spec)
    return [c["name"] for c in chans]


def stage_params(case_dir, case, log, seed=None, force=False):
    """从 spec 的道表生成 `params/free_params.toml` + `params/resonances_config.toml`。

    格式故意**贴近** `kk_dis/run/free_params.toml`（`path/value/range/error/arg_index`），
    但类别不同：这里每个道是一个**复耦合** `g_i`，写成 `constN`/`thetaN` 一对
    实参（实部/虚部），与 `dplex_dconstruct` 的约定一致。

    初值：`|g|` 在全道统一（随机相位），默认 0.1，可用
    `[case].coupling_scale` 覆盖；`fixed = true` 的道不参与拟合（便于先冻结一批）。
    """
    path = os.path.join(case_dir, "params", "free_params.toml")
    if os.path.exists(path) and not force:
        log("  参数表已存在 %s（--force 覆盖）" % os.path.relpath(path, case_dir))
        return path
    seed = int(seed if seed is not None else case["case"].get("seed", 0))
    rng = onp.random.default_rng(seed + 1000)
    scale = float(case["case"].get("coupling_scale", 0.1))

    rows, arg = [], 0
    for fam in case["family"]:
        spec = os.path.join(case_dir, "specs", fam["spec"] + ".toml")
        names = _channel_names(spec)
        for ch in names:
            phase = float(rng.uniform(0.0, 2.0 * onp.pi))
            re, im = scale * float(onp.cos(phase)), scale * float(onp.sin(phase))
            base = "couplings.%s.%s" % (fam["name"], ch)
            for part, val in (("const", re), ("theta", im)):
                rows.append({"path": "%s.%s" % (base, part), "value": val,
                             "error": 0.01 * scale, "arg_index": arg,
                             "fixed": bool(fam.get("frozen", False))})
                arg += 1
        log("  %-12s %2d 道 -> %2d 个自由实参" % (fam["name"], len(names), 2 * len(names)))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# 由 analyses/fit_examples/common/case.py 自动生成（不要手改）\n")
        fh.write("# 每个道一个复耦合：.const = Re g, .theta = Im g\n")
        fh.write("# fixed = true 的项冻结；range 先留空（高斯基线由 likelihood 侧决定）\n\n")
        for r in rows:
            fh.write("[[data]]\n")
            fh.write('path = "%s"\n' % r["path"])
            fh.write("value = %.12g\n" % r["value"])
            fh.write("error = %.12g\n" % r["error"])
            fh.write("arg_index = %d\n" % r["arg_index"])
            fh.write("fixed = %s\n\n" % ("true" if r["fixed"] else "false"))

    # resonances_config：只描述传播子，耦合留给 free_params
    rpath = os.path.join(case_dir, "params", "resonances_config.toml")
    with open(rpath, "w", encoding="utf-8") as fh:
        fh.write("# 由 analyses/fit_examples/common/case.py 自动生成（不要手改）\n")
        fh.write("# 只写传播子；耦合与初值在 free_params.toml\n\n")
        for fam in case["family"]:
            fh.write("[resonances.%s.propagators.A_propagator]\n" % fam["name"])
            fh.write('propagator_type = "%s"\n' % fam.get("propagator", "BW"))
            fh.write("mass = { value = %.6g, fixed = %s }\n"
                     % (float(fam.get("mass", 1.0)),
                        "false" if fam.get("fit_mass") else "true"))
            fh.write("width = { value = %.6g, fixed = %s }\n"
                     % (float(fam.get("width", 0.1)),
                        "false" if fam.get("fit_width") else "true"))
            fh.write('Sbc = "%s"\n\n' % fam.get("sbc", fam["name"]))
    log("  参数表 %s（%d 个实参）" % (os.path.relpath(path, case_dir), len(rows)))
    return path


def _scan_tensors(case_dir, case):
    """不改文件、只把**盘上已有的**产物扫成 report 结构（幂等）。"""
    report = []
    for fam in case["family"]:
        for basis in BASES:
            cands = sorted(glob.glob(os.path.join(
                case_dir, "tensors", fam["spec"], basis, "*.npy")))
            if not cands:
                continue
            arr = onp.load(cands[0], mmap_mode="r")
            report.append({"family": fam["name"], "spec": fam["spec"],
                           "basis": basis,
                           "file": os.path.basename(cands[0]),
                           "shape": list(arr.shape), "dtype": str(arr.dtype),
                           "seconds": None,
                           "finite": bool(onp.isfinite(arr).all()),
                           "max": float(onp.abs(arr).max())})
    return report


# ------------------------------------------------------------------ 4b) 标定
def _tensor_path(case_dir, family, basis):
    """(族, 表象) -> 唯一的 .npy。多份就报错 —— 静默取第一个会选错族。"""
    cands = sorted(glob.glob(os.path.join(case_dir, "tensors", family, basis,
                                          "*.npy")))
    if not cands:
        raise SystemExit("缺 tensors/%s/%s/*.npy" % (family, basis))
    if len(cands) > 1:
        raise SystemExit("tensors/%s/%s/ 下有 %d 个 .npy：%s"
                         % (family, basis, len(cands),
                            [os.path.basename(c) for c in cands]))
    return cands[0]


def stage_calibrate(case_dir, case, log):
    """反解 cartesian 与 helicity_cascade 两份 cache 的逐道复常数，写进 index.toml。

    这是**跨表象比较的前提**：两份 cache 不是同一个数组，差一个逐道复常数
    （`kk_dis` §5）。不标定就在固定参数点比 likelihood，会得到"差几个百分点"
    的假失败。

    残差是**判据**而不是附带信息：
        rel ~ 1e-15  -> 同一振幅，只差常数（标定有效）
        rel 大        -> 形状不同，是**真问题**（已知：ex3_rhopi、ex6 rr/f2s）
    所以每份 cache 的 `calib_max_rel` 都记下来，README 引用它。
    """
    out = {}
    for fam in case["family"]:
        try:
            # 目录按 **spec 名** 分（不是族名）：一份 spec = 一组道表 = 一组 cache。
            # 族名只在 case.toml / free_params.toml 里用来分组，是**逻辑**名字。
            pc = _tensor_path(case_dir, fam["spec"], "cartesian")
            ph = _tensor_path(case_dir, fam["spec"], "helicity_cascade")
        except SystemExit as e:
            log("  %-12s 跳过（%s）" % (fam["name"], str(e).splitlines()[0]))
            continue
        import numpy as onp
        T = onp.load(pc)
        C = onp.load(ph)
        EPS = onp.array([[1 / onp.sqrt(2), -1j / onp.sqrt(2), 0],
                         [0, 0, 1],
                         [-1 / onp.sqrt(2), -1j / onp.sqrt(2), 0]])
        W = onp.einsum("lm,iem->iel", EPS, T.astype(onp.complex128))
        cs, worst = [], 0.0
        for i in range(W.shape[0]):
            c = onp.vdot(C[i], W[i]) / max(abs(onp.vdot(C[i], C[i])), 1e-300)
            cs.append(c)
            worst = max(worst, float(onp.abs(W[i] - c * C[i]).max()
                                     / max(onp.abs(W[i]).max(), 1e-300)))
        out[fam["name"]] = {"c": cs, "max_rel": worst}
        if worst < 1e-9:
            verdict = "OK"
        elif worst < 1e-6:
            verdict = "OK（~1e-7：中间态离壳）"
        else:
            verdict = "**形状不同（已知边界）**"
        log("  %-12s 标定 |c| = %s  max_rel = %.2e  %s"
            % (fam["name"], " ".join("%.4g" % abs(c) for c in cs),
               worst, verdict))
    return out


# ------------------------------------------------------------------ 5) 事实清单
def write_index(case_dir, case, specs, tensors, log, calib=None):
    """把**实测事实**（形状、文件名、dtype、耗时）写成 `tensors/index.toml`。

    ⚠️ 不写回 `case.toml`：那是**人手写**的输入（族、道、冻结标记），
    build 覆盖它等于把配置吃掉。生成物与输入必须分开 —— 这是 kk_dis 里
    "`pipeline.toml` 是冻结输入、`manifest.json` 是产物清单"的同一条纪律。
    """
    # ⚠️ `calib={}` 意味着"这一轮没算标定"，**不是**"标定不存在"。
    # 张量一重建就必须重算标定；其余阶段（`--only params/readme`）必须把
    # 盘上已有的那份留住 —— 否则 index.toml 会被悄悄清空，下游
    # `likelihood.py` 会把所有族判成"不可比"，表现得像"跨表象对不上"。
    calib = dict(calib or {})
    old = os.path.join(case_dir, "tensors", "index.toml")
    if os.path.exists(old):
        with open(old, "rb") as fh:
            prev = _toml.load(fh)
        for f in prev.get("family", []):
            nm, raw = f.get("name"), f.get("calibration_cart_to_casc")
            if nm and nm not in calib and raw:
                arr = onp.asarray(raw, dtype=float)
                calib[nm] = {"c": arr[:, 0] + 1j * arr[:, 1],
                             "max_rel": float(f.get("calibration_max_rel", 1.0)),
                             "carried_over": True}

    by_fam = {}
    for t in tensors:
        by_fam.setdefault(t["family"], {})[t["basis"]] = t
    lines = ["# 由 common/case.py 生成：实测事实（形状/文件名/dtype）。不要手改。",
             "# 人手写的输入在 case.toml。", ""]
    lines.append("[case]")
    lines.append('name = %s' % json.dumps(case["case"]["name"]))
    lines.append("")
    for fam in case["family"]:
        t = by_fam.get(fam["name"], {})
        lines.append("[[family]]")
        lines.append('name = %s' % json.dumps(fam["name"]))
        lines.append('spec = %s' % json.dumps(fam["spec"]))
        for b in BASES:
            if b in t:
                lines.append('%s_file = %s' % (b, json.dumps(t[b]["file"])))
                lines.append('%s_shape = %s' % (b, json.dumps(t[b]["shape"])))
                lines.append('%s_dtype = %s' % (b, json.dumps(t[b]["dtype"])))
                lines.append('%s_max = %.10g' % (b, t[b]["max"]))
        cal = calib.get(fam["name"])
        if cal:
            # ⚠️ TOML **没有复数标量** —— 写 `(1+2j)` 会直接 TOMLDecodeError。
            # 存成 [[re, im], ...]。
            lines.append("calibration_cart_to_casc = [%s]"
                         % ", ".join("[%.12g, %.12g]" % (c.real, c.imag)
                                     for c in cal["c"]))
            lines.append("calibration_max_rel = %.6e" % cal["max_rel"])
        lines.append("")

    path = os.path.join(case_dir, "tensors", "index.toml")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    log("  tensors/index.toml 已更新（%d 个族）" % len(case["family"]))
    return path


# ------------------------------------------------------------------ 6) 自述
def stage_readme(case_dir, case, log):
    """从**盘上已有产物**写 `<case>/README.md`。不重算任何东西，幂等。

    数字全部现场读（`data/momenta.meta.json`、`tensors/report.json`、
    `params/*.toml`、`logs/likelihood.json`），所以 README 不会与产物漂移。
    缺哪份就不写那一节 —— 而不是写一个猜测值。
    """
    name = case["case"]["name"]
    L = ["# case `%s`" % name, "",
         case["case"].get("title", ""), "",
         "> 本文件由 `common/case.py build <case> --only readme` 生成。"
         "人手写的输入在 `case.toml`；实测事实在 `tensors/index.toml`。", ""]

    # ---- 族与道
    L += ["## 共振族（相干求和）", "",
          "| 族 | spec | 道数 | 复耦合 | m (GeV) | Γ (GeV) | 传播子 |",
          "|---|---|---|---|---|---|---|"]
    n_ch_total = 0
    for fam in case["family"]:
        n_ch = _n_channels(case_dir, fam["spec"])
        n_ch_total += n_ch
        L.append("| `%s` | `%s` | %d | %d | %.4g | %.4g | %s |"
                 % (fam["name"], fam["spec"], n_ch, n_ch,
                    float(fam.get("mass", 0)), float(fam.get("width", 0)),
                    fam.get("propagator", "-")))
    L += ["", "合计 **%d 道 -> %d 个自由实参**（每道一个复耦合 "
          "`g = const + i·theta`）。" % (n_ch_total, 2 * n_ch_total), ""]

    # ---- 四动量
    mp = os.path.join(case_dir, "data", "momenta.meta.json")
    if os.path.exists(mp):
        d = json.load(open(mp, encoding="utf-8"))
        L += ["## 四动量（tf-pwa 相空间）", "",
              "`data/momenta.npz`，布局 `%s`，键 `%s`。"
              % (d["layout"], ", ".join(d["keys"])), "",
              "| 项 | 值 |", "|---|---|",
              "| 事例数 | %d |" % d["n"],
              "| 生成器 | `%s` |" % d["generator"],
              "| 耗时 | %.2f s（%.3g 事例/秒）|"
              % (d["seconds"], d["events_per_s"]),
              "| 四动量守恒 | %.2e（绝对）/ %.2e（相对）|"
              % (d["conservation_abs"], d["conservation_rel"]),
              "| 在壳 | max %.2e GeV |" % max(d["onshell_max_abs"].values()),
              ""]
        worst = min(d["two_body_min"].items(), key=lambda kv: kv[1]["margin"])
        L += ["最紧的两体阈值余量：`%s`，min = %.6f GeV，阈值 %.6f，余量 %.2e GeV。"
              % (worst[0], worst[1]["min"], worst[1]["thr"], worst[1]["margin"]),
              "", "余量为正是**必要条件**：`Qabc` 变负会让 `B1/B2` 的 `sqrt` 出 NaN。", ""]

    # ---- 张量
    rp = os.path.join(case_dir, "tensors", "report.json")
    if os.path.exists(rp):
        rep = json.load(open(rp, encoding="utf-8"))
        L += ["## 张量 cache", "",
              "| 族 | 表象 | 形状 | dtype | max abs | 生成 |", "|---|---|---|---|---|---|"]
        for t in rep:
            sec = ("%.2f s" % t["seconds"]) if t.get("seconds") else "-"
            L.append("| `%s` | `%s` | `%s` | `%s` | %.3g | %s |"
                     % (t["family"], t["basis"], t["shape"], t["dtype"],
                        t["max"], sec))
        bases_here = sorted({t["basis"] for t in rep})
        L.append("")
        if len(bases_here) == 2:
            L += ["槽位语义是**承重约定**：协变侧末轴 = `(x, y, z)`，"
                  "级联侧末轴 = `(λ=+1, 0, -1)`。横向强度取 `(x,y)` 与 `(±1)`，"
                  "两边由 `W = EPS_A · T` 联系。", ""]
        else:
            L += ["本 case 只有 `%s` 一份 cache（spec 未声明另一个 `[[basis]]`），"
                  "因此没有跨表象比较。" % bases_here[0], ""]

        idx = os.path.join(case_dir, "tensors", "index.toml")
        if os.path.exists(idx):
            with open(idx, "rb") as fh:
                ix = _toml.load(fh)
            rows = [f for f in ix.get("family", [])
                    if f.get("calibration_max_rel") is not None]
            if rows:
                L += ["### 跨表象标定", "",
                      "`c_i` 定义为 `W_i = c_i · C_i`（协变侧经 `EPS_A` 后与级联侧比较）。"
                      "残差 ~1e-15 说明两份 cache 是**同一振幅**、只差逐道复常数；"
                      "残差大说明形状不同（真问题）。", "",
                      "| 族 | max_rel | 结论 |", "|---|---|---|"]
                for f in rows:
                    r = f["calibration_max_rel"]
                    # 三档，与 `likelihood.py` 的可比阈值（1e-6）一致：
                    #   <1e-9        逐位同一振幅
                    #   1e-9..1e-6   同一振幅，残差来自中间态离壳（已知边界）
                    #   >1e-6        形状不同，两份 cache 不是同一个振幅
                    if r < 1e-9:
                        verdict = "只差常数"
                    elif r < 1e-6:
                        verdict = "只差常数（~1e-7：中间态离壳）"
                    else:
                        verdict = "**形状不同**"
                    L.append("| `%s` | %.2e | %s |" % (f["name"], r, verdict))
                L.append("")

    # ---- 参数表
    fp = os.path.join(case_dir, "params", "free_params.toml")
    if os.path.exists(fp):
        with open(fp, "rb") as fh:
            pars = _toml.load(fh).get("data", [])
        n_fixed = sum(1 for p in pars if p.get("fixed"))
        L += ["## 参数表", "",
              "`params/free_params.toml`：%d 项，其中 %d 项冻结。"
              % (len(pars), n_fixed),
              "初值是 `coupling_scale = %s` 下的**随机相位**，不是拟合结果。"
              % case["case"].get("coupling_scale", "-"),
              "", "`params/resonances_config.toml`：只写传播子（质量/宽度先冻结）。",
              ""]

    # ---- 似然测试结果
    lp = os.path.join(case_dir, "logs", "likelihood.json")
    if os.path.exists(lp):
        d = json.load(open(lp, encoding="utf-8"))
        L += ["## 似然测试结果", "",
              "设备 `%s`，jax %s（x64=%s）。"
              % (", ".join(d["devices"]), d["jax"], d["x64"]), ""]
        L += ["| 表象 | 值+梯度 (s) | 事例/秒 |", "|---|---|---|"]
        for b, v in d["seconds_per_value_grad"].items():
            L.append("| `%s` | %.2e | %.3g |"
                     % (b, v, d.get("events_per_s_value_grad", {}).get(b, 0)))
        L.append("")
        if "derivative_check" in d:
            L += ["| 表象 | grad vs FD | HVP vs FD |", "|---|---|---|"]
            for b, v in d["derivative_check"].items():
                L.append("| `%s` | %.1e | %.1e |"
                         % (b, v["grad_vs_fd_rel"], v["hvp_vs_fd_rel"]))
            L.append("")
        if "cart_vs_casc" in d:
            e = d["cart_vs_casc"].get("initial", {})
            if "intensity_rel_on_comparable" in e:
                L += ["跨表象强度残差（仅可比族）**%.2e**。" %
                      e["intensity_rel_on_comparable"]]
            if e.get("excluded"):
                L += ["", "已排除（两份 cache 形状不同，不可比）：%s —— "
                      "见 `../../README.md` 的「已知边界」。" % e["excluded"]]
            L.append("")
        if "fit" in d:
            f = d["fit"]
            L += ["短跑 L-BFGS-B（%d 次迭代，**未收敛是预期的**）："
                  "L 从初值到 %.1f，%.2f s。" % (f["nit"], f["f0"], f["seconds"]), ""]

    # ---- 怎么重跑
    # `--basis both` 只在该 case **两份** cache 都在时才对；`psi_omega_kk`
    # 没有级联产物，写 `both` 会直接报 "缺 cache"。这里按盘上实际情况给命令。
    has = {b for b in BASES
           if any(glob.glob(os.path.join(case_dir, "tensors", fam["spec"],
                                         b, "*.npy"))
                  for fam in case["family"])}
    basis = "both" if len(has) == 2 else (sorted(has)[0] if has else "both")
    L += ["## 重跑", "", "```bash",
          "cd LLMPWA   # cwd 必须是仓库根",
          "python analyses/fit_examples/common/case.py build "
          "analyses/fit_examples/cases/%s" % name,
          "analyses/kk_dis/.venv/bin/python "
          "analyses/fit_examples/common/likelihood.py "
          "analyses/fit_examples/cases/%s --basis %s --check --fit" % (name, basis),
          "```", ""]

    path = os.path.join(case_dir, "README.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    log("  README.md 已更新（%d 行）" % len(L))
    return path


def _n_channels(case_dir, spec):
    with open(os.path.join(case_dir, "specs", _spec_name(spec) + ".toml"),
              "rb") as fh:
        raw = _toml.load(fh)
    return len(raw.get("channel") or [])


# ------------------------------------------------------------------ 7) jobs
def emit_jobs(case_dirs, out, log):
    """把若干 case 的相空间请求写成 `gen_phsp_tfpwa.py --jobs` 的格式。

    这是**离线再生**的入口：本机没有 tf-pwa，所以造四动量必须走 docker。
    把"要造什么"（case.toml 里的 n_events/seed + spec 的末态质量）抽成
    一个纯数据文件，docker 里那一侧就完全不需要读仓库代码。
    """
    jobs = []
    for cd in case_dirs:
        case = load_case(cd)
        fam0 = case["family"][0]
        spec = _spec_path(cd, fam0["spec"])
        with open(spec, "rb") as fh:
            raw = _toml.load(fh)
        keys = raw["input"]["keys"]
        layout = raw["input"].get("layout", "x,y,z,E")
        m0 = float(raw["chain"].get("psi_mass", 3.686))
        masses = []
        for k in keys:
            p = raw["particles"].get(k, {})
            if "mass" not in p:
                raise SystemExit("%s 的粒子 %s 没有 mass" % (spec, k))
            masses.append(float(p["mass"]))
        jobs.append({
            "name": case["case"]["name"], "m0": m0, "masses": masses,
            "keys": keys, "n": int(case["case"].get("n_events", 2000)),
            "seed": int(case["case"].get("seed", 0)),
            "layout": layout,
            "out": os.path.abspath(os.path.join(cd, "data", "momenta.npz")),
        })
        log("  job %-13s n=%d keys=%s" % (jobs[-1]["name"], jobs[-1]["n"],
                                          ",".join(keys)))
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(jobs, fh, indent=2, ensure_ascii=False)
    log("  已写 %s（%d 个 job）" % (out, len(jobs)))
    return out


# ------------------------------------------------------------------ main
def build(case_dir, only="", force=False, n_max=None):
    case = load_case(case_dir)
    logs = []

    def log(msg):
        print(msg, flush=True)
        logs.append(msg)

    log("== case %s（%s）" % (case["case"]["name"], case_dir))
    t0 = time.time()
    if only in ("", "specs"):
        specs = stage_specs(case_dir, case, log)
    else:
        specs = []
    if only in ("", "momenta"):
        stage_momenta(case_dir, case, log, force=force)
    if only in ("", "tensors"):
        tensors = stage_tensors(case_dir, case, log, force=force, n_max=n_max)
    else:
        # `--only calibrate` / `--only params` 时也要能把**已有**的产物事实
        # 重新写进 index.toml，所以从盘上重扫一遍（不重算任何东西）。
        tensors = _scan_tensors(case_dir, case)
    if only in ("", "params"):
        stage_params(case_dir, case, log, force=force)
    calib = {}
    # 张量重建后旧标定立刻失效（cache 换了），所以 tensors 阶段也要重算。
    if only in ("", "tensors", "calibrate"):
        calib = stage_calibrate(case_dir, case, log)
    if only in ("", "readme"):
        stage_readme(case_dir, case, log)
    if tensors:
        write_index(case_dir, case, specs, tensors, log, calib)
    os.makedirs(os.path.join(case_dir, "logs"), exist_ok=True)
    with open(os.path.join(case_dir, "logs", "build.log"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(logs) + "\n")
    print("== %.1fs" % (time.time() - t0))
    return 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "jobs"])
    ap.add_argument("case_dir", nargs="+")
    ap.add_argument("--only", default="",
                    choices=["", "specs", "momenta", "tensors", "params",
                             "calibrate", "readme"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--n-max", type=int, default=None,
                    help="张量只算前 n 个事例（调试用；产物会与四动量不一致）")
    ap.add_argument("--out", default="jobs.json",
                    help="`jobs` 子命令的输出路径")
    args = ap.parse_args()
    if args.cmd == "jobs":
        def log(msg):
            print(msg, flush=True)
        emit_jobs([os.path.abspath(p) for p in args.case_dir], args.out, log)
        return 0
    if len(args.case_dir) != 1:
        raise SystemExit("build 只接受一个 case 目录（收到 %d 个）"
                         % len(args.case_dir))
    return build(os.path.abspath(args.case_dir[0]), args.only, args.force,
                 args.n_max)


if __name__ == "__main__":
    sys.exit(main())
