import os
import sys

from base_functions import (
    BW, dplex_dabs, dplex_deinsum, dplex_deinsum_ord, dplex_dconstruct,
    flatte1270, flatte980, jit, make_initial_args, np, onp,
)

# MODE_BASIS = 每个 mode 内的 state（共振）数——与附件参考一致：
#   component 的 ljk 中 l 维 = state 数（f0_980=1, f0_BW=2, f2_BW=4），
#   权重键 "{mode}_{j}" 的 j 枚举 state 0..n-1
MODE_NAMES = ("phif0_kk_BW_flatte980", "phif0_kk_BW_BW", "phif2_kk_BW_BW")
MODE_BASIS = {MODE_NAMES[0]: 1, MODE_NAMES[1]: 2, MODE_NAMES[2]: 4}


def extract_parameters(args):
    args = np.asarray(args)
    return {
        "phif0_980": {"mass": args[0], "g_kk": args[1], "rg": args[2], "const": np.asarray([0.1, args[3]]), "theta": np.asarray([0.1, args[4]])},
        "phif0_1710": {"mass": args[5], "width": args[6], "const": args[7:9], "theta": args[9:11]},
        "phif2_1270": {"mass": args[11], "width": args[12], "const": args[13:18], "theta": args[18:23]},
        "phif2_1525": {"mass": args[23], "width": args[24], "const": args[25:30], "theta": args[30:35]},
        "phif2_2150": {"mass": args[35], "width": args[36], "const": args[37:42], "theta": args[42:47]},
        "phif2_2340": {"mass": args[47], "width": args[48], "const": args[49:54], "theta": args[54:59]},
        "phif0_2470": {"mass": args[59], "width": args[60], "const": args[61:63], "theta": args[63:65]},
    }


def _propagator_product(phi_s, f_props):
    phi_prop = BW(1.02, 0.004, phi_s)
    n_states = f_props.shape[1]
    phi_props = np.broadcast_to(phi_prop[:, None, :], (2, n_states, phi_s.shape[0]))
    return dplex_deinsum("le,le->le", phi_props, f_props)


def _coupled_component(tensor, const, theta, propagators):
    # tensor (i, E, K) 实数振幅；const/theta (l, i) l=state 数；propagators (2, l, E) 复数
    couplings = dplex_dconstruct(const, theta)          # (2, l, i)
    weighted = dplex_deinsum_ord("iek,li->lek", tensor, couplings)  # (2, l, E, K)
    return dplex_deinsum("lek,le->lek", weighted, propagators)


def component_phif0_kk_BW_flatte980(args, phi_s, f_s, phif0_s, phif2_s):
    del phif2_s
    p = extract_parameters(args)["phif0_980"]
    prop = _propagator_product(phi_s, flatte980(p["mass"], p["g_kk"], p["rg"], f_s)[:, None, :])
    # 单 state：const/theta (1, i) → l 展开为 1
    return _coupled_component(phif0_s, p["const"][None, :], p["theta"][None, :], prop)


def component_phif0_kk_BW_BW(args, phi_s, f_s, phif0_s, phif2_s):
    # 2 states (phif0_1710, phif0_2470)：堆叠 const/theta 与传播子，l 展开为 2
    del phif2_s
    p = extract_parameters(args)
    states = ("phif0_1710", "phif0_2470")
    const = np.stack([p[n]["const"] for n in states])          # (2, 2)
    theta = np.stack([p[n]["theta"] for n in states])          # (2, 2)
    f_props = np.stack([BW(p[n]["mass"], p[n]["width"], f_s) for n in states], axis=1)  # (2, 2, E)
    prop = _propagator_product(phi_s, f_props)
    return _coupled_component(phif0_s, const, theta, prop)


def component_phif2_kk_BW_BW(args, phi_s, f_s, phif0_s, phif2_s):
    # 4 states (f2_1270/1525/2150/2340)：堆叠，l 展开为 4
    del phif0_s
    p = extract_parameters(args)
    states = ("phif2_1270", "phif2_1525", "phif2_2150", "phif2_2340")
    const = np.stack([p[n]["const"] for n in states])          # (4, 5)
    theta = np.stack([p[n]["theta"] for n in states])          # (4, 5)
    f_props = np.stack([BW(p[n]["mass"], p[n]["width"], f_s) for n in states], axis=1)  # (2, 4, E)
    prop = _propagator_product(phi_s, f_props)
    return _coupled_component(phif2_s, const, theta, prop)


def _components(args, data, prefix):
    phi = np.asarray(data[prefix + "phi_kk"])
    f = np.asarray(data[prefix + "f_kk"])
    f0 = np.asarray(data[prefix + "phif0_kk"])
    f2 = np.asarray(data[prefix + "phif2_kk"])
    return (component_phif0_kk_BW_flatte980(args, phi, f, f0, f2),
            component_phif0_kk_BW_BW(args, phi, f, f0, f2),
            component_phif2_kk_BW_BW(args, phi, f, f0, f2))


def _weight_from_data(args, data, prefix):
    components = _components(args, data, prefix)
    total = sum(np.sum(component, axis=1) for component in components)
    return (np.sum(dplex_dabs(total), axis=-1),) + tuple(
        np.einsum("ljk->lj", dplex_dabs(component)) for component in components)


def _load_data_for_weight(prefix, n_events=None):
    # 权重施加在 MC 评估样本上（与 draw_weight 参考实现一致）：
    #   prefix="mc_"    → data/mc_truth/*.npy 全样本（weight_kk，mode="pass"）
    #   prefix="truth_" → data/mc_truth/*.npy 前 150000 事件（weight_truth_kk，mode="truth"）
    source = "data/mc_truth"
    data = {}
    if n_events is None:
        n_events = 150000 if prefix == "truth_" else None
    for var in ("phi_kk", "f_kk", "phif0_kk", "phif2_kk"):
        arr = onp.load(os.path.join(source, var + ".npy"))
        if n_events is None:
            data[prefix + var] = arr
        elif arr.ndim == 1:
            data[prefix + var] = arr[:n_events]
        else:
            data[prefix + var] = arr[:, :n_events, :]
    mc0 = onp.load("data/mc_truth/phif0_kk.npy")
    mc2 = onp.load("data/mc_truth/phif2_kk.npy")
    if n_events is not None:
        mc0 = mc0[:, :n_events, :]
        mc2 = mc2[:, :n_events, :]
    r0 = 1.0 / onp.mean(onp.sqrt(onp.sum(mc0 ** 2, axis=-1)), axis=1)
    r2 = 1.0 / onp.mean(onp.sqrt(onp.sum(mc2 ** 2, axis=-1)), axis=1)
    data[prefix + "phif0_kk"] = onp.einsum("c,cek->cek", r0, data[prefix + "phif0_kk"])
    data[prefix + "phif2_kk"] = onp.einsum("c,cek->cek", r2, data[prefix + "phif2_kk"])
    return data


def weight_kk(args):
    return _weight_from_data(args, _load_data_for_weight("mc_"), "mc_")


def weight_truth_kk(args):
    return _weight_from_data(args, _load_data_for_weight("truth_"), "truth_")


def run_weight(args_list, mode="pass"):
    args = np.array(args_list)
    if mode == "pass":
        wt_list, output_path = jit(weight_kk)(args), "output/draw/weight.npz"
    elif mode == "truth":
        wt_list, output_path = jit(weight_truth_kk)(args), "output/draw/weight_truth.npz"
    else:
        raise ValueError("mode must be 'pass' or 'truth'")
    wt_list = tuple(onp.asarray(x) for x in wt_list)
    sum_wt = float(onp.sum(wt_list[0]))
    total_weight = {"all_mods_wt": wt_list[0]}
    total_fit_frac = 0.0
    for j, name in enumerate(MODE_NAMES, 1):
        for basis_index in range(MODE_BASIS[name]):
            value = wt_list[j][basis_index]
            total_weight[f"{name}_{basis_index}"] = value
            fraction = float(onp.sum(value)) / sum_wt
            total_fit_frac += fraction
            print(f"{name}_{basis_index} fit fraction =", fraction)
    print("total fit fraction =", total_fit_frac)
    total_weight["fit_value"] = onp.asarray(args_list)
    total_weight["sum_wt"] = onp.asarray(sum_wt)
    os.makedirs("output/draw", exist_ok=True)
    onp.savez(output_path, **total_weight)
    return total_fit_frac


def _smoke():
    args, _, _ = make_initial_args()
    data = _load_data_for_weight("mc_", n_events=32)
    wt = jit(lambda x: _weight_from_data(x, data, "mc_"))(np.array(args))
    print("smoke wt shapes =", [tuple(onp.asarray(x).shape) for x in wt])
    print("smoke wt sums =", [float(onp.sum(x)) for x in wt])
    print("PASS smoke test")


if __name__ == "__main__":
    if "--smoke" in sys.argv:
        _smoke()
    else:
        args_list = onp.load("output/fit/fit_result_values.npy")
        print("计算数据权重 (mode=pass)...")
        run_weight(args_list, mode="pass")
        print("计算 truth MC 权重 (mode=truth)...")
        run_weight(args_list, mode="truth")
        print("权重计算完成，结果已保存至 output/draw/")
