import os
import sys

from base_functions import (
    BW,
    config,
    dplex_dabs,
    dplex_dconstruct,
    dplex_deinsum,
    dplex_deinsum_ord,
    flatte1270,
    flatte980,
    jit,
    make_initial_args,
    np,
    onp,
    setup_logging,
)


MODE_NAMES = (
    "phif0_kk_BW_flatte980",
    "phif0_kk_BW_BW",
    "phif2_kk_BW_BW",
)
MODE_BASIS = {
    "phif0_kk_BW_flatte980": 1,
    "phif0_kk_BW_BW": 2,
    "phif2_kk_BW_BW": 4,
}
TRUTH_MC_CAP = 150000
_DATA_CACHE = {}


def extract_parameters(args):
    """Map the 65 fitted arguments to this analysis's resonance parameters."""
    args = np.asarray(args)
    return {
        "phif0_980": {
            "A_mass": 1.02,
            "A_width": 0.004,
            "B_mass": args[0],
            "B_g_kk": args[1],
            "B_rg": args[2],
            "const": np.asarray([0.1, args[3]]),
            "theta": np.asarray([0.1, args[4]]),
        },
        "phif0_1710": {
            "A_mass": 1.02,
            "A_width": 0.004,
            "B_mass": args[5],
            "B_width": args[6],
            "const": args[7:9],
            "theta": args[9:11],
        },
        "phif2_1270": {
            "A_mass": 1.02,
            "A_width": 0.004,
            "B_mass": args[11],
            "B_width": args[12],
            "const": args[13:18],
            "theta": args[18:23],
        },
        "phif2_1525": {
            "A_mass": 1.02,
            "A_width": 0.004,
            "B_mass": args[23],
            "B_width": args[24],
            "const": args[25:30],
            "theta": args[30:35],
        },
        "phif2_2150": {
            "A_mass": 1.02,
            "A_width": 0.004,
            "B_mass": args[35],
            "B_width": args[36],
            "const": args[37:42],
            "theta": args[42:47],
        },
        "phif2_2340": {
            "A_mass": 1.02,
            "A_width": 0.004,
            "B_mass": args[47],
            "B_width": args[48],
            "const": args[49:54],
            "theta": args[54:59],
        },
        "phif0_2470": {
            "A_mass": 1.02,
            "A_width": 0.004,
            "B_mass": args[59],
            "B_width": args[60],
            "const": args[61:63],
            "theta": args[63:65],
        },
    }


def _propagator_product(phi_s, f_propagators):
    phi_propagator = BW(1.02, 0.004, phi_s)
    n_states = f_propagators.shape[1]
    phi_propagators = np.broadcast_to(
        phi_propagator[:, None, :], (2, n_states, phi_s.shape[0])
    )
    return dplex_deinsum("le,le->le", phi_propagators, f_propagators)


def _coupled_component(amplitude_tensor, const, theta, propagators):
    couplings = dplex_dconstruct(const, theta)
    weighted_amplitude = dplex_deinsum_ord(
        "iek,li->lek", amplitude_tensor, couplings
    )
    return dplex_deinsum("lek,le->lek", weighted_amplitude, propagators)


def component_phif0_kk_BW_flatte980(args, phi_s, f_s, phif0_s, phif2_s):
    del phif2_s
    parameter = extract_parameters(args)["phif0_980"]
    f_propagators = flatte980(
        parameter["B_mass"], parameter["B_g_kk"], parameter["B_rg"], f_s
    )[:, None, :]
    propagators = _propagator_product(phi_s, f_propagators)
    return _coupled_component(
        phif0_s,
        parameter["const"][None, :],
        parameter["theta"][None, :],
        propagators,
    )


def component_phif0_kk_BW_BW(args, phi_s, f_s, phif0_s, phif2_s):
    del phif2_s
    parameters = extract_parameters(args)
    state_names = ("phif0_1710", "phif0_2470")
    const = np.stack([parameters[name]["const"] for name in state_names])
    theta = np.stack([parameters[name]["theta"] for name in state_names])
    f_propagators = np.stack(
        [
            BW(
                parameters[name]["B_mass"],
                parameters[name]["B_width"],
                f_s,
            )
            for name in state_names
        ],
        axis=1,
    )
    propagators = _propagator_product(phi_s, f_propagators)
    return _coupled_component(phif0_s, const, theta, propagators)


def component_phif2_kk_BW_BW(args, phi_s, f_s, phif0_s, phif2_s):
    del phif0_s
    parameters = extract_parameters(args)
    state_names = ("phif2_1270", "phif2_1525", "phif2_2150", "phif2_2340")
    const = np.stack([parameters[name]["const"] for name in state_names])
    theta = np.stack([parameters[name]["theta"] for name in state_names])

    first = parameters["phif2_1270"]
    f_propagators = [
        flatte1270(first["B_mass"], first["B_width"], f_s)
    ]
    for name in state_names[1:]:
        parameter = parameters[name]
        f_propagators.append(
            BW(parameter["B_mass"], parameter["B_width"], f_s)
        )
    f_propagators = np.stack(f_propagators, axis=1)
    propagators = _propagator_product(phi_s, f_propagators)
    return _coupled_component(phif2_s, const, theta, propagators)


def _components(args, data, prefix):
    phi_s = np.asarray(data[prefix + "phi_kk"])
    f_s = np.asarray(data[prefix + "f_kk"])
    phif0_s = np.asarray(data[prefix + "phif0_kk"])
    phif2_s = np.asarray(data[prefix + "phif2_kk"])
    return (
        component_phif0_kk_BW_flatte980(args, phi_s, f_s, phif0_s, phif2_s),
        component_phif0_kk_BW_BW(args, phi_s, f_s, phif0_s, phif2_s),
        component_phif2_kk_BW_BW(args, phi_s, f_s, phif0_s, phif2_s),
    )


def _weight_from_data(args, data, prefix):
    components = _components(args, data, prefix)
    total_amplitude = sum(np.sum(component, axis=1) for component in components)
    total_intensity = np.sum(dplex_dabs(total_amplitude), axis=-1)
    component_intensities = tuple(
        np.einsum("ljk->lj", dplex_dabs(component))
        for component in components
    )
    return (total_intensity,) + component_intensities


def _normalization_factors():
    factors = {}
    for variable in ("phif0_kk", "phif2_kk"):
        mc_array = onp.load(
            os.path.join("data/mc_truth", variable + ".npy"), mmap_mode="r"
        )
        factors[variable] = 1.0 / onp.mean(
            onp.sqrt(onp.sum(mc_array ** 2, axis=-1)), axis=1
        )
    return factors


def _load_data_for_weight(prefix, n_events=None, use_cache=True):
    if prefix not in ("mc_", "truth_"):
        raise ValueError("weight data prefix must be 'mc_' or 'truth_'")
    if n_events is None and prefix == "truth_":
        n_events = TRUTH_MC_CAP

    cache_key = (prefix, n_events)
    if use_cache and cache_key in _DATA_CACHE:
        return _DATA_CACHE[cache_key]

    data = {}
    for variable in ("phi_kk", "f_kk"):
        array = onp.load(
            os.path.join("data/mc_truth", variable + ".npy"), mmap_mode="r"
        )
        data[prefix + variable] = onp.asarray(
            array if n_events is None else array[:n_events]
        )

    normalization = _normalization_factors()
    for variable in ("phif0_kk", "phif2_kk"):
        array = onp.load(
            os.path.join("data/mc_truth", variable + ".npy"), mmap_mode="r"
        )
        selected = onp.asarray(
            array if n_events is None else array[:, :n_events, :]
        )
        data[prefix + variable] = onp.einsum(
            "c,cek->cek", normalization[variable], selected
        )

    if use_cache:
        _DATA_CACHE[cache_key] = data
    return data


def weight_kk(args):
    return _weight_from_data(args, _load_data_for_weight("mc_"), "mc_")


def weight_truth_kk(args):
    return _weight_from_data(args, _load_data_for_weight("truth_"), "truth_")


def run_weight(args_list, mode="pass"):
    args = np.array(args_list)
    if args.ndim != 1 or args.shape[0] != 65:
        raise ValueError("expected a one-dimensional fitted parameter vector of length 65")

    if mode == "pass":
        wt_list = jit(weight_kk)(args)
        output_path = "output/draw/weight.npz"
    elif mode == "truth":
        wt_list = jit(weight_truth_kk)(args)
        output_path = "output/draw/weight_truth.npz"
    else:
        raise ValueError("mode must be 'pass' or 'truth'")

    wt_list = tuple(onp.asarray(weight) for weight in wt_list)
    sum_wt = float(onp.sum(wt_list[0]))
    if not onp.isfinite(sum_wt) or sum_wt <= 0.0:
        raise ValueError("total intensity sum is non-positive or non-finite")

    total_weight = {"all_mods_wt": wt_list[0]}
    total_fit_frac = 0.0
    for list_index, mode_name in enumerate(MODE_NAMES, start=1):
        mode_weights = wt_list[list_index]
        expected_basis = MODE_BASIS[mode_name]
        if mode_weights.shape[0] != expected_basis:
            raise ValueError(
                "%s produced %d bases, expected %d"
                % (mode_name, mode_weights.shape[0], expected_basis)
            )
        for basis_index in range(expected_basis):
            key = "%s_%d" % (mode_name, basis_index)
            value = mode_weights[basis_index]
            total_weight[key] = value
            fraction = float(onp.sum(value)) / sum_wt
            total_fit_frac += fraction
            print("%s fit fraction = %.12g" % (key, fraction))

    print("total fit fraction = %.12g" % total_fit_frac)
    total_weight["fit_value"] = onp.asarray(args_list)
    total_weight["sum_wt"] = onp.asarray(sum_wt)
    os.makedirs("output/draw", exist_ok=True)
    onp.savez(output_path, **total_weight)
    print("saved", output_path, "with", len(total_weight), "keys")
    return total_fit_frac


def _smoke(n_events=512):
    args, _, _ = make_initial_args()
    data = _load_data_for_weight("mc_", n_events=n_events, use_cache=False)
    wt_list = jit(lambda values: _weight_from_data(values, data, "mc_"))(
        np.array(args)
    )
    shapes = [tuple(onp.asarray(weight).shape) for weight in wt_list]
    sums = [float(onp.sum(onp.asarray(weight))) for weight in wt_list]
    expected_shapes = [
        (n_events,),
        (1, n_events),
        (2, n_events),
        (4, n_events),
    ]
    print("smoke event count =", n_events)
    print("smoke wt shapes =", shapes)
    print("smoke wt sums =", sums)
    if shapes != expected_shapes or not all(onp.isfinite(value) for value in sums):
        raise RuntimeError("smoke-test shape or finiteness check failed")
    print("PASS: draw-weight smoke test completed with finite weights")


if __name__ == "__main__":
    config.update("jax_enable_x64", True)
    if "--smoke" in sys.argv:
        _smoke()
    else:
        logger = setup_logging()
        args_list = onp.load("output/fit/fit_result_values.npy")
        logger.info("计算数据权重 (mode=pass)...")
        run_weight(args_list, mode="pass")
        logger.info("计算 truth MC 权重 (mode=truth)...")
        run_weight(args_list, mode="truth")
        logger.info("权重计算完成，结果已保存至 output/draw/")
