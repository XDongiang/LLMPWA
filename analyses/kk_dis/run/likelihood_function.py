from base_functions import (
    BW,
    dplex_dabs,
    dplex_dconstruct,
    dplex_deinsum,
    dplex_deinsum_ord,
    flatte1270,
    flatte980,
    jvp,
    grad,
    make_initial_args,
    np,
    onp,
)


GAUSSIAN_CONSTRAINTS = (
    (0, 0.98, 10.0),
    (5, 1.704, 1.0),
    (6, 0.123, 1.0),
    (11, 1.2755, 1.0),
    (12, 0.1867, 1.0),
    (23, 1.517, 1.0),
    (24, 0.086, 1.0),
    (35, 2.157, 1.0),
    (36, 0.152, 1.0),
    (47, 2.345, 0.01),
    (48, 0.322, 1.0),
    (59, 2.47, 0.007),
    (60, 0.075, 0.011),
)


def extract_parameters(args):
    args = np.asarray(args)
    return {
        "f0_980": {
            "mass": args[0], "g_pipi": args[1], "rg": args[2],
            "const": np.asarray([0.1, args[3]]),
            "theta": np.asarray([0.1, args[4]]),
        },
        "f0_bw": {
            "mass": args[np.asarray([5, 59])],
            "width": args[np.asarray([6, 60])],
            "const": np.stack((args[np.asarray([7, 8])], args[np.asarray([61, 62])])),
            "theta": np.stack((args[np.asarray([9, 10])], args[np.asarray([63, 64])])),
        },
        "f2_1270": {
            "mass": args[11], "width": args[12],
            "const": args[13:18], "theta": args[18:23],
        },
        "f2_bw": {
            "mass": args[np.asarray([23, 35, 47])],
            "width": args[np.asarray([24, 36, 48])],
            "const": np.stack((args[25:30], args[37:42], args[49:54])),
            "theta": np.stack((args[30:35], args[42:47], args[54:59])),
        },
    }


def _propagator_product(phi_mass, phi_width, phi_s, f_propagators):
    phi_prop = BW(phi_mass, phi_width, phi_s)
    n_states = f_propagators.shape[1]
    phi_props = np.broadcast_to(phi_prop[:, None, :], (2, n_states, phi_s.shape[0]))
    return dplex_deinsum("le,le->le", phi_props, f_propagators)


def _coupled_components(tensor, const, theta, propagators):
    couplings = dplex_dconstruct(const, theta)
    weighted = dplex_deinsum_ord("iek,li->lek", tensor, couplings)
    return dplex_deinsum("lek,le->lek", weighted, propagators)


def _category_components(params, phi_s, f_s, f0_tensor, f2_tensor):
    phi_mass = 1.02
    phi_width = 0.004

    p = params["f0_980"]
    f_prop = flatte980(p["mass"], p["g_pipi"], p["rg"], f_s)[:, None, :]
    prop = _propagator_product(phi_mass, phi_width, phi_s, f_prop)
    f0_980 = _coupled_components(f0_tensor, p["const"][None, :], p["theta"][None, :], prop)

    p = params["f0_bw"]
    f_prop = np.stack([BW(p["mass"][i], p["width"][i], f_s) for i in range(2)], axis=1)
    prop = _propagator_product(phi_mass, phi_width, phi_s, f_prop)
    f0_bw = _coupled_components(f0_tensor, p["const"], p["theta"], prop)

    p = params["f2_1270"]
    f_prop = flatte1270(p["mass"], p["width"], f_s)[:, None, :]
    prop = _propagator_product(phi_mass, phi_width, phi_s, f_prop)
    f2_1270 = _coupled_components(f2_tensor, p["const"][None, :], p["theta"][None, :], prop)

    p = params["f2_bw"]
    f_prop = np.stack([BW(p["mass"][i], p["width"][i], f_s) for i in range(3)], axis=1)
    prop = _propagator_product(phi_mass, phi_width, phi_s, f_prop)
    f2_bw = _coupled_components(f2_tensor, p["const"], p["theta"], prop)
    return f0_980, f0_bw, f2_1270, f2_bw


def _total_amplitude(components):
    return sum(np.sum(component, axis=1) for component in components)


def _components_for_prefix(args, data, prefix):
    params = extract_parameters(args)
    return _category_components(
        params,
        np.asarray(data[prefix + "phi_kk"]),
        np.asarray(data[prefix + "f_kk"]),
        np.asarray(data[prefix + "phif0_kk"]),
        np.asarray(data[prefix + "phif2_kk"]),
    )


def event_intensity(args, data, prefix):
    total = _total_amplitude(_components_for_prefix(args, data, prefix))
    return np.sum(dplex_dabs(total), axis=-1)


def fraction_values(args, data):
    components = _components_for_prefix(args, data, "truth_")
    denominator = np.sum(dplex_dabs(_total_amplitude(components)))
    denominator = np.maximum(denominator, np.finfo(np.asarray(denominator).dtype).tiny)
    fractions = np.stack([np.sum(dplex_dabs(component)) / denominator for component in components])
    return fractions


def gaussian_penalty(args):
    args = np.asarray(args)
    return sum((args[index] - center) ** 2 / (2.0 * sigma ** 2)
               for index, center, sigma in GAUSSIAN_CONSTRAINTS)


def data_likelihood_kk(args, data_or_jax_data):
    intensity = event_intensity(args, data_or_jax_data, "data_")
    tiny = np.finfo(intensity.dtype).tiny
    fractions = fraction_values(args, data_or_jax_data)
    fraction_penalty = 1000.0 * (np.sum(fractions) - 1.1) ** 2
    return -np.sum(np.log(np.maximum(intensity, tiny))) + fraction_penalty + gaussian_penalty(args)


def mc_likelihood_kk(args, data_or_jax_data):
    intensity = event_intensity(args, data_or_jax_data, "mc_")
    return np.mean(intensity)


def combined_likelihood(args, data_or_jax_data):
    data_size = data_or_jax_data["data_phi_kk"].shape[0]
    mc_norm = mc_likelihood_kk(args, data_or_jax_data)
    tiny = np.finfo(mc_norm.dtype).tiny
    return data_likelihood_kk(args, data_or_jax_data) + data_size * np.log(np.maximum(mc_norm, tiny))


def make_distributed_likelihood(data_size):
    def likelihood(args, data_or_jax_data):
        mc_norm = mc_likelihood_kk(args, data_or_jax_data)
        tiny = np.finfo(mc_norm.dtype).tiny
        return data_likelihood_kk(args, data_or_jax_data) + data_size * np.log(np.maximum(mc_norm, tiny))
    return likelihood


def hvp_combined_likelihood(args, vector, data_or_jax_data):
    gradient = grad(combined_likelihood, argnums=0)
    return jvp(lambda x: gradient(x, data_or_jax_data), (args,), (vector,))[1]


def _load_smoke_subset(n_data=128, n_mc=512, n_truth=256):
    data = {}
    for var in ("phi_kk", "f_kk", "b123_kk", "b124_kk"):
        real = onp.load("data/real_data/" + var + ".npy", mmap_mode="r")
        mc = onp.load("data/mc_truth/" + var + ".npy", mmap_mode="r")
        data["data_" + var] = onp.asarray(real[:n_data])
        data["mc_" + var] = onp.asarray(mc[:n_mc])
        data["truth_" + var] = onp.asarray(mc[:n_truth])
    for var in ("phif0_kk", "phif2_kk"):
        real = onp.load("data/real_data/" + var + ".npy", mmap_mode="r")
        mc = onp.load("data/mc_truth/" + var + ".npy", mmap_mode="r")
        data["data_" + var] = onp.asarray(real[:, :n_data, :])
        data["mc_" + var] = onp.asarray(mc[:, :n_mc, :])
        data["truth_" + var] = onp.asarray(mc[:, :n_truth, :])
        regular = 1.0 / onp.mean(onp.sqrt(onp.sum(data["mc_" + var] ** 2, axis=-1)), axis=1)
        for prefix in ("data_", "mc_", "truth_"):
            data[prefix + var] = onp.einsum("i,iek->iek", regular, data[prefix + var])
    return data


if __name__ == "__main__":
    initial_args, _, _ = make_initial_args()
    smoke_data = _load_smoke_subset()
    data_value = data_likelihood_kk(initial_args, smoke_data)
    mc_value = mc_likelihood_kk(initial_args, smoke_data)
    combined_value = combined_likelihood(initial_args, smoke_data)
    print("n_args =", initial_args.size)
    print("data_size =", smoke_data["data_phi_kk"].shape[0])
    print("data_likelihood =", float(data_value))
    print("mc_likelihood =", float(mc_value))
    print("combined_likelihood =", float(combined_value))
