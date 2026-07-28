from base_functions import (
    np,
    grad,
    jvp,
    BW,
    flatte980,
    flatte1270,
    dplex_dabs,
    dplex_dconstruct,
    dplex_deinsum,
    dplex_deinsum_ord,
    make_initial_args,
    load_data,
    normalize_data,
)


PHI_MASS = 1.02
PHI_WIDTH = 0.004
FRACTION_TARGET = 1.1
FRACTION_PENALTY_SCALE = 1000.0
LOG_EPSILON = 1.0e-30

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
            "mass": args[0],
            "g_pipi": args[1],
            "rg": args[2],
            "const": np.asarray([0.1, args[3]]),
            "theta": np.asarray([0.1, args[4]]),
        },
        "f0_bw": {
            "mass": np.asarray([args[5], args[59]]),
            "width": np.asarray([args[6], args[60]]),
            "const": np.asarray([[args[7], args[8]], [args[61], args[62]]]),
            "theta": np.asarray([[args[9], args[10]], [args[63], args[64]]]),
        },
        "f2_1270": {
            "mass": args[11],
            "width": args[12],
            "const": args[13:18],
            "theta": args[18:23],
        },
        "f2_bw": {
            "mass": np.asarray([args[23], args[35], args[47]]),
            "width": np.asarray([args[24], args[36], args[48]]),
            "const": np.stack((args[25:30], args[37:42], args[49:54])),
            "theta": np.stack((args[30:35], args[42:47], args[54:59])),
        },
    }


def _multiply_event_propagators(left, right):
    if right.ndim == 2:
        return dplex_deinsum("e,e->e", left, right)
    return dplex_deinsum("e,le->le", left, right)


def _single_component(tensor, propagator, const, theta):
    coupling = dplex_dconstruct(const, theta)
    weighted = dplex_deinsum_ord("iek,i->ek", tensor, coupling)
    return dplex_deinsum("e,ek->ek", propagator, weighted)[:, None, ...]


def _multiple_components(tensor, propagators, const, theta):
    coupling = dplex_dconstruct(const, theta)
    weighted = dplex_deinsum_ord("iek,li->lek", tensor, coupling)
    return dplex_deinsum("le,lek->lek", propagators, weighted)


def component_BW_flatte980(phi_sbc, f_sbc, tensor, parameters):
    phi_prop = BW(PHI_MASS, PHI_WIDTH, phi_sbc)
    f_prop = flatte980(
        parameters["mass"], parameters["g_pipi"], parameters["rg"], f_sbc
    )
    propagator = _multiply_event_propagators(phi_prop, f_prop)
    return _single_component(
        tensor, propagator, parameters["const"], parameters["theta"]
    )


def component_BW_flatte1270(phi_sbc, f_sbc, tensor, parameters):
    phi_prop = BW(PHI_MASS, PHI_WIDTH, phi_sbc)
    f_prop = flatte1270(parameters["mass"], parameters["width"], f_sbc)
    propagator = _multiply_event_propagators(phi_prop, f_prop)
    return _single_component(
        tensor, propagator, parameters["const"], parameters["theta"]
    )


def component_BW_BW(phi_sbc, f_sbc, tensor, parameters):
    phi_prop = BW(PHI_MASS, PHI_WIDTH, phi_sbc)
    masses = parameters["mass"]
    widths = parameters["width"]
    f_prop = np.stack(
        [BW(masses[index], widths[index], f_sbc) for index in range(masses.shape[0])],
        axis=1,
    )
    propagators = _multiply_event_propagators(phi_prop, f_prop)
    return _multiple_components(
        tensor, propagators, parameters["const"], parameters["theta"]
    )


def calculate_BW_flatte980(phi_sbc, f_sbc, tensor, parameters):
    return np.sum(component_BW_flatte980(phi_sbc, f_sbc, tensor, parameters), axis=1)


def calculate_BW_flatte1270(phi_sbc, f_sbc, tensor, parameters):
    return np.sum(component_BW_flatte1270(phi_sbc, f_sbc, tensor, parameters), axis=1)


def calculate_BW_BW(phi_sbc, f_sbc, tensor, parameters):
    return np.sum(component_BW_BW(phi_sbc, f_sbc, tensor, parameters), axis=1)


def _class_components(args, data, prefix):
    parameters = extract_parameters(args)
    phi_sbc = data[prefix + "phi_kk"]
    f_sbc = data[prefix + "f_kk"]
    f0_tensor = data[prefix + "phif0_kk"]
    f2_tensor = data[prefix + "phif2_kk"]
    return (
        component_BW_flatte980(phi_sbc, f_sbc, f0_tensor, parameters["f0_980"]),
        component_BW_BW(phi_sbc, f_sbc, f0_tensor, parameters["f0_bw"]),
        component_BW_flatte1270(phi_sbc, f_sbc, f2_tensor, parameters["f2_1270"]),
        component_BW_BW(phi_sbc, f_sbc, f2_tensor, parameters["f2_bw"]),
    )


def _total_amplitude(args, data, prefix):
    components = _class_components(args, data, prefix)
    return sum(np.sum(component, axis=1) for component in components)


def _event_intensity(amplitude):
    return np.sum(dplex_dabs(amplitude), axis=-1)


def fraction_total(args, data):
    components = _class_components(args, data, "truth_")
    total_amplitude = sum(np.sum(component, axis=1) for component in components)
    denominator = np.sum(dplex_dabs(total_amplitude))
    numerator = sum(np.sum(dplex_dabs(component)) for component in components)
    return numerator / np.maximum(denominator, LOG_EPSILON)


def gaussian_constraint(args):
    return sum(
        (args[index] - center) ** 2 / (2.0 * sigma ** 2)
        for index, center, sigma in GAUSSIAN_CONSTRAINTS
    )


def data_step_function(args, data):
    f_total = fraction_total(args, data)
    fraction_penalty = FRACTION_PENALTY_SCALE * (f_total - FRACTION_TARGET) ** 2
    return fraction_penalty + gaussian_constraint(args)


def data_likelihood_kk(args, data_or_jax_data):
    amplitude = _total_amplitude(args, data_or_jax_data, "data_")
    intensity = _event_intensity(amplitude)
    nll = -np.sum(np.log(np.maximum(intensity, LOG_EPSILON)))
    return nll + data_step_function(args, data_or_jax_data)


def mc_likelihood_kk(args, data_or_jax_data):
    amplitude = _total_amplitude(args, data_or_jax_data, "mc_")
    return np.mean(_event_intensity(amplitude))


def combined_likelihood(args, data_or_jax_data):
    data_size = data_or_jax_data["data_phi_kk"].shape[0]
    normalization = mc_likelihood_kk(args, data_or_jax_data)
    return data_likelihood_kk(args, data_or_jax_data) + data_size * np.log(
        np.maximum(normalization, LOG_EPSILON)
    )


def make_distributed_likelihood(data_size):
    def distributed_likelihood(args, data_or_jax_data):
        normalization = mc_likelihood_kk(args, data_or_jax_data)
        return data_likelihood_kk(args, data_or_jax_data) + data_size * np.log(
            np.maximum(normalization, LOG_EPSILON)
        )

    return distributed_likelihood


def hvp_combined_likelihood(args, tangent, data_or_jax_data):
    return jvp(
        lambda values: grad(combined_likelihood)(values, data_or_jax_data),
        (args,),
        (tangent,),
    )[1]


if __name__ == "__main__":
    args, _, _ = make_initial_args()
    data = normalize_data(load_data())
    values = {
        "data_likelihood": data_likelihood_kk(args, data),
        "mc_likelihood": mc_likelihood_kk(args, data),
        "combined_likelihood": combined_likelihood(args, data),
        "fraction_total": fraction_total(args, data),
    }
    for name, value in values.items():
        print(name + " =", float(value))
