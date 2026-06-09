from agent.config_accessor import ConfigAccessor


def split_args_handler(cfg: ConfigAccessor):
    free_params = cfg.resolver.resolve(cfg.read_input("free_params"))["data"]

    # Build ordered index lists: for each channel, collect (local_arg_index -> total_arg_index)
    kk_slots: list[tuple[int, int]] = []    # (arg_index_kk, arg_index)
    pipi_slots: list[tuple[int, int]] = []  # (arg_index_pipi, arg_index)

    for p in free_params:
        total_idx = p["arg_index"]
        if p.get("arg_index_kk") is not None:
            kk_slots.append((p["arg_index_kk"], total_idx))
        if p.get("arg_index_pipi") is not None:
            pipi_slots.append((p["arg_index_pipi"], total_idx))

    # Sort by local channel index so the resulting sub-array preserves original order
    kk_slots.sort(key=lambda x: x[0])
    pipi_slots.sort(key=lambda x: x[0])

    kk_indices = [total_idx for _, total_idx in kk_slots]
    pipi_indices = [total_idx for _, total_idx in pipi_slots]

    code = f"""\
_KK_INDICES = np.array({kk_indices})
_PIPI_INDICES = np.array({pipi_indices})

def split_args(total_args):
    args_kk = total_args[_KK_INDICES]
    args_pipi = total_args[_PIPI_INDICES]
    return args_kk, args_pipi
"""

    return code
