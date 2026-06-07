from agent.config_accessor import ConfigAccessor
from agent.config_parser import parse_config


def config_strip_handler(cfg: ConfigAccessor):
    resonances_config = cfg.resolver.resolve("ref.resonances_config")

    stripped_config, free_params, _ = parse_config(resonances_config)

    free_params_range = []

    for i in free_params:
        if i.get("range") != None:
            free_params_range.append({"arg_index":i["arg_index"],"range":i["range"]})

    # collect all_sbc and all_amp (mirrors generator_base.get_all_resonance_data)
    sbc_list = []
    amp_list = []
    for resonance in resonances_config.get("resonances", {}).values():
        for prop in resonance.get("propagators", {}).values():
            if "Sbc" in prop:
                sbc_list.append(prop["Sbc"])
        if resonance.get("kind") == "shared_state":
            for amp_entry in resonance.get("amplitudes", []):
                if "AMP" in amp_entry:
                    amp_list.append(amp_entry["AMP"])
                if "B_Sbc" in amp_entry:
                    sbc_list.append(amp_entry["B_Sbc"])
        else:
            if "AMP" in resonance.get("Amplitude", {}):
                amp_list.append(resonance["Amplitude"]["AMP"])

    all_sbc = list(dict.fromkeys(sbc_list))
    all_amp = list(dict.fromkeys(amp_list))

    return {
        "free_params": free_params,   # → 写入 run/free_params.toml
        "free_params_range":  free_params_range,
        "stripped_config": stripped_config,             # → 写入 gen/fragments/stripped_config.toml
        "all_sbc": all_sbc,                             # → direct，存入 manifest
        "all_amp": all_amp,                             # → direct，存入 manifest
    }
