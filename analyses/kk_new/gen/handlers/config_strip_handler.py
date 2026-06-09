from agent.config_accessor import ConfigAccessor
from agent.config_parser import parse_config


def config_strip_handler(cfg: ConfigAccessor):
    try:
        resonances_config_path = cfg.read_input("resonances_config")
        resonances_config = cfg.resolver.resolve(resonances_config_path)
    except:
        print("unable to read input, fall back to default")
        resonances_config = cfg.resolver.resolve("ref.resonances_config")

    stripped_config, free_params, _ = parse_config(resonances_config)

    free_params_range = []

    for i in free_params:
        if i.get("range") != None:
            free_params_range.append({"arg_index":i["arg_index"],"range":i["range"]})

    # collect all_sbc and all_amp (mirrors generator_base.get_all_resonance_data)
    sbc_list = []
    amp_list = []
    # amp -> [{sbc, propagator_type}, ...]
    amp_sbc_map: dict[str, list[dict]] = {}

    for resonance in resonances_config.get("resonances", {}).values():
        for prop in resonance.get("propagators", {}).values():
            if "Sbc" in prop:
                sbc_list.append(prop["Sbc"])

        if resonance.get("kind") == "shared_state":
            # shared propagators contribute sbc/propagator_type for all sub-amplitudes
            shared_sbc_entries = [
                {"sbc": prop["Sbc"], "propagator_type": prop["propagator_type"]}
                for prop in resonance.get("propagators", {}).values()
                if "Sbc" in prop and "propagator_type" in prop
            ]
            for amp_entry in resonance.get("amplitudes", []):
                amp_name = amp_entry.get("AMP")
                if amp_name:
                    amp_list.append(amp_name)
                    entries = list(shared_sbc_entries)
                    if "B_Sbc" in amp_entry:
                        sbc_list.append(amp_entry["B_Sbc"])
                        # B_propagator propagator_type from shared propagators
                        b_prop_type = resonance.get("propagators", {}).get(
                            "B_propagator", {}
                        ).get("propagator_type")
                        entries.append({
                            "sbc": amp_entry["B_Sbc"],
                            "propagator_type": b_prop_type,
                        })
                    amp_sbc_map.setdefault(amp_name, [])
                    for e in entries:
                        if e not in amp_sbc_map[amp_name]:
                            amp_sbc_map[amp_name].append(e)
        else:
            amp_name = resonance.get("Amplitude", {}).get("AMP")
            if amp_name:
                amp_list.append(amp_name)
                entries = [
                    {"sbc": prop["Sbc"], "propagator_type": prop["propagator_type"]}
                    for prop in resonance.get("propagators", {}).values()
                    if "Sbc" in prop and "propagator_type" in prop
                ]
                amp_sbc_map.setdefault(amp_name, [])
                for e in entries:
                    if e not in amp_sbc_map[amp_name]:
                        amp_sbc_map[amp_name].append(e)

    all_sbc = list(dict.fromkeys(sbc_list))
    all_amp = list(dict.fromkeys(amp_list))
    # {amp: [{sbc, propagator_type}, ...]} — deduped per amp
    all_sbc_amp = {amp: amp_sbc_map.get(amp, []) for amp in all_amp}

    return {
        "free_params": free_params,   # → 写入 run/free_params.toml
        "free_params_range":  free_params_range,
        "stripped_config": stripped_config,             # → 写入 gen/fragments/stripped_config.toml
        "all_sbc": all_sbc,                             # → direct，存入 manifest
        "all_amp": all_amp,                             # → direct，存入 manifest
        "all_sbc_amp": all_sbc_amp,                     # → {amp: [{sbc, propagator_type}, ...]}
    }

