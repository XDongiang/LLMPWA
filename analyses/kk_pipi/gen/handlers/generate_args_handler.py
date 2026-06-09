from agent.config_accessor import ConfigAccessor


def generate_args_handler(cfg: ConfigAccessor):
    param_kk = cfg.resolver.resolve(cfg.read_input("param_kk"))["data"]
    param_pipi = cfg.resolver.resolve(cfg.read_input("param_pipi"))["data"]
    ctrl = cfg.resolver.resolve(cfg.read_input("ctrl"))

    resonances_binding = ctrl.get("binding", {}).get("resonances", {})

    # Build lookup: (res_name, param_name) -> {usevalue, diff}
    # "param_name" is the path after "resonances.<res_name>.", e.g.
    # "propagators.B_propagator.mass"
    shared_rules: dict = {}
    for res_name, res_cfg in resonances_binding.items():
        for rule in res_cfg.get("param", []):
            shared_rules[(res_name, rule["name"])] = {
                "usevalue": rule["usevalue"],
                "diff": rule.get("diff", None),
            }

    def _res_and_param(p: dict):
        parts = p["path"].split(".")
        # path: "resonances.<res_name>.<rest...>"
        return parts[1], ".".join(parts[2:])

    # Index each channel's params by (res_name, param_name)
    kk_idx = {_res_and_param(p): p for p in param_kk}

    total_params: list = []
    # (channel, res_name, param_name) -> total arg_index
    slot_map: dict = {}

    def _add_slot(channel: str, p: dict, res_name: str, param_name: str) -> int:
        idx = len(total_params)
        entry = {
            "path": p["path"],
            "value": p["value"],
            "range": p.get("range"),
            "error": p.get("error"),
            "arg_index": idx,
            "arg_index_kk": p["arg_index"] if channel == "kk" else None,
            "arg_index_pipi": p["arg_index"] if channel == "pipi" else None,
        }
        total_params.append(entry)
        slot_map[(channel, res_name, param_name)] = idx
        return idx

    # Pass 1: kk params — skip those bound to use pipi's value
    for p in param_kk:
        res_name, param_name = _res_and_param(p)
        rule = shared_rules.get((res_name, param_name))
        if rule and rule["usevalue"] == "pipi":
            slot_map[("kk_deferred", res_name, param_name)] = True
            continue
        _add_slot("kk", p, res_name, param_name)

    # Pass 2: pipi params
    for p in param_pipi:
        res_name, param_name = _res_and_param(p)
        rule = shared_rules.get((res_name, param_name))

        if rule and rule["usevalue"] == "kk":
            # Share the kk slot — annotate arg_index_pipi
            kk_slot = slot_map.get(("kk", res_name, param_name))
            if kk_slot is not None:
                total_params[kk_slot]["arg_index_pipi"] = p["arg_index"]
                slot_map[("pipi", res_name, param_name)] = kk_slot
            continue

        slot_idx = _add_slot("pipi", p, res_name, param_name)

        # If kk deferred to this pipi slot, back-fill arg_index_kk
        if ("kk_deferred", res_name, param_name) in slot_map:
            kk_p = kk_idx.get((res_name, param_name))
            if kk_p is not None:
                total_params[slot_idx]["arg_index_kk"] = kk_p["arg_index"]
            slot_map[("kk", res_name, param_name)] = slot_idx

    return {"free_params": total_params}
