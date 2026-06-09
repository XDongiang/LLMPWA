import sys
from agent.config_accessor import ConfigAccessor


def _get_nested(d: dict, dotpath: str):
    """Navigate a dict by dot-separated key path. Returns (value, True) or (None, False)."""
    keys = dotpath.split(".")
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None, False
        cur = cur[k]
    return cur, True


def combine_check_handler(cfg: ConfigAccessor):
    ctrl = cfg.resolver.resolve("ref.resonances_config_ctrl")

    binding_ref = ctrl["binding"]["ref"]   # {"kk": "ref.resonances_config_kk", "pipi": "ref.resonances_config_pipi"}
    binding_resonances = ctrl["binding"].get("resonances", {})

    # resolve each config referenced in binding.ref
    configs = {alias: cfg.resolver.resolve(ref_path) for alias, ref_path in binding_ref.items()}

    errors = []
    for resonance_name, resonance_binding in binding_resonances.items():
        for param_entry in resonance_binding.get("param", []):
            param_name = param_entry["name"]          # e.g. "propagators.B_propagator.mass"
            usevalue   = param_entry["usevalue"]      # e.g. "kk" or "pipi"

            # check the param exists in the usevalue config
            source_cfg = configs.get(usevalue)
            if source_cfg is None:
                errors.append(f"[{resonance_name}] param '{param_name}': alias '{usevalue}' not found in binding.ref")
                continue

            source_resonances = source_cfg.get("resonances", {})
            if resonance_name not in source_resonances:
                errors.append(f"[{resonance_name}] not found in '{usevalue}' config")
                continue

            # param must exist in all configs
            for alias, check_cfg in configs.items():
                check_resonances = check_cfg.get("resonances", {})
                if resonance_name not in check_resonances:
                    errors.append(f"[{resonance_name}] not found in '{alias}' config")
                    continue
                _, found = _get_nested(check_resonances[resonance_name], param_name)
                if not found:
                    errors.append(f"[{resonance_name}] param '{param_name}' not found in '{alias}' config")

    if errors:
        print("combine_check_handler: shared parameter check failed:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    return {"checked": True}