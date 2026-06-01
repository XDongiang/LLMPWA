"""
config_parser.py — pre-process resonances_config.toml before LLM stages.

Produces:
  stripped_config  — same structure as the raw config, but free-parameter
                     fields are replaced with {arg_index = i} (fixed params
                     keep their full dict).
  free_params      — ordered list of {path, value, range, error, arg_index}
                     for every free parameter, in traversal order.
  args_list        — [value, ...] in the same order as free_params
"""

import copy
from typing import Any


def _is_param_dict(v: Any) -> bool:
    """True if v looks like a parameter dict (has 'fixed' key)."""
    return isinstance(v, dict) and "fixed" in v


def _process_section(section: dict, free_params: list, path_prefix: str):
    """Collect free params from a flat section dict (propagator or Amplitude)."""
    for key in list(section.keys()):
        v = section[key]
        if _is_param_dict(v) and not v.get("fixed", True):
            free_params.append({
                "path": f"{path_prefix}.{key}",
                "value": v.get("value", None),
                "range": v.get("range", None),
                "error": v.get("error", None),
            })


def _set_by_path(root: dict, path: str, value: Any) -> None:
    """Set root[k1][k2]...[kN] = value, where path = 'k1.k2...kN'."""
    keys = path.split(".")
    d = root
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value


def parse_config(config: dict) -> tuple:
    """
    Parameters
    ----------
    config : dict
        Raw TOML config loaded by toml.load().

    Returns
    -------
    stripped_config : dict
        Deep copy of config where every free-parameter field is replaced
        with ``{"arg_index": i}`` (fixed params keep their full dict).
    free_params : list of dict
        Ordered list of {path, value, range, error, arg_index}.
    args_list : list of float
        Plain list of values in the same order as free_params.
    """
    stripped = copy.deepcopy(config)
    free_params: list = []

    resonances = stripped.get("resonances", {})
    for res_name, res in resonances.items():
        kind = res.get("kind", "normal")

        for prop_name, prop in res.get("propagators", {}).items():
            prefix = f"resonances.{res_name}.propagators.{prop_name}"
            _process_section(prop, free_params, prefix)

        if kind == "shared_state":
            sap = res.get("shared_amplitude_parameters", {})
            prefix = f"resonances.{res_name}.shared_amplitude_parameters"
            _process_section(sap, free_params, prefix)
        else:
            amp = res.get("Amplitude", {})
            prefix = f"resonances.{res_name}.Amplitude"
            _process_section(amp, free_params, prefix)

    args_list = [p["value"] for p in free_params]

    for i, p in enumerate(free_params):
        p["arg_index"] = i
        _set_by_path(stripped, p["path"], {"arg_index": i})

    return stripped, free_params, args_list
