"""
config_parser.py — pre-process resonances_config.toml before LLM stages.

Produces:
  stripped_config  — same structure as the raw config, but free-parameter
                     `value` fields are removed (fixed params keep their values).
  free_params      — ordered list of (resonance_path, param_name, value, range)
  args_list        — [value, ...] in the same order as free_params
"""

import copy
from typing import Any


# Parameter field names that may carry value/fixed/range
_PARAM_FIELDS = ("mass", "width", "g_kk", "rg")
_PARAM_PREFIX_FIELDS = ("const", "theta")


def _is_param_dict(v: Any) -> bool:
    """True if v looks like a parameter dict (has 'fixed' key)."""
    return isinstance(v, dict) and "fixed" in v


def _iter_param_keys(section: dict):
    """Yield keys in a section dict that are parameter dicts."""
    for k, v in section.items():
        if _is_param_dict(v):
            yield k


def _strip_param(param: dict, free_params: list, path: str):
    """
    Mutate param in-place: if fixed=false, pop 'value' and record it in free_params.
    Returns the (possibly mutated) param dict.
    """
    if not param.get("fixed", True):
        value = param.pop("value", None)
        range_ = param.get("range", None)
        free_params.append({
            "path": path,
            "value": value,
            "range": range_,
        })
    return param


def _process_section(section: dict, free_params: list, path_prefix: str):
    """Strip free-param values from a flat section dict (propagator or Amplitude)."""
    for key in list(section.keys()):
        v = section[key]
        if _is_param_dict(v):
            _strip_param(v, free_params, f"{path_prefix}.{key}")


def parse_config(config: dict) -> tuple:
    """
    Parameters
    ----------
    config : dict
        Raw TOML config loaded by toml.load().

    Returns
    -------
    stripped_config : dict
        Deep copy of config with free-parameter `value` fields removed.
    free_params : list of dict
        Ordered list of {"path": str, "value": float, "range": list|None}
        for every free parameter, in traversal order.
    args_list : list of float
        Plain list of values in the same order as free_params.
    """
    stripped = copy.deepcopy(config)
    free_params: list = []

    resonances = stripped.get("resonances", {})
    for res_name, res in resonances.items():
        kind = res.get("kind", "normal")

        # --- propagators ---
        for prop_name, prop in res.get("propagators", {}).items():
            prefix = f"resonances.{res_name}.propagators.{prop_name}"
            _process_section(prop, free_params, prefix)

        if kind == "shared_state":
            # shared_amplitude_parameters
            sap = res.get("shared_amplitude_parameters", {})
            prefix = f"resonances.{res_name}.shared_amplitude_parameters"
            _process_section(sap, free_params, prefix)
        else:
            # Amplitude section (const*, theta*, ...)
            amp = res.get("Amplitude", {})
            prefix = f"resonances.{res_name}.Amplitude"
            _process_section(amp, free_params, prefix)

    args_list = [p["value"] for p in free_params]
    return stripped, free_params, args_list
