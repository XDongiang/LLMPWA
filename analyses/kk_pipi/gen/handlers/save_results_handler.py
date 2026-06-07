from agent.config_accessor import ConfigAccessor


def save_results_handler(cfg: ConfigAccessor):

    input_path = "run/free_params.toml"
    output_path = "run/free_params.toml"

    code = f"""
def save_result(args, errors, path="{output_path}"):
    _data = toml.load("{input_path}")["data"]
    _data.sort(key=lambda x: x["arg_index"])

    for i, d in enumerate(_data):
        d["value"] = float(args[i])
        d["error"] = float(errors[i])

    with open(path, "w") as f:
        toml.dump({{"data": _data}}, f)
"""

    return code
