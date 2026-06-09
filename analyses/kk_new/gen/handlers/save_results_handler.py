from agent.config_accessor import ConfigAccessor


def save_results_handler(cfg: ConfigAccessor):
    try:
        input_path = cfg.read_input("input_path")
        output_path = cfg.read_input("output_path")
    except:
        print("fail to get free_params.toml path from imput, fail back to default")
        input_path = "run/free_params.toml"
        output_path = "output/free_params_fitted.toml"

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
