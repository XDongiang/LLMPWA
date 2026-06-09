from agent.config_accessor import ConfigAccessor


def make_initial_args_handler(cfg: ConfigAccessor):

    try:
        path = cfg.read_input("path")
    except:
        print("fail to get free_params.toml path from imput, fail back to default")
        path = "run/free_params.toml"

    code = f"""
def make_initial_args():
    _data = toml.load("{path}")["data"]
    _data.sort(key=lambda x: x["arg_index"])

    args = [d["value"] for d in _data]
    ranges = [d.get("range", [-onp.inf, onp.inf]) for d in _data]
    errors = [d["error"] for d in _data]

    return onp.array(args), onp.array(ranges), onp.array(errors)
"""

    return code