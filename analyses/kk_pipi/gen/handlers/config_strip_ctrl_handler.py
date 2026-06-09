from agent.config_accessor import ConfigAccessor
from agent.config_parser import parse_config


def config_strip_ctrl_handler(cfg: ConfigAccessor):

    try:
        resonances_config = cfg.read_input("resonances_config")
    except:
        print("unable to read input, fall back to default")
        resonances_config = cfg.resolver.resolve("ref.resonances_config")





    return {"test":True}
