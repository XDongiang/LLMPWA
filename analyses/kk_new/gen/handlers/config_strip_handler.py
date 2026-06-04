import toml
from agent.config_accessor import ConfigAccessor
from agent.resolver import Resolver

def config_strip_handler(cfg: ConfigAccessor):
    
    resonances_config = cfg.resolver.resolve("ref.resonances_config")

    print("test")
    print(resonances_config)




    return {
        "free_params": {...},        # → 写入 run/free_params.toml
        "stripped_config": {...},    # → 写入 gen/fragments/stripped_config.toml
        "all_sbc": ["xxx", "xxx"],   # → direct，存入 manifest
        "all_amp": ["xxx", "xxx"],   # → direct，存入 manifest
    }
