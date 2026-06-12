from agent.config_accessor import ConfigAccessor



def assemble_final_code_handler(cfg: ConfigAccessor):

    input = cfg.read_input("list")

    code = ""

    for ref in input:

        res = cfg.resolver.resolve(ref)

        if type(res) == str :
            code = code + res +"\n\n"
        elif type(res) == list:
            for i in res :
                if type(i) == str :
                    code = code + i +"\n\n"


    return code