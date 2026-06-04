# LLMPWA 代码生成框架设计
## 总体思路
用 llm_config.toml 完全驱动流程，引擎（engine.py）是通用的，不包含任何物理分析逻辑。所有分析相关的逻辑全部在 gen/ 目录里，换一个分析只需要换一套 gen/。

目录结构
```
analyses/kk_new/
├── llm_config_fit.toml          # 流程配置，唯一入口
├── resonances_config.toml       # 物理输入
├── gen/
│   ├── manifest.toml            # 运行时状态（引擎自动维护）
│   ├── handlers/                # python stage 的处理函数
│   │   ├── config_strip_handler.py
│   │   ├── make_initial_args_handler.py
│   │   ├── save_results_handler.py
│   │   └── assemble_final_code_handler.py
│   ├── prompts/                 # llm stage 的 prompt 文件
│   │   ├── classification_prompt.txt
│   │   ├── data_load_prompt.txt
│   │   └── ...
│   ├── templates/               # 发给 LLM 参考的代码模板（按 # SECTION: xxx 切割）
│   │   ├── fit.py
│   │   ├── shared.py
│   │   └── physics_functions.py
│   ├── fragments/               # 各 stage 输出的中间代码片段（引擎写入）
│   └── llm_logs/                # LLM 对话记录（引擎写入）
├── run/
│   ├── fit_script.py            # 最终生成脚本
│   └── free_params.toml         # 自由参数初始值（config_strip 输出）
└── data/ ...
```

## 核心模块划分

```
agent/
├── engine.py          # 新主引擎，通用，驱动 llm_config.toml
├── config_loader.py   # 解析 llm_config.toml，验证结构
├── resolver.py        # <<...>> 引用解析、manifest 路径穿透
├── validator.py       # 静态依赖检查 + 运行时字段校验
├── stage_runner.py    # 执行单个 stage（python/llm/foreach）
├── manifest.py        # manifest.toml 读写，原子写入
├── llm_client.py      # 现有 easytrans_client.py，改名
└── config_parser.py   # 保持不变，被 config_strip_handler.py 调用
```
### 引擎主流程（engine.py）
```
class Engine:
    def __init__(self, workdir: str, config_name: str = "llm_config_fit.toml"):
        self.workdir = Path(workdir).resolve()
        self.config = ConfigLoader.load(self.workdir / config_name)
        self.manifest = Manifest(self.workdir / "gen" / "manifest.toml")
        self.resolver = Resolver(self.workdir, self.manifest, self.config)
        self.validator = Validator(self.config)

    def run(self):
        # 1. 静态检查：在任何 stage 运行前
        self.validator.static_check()

        # 2. 按 toml 中定义的顺序逐 stage 执行
        for stage_name, stage_cfg in self.config.stages.items():
            runner = StageRunner(stage_name, stage_cfg, self)
            runner.execute()
```
### 引用解析（resolver.py）
这是整个框架的核心，负责把 <<path>> 解析为实际值。

解析路径语法：
```
语法	含义
<<ref.code_template.SECTION_NAME>>	从 template 文件取 # SECTION: xxx 片段
<<stages.config_strip.all_sbc>>	从 manifest 取 direct 类型字段
<<stages.config_strip.stripped_config>>	读取 file 类型字段，返回文件全文
<<stages.config_strip.stripped_config.resonances>>	读文件后取内部字段
<<stages.resonance_calculation[*].all>>	foreach stage 所有 key 的输出，按序拼接
<<stages.resonance_calculation[BW_BW].all>>	foreach stage 特定 key 的输出
```
#### 解析规则（resolve 函数）：

```
def resolve(dotpath: str) -> Any:
    # 1. 分段解析
    parts = parse_path(dotpath)  # 处理 [*] [key] 等特殊语法

    # 2. 从 manifest 或 config 根节点开始遍历
    node = lookup_root(parts[0])  # "ref" → config.ref, "stages" → manifest.stages

    for part in parts[1:]:
        if isinstance(node, FileRef):
            # 穿透：读文件，继续在文件内容里取剩余路径
            node = load_file(node.path)
            node = get_subkey(node, part)
        elif part == "[*]":
            # foreach 展开：返回所有子节点的列表
            node = list(node.values())
        elif part.startswith("[") and part.endswith("]"):
            key = part[1:-1]
            node = node[key]  # KeyError → 立刻抛，带完整路径
        else:
            node = node[part]  # KeyError → 立刻抛，带完整路径

        if node is None:
            raise ResolutionError(f"Path '{dotpath}' resolved to None at '{part}'")

    return node
```
关键约束： 只有 manifest 中显式声明 type=file 的字段才触发文件读取，文件内部的字段是普通数据，不做二次穿透。

### 静态检查器（validator.py）
在所有 stage 运行前执行，分两轮：

第一轮（纯静态，启动时）：

- 所有 output_type, kind, handler 字段的合法性
- 所有 {"type"="file", "path"=...} 引用的文件是否存在（prompts、handlers、templates）
- 所有 `<<stages.X.Y>>` 引用中，X 对应的 stage 是否在当前 config 中定义
- 所有 `<<stages.X.Y>>` 引用中，Y 字段是否在 stage X 的 output 声明里
- foreach 的 ref 来源 stage 是否在当前 stage 之前定义（DAG 拓扑检查）

- 如果 classification 在 resonance_calculation 之后定义，这里直接报错
- ConfigError: stage 'resonance_calculation' foreach depends on
- 'stages.classification.propagator_classification', but 'classification'
- is not defined before 'resonance_calculation' in llm_config.toml

第二轮（foreach 展开后，classification stage 完成后）：

- `<<stages.X.Y.field_in_file>>` 中文件内字段的存在性（此时文件已生成）
- foreach 的 [BW_BW] 等具体 key 是否在展开结果里

运行时校验（每个字段读取时）：

```
def read_field(stage, field):
    value = manifest.get(stage, field)
    if value is None:
        raise FieldError(
            f"Stage '{stage}' field '{field}' is None in manifest.
"
            f"  This usually means the stage didn't run or its output was empty.
"
            f"  Check gen/llm_logs/{stage}.md for the LLM response."
        )
    if isinstance(value, FileRef) and not value.path.exists():
        raise FieldError(
            f"Stage '{stage}' field '{field}' points to '{value.path}' which does not exist."
        )
    return value
```
### Stage 执行器（stage_runner.py）
处理三种 stage 类型：

#### python stage：

```
def execute_python(self):
    handler_fn = load_handler(self.cfg.handler)  # importlib 动态加载
    # handler 签名: fn(config, resolver) -> dict
    # config 是 ConfigAccessor，通过 read_config(cfg.free_params) 自动穿透 file 类型
    result = handler_fn(ConfigAccessor(self.cfg, self.resolver), self.resolver)
    self.write_outputs(result)
```

#### llm stage：

```
def execute_llm(self):
    prompt_template = read_file(self.cfg.prompt)
    prompt = self.resolver.render(prompt_template)  # 替换所有 <<...>>
    if self.cfg.human_prompt:
        human = self.resolver.render(self.cfg.human_prompt)
        prompt = prompt  + human

    cache_hash = compute_hash(prompt)  # prompt 本身就是 hash 输入，天然完备
    if self.manifest.is_cached(self.name, cache_hash):
        return self.manifest.load_cached(self.name)

    code = self.llm_client.call(prompt)
    self.write_outputs({"all": code})
    self.manifest.save_llm_log(self.name, prompt, code)
```

注意 hash 直接用渲染后的 prompt，不需要手动列 hash_inputs，引用内容变了 prompt 自然变，hash 自然失效。这比现有的手动 hash_inputs=[stripped_json, prompt_template] 更安全。

#### foreach stage：

```
def execute_foreach(self):
    keys = self.resolver.resolve(self.cfg.foreach.path)  # 拿到 dict 的 keys
    if not keys:
        raise ForeachError(f"foreach source '{self.cfg.foreach.path}' is empty")

    for key in keys:
        sub_runner = StageRunner(
            name=f"{self.name}.{key}",
            cfg=self.cfg,
            key=key,          # 注入 <<key>> 变量
            engine=self.engine,
        )
        sub_runner.execute_llm()
```
### manifest.py
原子写入，结构清晰：

```
class Manifest:
    def save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(toml.dumps(self._data))
        tmp.replace(self.path)  # 原子替换

    def is_cached(self, stage: str, hash: str) -> bool:
        node = self._get(stage)
        if not node:
            return False
        if node.get("prompt_hash") != hash:
            return False
        out = node.get("output")
        if isinstance(out, dict) and out.get("type") == "file":
            return (self.workdir / out["path"]).exists()
        return out is not None

    def write_stage(self, stage: str, outputs: dict, prompt_hash: str = ""):
        # outputs 对应 llm_config 里的 output.xxx 声明
        # 引擎根据声明的 type 决定是写文件还是直接存 manifest
        ...
```
### ConfigAccessor（handler 内使用）
python stage 的 handler 通过这个对象访问配置，屏蔽 file 穿透细节：

```
class ConfigAccessor:
    def read(self, field_descriptor: dict) -> Any:
        """
        field_descriptor 是 llm_config 里的字段定义，如：
          {"type": "file", "path": "resonances_config.toml"}
          {"type": "ref", "path": "stages.config_strip.free_params"}
        """
        if field_descriptor.get("type") == "file":
            return load_file(self.workdir / field_descriptor["path"])
        if field_descriptor.get("type") == "ref":
            return self.resolver.resolve(field_descriptor["path"])
        return field_descriptor  # 直接值
```
handler 写法示例：

```
# gen/handlers/config_strip_handler.py
def config_strip_handler(cfg: ConfigAccessor, resolver) -> dict:
    raw_config = cfg.read(cfg.stage["ref.resonances_config"])
    stripped, free_params, _ = parse_config(raw_config)
    sbc, amp = extract_sbc_amp(raw_config)
    return {
        "free_params": free_params,          # → run/free_params.toml
        "stripped_config": stripped,         # → gen/fragments/stripped_config.toml
        "save_results": gen_save_results(free_params),  # → gen/fragments/save_results.py
        "all_sbc": sbc,                      # → manifest direct
        "all_amp": amp,                      # → manifest direct
    }
    # 引擎根据 llm_config 里 output.xxx 的声明决定每个字段如何存储
```
## 入口（cli.py）

```
# 用法
python -m agent.cli --workdir analyses/kk_new --config llm_config_fit.toml

# 支持单独重跑某个 stage（跳过缓存）
python -m agent.cli --workdir analyses/kk_new --stage likelihood_function --force

# 静态检查（不实际运行）
python -m agent.cli --workdir analyses/kk_new --check-only
```

## 与现有代码的关系
现有文件	新框架对应
generator_base.py	拆分为 engine.py + resolver.py + manifest.py + stage_runner.py
generator_fit.py	删除，逻辑移入 gen/handlers/
config_parser.py	保留，被 config_strip_handler.py 调用
easytrans_client.py	保留，改名 llm_client.py
agent/prompts/	移入 analyses/kk_new/gen/prompts/（已完成）
agent/prompts/templates/	移入 analyses/kk_new/gen/templates/（已完成）
关键设计决策汇总
hash = 渲染后的 prompt，不手动列 hash_inputs，引用内容变化自动失效
<<>> 定界符，{} 留给 Python 代码和物理公式
foreach 输出独立文件（resonance_calculation_{key}.py），[*] 语法在引用时合并
严格校验：静态检查 + 运行时每次字段读取都校验，空值/KeyError 立刻抛带上下文的异常
handler 返回 dict，引擎根据 llm_config 声明决定每个字段如何持久化，handler 不关心路径
所有路径相对 workdir，引擎统一处理，handler 不拼路径