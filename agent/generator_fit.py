"""
generator_fit.py — Fit mode pipeline (8 stages).

Stage 0  config_strip     python  strip free-param values → stripped_config, free_params
Stage 1  classification   llm     classify resonances → manifest [clasad        llm     generate data loading functions
Stage 3  resonance_calc   llm     generate calculate_*/component_* per propagator group
Stage 4  extract_params   python  generate extract_parameters() by template
Stage 5  likelihood       llm     generate likelihood functions
Stage 6  run_load_data    llm     generate data loading execution block
Stage 7  main_section     llm     generate main() entry point
"""

import json
import os
from typing import Dict, List, Optional

from generator_base import StageRunner, load_prompt
from config_parser import parse_config


class FitGenerator(StageRunner):
    """Fit mode pipeline."""

    def __init__(self, workdir: str, model: Optional[str] = None,
                 model_check: Optional[str] = None,
                 config_path: Optional[str] = None) -> None:
        super().__init__(workdir=workdir, mode="fit", model=model,
                         model_check=model_check, config_path=config_path)

    # ------------------------------------------------------------------
    # Stage 0: config_strip (pure Python, no cache needed)
    # ------------------------------------------------------------------

    def stage0_config_strip(self) -> None:
        """Strip free-param values from raw_config; persist two fragment files."""
        stripped_config, free_params, _ = parse_config(self.raw_config)

        # field: free_params — {path, value, range, error, arg_index}
        self.write_stage_field(
            "config_strip", "free_params",
            {"free_params": free_params},
            fragment_path="fragments/free_params.toml",
        )

        # field: stripped_config — free fields replaced with {arg_index = i}
        self.write_stage_field(
            "config_strip", "stripped_config",
            stripped_config,
            fragment_path="fragments/stripped_config.toml",
        )

    # ------------------------------------------------------------------
    # Stage 1: classification (LLM)
    # ------------------------------------------------------------------

    def stage1_classification(self) -> None:
        stripped_config = self.read_stage_output("config_strip", field="stripped_config")
        stripped_json = json.dumps(stripped_config.get("resonances", {}), indent=2)
        prompt_template = load_prompt("analysis_toml_config")

        prompt = self.build_stage_prompt(
            "classification", prompt_template,
            all_resonances_info=stripped_json,
        )

        self.run_llm_stage(
            name="classification",
            hash_inputs=[stripped_json, prompt_template],
            prompt=prompt,
            fragment_name="fragments/classification.json",
            check=False,
        )

    # ------------------------------------------------------------------
    # Stage 2: data_load (LLM)
    # ------------------------------------------------------------------

    def stage2_data_load(self) -> None:
        stripped_config = self.read_stage_output("config_strip", field="stripped_config")
        stripped_json = json.dumps(stripped_config.get("resonances", {}), indent=2)
        sbc, amp = self.get_all_resonance_data()
        prompt_template = load_prompt("load_data")

        prompt = self.build_stage_prompt(
            "data_load", prompt_template,
            data_loading_section=self.sections.get("DATA_LOADING", ""),
            sbc=sbc,
            amp=amp,
        )

        self.run_llm_stage(
            name="data_load",
            hash_inputs=[stripped_json, prompt_template],
            prompt=prompt,
            fragment_name="fragments/data_load.py",
        )

    # ------------------------------------------------------------------
    # Stage 3: resonance_calculation (LLM, one per propagator group)
    # ------------------------------------------------------------------

    def stage3_resonance_calculation(self) -> None:
        stripped_config = self.read_stage_output("config_strip", field="stripped_config")
        classification = self.read_stage_output("classification")
        prop_classification = classification.get("propagator_classification", {})
        classification_json = json.dumps(prop_classification, indent=2)
        prompt_template = load_prompt("calculate_function")

        for group_key, resonance_names in prop_classification.items():
            resonance_names = sorted(resonance_names)
            rep_name = resonance_names[0]
            resonance_info = json.dumps(
                stripped_config.get("resonances", {}).get(rep_name, {}), indent=2
            )

            prompt = self.build_stage_prompt(
                f"resonance_calculation.{group_key}", prompt_template,
                physics_propagator=self.sections.get("PHYSICS_FUNCTIONS", ""),
                calculate_function_template=self.sections.get("calculate_functions", ""),
                ana_key=group_key,
                resonance_len_in_group=len(resonance_names) > 1,
                resonance_info=resonance_info,
            )

            self.run_llm_stage(
                name=f"resonance_calculation.{group_key}",
                hash_inputs=[classification_json, resonance_info, prompt_template],
                prompt=prompt,
                fragment_name=f"fragments/resonance_{group_key}.py",
                check=False,
            )

    # ------------------------------------------------------------------
    # Stage 4a: make_initial_args (pure Python)
    # ------------------------------------------------------------------

    def stage4a_make_initial_args(self) -> None:
        free_params = self.read_stage_output("config_strip", field="free_params")["free_params"]
        schema_json = json.dumps(
            [{"path": p["path"], "value": p.get("value")} for p in free_params], indent=2
        )

        def build_fn():
            return _generate_make_initial_args(free_params)

        self.run_python_stage(
            name="make_initial_args",
            fn=build_fn,
            hash_inputs=[schema_json],
            fragment_name="fragments/make_initial_args.py",
        )

    # ------------------------------------------------------------------
    # Stage 4b: extract_parameters (LLM)
    # ------------------------------------------------------------------

    def stage4b_extract_parameters(self) -> None:
        classification = self.read_stage_output("classification")
        parameter_lists = classification.get("parameter_lists", {})
        parameter_lists_json = json.dumps(parameter_lists, indent=2)
        prompt_template = load_prompt("extract_parameters")

        prompt = self.build_stage_prompt(
            "extract_parameters", prompt_template,
            parameter_lists=parameter_lists_json,
        )

        self.run_llm_stage(
            name="extract_parameters",
            hash_inputs=[parameter_lists_json, prompt_template],
            prompt=prompt,
            fragment_name="fragments/extract_parameters.py",
        )

    # ------------------------------------------------------------------
    # Stage 4c: save_results (pure Python)
    # ------------------------------------------------------------------

    def stage4c_save_results(self) -> None:
        free_params = self.read_stage_output("config_strip", field="free_params")["free_params"]
        schema_json = json.dumps(
            [{"path": p["path"], "range": p.get("range"), "arg_index": p["arg_index"]}
             for p in free_params], indent=2
        )

        def build_fn():
            return _generate_save_results(free_params)

        self.run_python_stage(
            name="save_results",
            fn=build_fn,
            hash_inputs=[schema_json],
            fragment_name="fragments/save_results.py",
        )

    # ------------------------------------------------------------------
    # Stage 5: likelihood_function (LLM)
    # ------------------------------------------------------------------

    def stage5_likelihood_function(self) -> None:
        classification = self.read_stage_output("classification")
        free_params = self.read_stage_output("config_strip", field="free_params")["free_params"]
        resonance_fragments = self._load_resonance_fragments(classification)
        extract_parameters_code = self.load_fragment("fragments/extract_parameters.py")

        schema = [{"path": p["path"], "range": p.get("range"), "arg_index": p.get("arg_index")} for p in free_params]
        schema_json = json.dumps(schema, indent=2)
        classification_json = json.dumps(
            classification.get("propagator_classification", {}), indent=2
        )
        prompt_template = load_prompt("likelihood_function")

        prompt = self.build_stage_prompt(
            "likelihood_function", prompt_template,
            parameter_info_str=schema_json,
            resonance_calculation="\n\n".join(resonance_fragments),
            extract_parameters=extract_parameters_code,
            likelihood_functions_section=self.sections.get("likelihood_functions", ""),
        )

        self.run_llm_stage(
            name="likelihood_function",
            hash_inputs=[classification_json, schema_json, prompt_template],
            prompt=prompt,
            fragment_name="fragments/likelihood_function.py",
        )

    # ------------------------------------------------------------------
    # Stage 6: run_load_data (LLM)
    # ------------------------------------------------------------------

    def stage6_run_load_data(self) -> None:
        data_load_code = self.load_fragment("fragments/data_load.py")
        data_load_hash = self.compute_hash(data_load_code)
        prompt_template = load_prompt("run_load_data")

        prompt = self.build_stage_prompt(
            "run_load_data", prompt_template,
            load_data_section=self.sections.get("load_data_section", ""),
            load_data=data_load_code,
        )

        self.run_llm_stage(
            name="run_load_data",
            hash_inputs=[data_load_hash, prompt_template],
            prompt=prompt,
            fragment_name="fragments/run_load_data.py",
        )

    # ------------------------------------------------------------------
    # Stage 7: main_section (LLM) — called after assembly
    # ------------------------------------------------------------------

    def stage7_main_section(self) -> None:
        assembled_code = self._read_assembled_code()
        assembled_hash = self.compute_hash(assembled_code)
        prompt_template = load_prompt("main_section")

        prompt = self.build_stage_prompt(
            "main_section", prompt_template,
            main_section=self.sections.get("main_section", ""),
            full_code=assembled_code,
        )

        self.run_llm_stage(
            name="main_section",
            hash_inputs=[assembled_hash, prompt_template],
            prompt=prompt,
            fragment_name="fragments/main_section.py",
        )

    # ------------------------------------------------------------------
    # Internal: load resonance fragments from manifest-driven cache files
    # ------------------------------------------------------------------

    def _load_resonance_fragments(self, classification: dict) -> List[str]:
        prop_classification = classification.get("propagator_classification", {})
        fragments: List[str] = []
        for group_key in prop_classification.keys():
            fragments.append(self.load_fragment(f"fragments/resonance_{group_key}.py"))
        return fragments

    def _read_assembled_code(self) -> str:
        classification = self.read_stage_output("classification")
        fragments = {
            "data_load": self.load_fragment("fragments/data_load.py"),
            "resonance_calculation": "\n\n".join(self._load_resonance_fragments(classification)),
            "make_initial_args": self.load_fragment("fragments/make_initial_args.py"),
            "extract_parameters": self.load_fragment("fragments/extract_parameters.py"),
            "save_results": self.load_fragment("fragments/save_results.py"),
            "likelihood_function": self.load_fragment("fragments/likelihood_function.py"),
            "run_load_data": self.load_fragment("fragments/run_load_data.py"),
        }
        return self.assemble_code(fragments)

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------

    def assemble_code(self, fragments: Dict[str, str]) -> str:
        header = "\n".join([
            "# Auto-generated by LLMResonanceGenerator — do not edit manually",
            self.sections.get("COMMON_UTILITIES", ""),
            self.sections.get("PATH_CONFIG", ""),
            self.sections.get("LOGGING_CONFIG", ""),
            self.sections.get("DPLEX_FUNCTIONS", ""),
            self.sections.get("PHYSICS_FUNCTIONS", ""),
        ])

        parts = [header]
        for key in ("data_load", "resonance_calculation", "make_initial_args", "extract_parameters", "save_results"):
            if key in fragments:
                parts.append(fragments[key])

        if "likelihood_function" in fragments:
            parts.append(
                fragments["likelihood_function"] + "\n\n"
                + self.sections.get("combined_likelihood_function", "")
            )

        if "run_load_data" in fragments:
            parts.append(fragments["run_load_data"])

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def run(self, output_path: Optional[str] = None) -> str:
        self.print_config_summary()

        # Stage 0
        self.stage0_config_strip()

        # Stage 1
        self.stage1_classification()
        print("Classification:", self.read_stage_output("classification"))

        # Stage 2
        self.stage2_data_load()

        # Stage 3
        self.stage3_resonance_calculation()

        # Stage 4a/4b/4c
        self.stage4a_make_initial_args()
        self.stage4b_extract_parameters()
        self.stage4c_save_results()

        # Stage 5
        self.stage5_likelihood_function()

        # Stage 6
        self.stage6_run_load_data()

        # Assemble without main_section first
        assembled = self._read_assembled_code()

        # Stage 7 (needs assembled code as context)
        self.stage7_main_section()

        main_section = self.load_fragment("fragments/main_section.py")
        full_code = assembled + "\n\n" + main_section

        out = output_path or os.path.join(self.workdir, "run", "generated_script.py")
        return self.save_code(full_code, out)


# ---------------------------------------------------------------------------
# Stage 4 helpers: pure-Python code generators
# ---------------------------------------------------------------------------

def _generate_make_initial_args(free_params: list) -> str:
    lines = ["def make_initial_args():"]
    if not free_params:
        lines.append("    return onp.array([])")
        return "\n".join(lines)

    lines.append("    return onp.array([")
    for p in free_params:
        value = p.get("value", 0.0)
        if value is None:
            value = 0.0
        lines.append(f"        {repr(value)},  # {p['path']}")
    lines.append("    ])")
    return "\n".join(lines)


def _generate_save_results(free_params: list) -> str:
    lines = [
        "def save_results(args, errors, output_path):",
        "    import toml, os",
        "    records = []",
    ]
    for p in free_params:
        i = p["arg_index"]
        path = p["path"]
        range_val = p.get("range")
        lines.append(f"    rec = {{")
        lines.append(f"        'path': {repr(path)},")
        lines.append(f"        'value': float(args[{i}]),")
        if range_val is not None:
            lines.append(f"        'range': {repr(range_val)},")
        lines.append(f"        'error': float(errors[{i}]),")
        lines.append(f"        'arg_index': {i},")
        lines.append(f"    }}")
        lines.append(f"    records.append(rec)")
    lines += [
        "    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)",
        "    with open(output_path, 'w', encoding='utf-8') as f:",
        "        toml.dump({'free_params': records}, f)",
    ]
    return "\n".join(lines)
