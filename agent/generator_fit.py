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
        """Strip free-param values from raw_config and persist to manifest."""
        stripped_config, free_params, _ = parse_config(self.raw_config)
        self.write_stage_output(
            "config_strip",
            {"stripped_config": stripped_config, "free_params": free_params},
        )

    # ------------------------------------------------------------------
    # Stage 1: classification (LLM)
    # ------------------------------------------------------------------

    def stage1_classification(self) -> None:
        stripped_config = self.read_stage_output("config_strip")["stripped_config"]
        stripped_json = json.dumps(stripped_config.get("resonances", {}), indent=2)
        prompt_template = load_prompt("analysis_toml_config")

        def build_prompt():
            return prompt_template.format(all_resonances_info=stripped_json)

        raw = self.run_llm_stage(
            name="classification",
            hash_inputs=[stripped_json, prompt_template],
            build_prompt_fn=build_prompt,
            fragment_name="fragments/classification.json",
            check=False,
        )

        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            lines = [l for l in raw.splitlines() if not l.startswith("# ===")]
            result = json.loads("\n".join(lines))

        self.write_stage_output("classification", result)

    # ------------------------------------------------------------------
    # Stage 2: data_load (LLM)
    # ------------------------------------------------------------------

    def stage2_data_load(self) -> None:
        stripped_config = self.read_stage_output("config_strip")["stripped_config"]
        stripped_json = json.dumps(stripped_config.get("resonances", {}), indent=2)
        sbc, amp = self.get_all_resonance_data()
        prompt_template = load_prompt("load_data")

        def build_prompt():
            return prompt_template.format(
                data_loading_section=self.sections.get("DATA_LOADING", ""),
                sbc=sbc,
                amp=amp,
            )

        self.run_llm_stage(
            name="data_load",
            hash_inputs=[stripped_json, prompt_template],
            build_prompt_fn=build_prompt,
            fragment_name="fragments/data_load.py",
        )

    # ------------------------------------------------------------------
    # Stage 3: resonance_calculation (LLM, one per propagator group)
    # ------------------------------------------------------------------

    def stage3_resonance_calculation(self) -> None:
        stripped_config = self.read_stage_output("config_strip")["stripped_config"]
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

            def build_prompt(key=group_key, names=resonance_names, info=resonance_info):
                return prompt_template.format(
                    physics_propagator=self.sections.get("PHYSICS_FUNCTIONS", ""),
                    calculate_function_template=self.sections.get("calculate_functions", ""),
                    ana_key=key,
                    resonance_len_in_group=len(names) > 1,
                    resonance_info=info,
                )

            self.run_llm_stage(
                name=f"resonance_calculation.{group_key}",
                hash_inputs=[classification_json, resonance_info, prompt_template],
                build_prompt_fn=build_prompt,
                fragment_name=f"fragments/resonance_{group_key}.py",
                check=True,
            )

    # ------------------------------------------------------------------
    # Stage 4: extract_parameters (pure Python template)
    # ------------------------------------------------------------------

    def stage4_extract_parameters(self) -> None:
        free_params = self.read_stage_output("config_strip")["free_params"]
        schema = [{"path": p["path"], "range": p.get("range")} for p in free_params]
        schema_json = json.dumps(schema, indent=2)

        def build_fn():
            return _generate_extract_parameters(free_params)

        self.run_python_stage(
            name="extract_parameters",
            fn=build_fn,
            hash_inputs=[schema_json],
            fragment_name="fragments/extract_parameters.py",
        )

    # ------------------------------------------------------------------
    # Stage 5: likelihood_function (LLM)
    # ------------------------------------------------------------------

    def stage5_likelihood_function(self) -> None:
        classification = self.read_stage_output("classification")
        free_params = self.read_stage_output("config_strip")["free_params"]
        resonance_fragments = self._load_resonance_fragments(classification)
        extract_parameters_code = self.load_fragment("fragments/extract_parameters.py")

        schema = [{"path": p["path"], "range": p.get("range")} for p in free_params]
        schema_json = json.dumps(schema, indent=2)
        classification_json = json.dumps(
            classification.get("propagator_classification", {}), indent=2
        )
        prompt_template = load_prompt("likelihood_function")

        def build_prompt():
            return prompt_template.format(
                parameter_info_str=schema_json,
                resonance_calculation="\n\n".join(resonance_fragments),
                extract_parameters=extract_parameters_code,
                likelihood_functions_section=self.sections.get("likelihood_functions", ""),
            )

        self.run_llm_stage(
            name="likelihood_function",
            hash_inputs=[classification_json, schema_json, prompt_template],
            build_prompt_fn=build_prompt,
            fragment_name="fragments/likelihood_function.py",
        )

    # ------------------------------------------------------------------
    # Stage 6: run_load_data (LLM)
    # ------------------------------------------------------------------

    def stage6_run_load_data(self) -> None:
        data_load_code = self.load_fragment("fragments/data_load.py")
        data_load_hash = self.compute_hash(data_load_code)
        prompt_template = load_prompt("run_load_data")

        def build_prompt():
            return prompt_template.format(
                load_data_section=self.sections.get("load_data_section", ""),
                load_data=data_load_code,
            )

        self.run_llm_stage(
            name="run_load_data",
            hash_inputs=[data_load_hash, prompt_template],
            build_prompt_fn=build_prompt,
            fragment_name="fragments/run_load_data.py",
        )

    # ------------------------------------------------------------------
    # Stage 7: main_section (LLM) — called after assembly
    # ------------------------------------------------------------------

    def stage7_main_section(self) -> None:
        assembled_code = self._read_assembled_code()
        assembled_hash = self.compute_hash(assembled_code)
        prompt_template = load_prompt("main_section")

        def build_prompt():
            return prompt_template.format(
                main_section=self.sections.get("main_section", ""),
                full_code=assembled_code,
            )

        self.run_llm_stage(
            name="main_section",
            hash_inputs=[assembled_hash, prompt_template],
            build_prompt_fn=build_prompt,
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
            "extract_parameters": self.load_fragment("fragments/extract_parameters.py"),
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
        for key in ("data_load", "resonance_calculation", "extract_parameters"):
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

        # Stage 4
        self.stage4_extract_parameters()

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
# Stage 4 helper: pure-Python extract_parameters generator
# ---------------------------------------------------------------------------

def _generate_extract_parameters(free_params: list) -> str:
    lines = ["def extract_parameters(args):"]
    if not free_params:
        lines.append("    return")
        return "\n".join(lines)

    return_names = []
    for i, p in enumerate(free_params):
        var_name = _path_to_varname(p["path"])
        lines.append(f"    {var_name} = args[{i}]  # {p['path']}")
        return_names.append(var_name)

    lines.append(f"    return {', '.join(return_names)}")
    return "\n".join(lines)


def _path_to_varname(path: str) -> str:
    """
    "resonances.phif0_980.propagators.B_propagator.mass" → "phif0_980_B_propagator_mass"
    """
    skip = {"resonances", "propagators", "Amplitude", "shared_amplitude_parameters"}
    kept = [p for p in path.split(".") if p not in skip]
    return "_".join(kept)
