"""
generator_draw.py — Draw weight generation pipeline.

Reuses stages 0/1/3/4 from FitGenerator (same fragment files in cache/fragments/),
then runs draw-specific stages:
  - draw_load_data   (LLM, includes extra_sbc)
  - draw_weight      (LLM)
  - draw_run_load    (LLM)
  - draw_main        (LLM, called after assembly)
"""

import json
import os
from typing import Dict, List, Optional

from generator_base import StageRunner, load_prompt
from config_parser import parse_config
from generator_fit import FitGenerator


class DrawGenerator(FitGenerator):
    """Draw mode pipeline; inherits shared stages from FitGenerator."""

    def __init__(self, workdir: str, model: Optional[str] = None,
                 model_check: Optional[str] = None,
                 config_path: Optional[str] = None) -> None:
        super().__init__(workdir=workdir, model=model,
                         model_check=model_check, config_path=config_path)
        # Override mode so template sections are loaded for draw
        self.mode = "draw"
        self.sections = __import__("agent.generator_base", fromlist=["parse_template_sections"]).parse_template_sections(mode="draw")
        physics_file = os.path.join(self.workdir, "physics_functions.py")
        if os.path.exists(physics_file):
            with open(physics_file, encoding="utf-8") as f:
                self.sections["PHYSICS_FUNCTIONS"] = f.read()

    # ------------------------------------------------------------------
    # Reused stages: stage0/1/3/4 are identical to fit (same fragment files)
    # ------------------------------------------------------------------
    # We inherit stage0_config_strip, stage1_classification,
    # stage3_resonance_calculation, stage4_extract_parameters as-is.

    # ------------------------------------------------------------------
    # Draw-specific: draw_load_data (LLM)
    # ------------------------------------------------------------------

    def stage_draw_load_data(self) -> None:
        stripped_config = self.read_stage_output("config_strip.stripped_config")
        stripped_json = json.dumps(stripped_config.get("resonances", {}), indent=2)
        sbc, amp = self.get_all_resonance_data()
        extra_sbc = self.get_draw_extra_sbc()
        prompt_template = load_prompt("draw_load_data")

        def build_prompt():
            return prompt_template.format(
                data_loading_section=self.sections.get("DATA_LOADING", ""),
                sbc=sbc,
                amp=amp,
                extra_sbc=extra_sbc,
            )

        self.run_llm_stage(
            name="draw_load_data",
            hash_inputs=[stripped_json, json.dumps(extra_sbc), prompt_template],
            build_prompt_fn=build_prompt,
            fragment_name="fragments/draw_load_data.py",
        )

    # ------------------------------------------------------------------
    # Draw-specific: draw_weight_function (LLM)
    # ------------------------------------------------------------------

    def stage_draw_weight_function(self) -> None:
        classification = self.read_stage_output("classification")
        free_params = self.read_stage_output("config_strip.free_params")["free_params"]
        resonance_fragments = self._load_resonance_fragments(classification)
        extract_parameters_code = self.load_fragment("fragments/extract_parameters.py")

        schema = [{"path": p["path"], "range": p.get("range")} for p in free_params]
        schema_json = json.dumps(schema, indent=2)
        classification_json = json.dumps(
            classification.get("propagator_classification", {}), indent=2
        )
        prompt_template = load_prompt("draw_weight_function")

        def build_prompt():
            return prompt_template.format(
                parameter_info_str=schema_json,
                resonance_calculation="\n\n".join(resonance_fragments),
                extract_parameters=extract_parameters_code,
                draw_weight_functions_section=self.sections.get("draw_weight_functions", ""),
            )

        self.run_llm_stage(
            name="draw_weight_function",
            hash_inputs=[classification_json, schema_json, prompt_template],
            build_prompt_fn=build_prompt,
            fragment_name="fragments/draw_weight_function.py",
        )

    # ------------------------------------------------------------------
    # Draw-specific: draw_run_load_data (LLM)
    # ------------------------------------------------------------------

    def stage_draw_run_load_data(self) -> None:
        draw_load_data_code = self.load_fragment("fragments/draw_load_data.py")
        data_hash = self.compute_hash(draw_load_data_code)
        prompt_template = load_prompt("draw_run_load_data")

        def build_prompt():
            return prompt_template.format(
                draw_load_data_section=self.sections.get("draw_load_data_section", ""),
                load_data=draw_load_data_code,
            )

        self.run_llm_stage(
            name="draw_run_load_data",
            hash_inputs=[data_hash, prompt_template],
            build_prompt_fn=build_prompt,
            fragment_name="fragments/draw_run_load_data.py",
        )

    # ------------------------------------------------------------------
    # Draw-specific: draw_main_section (LLM) — called after assembly
    # ------------------------------------------------------------------

    def stage_draw_main_section(self) -> None:
        assembled_code = self._read_draw_assembled_code()
        assembled_hash = self.compute_hash(assembled_code)
        prompt_template = load_prompt("draw_main_section")

        def build_prompt():
            return prompt_template.format(
                draw_main_section=self.sections.get("draw_main_section", ""),
                full_code=assembled_code,
            )

        self.run_llm_stage(
            name="draw_main_section",
            hash_inputs=[assembled_hash, prompt_template],
            build_prompt_fn=build_prompt,
            fragment_name="fragments/draw_main_section.py",
        )

    # ------------------------------------------------------------------
    # Internal: load resonance fragments from manifest-driven cache files
    # ------------------------------------------------------------------

    def _read_draw_assembled_code(self) -> str:
        classification = self.read_stage_output("classification")
        fragments = {
            "draw_load_data": self.load_fragment("fragments/draw_load_data.py"),
            "resonance_calculation": "\n\n".join(self._load_resonance_fragments(classification)),
            "extract_parameters": self.load_fragment("fragments/extract_parameters.py"),
            "draw_weight_function": self.load_fragment("fragments/draw_weight_function.py"),
            "draw_run_load_data": self.load_fragment("fragments/draw_run_load_data.py"),
        }
        return self.assemble_draw_code(fragments)

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------

    def assemble_draw_code(self, fragments: Dict[str, str]) -> str:
        header = "\n".join([
            "# Auto-generated draw weight script by LLMResonanceGenerator — do not edit manually",
            self.sections.get("COMMON_UTILITIES", ""),
            self.sections.get("PATH_CONFIG", ""),
            self.sections.get("LOGGING_CONFIG", ""),
            self.sections.get("DPLEX_FUNCTIONS", ""),
            self.sections.get("PHYSICS_FUNCTIONS", ""),
        ])

        parts = [header]
        for key in ("draw_load_data", "resonance_calculation", "extract_parameters",
                    "draw_weight_function"):
            if key in fragments:
                parts.append(fragments[key])

        if "draw_run_load_data" in fragments:
            parts.append(fragments["draw_run_load_data"])

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def run(self, output_path: Optional[str] = None) -> str:
        self.print_config_summary()

        # Reuse fit stages (all read from / write to manifest)
        self.stage0_config_strip()
        self.stage1_classification()
        self.stage3_resonance_calculation()
        self.stage4_extract_parameters()

        # Draw-specific stages
        self.stage_draw_load_data()
        self.stage_draw_weight_function()
        self.stage_draw_run_load_data()

        assembled = self._read_draw_assembled_code()
        self.stage_draw_main_section()
        draw_main = self.load_fragment("fragments/draw_main_section.py")
        full_code = assembled + "\n\n" + draw_main

        out = output_path or os.path.join(self.workdir, "run", "draw_weight_script.py")
        return self.save_code(full_code, out)
