"""
generator_plot.py — Draw plot generation pipeline.

Reuses stage 0/1 from FitGenerator (same manifest, same fragments),
then runs plot-specific stages:
  - per-resonance plot function (LLM, one per resonance)
  - plot_main (LLM, integrates all per-resonance fragments)
"""

import json
import os
from typing import Dict, List, Optional

from agent.prompts.generator_base import StageRunner, load_prompt
from generator_fit import FitGenerator


class PlotGenerator(FitGenerator):
    """Plot mode pipeline; inherits shared stages from FitGenerator."""

    def __init__(self, workdir: str, model: Optional[str] = None,
                 model_check: Optional[str] = None,
                 config_path: Optional[str] = None) -> None:
        super().__init__(workdir=workdir, model=model,
                         model_check=model_check, config_path=config_path)
        # Override mode so template sections are loaded for plot
        self.mode = "plot"
        from agent.prompts.generator_base import parse_template_sections
        self.sections = parse_template_sections(mode="plot")

    # ------------------------------------------------------------------
    # Helper: derive weight.npz key for a resonance from classification
    # ------------------------------------------------------------------

    def _get_weight_key(self, resonance_name: str, classification: dict) -> str:
        amplitude_classification = classification.get("amplitude_classification", {})
        for amp_key, resonance_list in amplitude_classification.items():
            if resonance_name in resonance_list:
                idx = sorted(resonance_list).index(resonance_name)
                return f"{amp_key}_{idx}"

        # Fallback: construct from raw_config
        res = self.raw_config["resonances"][resonance_name]
        A_prop = res["propagators"]["A_propagator"]["propagator_type"]
        B_prop = res["propagators"]["B_propagator"]["propagator_type"]
        if res.get("kind") == "shared_state":
            amp_value = res["amplitudes"][0]["AMP"]
        else:
            amp_value = res["Amplitude"]["AMP"]
        prop_key = f"{A_prop}_{B_prop}"
        return f"{amp_value}_{prop_key}_0"

    # ------------------------------------------------------------------
    # Per-resonance plot (LLM, one per resonance)
    # ------------------------------------------------------------------

    def stage_per_resonance_plot(self, resonance_name: str) -> None:
        stripped_config = self.read_stage_output("config_strip.stripped_config")
        classification = self.read_stage_output("classification")
        resonance_info = json.dumps(
            stripped_config.get("resonances", {}).get(resonance_name, {}), indent=2
        )
        weight_key = self._get_weight_key(resonance_name, classification)
        prompt_template = load_prompt("draw_plot_resonance")

        prompt = self.build_stage_prompt(
            f"draw_plot.{resonance_name}", prompt_template,
            resonance_name=resonance_name,
            resonance_info=resonance_info,
            weight_key=weight_key,
            draw_plot_resonance_template=self.sections.get("draw_plot_resonance_template", ""),
        )

        self.run_llm_stage(
            name=f"draw_plot.{resonance_name}",
            hash_inputs=[resonance_info, weight_key, prompt_template],
            prompt=prompt,
            fragment_name=f"fragments/draw_plot_{resonance_name}.py",
        )

    # ------------------------------------------------------------------
    # plot_main (LLM)
    # ------------------------------------------------------------------

    def stage_plot_main(self) -> None:
        resonance_fragments = [
            self.load_fragment(f"fragments/draw_plot_{r}.py")
            for r in self.raw_config.get("resonances", {}).keys()
        ]
        sbc, _ = self.get_all_resonance_data()
        extra_sbc = self.get_draw_extra_sbc()

        sbc_json = json.dumps(sbc)
        extra_sbc_json = json.dumps(extra_sbc)
        fragments_hash = self.compute_hash("\n---\n".join(resonance_fragments))
        prompt_template = load_prompt("draw_plot_main")

        prompt = self.build_stage_prompt(
            "draw_plot_main", prompt_template,
            resonance_fragments="\n\n".join(resonance_fragments),
            sbc_list=sbc,
            extra_sbc_list=extra_sbc,
            draw_plot_main_template=self.sections.get("draw_plot_main_template", ""),
        )

        self.run_llm_stage(
            name="draw_plot_main",
            hash_inputs=[fragments_hash, sbc_json, extra_sbc_json, prompt_template],
            prompt=prompt,
            fragment_name="fragments/draw_plot_main.py",
        )

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def run(self, output_path: Optional[str] = None) -> str:
        self.print_config_summary()

        # Reuse fit stages (read from / write to manifest)
        self.stage0_config_strip()
        self.stage1_classification()

        # Per-resonance plot fragments
        for resonance_name in self.raw_config.get("resonances", {}).keys():
            self.stage_per_resonance_plot(resonance_name)

        # Plot main
        self.stage_plot_main()
        main_code = self.load_fragment("fragments/draw_plot_main.py")

        # Assemble
        header = "\n".join([
            "# Auto-generated draw plot script by LLMResonanceGenerator — do not edit manually",
            self.sections.get("draw_plot_imports", ""),
            self.sections.get("PATH_CONFIG", ""),
            self.sections.get("LOGGING_CONFIG", ""),
        ])
        resonance_fragments = [
            self.load_fragment(f"fragments/draw_plot_{r}.py")
            for r in self.raw_config.get("resonances", {}).keys()
        ]
        full_code = "\n\n".join([header] + resonance_fragments + [main_code])

        out = output_path or os.path.join(self.workdir, "run", "draw_plot_script.py")
        return self.save_code(full_code, out)
