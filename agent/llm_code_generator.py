#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

load_dotenv()
import toml
import hashlib
import argparse

foo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(foo_path)
sys.path.append(foo_path)

from agent.easytrans_client import EasyTransClient, EasyTransError

PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")


def load_prompt(name: str) -> str:
    """Load a prompt template from agent/prompts/<name>.txt"""
    path = os.path.join(PROMPTS_DIR, f"{name}.txt")
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def parse_template_sections(file_path=None):
    """Parse # SECTION: blocks from a Python template file."""
    if file_path is None:
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "common_template.py")

    sections = {}
    current_section = None
    section_content = []

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        if line.strip().startswith('# SECTION:'):
            if current_section:
                sections[current_section] = ''.join(section_content)
            current_section = line.strip().split('SECTION:')[1].strip()
            section_content = []
        elif current_section:
            section_content.append(line)

    if current_section:
        sections[current_section] = ''.join(section_content)

    return sections


def check_directory_structure(workdir: str):
    for subdir in ["run", "cache", "results", "data"]:
        path = os.path.join(workdir, subdir)
        if not os.path.isdir(path):
            print(f"Directory {path} not exists! create directory")
            os.makedirs(path, exist_ok=True)


class LLMResonanceGenerator:
    """LLM-driven PWA resonance code generator (generation only, no execution)."""

    def __init__(self, workdir: str = ".", config_path: str = None,
                 model: Optional[str] = None, model_check: Optional[str] = None):
        self.workdir = os.path.abspath(workdir)
        check_directory_structure(self.workdir)

        self.model = model or os.getenv('EASYTRANS_MODEL', 'gemini-2.5-pro')
        self.model_check = model_check or os.getenv('EASYTRANS_MODEL_CHECK', self.model)

        self.llm_client = EasyTransClient()
        print(f"LLM engine: {self.model}")

        self.config_path = config_path or os.path.join(self.workdir, "resonances_config.toml")
        self.config = self._load_config()

        self.sections = parse_template_sections()

        physics_file = os.path.join(self.workdir, "physics_functions.py")
        if os.path.exists(physics_file):
            with open(physics_file, encoding='utf-8') as f:
                self.sections['PHYSICS_FUNCTIONS'] = f.read()
            print(f"Loaded physics functions: {physics_file}")
        else:
            print(f"Warning: {physics_file} not found, using default from common_template.py")

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    def _load_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = toml.load(f)
            print(f"Config loaded: {self.config_path}")
            print(f"Resonances: {len(config_data.get('resonances', {}))}")
            return config_data
        except FileNotFoundError:
            print(f"Config not found: {self.config_path}")
            raise
        except toml.TomlDecodeError as e:
            print(f"TOML parse error: {e}")
            raise

    def get_all_resonance_names(self) -> List[str]:
        return list(self.config.get('resonances', {}).keys())

    def get_all_resonance_data(self):
        sbc_list = [
            prop["Sbc"]
            for resonance in self.config["resonances"].values()
            for prop in resonance.get("propagators", {}).values()
            if "Sbc" in prop
        ]
        amp_list = [
            resonance["Amplitude"]["AMP"]
            for resonance in self.config["resonances"].values()
            if "AMP" in resonance.get("Amplitude", {})
        ]
        return list(dict.fromkeys(sbc_list)), list(dict.fromkeys(amp_list))

    def print_config_summary(self):
        print("Config summary:")
        print("=" * 40)
        for name, config in self.config.get('resonances', {}).items():
            print(f"  {name}")
            for prop_name, prop_config in config.get('propagators', {}).items():
                print(f"    - {prop_name}: {prop_config.get('propagator_type', 'unknown')}")

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _load_cache(self, cache_file: str) -> dict:
        try:
            if os.path.exists(cache_file):
                with open(cache_file, 'r', encoding='utf-8') as f:
                    print(f"Cache loaded: {cache_file}")
                    return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Cache load failed ({cache_file}): {e}")
        return {}

    def _save_cache(self, cache: dict, cache_file: str):
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache, f, ensure_ascii=False, indent=4)
            print(f"Cache saved: {cache_file}")
        except IOError as e:
            print(f"Cache save failed ({cache_file}): {e}")

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _prompt_analysis_toml_config(self) -> str:
        all_resonances_info = json.dumps(self.config.get('resonances', {}), indent=4)
        return load_prompt("analysis_toml_config").format(
            all_resonances_info=all_resonances_info
        )

    def _prompt_load_data(self) -> str:
        sbc, amp = self.get_all_resonance_data()
        return load_prompt("load_data").format(
            data_loading_section=self.sections.get('DATA_LOADING', ''),
            sbc=sbc,
            amp=amp
        )

    def _prompt_calculate_function(self, ana_key: str, ana_value: list, resonance_name: str) -> str:
        all_resonances_info = self.config.get('resonances', {})
        resonance_info = json.dumps(all_resonances_info.get(resonance_name, {}), indent=4)
        return load_prompt("calculate_function").format(
            physics_propagator=self.sections.get('PHYSICS_FUNCTIONS', ''),
            calculate_function_template=self.sections.get('calculate_functions', ''),
            ana_key=ana_key,
            resonance_len_in_group=len(ana_value) > 1,
            resonance_info=resonance_info
        )

    def _prompt_extract_parameters(self, parameter_info: dict) -> str:
        return load_prompt("extract_parameters").format(
            prepare_data_parameters=self.sections.get('prepare_data_parameters', ''),
            args_list=json.dumps(parameter_info["parameter_lists"], indent=4),
            fixed_list=json.dumps(parameter_info["fixed_parameter"], indent=4),
            full_config=self.config
        )

    def _prompt_run_load_data(self, load_data: str) -> str:
        return load_prompt("run_load_data").format(
            load_data_section=self.sections.get('load_data_section', ''),
            load_data=load_data
        )

    def _prompt_likelihood_function(self, parameter_info: dict,
                                    resonance_calculation: list, extract_parameters: str) -> str:
        return load_prompt("likelihood_function").format(
            parameter_info_str=json.dumps(parameter_info["parameter_lists"], indent=4),
            resonance_calculation="\n\n".join(resonance_calculation),
            extract_parameters=extract_parameters,
            likelihood_functions_section=self.sections.get('likelihood_functions', '')
        )

    def _prompt_main_section(self, full_code: str) -> str:
        return load_prompt("main_section").format(
            main_section=self.sections.get('main_section', ''),
            full_code=full_code
        )

    # ------------------------------------------------------------------
    # Draw prompt builders
    # ------------------------------------------------------------------

    def get_draw_extra_sbc(self) -> List[str]:
        return self.config.get('draw', {}).get('extra_sbc', [])

    def _prompt_draw_load_data(self) -> str:
        sbc, amp = self.get_all_resonance_data()
        extra_sbc = self.get_draw_extra_sbc()
        return load_prompt("draw_load_data").format(
            data_loading_section=self.sections.get('DATA_LOADING', ''),
            sbc=sbc,
            amp=amp,
            extra_sbc=extra_sbc
        )

    def _prompt_draw_weight_function(self, parameter_info: dict,
                                     resonance_calculation: list, extract_parameters: str) -> str:
        return load_prompt("draw_weight_function").format(
            parameter_info_str=json.dumps(parameter_info["parameter_lists"], indent=4),
            resonance_calculation="\n\n".join(resonance_calculation),
            extract_parameters=extract_parameters,
            draw_weight_functions_section=self.sections.get('draw_weight_functions', '')
        )

    def _prompt_draw_run_load_data(self, load_data: str) -> str:
        return load_prompt("draw_run_load_data").format(
            draw_load_data_section=self.sections.get('draw_load_data_section', ''),
            load_data=load_data
        )

    def _prompt_draw_main_section(self, full_code: str) -> str:
        return load_prompt("draw_main_section").format(
            draw_main_section=self.sections.get('draw_main_section', ''),
            full_code=full_code
        )

    # ------------------------------------------------------------------
    # LLM call with caching
    # ------------------------------------------------------------------

    def _generate(self, prompt: str, cache_file: str, check: bool = False) -> str:
        cache = self._load_cache(cache_file)
        prompt_hash = hashlib.sha256(prompt.encode('utf-8')).hexdigest()

        if prompt_hash in cache:
            print(f"Cache hit ({cache_file}, {prompt_hash[:8]}...)")
            return cache[prompt_hash]['generated_code']

        print(f"Prompt length: {len(prompt)} chars")

        response = self.llm_client.responses(input_text=prompt, model=self.model)
        if not self.llm_client.validate_response(response):
            raise EasyTransError("LLM response validation failed")

        generated_code = self.llm_client.extract_content(response)
        if not generated_code:
            raise EasyTransError("LLM returned empty content")

        generated_code = re.sub(r'^```\w*\n?', '', generated_code.strip())
        generated_code = re.sub(r'\n?```$', '', generated_code.strip())

        print(f"Generated: {len(generated_code)} chars")

        if check:
            check_prompt = (
                f"{prompt}\nPlease strictly check the following code for compliance with all "
                f"the rules mentioned in the prompt. If any rule is violated, regenerate the "
                f"code until it fully complies with all the rules.\nCode:\n{generated_code}"
            )
            check_out = self.llm_client.responses(input_text=check_prompt, model=self.model_check)
            generated_code = self.llm_client.extract_content(check_out)
            generated_code = re.sub(r'^```\w*\n?', '', generated_code.strip())
            generated_code = re.sub(r'\n?```$', '', generated_code.strip())

        cache[prompt_hash] = {"prompt": prompt, "generated_code": generated_code}
        self._save_cache(cache, cache_file)
        return generated_code

    # ------------------------------------------------------------------
    # Generation pipeline
    # ------------------------------------------------------------------

    def generate_functions(self) -> Dict[str, str]:
        """Run the full generation pipeline. Returns a dict of named code fragments."""
        cache_dir = os.path.join(self.workdir, "cache")
        functions = {}

        # Stage 1: analyse TOML config
        ana_result_raw = self._generate(
            self._prompt_analysis_toml_config(),
            os.path.join(cache_dir, "ana_cache.json"),
            check=False
        )
        ana_result = json.loads(ana_result_raw)
        print("Analysis result:", ana_result)
        time.sleep(1)

        # Stage 2: data loading
        functions['data_load'] = self._generate(
            self._prompt_load_data(),
            os.path.join(cache_dir, "load_data_cache.json"),
            check=False
        )
        time.sleep(1)

        # Stage 3: resonance calculate/component functions
        resonance_calculation_fragments = []
        for ana_key, ana_value in ana_result["propagator_classification"].items():
            ana_value = sorted(ana_value)
            fragment = self._generate(
                self._prompt_calculate_function(ana_key, ana_value, ana_value[0]),
                os.path.join(cache_dir, f"resonance_calculation_{ana_key}.json"),
                check=True
            )
            resonance_calculation_fragments.append(fragment)
        functions['resonance_calculation'] = "\n\n".join(resonance_calculation_fragments)
        time.sleep(1)

        # Stage 4: extract_parameters + run_load_data
        functions['extract_parameters'] = self._generate(
            self._prompt_extract_parameters(ana_result),
            os.path.join(cache_dir, "extract_parameters_cache.json"),
            check=False
        )
        functions['run_load_data'] = self._generate(
            self._prompt_run_load_data(functions['data_load']),
            os.path.join(cache_dir, "run_load_data_cache.json"),
            check=False
        )
        time.sleep(1)

        # Stage 5: likelihood functions
        functions['likelihood_function'] = self._generate(
            self._prompt_likelihood_function(
                ana_result, resonance_calculation_fragments, functions['extract_parameters']
            ),
            os.path.join(cache_dir, "analysis_likelihood_cache.json"),
            check=False
        )
        time.sleep(1)

        return functions

    def assemble_code(self, functions: Dict[str, str]) -> str:
        """Assemble all fragments into a single script string."""
        cache_dir = os.path.join(self.workdir, "cache")

        header = "\n".join([
            "# Auto-generated by LLMResonanceGenerator — do not edit manually",
            self.sections.get('COMMON_UTILITIES', ''),
            self.sections.get('PATH_CONFIG', ''),
            self.sections.get('LOGGING_CONFIG', ''),
            self.sections.get('DPLEX_FUNCTIONS', ''),
            self.sections.get('PHYSICS_FUNCTIONS', ''),
        ])

        parts = [header]
        for key in ('data_load', 'resonance_calculation', 'extract_parameters',):
            if key in functions:
                parts.append(functions[key])

        if 'likelihood_function' in functions:
            parts.append(
                functions['likelihood_function'] + "\n\n"
                + self.sections.get('combined_likelihood_function', '')
            )

        if 'run_load_data' in functions:
            parts.append(functions['run_load_data'])

        full_code = "\n\n".join(parts)

        # Stage 7: main entry point (needs full_code as context)
        functions['main_section'] = self._generate(
            self._prompt_main_section(full_code),
            os.path.join(cache_dir, "main_section_cache.json"),
            check=False
        )
        time.sleep(1)

        full_code += "\n\n" + functions['main_section']
        return full_code

    def save_code(self, code: str, output_path: Optional[str] = None) -> str:
        """Write the assembled script to disk and return the path."""
        if output_path is None:
            output_path = os.path.join(self.workdir, "run", "generated_script.py")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(code)
        print(f"Script saved: {output_path}")
        return output_path

    # ------------------------------------------------------------------
    # Draw generation pipeline
    # ------------------------------------------------------------------

    def generate_draw_functions(self) -> Dict[str, str]:
        """Run the draw weight generation pipeline."""
        cache_dir = os.path.join(self.workdir, "cache")
        functions = {}

        # Stage 1: analyse TOML config (reuse fit cache)
        ana_result_raw = self._generate(
            self._prompt_analysis_toml_config(),
            os.path.join(cache_dir, "ana_cache.json"),
            check=False
        )
        ana_result = json.loads(ana_result_raw)
        print("Analysis result:", ana_result)
        time.sleep(1)

        # Stage 2: draw data loading (includes extra_sbc)
        functions['draw_load_data'] = self._generate(
            self._prompt_draw_load_data(),
            os.path.join(cache_dir, "draw_load_data_cache.json"),
            check=False
        )
        time.sleep(1)

        # Stage 3: resonance calculate/component functions (reuse fit cache)
        resonance_calculation_fragments = []
        for ana_key, ana_value in ana_result["propagator_classification"].items():
            ana_value = sorted(ana_value)
            fragment = self._generate(
                self._prompt_calculate_function(ana_key, ana_value, ana_value[0]),
                os.path.join(cache_dir, f"resonance_calculation_{ana_key}.json"),
                check=True
            )
            resonance_calculation_fragments.append(fragment)
        functions['resonance_calculation'] = "\n\n".join(resonance_calculation_fragments)
        time.sleep(1)

        # Stage 4: extract_parameters (reuse fit cache)
        functions['extract_parameters'] = self._generate(
            self._prompt_extract_parameters(ana_result),
            os.path.join(cache_dir, "extract_parameters_cache.json"),
            check=False
        )
        time.sleep(1)

        # Stage 5: draw weight functions
        functions['draw_weight_function'] = self._generate(
            self._prompt_draw_weight_function(
                ana_result, resonance_calculation_fragments, functions['extract_parameters']
            ),
            os.path.join(cache_dir, "draw_weight_function_cache.json"),
            check=False
        )
        time.sleep(1)

        # Stage 6: draw run_load_data
        functions['draw_run_load_data'] = self._generate(
            self._prompt_draw_run_load_data(functions['draw_load_data']),
            os.path.join(cache_dir, "draw_run_load_data_cache.json"),
            check=False
        )
        time.sleep(1)

        return functions

    def assemble_draw_code(self, functions: Dict[str, str]) -> str:
        """Assemble draw weight script from fragments."""
        cache_dir = os.path.join(self.workdir, "cache")

        header = "\n".join([
            "# Auto-generated draw weight script by LLMResonanceGenerator — do not edit manually",
            self.sections.get('COMMON_UTILITIES', ''),
            self.sections.get('PATH_CONFIG', ''),
            self.sections.get('LOGGING_CONFIG', ''),
            self.sections.get('DPLEX_FUNCTIONS', ''),
            self.sections.get('PHYSICS_FUNCTIONS', ''),
        ])

        parts = [header]
        for key in ('draw_load_data', 'resonance_calculation', 'extract_parameters',
                    'draw_weight_function'):
            if key in functions:
                parts.append(functions[key])

        if 'draw_run_load_data' in functions:
            parts.append(functions['draw_run_load_data'])

        full_code = "\n\n".join(parts)

        # Stage 7: draw main section (needs full_code as context)
        functions['draw_main_section'] = self._generate(
            self._prompt_draw_main_section(full_code),
            os.path.join(cache_dir, "draw_main_section_cache.json"),
            check=False
        )
        time.sleep(1)

        full_code += "\n\n" + functions['draw_main_section']
        return full_code


def main():
    parser = argparse.ArgumentParser(description="LLM-driven PWA resonance code generator")
    parser.add_argument("--workdir", default=".",
                        help="Analysis directory containing resonances_config.toml")
    parser.add_argument("--config", default=None,
                        help="TOML config path (default: workdir/resonances_config.toml)")
    parser.add_argument("--model", default=os.getenv('EASYTRANS_MODEL', 'gemini-2.5-pro'))
    parser.add_argument("--model-check", default=os.getenv('EASYTRANS_MODEL_CHECK',
                                                            os.getenv('EASYTRANS_MODEL', 'gemini-2.5-pro')))
    parser.add_argument("--output", default=None,
                        help="Output script path")
    parser.add_argument("--mode", default="fit", choices=["fit", "draw"],
                        help="Generation mode: 'fit' for fit script, 'draw' for draw weight script")
    args = parser.parse_args()

    generator = LLMResonanceGenerator(
        workdir=args.workdir,
        config_path=args.config,
        model=args.model,
        model_check=args.model_check,
    )
    generator.print_config_summary()

    if args.mode == "fit":
        functions = generator.generate_functions()
        full_code = generator.assemble_code(functions)
        output_path = args.output or os.path.join(args.workdir, "run", "generated_script.py")
        generator.save_code(full_code, output_path=output_path)
    elif args.mode == "draw":
        functions = generator.generate_draw_functions()
        full_code = generator.assemble_draw_code(functions)
        output_path = args.output or os.path.join(args.workdir, "run", "draw_weight_script.py")
        generator.save_code(full_code, output_path=output_path)


if __name__ == "__main__":
    main()
