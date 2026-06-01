"""
generator_base.py — StageRunner base class for the LLMPWA pipeline.

Every stage (LLM or pure Python) goes through run_llm_stage / run_python_stage,
which handle manifest.toml r/w, hash-based cache, and fragment file management.
"""

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()
import toml

_AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_AGENT_DIR)

PROMPTS_DIR = os.path.join(_AGENT_DIR, "prompts")
TEMPLATES_DIR = os.path.join(PROMPTS_DIR, "templates")

_MODE_TEMPLATES = {
    "fit":  ["shared", "fit"],
    "draw": ["shared", "fit", "draw"],
    "plot": ["plot"],
}


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def load_prompt(name: str) -> str:
    path = os.path.join(PROMPTS_DIR, f"{name}.txt")
    with open(path, encoding="utf-8") as f:
        return f.read()


def _parse_sections_from_file(file_path: str) -> dict:
    sections: dict = {}
    current: Optional[str] = None
    buf: list = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            if line.strip().startswith("# SECTION:"):
                if current:
                    sections[current] = "".join(buf)
                current = line.strip().split("SECTION:")[1].strip()
                buf = []
            elif current:
                buf.append(line)
    if current:
        sections[current] = "".join(buf)
    return sections


def parse_template_sections(mode: str = "fit") -> dict:
    sections: dict = {}
    for name in _MODE_TEMPLATES.get(mode, ["shared", "fit"]):
        path = os.path.join(TEMPLATES_DIR, f"{name}.py")
        sections.update(_parse_sections_from_file(path))
    return sections


def check_directory_structure(workdir: str) -> None:
    for sub in ("run", "cache", "results", "data"):
        p = os.path.join(workdir, sub)
        if not os.path.isdir(p):
            print(f"Creating directory: {p}")
            os.makedirs(p, exist_ok=True)


# ---------------------------------------------------------------------------
# StageRunner
# ---------------------------------------------------------------------------

class StageRunner:
    """
    Base class for fit/draw/plot generators.

    Provides:
      - manifest.toml read/write
      - fragment file read/write under cache/fragments/
      - hash-based cache invalidation
      - LLM call with optional check pass
      - human_prompt injection from llm_config_fit.toml (Goal 4)
    """

    def __init__(
        self,
        workdir: str,
        mode: str,
        model: Optional[str] = None,
        model_check: Optional[str] = None,
        config_path: Optional[str] = None,
    ) -> None:
        from easytrans_client import EasyTransClient

        self.workdir = os.path.abspath(workdir)
        self.mode = mode
        check_directory_structure(self.workdir)

        self.model = model or os.getenv("EASYTRANS_MODEL", "gemini-2.5-pro")
        self.model_check = model_check or os.getenv("EASYTRANS_MODEL_CHECK", self.model)
        self.llm_client = EasyTransClient()
        print(f"LLM engine: {self.model}")

        self.config_path = config_path or os.path.join(self.workdir, "resonances_config.toml")
        self.raw_config = self._load_toml(self.config_path)

        self.sections = parse_template_sections(mode=self.mode)
        physics_file = os.path.join(self.workdir, "physics_functions.py")
        if os.path.exists(physics_file):
            with open(physics_file, encoding="utf-8") as f:
                self.sections["PHYSICS_FUNCTIONS"] = f.read()
            print(f"Loaded physics functions: {physics_file}")

        # Goal 4: optional per-analysis LLM config
        llm_cfg_path = os.path.join(self.workdir, "llm_config_fit.toml")
        self.llm_config: dict = {}
        if os.path.exists(llm_cfg_path):
            self.llm_config = self._load_toml(llm_cfg_path)
            print(f"Loaded LLM config: {llm_cfg_path}")

        self._cache_dir = os.path.join(self.workdir, "cache")
        self._fragments_dir = os.path.join(self._cache_dir, "fragments")
        os.makedirs(self._fragments_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # TOML helpers
    # ------------------------------------------------------------------

    def _load_toml(self, path: str) -> dict:
        with open(path, encoding="utf-8") as f:
            return toml.load(f)

    # ------------------------------------------------------------------
    # Manifest helpers
    # ------------------------------------------------------------------

    def _manifest_path(self) -> str:
        return os.path.join(self._cache_dir, "manifest.toml")

    def load_manifest(self) -> dict:
        p = self._manifest_path()
        if os.path.exists(p):
            try:
                return self._load_toml(p)
            except Exception as e:
                print(f"manifest.toml load failed: {e}")
        return {}

    def save_manifest(self, manifest: dict) -> None:
        p = self._manifest_path()
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            toml.dump(manifest, f)
        os.replace(tmp, p)
        print(f"manifest.toml saved: {p}")

    # ------------------------------------------------------------------
    # Stage output helpers (read/write manifest.stages.<name>.output)
    # ------------------------------------------------------------------

    def read_stage_output(self, name: str) -> Any:
        """Read a stage's structured output from manifest.toml.

        Stages are forbidden from passing data through function arguments;
        every stage's output must be persisted here and read back by
        downstream stages via this method.
        """
        manifest = self.load_manifest()
        return _nested_get(manifest, "stages", *name.split("."), "output")

    def write_stage_output(self, name: str, output: Any) -> None:
        """Persist a stage's structured output to manifest.toml."""
        manifest = self.load_manifest()
        _nested_set(manifest, output, "stages", *name.split("."), "output")
        self.save_manifest(manifest)

    # ------------------------------------------------------------------
    # Hash helpers
    # ------------------------------------------------------------------

    def compute_hash(self, *parts: str) -> str:
        combined = "\n---\n".join(parts)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Fragment helpers
    # ------------------------------------------------------------------

    def _fragment_abs(self, rel_path: str) -> str:
        return os.path.join(self._cache_dir, rel_path)

    def load_fragment(self, rel_path: str) -> str:
        with open(self._fragment_abs(rel_path), encoding="utf-8") as f:
            return f.read()

    def save_fragment(self, rel_path: str, code: str) -> None:
        abs_path = self._fragment_abs(rel_path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(code)

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    def llm_call(self, prompt: str, check: bool = False) -> str:
        from easytrans_client import EasyTransError

        print(f"Prompt length: {len(prompt)} chars")
        response = self.llm_client.responses(input_text=prompt, model=self.model)
        if not self.llm_client.validate_response(response):
            raise EasyTransError("LLM response validation failed")
        code = self.llm_client.extract_content(response)
        if not code:
            raise EasyTransError("LLM returned empty content")
        code = _strip_fences(code)

        if check:
            check_prompt = (
                f"{prompt}\nPlease strictly check the following code for compliance with all "
                f"the rules mentioned in the prompt. If any rule is violated, regenerate the "
                f"code until it fully complies with all the rules.\nCode:\n{code}"
            )
            resp2 = self.llm_client.responses(input_text=check_prompt, model=self.model_check)
            code = _strip_fences(self.llm_client.extract_content(resp2))

        return code

    # ------------------------------------------------------------------
    # Stage runners
    # ------------------------------------------------------------------

    def _human_prompt_for(self, stage_name: str) -> str:
        return self.llm_config.get("stages", {}).get(stage_name, {}).get("human_prompt", "")

    def run_llm_stage(
        self,
        name: str,
        hash_inputs: List[str],
        build_prompt_fn: Callable[[], str],
        fragment_name: str,
        check: bool = False,
    ) -> str:
        """
        Run an LLM stage with manifest-based caching.

        name          — stage key in manifest (e.g. "data_load", "resonance_calculation.BW_BW")
        hash_inputs   — list of strings that determine cache validity (stripped config, templates…)
        build_prompt_fn — called only on cache miss; returns the full prompt string
        fragment_name — relative path under cache/ (e.g. "fragments/data_load.py")
        check         — whether to run a second LLM check pass
        """
        human_prompt = self._human_prompt_for(name)
        current_hash = self.compute_hash(*hash_inputs, human_prompt)

        manifest = self.load_manifest()
        stage_meta = _nested_get(manifest, "stages", *name.split("."))

        if stage_meta and stage_meta.get("input_schema_hash") == current_hash:
            frag_path = stage_meta.get("fragment", fragment_name)
            if os.path.exists(self._fragment_abs(frag_path)):
                print(f"Cache hit: {name}")
                return self.load_fragment(frag_path)

        print(f"Cache miss: {name} — calling LLM")
        prompt = build_prompt_fn()
        if human_prompt:
            prompt = prompt + "\n\n" + human_prompt
        code = self.llm_call(prompt, check=check)
        time.sleep(1)

        self.save_fragment(fragment_name, code)
        manifest = self.load_manifest()
        _nested_set(manifest, {"input_schema_hash": current_hash, "fragment": fragment_name},
                    "stages", *name.split("."))
        self.save_manifest(manifest)
        return code

    def run_python_stage(
        self,
        name: str,
        fn: Callable[[], str],
        hash_inputs: List[str],
        fragment_name: str,
    ) -> str:
        """
        Run a pure-Python stage with manifest-based caching.

        fn must return the code string to write as the fragment.
        """
        current_hash = self.compute_hash(*hash_inputs)

        manifest = self.load_manifest()
        stage_meta = _nested_get(manifest, "stages", *name.split("."))

        if stage_meta and stage_meta.get("input_schema_hash") == current_hash:
            frag_path = stage_meta.get("fragment", fragment_name)
            if os.path.exists(self._fragment_abs(frag_path)):
                print(f"Cache hit: {name}")
                return self.load_fragment(frag_path)

        print(f"Running python stage: {name}")
        code = fn()

        self.save_fragment(fragment_name, code)
        manifest = self.load_manifest()
        _nested_set(manifest, {"input_schema_hash": current_hash, "fragment": fragment_name},
                    "stages", *name.split("."))
        self.save_manifest(manifest)
        return code

    # ------------------------------------------------------------------
    # Config helpers (shared across modes)
    # ------------------------------------------------------------------

    def get_all_resonance_data(self):
        sbc_list: list = []
        amp_list: list = []
        for resonance in self.raw_config["resonances"].values():
            for prop in resonance.get("propagators", {}).values():
                if "Sbc" in prop:
                    sbc_list.append(prop["Sbc"])
            if resonance.get("kind") == "shared_state":
                for amp_entry in resonance.get("amplitudes", []):
                    if "AMP" in amp_entry:
                        amp_list.append(amp_entry["AMP"])
                    if "B_Sbc" in amp_entry:
                        sbc_list.append(amp_entry["B_Sbc"])
            else:
                if "AMP" in resonance.get("Amplitude", {}):
                    amp_list.append(resonance["Amplitude"]["AMP"])
        return list(dict.fromkeys(sbc_list)), list(dict.fromkeys(amp_list))

    def get_draw_extra_sbc(self) -> List[str]:
        return self.raw_config.get("draw", {}).get("extra_sbc", [])

    def print_config_summary(self) -> None:
        print("Config summary:")
        print("=" * 40)
        for name, cfg in self.raw_config.get("resonances", {}).items():
            kind = cfg.get("kind", "normal")
            print(f"  {name} [{kind}]")
            for prop_name, prop_cfg in cfg.get("propagators", {}).items():
                print(f"    - {prop_name}: {prop_cfg.get('propagator_type', 'unknown')}")
            if kind == "shared_state":
                for amp_entry in cfg.get("amplitudes", []):
                    print(f"    - amplitude: {amp_entry.get('name')} "
                          f"AMP={amp_entry.get('AMP')} B_Sbc={amp_entry.get('B_Sbc')}")

    def save_code(self, code: str, output_path: Optional[str] = None) -> str:
        if output_path is None:
            output_path = os.path.join(self.workdir, "run", "generated_script.py")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(code)
        print(f"Script saved: {output_path}")
        return output_path


# ---------------------------------------------------------------------------
# Private utilities
# ---------------------------------------------------------------------------

def _strip_fences(code: str) -> str:
    code = re.sub(r"^```\w*\n?", "", code.strip())
    code = re.sub(r"\n?```$", "", code.strip())
    return code


def _nested_get(d: dict, *keys: str):
    """Return d[k1][k2]... or None if any key is missing."""
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def _nested_set(d: dict, value: Any, *keys: str) -> None:
    """Set d[k1][k2]... = value, creating intermediate dicts as needed."""
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value
