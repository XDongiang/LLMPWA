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
        self._prompts_dir = os.path.join(self._cache_dir, "prompts")
        os.makedirs(self._fragments_dir, exist_ok=True)
        os.makedirs(self._prompts_dir, exist_ok=True)

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
            f.write(_dump_manifest(manifest))
        os.replace(tmp, p)
        print(f"manifest.toml saved: {p}")

    # ------------------------------------------------------------------
    # Stage output helpers (read/write manifest.stages.<name>.output)
    # ------------------------------------------------------------------

    def read_stage_output(self, name: str, field: str = "output") -> Any:
        """Read a stage field from manifest.toml.

        *field* defaults to ``"output"`` for normal stages. For stages that
        store multiple file refs as sibling fields (e.g. config_strip), pass
        the field name directly (e.g. ``field="free_params"``).

        A ``{"type": "file", "path": "..."}`` value is transparently loaded
        from disk. Legacy ``{"_fragment_ref": "..."}`` is also supported.
        """
        manifest = self.load_manifest()
        value = _nested_get(manifest, "stages", *name.split("."), field)
        if isinstance(value, dict):
            if value.get("type") == "file" and "path" in value:
                ref = value["path"]
                abs_path = self._fragment_abs(ref)
                if ref.endswith(".json"):
                    with open(abs_path, encoding="utf-8") as f:
                        return json.load(f)
                return self._load_toml(abs_path)
            if "_fragment_ref" in value:
                ref = value["_fragment_ref"]
                abs_path = self._fragment_abs(ref)
                if ref.endswith(".json"):
                    with open(abs_path, encoding="utf-8") as f:
                        return json.load(f)
                return self._load_toml(abs_path)
        return value

    def write_stage_field(self, name: str, field: str, output: Any,
                          fragment_path: Optional[str] = None) -> None:
        """Write an arbitrary field into a stage's manifest node.

        If *fragment_path* is given the data is serialised to disk and a
        ``{"type": "file", "path": path}`` pointer is stored under *field*.
        Otherwise *output* is stored inline.
        """
        if fragment_path is not None:
            abs_path = self._fragment_abs(fragment_path)
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            tmp = abs_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                toml.dump(output, f)
            os.replace(tmp, abs_path)
            stored: Any = {"type": "file", "path": fragment_path}
        else:
            stored = output

        manifest = self.load_manifest()
        # ensure the stage node has input_schema_hash (stage0 uses "")
        stage_node = _nested_get(manifest, "stages", *name.split(".")) or {}
        if "input_schema_hash" not in stage_node:
            _nested_set(manifest, "", "stages", *name.split("."), "input_schema_hash")
        _nested_set(manifest, stored, "stages", *name.split("."), field)
        self.save_manifest(manifest)

    def write_stage_output(self, name: str, output: Any,
                           fragment_path: Optional[str] = None) -> None:
        """Shorthand for write_stage_field(..., field='output')."""
        self.write_stage_field(name, "output", output, fragment_path=fragment_path)

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
        stages = self.llm_config.get("stages", {})
        # Exact match first, then fall back to parent key (e.g. "resonance_calculation" for "resonance_calculation.BW_BW")
        if stage_name in stages and "human_prompt" in stages[stage_name]:
            return stages[stage_name]["human_prompt"]
        parent = stage_name.rsplit(".", 1)[0] if "." in stage_name else ""
        if parent in stages:
            return stages[parent].get("human_prompt", "")
        return ""

    def resolve_prompt(self, template: str) -> str:
        """Replace {stages.A.B.C} placeholders with values from manifest.

        Resolution rules:
        - {stages.X.fragment} — load the fragment file at that path and insert its content
        - {stages.X.output} or deeper — resolve output (following _fragment_ref if present),
          then JSON-serialize the result (or sub-key if path goes deeper)
        - Any other {stages.*} path — look up the raw manifest value and str() it
        - Non-stages.* placeholders are left untouched (caller handles them)
        """
        import re

        manifest = self.load_manifest()

        def _replace(m: re.Match) -> str:
            key_path = m.group(1)  # e.g. "stages.data_load.fragment"
            if not key_path.startswith("stages."):
                return m.group(0)  # leave non-stages placeholders alone
            parts = key_path.split(".")  # ["stages", "data_load", "fragment"]
            # Special case: last segment is "fragment" — load the file content
            if parts[-1] == "fragment":
                val = _nested_get(manifest, *parts)
                if val and isinstance(val, str):
                    abs_path = self._fragment_abs(val)
                    if os.path.exists(abs_path):
                        return self.load_fragment(val)
                return m.group(0)
            # "output" anywhere in path — resolve output then optionally descend
            if "output" in parts:
                out_idx = parts.index("output")
                stage_parts = parts[1:out_idx]  # keys between "stages" and "output"
                sub_parts = parts[out_idx + 1:]  # keys after "output"
                raw_out = _nested_get(manifest, "stages", *stage_parts, "output")
                if isinstance(raw_out, dict) and raw_out.get("type") == "file" and "path" in raw_out:
                    ref = raw_out["path"]
                    abs_path = self._fragment_abs(ref)
                    if ref.endswith(".json"):
                        import json as _json
                        with open(abs_path, encoding="utf-8") as f:
                            raw_out = _json.load(f)
                    else:
                        raw_out = self._load_toml(abs_path)
                elif isinstance(raw_out, dict) and "_fragment_ref" in raw_out:
                    ref = raw_out["_fragment_ref"]
                    abs_path = self._fragment_abs(ref)
                    if ref.endswith(".json"):
                        import json as _json
                        with open(abs_path, encoding="utf-8") as f:
                            raw_out = _json.load(f)
                    else:
                        raw_out = self._load_toml(abs_path)
                for sp in sub_parts:
                    if isinstance(raw_out, dict):
                        raw_out = raw_out.get(sp)
                    else:
                        raw_out = None
                        break
                if raw_out is None:
                    return m.group(0)
                if isinstance(raw_out, (dict, list)):
                    import json as _json
                    return _json.dumps(raw_out, indent=2, ensure_ascii=False)
                return str(raw_out)
            # Generic path lookup
            val = _nested_get(manifest, *parts)
            if val is None:
                return m.group(0)
            if isinstance(val, (dict, list)):
                import json as _json
                return _json.dumps(val, indent=2, ensure_ascii=False)
            return str(val)

        return re.sub(r"\{(stages\.[^}]+)\}", _replace, template)

    def write_stage_meta(
        self,
        name: str,
        fragment_name: str,
        input_schema_hash: str,
        prompt_log_rel: str,
    ) -> None:
        """Write stage metadata (hash, output, prompt_log) into manifest."""
        manifest = self.load_manifest()
        _nested_set(
            manifest,
            {
                "input_schema_hash": input_schema_hash,
                "output": {"type": "file", "path": fragment_name},
                "prompt_log": {"type": "file", "path": prompt_log_rel},
            },
            "stages",
            *name.split("."),
        )
        self.save_manifest(manifest)

    def run_llm_stage(
        self,
        name: str,
        hash_inputs: List[str],
        prompt: str,
        fragment_name: str,
        check: bool = False,
    ) -> str:
        """Run an LLM stage with manifest-based caching.

        name          — stage key in manifest (e.g. "data_load", "resonance_calculation.BW_BW")
        hash_inputs   — list of strings that determine cache validity
        prompt        — fully constructed prompt string (caller handles human_prompt injection)
        fragment_name — relative path under cache/ (e.g. "fragments/data_load.py")
        check         — whether to run a second LLM check pass
        """
        current_hash = self.compute_hash(*hash_inputs)

        manifest = self.load_manifest()
        stage_meta = _nested_get(manifest, "stages", *name.split("."))

        if stage_meta and stage_meta.get("input_schema_hash") == current_hash:
            out = stage_meta.get("output")
            if isinstance(out, dict) and out.get("type") == "file":
                frag_path = out["path"]
            else:
                frag_path = stage_meta.get("fragment", fragment_name)
            if os.path.exists(self._fragment_abs(frag_path)):
                print(f"Cache hit: {name}")
                return self.load_fragment(frag_path)

        print(f"Cache miss: {name} — calling LLM")
        code = self.llm_call(prompt, check=check)
        time.sleep(1)

        self.save_fragment(fragment_name, code)

        prompt_log_rel = f"prompts/{name.replace('.', '_')}.md"
        prompt_log_abs = os.path.join(self._cache_dir, prompt_log_rel)
        os.makedirs(os.path.dirname(prompt_log_abs), exist_ok=True)
        with open(prompt_log_abs, "w", encoding="utf-8") as f:
            f.write(f"# Stage: {name}\n\n")
            f.write("## Prompt\n\n")
            f.write("```\n")
            f.write(prompt)
            f.write("\n```\n\n")
            f.write("## Response\n\n")
            f.write("```python\n")
            f.write(code)
            f.write("\n```\n")

        self.write_stage_meta(name, fragment_name, current_hash, prompt_log_rel)
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
            out = stage_meta.get("output")
            if isinstance(out, dict) and out.get("type") == "file":
                frag_path = out["path"]
            else:
                frag_path = stage_meta.get("fragment", fragment_name)
            if os.path.exists(self._fragment_abs(frag_path)):
                print(f"Cache hit: {name}")
                return self.load_fragment(frag_path)

        print(f"Running python stage: {name}")
        code = fn()

        self.save_fragment(fragment_name, code)
        manifest = self.load_manifest()
        _nested_set(
            manifest,
            {"input_schema_hash": current_hash, "output": {"type": "file", "path": fragment_name}},
            "stages", *name.split("."),
        )
        self.save_manifest(manifest)
        return code

    def build_stage_prompt(self, stage_name: str, template: str, **kwargs) -> str:
        """Construct a full prompt: resolve manifest refs, apply kwargs, append human_prompt.

        1. resolve_prompt handles {stages.X.Y} manifest references
        2. format_map applies caller-supplied kwargs (non-stages placeholders)
        3. human_prompt from llm_config_fit.toml is appended if present
        """
        prompt = self.resolve_prompt(template)
        if kwargs:
            prompt = prompt.format_map(kwargs)
        human = self._human_prompt_for(stage_name)
        if human:
            human = self.resolve_prompt(human)
            prompt = prompt + "\n\n" + human
        return prompt

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


def _toml_scalar(v: Any) -> str:
    """Serialize a scalar or list value to a TOML literal."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return str(v)
    if isinstance(v, list):
        items = ", ".join(_toml_scalar(i) for i in v)
        return f"[{items}]"
    escaped = str(v).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _toml_inline_table(d: dict) -> str:
    """Serialize a flat dict as a TOML inline table."""
    pairs = ", ".join(f"{k} = {_toml_scalar(v)}" for k, v in d.items())
    return "{ " + pairs + " }"


_STAGE_OPTIONAL_KEYS = {"output", "prompt_log", "free_params", "sbc_and_amp", "text"}


def _is_stage_node(d: dict) -> bool:
    """True if d is a stage metadata block (must have input_schema_hash)."""
    return "input_schema_hash" in d


def _emit_stage_block(lines: List[str], header: str, data: dict) -> None:
    lines.append(f"[{header}]")
    if "input_schema_hash" in data:
        lines.append(f"input_schema_hash = {_toml_scalar(data['input_schema_hash'])}")
    for k, v in data.items():
        if k == "input_schema_hash":
            continue
        if isinstance(v, dict):
            # inline table: values must be scalars or lists of scalars
            pairs = []
            for dk, dv in v.items():
                pairs.append(f"{dk} = {_toml_scalar(dv)}")
            lines.append(f"{k} = {{ {', '.join(pairs)} }}")
        else:
            lines.append(f"{k} = {_toml_scalar(v)}")
    lines.append("")


def _walk_stages(lines: List[str], prefix: str, node: dict) -> None:
    if _is_stage_node(node):
        # This node is a stage block — emit it directly, dict values as inline tables
        _emit_stage_block(lines, prefix, node)
    else:
        # This node is a grouping level (e.g. resonance_calculation) — recurse
        for k, v in node.items():
            if isinstance(v, dict):
                _walk_stages(lines, f"{prefix}.{k}", v)
            else:
                # scalar at grouping level — unlikely but handle gracefully
                lines.append(f"# {prefix}.{k} = {_toml_scalar(v)}")


def _dump_manifest(manifest: dict) -> str:
    """Serialize manifest to TOML with stages in order and output fields inlined."""
    lines: List[str] = []

    for k, v in manifest.items():
        if k == "stages":
            continue
        if isinstance(v, dict):
            lines.append(f"[{k}]")
            for ik, iv in v.items():
                lines.append(f"{ik} = {_toml_scalar(iv)}")
            lines.append("")
        else:
            lines.append(f"{k} = {_toml_scalar(v)}")

    for stage_name, stage_data in manifest.get("stages", {}).items():
        if isinstance(stage_data, dict):
            _walk_stages(lines, f"stages.{stage_name}", stage_data)

    return "\n".join(lines)
