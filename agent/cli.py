"""
cli.py — 命令行入口。

用法：
  python -m agent.cli --workdir analyses/kk_new --config llm_config_fit.toml
  python -m agent.cli --workdir analyses/kk_new --config llm_config_fit.toml --check-only
  python -m agent.cli --workdir analyses/kk_new --config llm_config_fit.toml --stage classification
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLMPWA engine — LLM-driven code generation pipeline"
    )
    parser.add_argument(
        "--workdir", "-w", required=True,
        help="Analysis working directory (e.g. analyses/kk_new)"
    )
    parser.add_argument(
        "--config", "-c", default="llm_config_fit.toml",
        help="Config filename inside workdir (default: llm_config_fit.toml)"
    )
    parser.add_argument(
        "--check-only", action="store_true",
        help="Run static checks only, do not execute any stages"
    )
    parser.add_argument(
        "--stage", "-s", default=None,
        help="Run a single stage by name (e.g. classification)"
    )
    parser.add_argument(
        "--model", default=None,
        help="Override LLM model (e.g. gemini-2.5-pro)"
    )
    parser.add_argument(
        "--model-check", default=None,
        help="Override check-pass LLM model"
    )
    args = parser.parse_args()

    # 把 agent/ 加入 sys.path，让模块间 import 正常工作
    agent_dir = str(Path(__file__).parent.resolve())
    if agent_dir not in sys.path:
        sys.path.insert(0, agent_dir)

    from engine import Engine
    from stage_runner import StageRunner

    engine = Engine(
        workdir=args.workdir,
        config_name=args.config,
        model=args.model,
        model_check=args.model_check,
    )

    if args.check_only:
        engine.check()
        return

    if args.stage:
        stage_cfg = engine.config.get_stage(args.stage)
        if stage_cfg is None:
            print(f"Error: stage '{args.stage}' not found in config.")
            print(f"Available stages: {engine.config.stage_names_in_order()}")
            sys.exit(1)
        engine.validator.static_check()
        runner = StageRunner(args.stage, stage_cfg, engine)
        runner.execute()
        print(f"Stage '{args.stage}' completed.")
        return

    engine.run()


if __name__ == "__main__":
    main()
