#!/usr/bin/env python3
"""
llm_code_generator.py — thin entry point for the LLMPWA code generator.

Parses CLI args, dispatches to FitGenerator / DrawGenerator / PlotGenerator
based on --mode. All real logic lives in agent/generator_*.py.
"""

import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="LLM-driven PWA resonance code generator"
    )
    parser.add_argument("--workdir", default=".",
                        help="Analysis directory containing resonances_config.toml")
    parser.add_argument("--config", default=None,
                        help="TOML config path (default: workdir/resonances_config.toml)")
    parser.add_argument("--model", default=os.getenv("EASYTRANS_MODEL", "gemini-2.5-pro"))
    parser.add_argument("--model-check",
                        default=os.getenv("EASYTRANS_MODEL_CHECK",
                                          os.getenv("EASYTRANS_MODEL", "gemini-2.5-pro")))
    parser.add_argument("--output", default=None,
                        help="Output script path")
    parser.add_argument("--mode", default="fit", choices=["fit", "draw", "plot"],
                        help="Generation mode: 'fit' for fit script, 'draw' for draw "
                             "weight script, 'plot' for draw plot script")
    args = parser.parse_args()

    if args.mode == "fit":
        from generator_fit import FitGenerator
        gen = FitGenerator(
            workdir=args.workdir,
            model=args.model,
            model_check=args.model_check,
            config_path=args.config,
        )
        gen.run(output_path=args.output)
    elif args.mode == "draw":
        from generator_draw import DrawGenerator
        gen = DrawGenerator(
            workdir=args.workdir,
            model=args.model,
            model_check=args.model_check,
            config_path=args.config,
        )
        gen.run(output_path=args.output)
    elif args.mode == "plot":
        from generator_plot import PlotGenerator
        gen = PlotGenerator(
            workdir=args.workdir,
            model=args.model,
            model_check=args.model_check,
            config_path=args.config,
        )
        gen.run(output_path=args.output)


if __name__ == "__main__":
    main()
