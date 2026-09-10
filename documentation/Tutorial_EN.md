# LLMPWA Tutorial

This tutorial uses `analyses/kk_dis` as the example: configure the environment, run the
`agent/` code-generation pipeline, and run the generated distributed fit. For the full
stage list and runtime contracts, see
[`analyses/kk_dis/README.md`](../analyses/kk_dis/README.md).

## 1. Environment setup

### 1.1 Generator dependencies

The code **generator** only needs three third-party packages:

```bash
pip install -r requirements.txt
```

That is `openai`, `python-dotenv`, and `toml`.

### 1.2 Dependencies to run the generated code (optional)

To actually **run** the fit after generation, install the physics/numeric stack from the
`runtime` extra in `pyproject.toml`:

```bash
pip install -e ".[runtime]"
```

and set up a CUDA + JAX GPU environment (`jax[cuda...]`), which the later steps use.

### 1.3 Configure the LLM API

```bash
export EASYTRANS_API_KEY="your_api_key_here"
# optional: these are the defaults
export EASYTRANS_BASE_URL="https://api.easytransnote.com/v1"
export EASYTRANS_MODEL="gemini-2.5-pro"
```

You can also put these in a `.env` file at the repo root (`agent/` reads it via
`load_dotenv()`).

## 2. Run the code-generation pipeline

From the repo root:

```bash
# static checks only (no LLM calls)
python -m agent.cli --workdir analyses/kk_dis --config llm_config_fit.toml --check-only

# run the full pipeline
python -m agent.cli --workdir analyses/kk_dis --config llm_config_fit.toml

# re-run a single stage
python -m agent.cli --workdir analyses/kk_dis --config llm_config_fit.toml --stage generate_fit_script
```

Pipeline stages (in `llm_config_fit.toml` order):

| stage | kind | output |
|-------|------|--------|
| `config_strip` | python | `run/free_params.toml`, `gen/fragments/stripped_config.toml`, ... |
| `inspect_data_shapes` | agent | shape report + `load_data`/`normalize_data`/`shard_data_distributed` fragments |
| `make_initial_args` / `save_results` | python | initial-args and save helper functions |
| `assemble_help_functions` | python | `run/base_functions.py` |
| `classification` | llm | `gen/fragments/classification.json` |
| `generate_likelihood` | agent | `run/likelihood_function.py` (requires approval) |
| `generate_fit_script` | agent | `run/fit_script.py` (requires approval) |

Notes:

- `inspect_data_shapes` defaults to `cache = true`, so it is reused when the config/data
  have not changed.
- `generate_likelihood` / `generate_fit_script` are interactive and need human review; if
  rejected, the Agent can revise and call `task.finish` again.

## 3. Run the generated fit (single node)

```bash
cd analyses/kk_dis
python run/fit_script.py
```

`JAX_NUM_PROCESSES` defaults to 1 in single-process mode, so `jax.distributed.initialize`
is not forced.

## 4. Run the generated fit (multinode Docker)

Node/container configuration lives in `node_config.toml` (header IP/PORT, worker list,
`container.script = "run/fit_script.py"`, ...).

```bash
# from the repo root
./docker/run_fit_multinode.sh --workdir analyses/kk_dis
# or explicitly
./docker/run_fit_multinode.sh --workdir analyses/kk_dis --config analyses/kk_dis/node_config.toml
```

The script:

1. rsyncs the workdir to each node and mounts it as the container workspace;
2. injects `JAX_COORDINATOR_ADDRESS` / `JAX_NUM_PROCESSES` / `JAX_PROCESS_ID` and NCCL
   environment;
3. runs the same `run/fit_script.py` on every node;
4. copies the header node's `output` back into the local workdir.

## 5. Inspect and visualize the pipeline

```bash
python agent/pipeline_state.py -w analyses/kk_dis -c llm_config_fit.toml -o gen/pipeline_state.json
python agent/pipeline_viz.py -w analyses/kk_dis -c llm_config_fit.toml -o gen/pipeline_report
```

See [pipeline_state_exporter.md](pipeline_state_exporter.md) and
[pipeline_visualization_plan.md](pipeline_visualization_plan.md).

## 6. Create a new analysis

Copy `analyses/kk_dis`, replace `resonances_config.toml` and `data/`, adjust
`gen/prompts/` as needed, then run the same engine command to reuse the whole flow.
