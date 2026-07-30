# Development

Use this reference for tests, docs, packaging, and contribution-style changes.

## Environment

Package metadata lives in `pyproject.toml`. AccelForge requires Python `>=3.8` and depends on scientific/Pydantic/YAML/ISL/HWComponents packages including:

- `numpy`, `pandas`, `sympy`, `symengine`
- `pydantic`, `pydantic_core`
- `ruamel.yaml`, `jinja2`
- `islpy-barvinok`
- `matplotlib`, `plotly`, `pydot`
- `hwcomponents`, `hwcomponents-adc`, `hwcomponents-cacti`, `hwcomponents-library`, `hwcomponents-neurosim`

Development extras include `pytest`, `pytest-cov`, `black`, `flake8`, `mypy`, `pydocstyle`, `nbconvert`, `nbformat`, and `ipykernel`.

## Tests

Use focused pytest runs:

```bash
pytest tests/test_model.py
pytest tests/test_mapper.py
pytest tests/test_plotting.py
pytest tests/test_tracegen.py
pytest tests/vibe_see_readme_in_this_dir/test_workload_parsing.py
pytest tests/vibe_see_readme_in_this_dir/values_per_action
pytest tests/network
```

Run full `pytest` only when the change is broad or before final confidence on shared behavior.

`tests/vibe_see_readme_in_this_dir/README.md` says those tests were written by an LLM with minimal human oversight. If one fails, inspect whether the test is meaningful; changing/deleting bad tests is acceptable, but PRs should still pass all tests and mention test changes.

## Docs

Docs are Sphinx sources under `docs/source/`.

Important pages:

- `index.rst`: top-level docs entry.
- `guide/guide.rst`: installation, examples, workflow, support, migration from Timeloop.
- `guide/spec*.rst`: input specifications.
- `guide/modeling*.rst`: modeling workflow and assumptions.
- `guide/parsing/*.rst`: expression and YAML parsing.
- `modules.rst`: generated API reference landing page.

The Makefile `generate-docs` target removes old generated API docs, runs `sphinx-apidoc`, then `sphinx-autobuild`. Inspect indentation before using the target because Makefile recipes require tabs.

## Examples And Notebooks

Examples are part of the project contract:

- `examples/arches/`: simple, TPU, Eyeriss, NVDLA, Simba, Snowcat, compute-in-memory, fanout variations.
- `examples/workloads/basic/`: matmuls/matvecs and annotated examples.
- `examples/workloads/transformers/`: GPT, Llama, Qwen, Mixtral, DeepSeek, Mamba, attention/feedforward fragments.
- `examples/workloads/compute_in_memory/`: CNNs and transformer workloads.
- `examples/mappings/`: fused/unfused LoopTree examples.
- `notebooks/tutorials/`: component energy/area, mapper, model, DSE, memory tradeoff tutorials.

When changing example syntax, update docs and tests that load those examples.

## Style

- Use existing Pydantic model style and typed fields.
- Keep user-visible errors specific and include field paths when possible.
- Prefer `rg` for code search.
- Format Python with `black .` if the edit touches enough Python to justify it.
- Avoid unrelated modernization; this repo has active experimental areas in `tests/not_working/` and deprecated plotting/mapper subfolders.
