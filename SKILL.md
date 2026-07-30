---
name: accelforge
description: Work effectively in the AccelForge project, a Python framework for modeling, mapping, plotting, and designing tensor algebra and deep neural network accelerators. Use when Codex needs to edit AccelForge source code, YAML architecture/workload/mapping specs, examples, notebooks, Sphinx docs, tests, mapper/model flows, component cost modeling, HWComponents integration, or migration from Timeloop-style accelerator models.
---

# AccelForge

AccelForge is a Python package for accelerator architecture modeling. It centers on YAML/Python specs for architectures, workloads, mappings, variables, renames, mapper config, and model config.

## Workflow

1. Locate the repo root and read local files before relying on memory. Prefer local `docs/source/`, `examples/`, and `tests/` over hosted docs when working in a checkout.
2. Identify the task area:
   - YAML specs, examples, or user modeling workflows: read `references/specs-and-workflows.md`.
   - Python implementation, APIs, or validation behavior: read `references/codebase-map.md`.
   - Tests, docs, packaging, or contribution changes: read `references/development.md`.
3. Preserve existing project patterns. Most user-facing input objects are Pydantic-style models under `accelforge/frontend/`; YAML loading and expression evaluation are shared infrastructure, not one-off parsing code.
4. Validate narrow changes with focused tests first, then broader tests if the touched code affects shared parsing, mapping, modeling, or public APIs.

## Project Facts

- Public entry points are exported from `accelforge/__init__.py`: `Spec`, `Arch`, `Workload`, `Mapping`, `Renames`, `Variables`, `Config`, `Metrics`, `set_n_parallel_jobs`, and examples.
- `Spec` is the top-level object. It owns `arch`, `workload`, `mapping`, `variables`, `renames`, `config`, `mapper`, and `model`.
- The primary workflow is: define architecture, define workload, map workload to architecture, analyze energy/area/latency.
- Component area, leak power, per-action energy, and throughput come from HWComponents models. Current code prefers `throughput`; legacy `latency` fields are deprecated and auto-converted.
- Workloads are cascades of Einsums. They can use verbose tensor-access dictionaries or concise string notation such as `Y[m, n] = A[m, k] * B[k, n]`.
- Mappings use LoopTree notation with loop, storage, compute, nested, and split concepts.

## Validation

Use the smallest command that covers the change:

```bash
pytest tests/path/to/test_file.py
pytest tests/vibe_see_readme_in_this_dir/path/to/test.py
pytest tests/test_model.py tests/test_mapper.py
pytest
```

For docs-only changes, build or inspect Sphinx source when practical. For notebook-facing behavior, inspect tutorial notebooks and run relevant notebook tests only when dependencies are available.

## Cautions

- Do not hand-roll YAML parsing with string manipulation; AccelForge uses `ruamel.yaml`, Pydantic models, custom YAML tags, includes, and expression evaluation helpers.
- Do not treat `tests/vibe_see_readme_in_this_dir/` failures as automatically authoritative. Its README says those tests were LLM-written with minimal oversight; inspect and fix/delete nonsensical tests when appropriate, while keeping PRs green.
- Be careful changing mapper defaults. Mapper search controls directly affect runtime, memory, optimality, and reproducibility.
- Keep examples and docs aligned with source model fields; Sphinx docs often include live attributes from code.
