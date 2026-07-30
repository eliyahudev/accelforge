# Codebase Map

Use this reference for source edits, API changes, and implementation navigation.

## Package Layout

- `accelforge/__init__.py`: public imports and `__all__`.
- `accelforge/frontend/`: user-facing spec model classes.
- `accelforge/frontend/arch/`: architecture nodes, components, constraints, spatial fanouts, flattening.
- `accelforge/frontend/workload.py`: Workload, Einsum, TensorAccess, projections, workload rendering.
- `accelforge/frontend/mapping/`: LoopTree mapping model.
- `accelforge/frontend/mapper/`: `FFM` config and mapper metrics config.
- `accelforge/frontend/spec.py`: top-level `Spec`, expression evaluation orchestration, component cost calculation, mapping/evaluation methods.
- `accelforge/model/`: evaluates mappings and runs the analytical model.
- `accelforge/model/_looptree/`: LoopTree analysis, capacity, accesses, energy, latency, reuse, distributed binding, visualization.
- `accelforge/mapper/FFM/`: Fast and Fusiest Mapper implementation.
- `accelforge/plotting/`: plotting helpers for specs, mappings, rooflines, and ski slopes.
- `accelforge/util/`: YAML parsing, expression evaluation, set expressions, ISL, permutations, base types, parallel helpers.

## Public API Pattern

Most public inputs are Pydantic-like `EvalableModel` classes with:

- typed fields and docstrings used by docs,
- `from_yaml()` / `to_yaml()` inherited or implemented locally,
- expression-evaluation hooks through `_eval_expressions`,
- YAML tag support through shared serialization infrastructure.

Prefer editing the relevant model class and shared helper APIs rather than adding caller-side special cases.

## Spec Lifecycle

`Spec._spec_eval_expressions()` evaluates in this order:

1. variables
2. renames
3. workload
4. workload-derived empty or per-Einsum renames
5. architecture
6. remaining spec fields

This order matters because architecture expressions may depend on spec variables and workload symbols. If architecture expressions need workload-specific symbols, pass an `einsum_name` into cost/flattening paths.

`Spec.calculate_component_costs()`:

- loads HWComponents models from `config.component_models`,
- includes installed component models when `config.use_installed_component_models` is true,
- evaluates expressions if needed,
- calculates area, dynamic action energy, action throughput, leak power,
- writes calculated values back onto the original architecture nodes, accounting for spatial fanout totals.

`Spec.map_workload_to_arch()` delegates to `accelforge.mapper.FFM.main.map_workload_to_arch()`.

`Spec.evaluate_mapping()` delegates to `accelforge.model.evaluate_mapping()`.

## Parsing And YAML

Do not parse AccelForge YAML with ad hoc string code. The project supports:

- YAML tags such as `!Memory`, `!Compute`, `!Toll`, `!Fork`, LoopTree tags, and custom model tags.
- extended YAML loading and file composition through `accelforge.util._yaml`.
- Jinja parsing hooks and `from_yaml(*files, jinja_parse_data=None, top_key=None, **kwargs)`.
- expression evaluation through `accelforge.util._eval_expressions`.
- set expressions through `accelforge.util._setexpressions`.

When changing YAML behavior, inspect existing tests under `tests/vibe_see_readme_in_this_dir/`, `tests/input_files/`, and examples.

For workload concise Einsum notation, start in `accelforge/frontend/workload.py`:

- `_parse_einsum_entry()`: combines an `einsum:` string with extra entry fields.
- `_parse_einsum_string()`: parses `Y[...] = A[...] * B[...]` strings.
- `_parse_projection()`: handles shorthand projections and explicit rank-name projections such as `K[b, M: p, h, e]`.
- `_projection_factory()` and `TensorAccess.model_post_init()`: convert list projections to implied uppercase-rank dictionaries while preserving explicit dictionary projections.

## HWComponents Integration

`Config` has:

- `expression_custom_functions`: functions or Python files that add expression functions.
- `component_models`: paths, modules, lists, or `hwcomponents.ComponentModel` classes.
- `use_installed_component_models`: defaults true.

Components cache and call HWComponents models for area, action energy, action throughput, and leak power. Current code checks that `hwcomponents.get_action_cost` exists.

Use `throughput` APIs. Deprecated `latency` fields still exist for migration compatibility and emit warnings.

## Common Edit Guidelines

- Keep field docstrings accurate; docs can include them directly.
- Preserve backwards-compatible YAML where existing examples/tests imply it.
- Use structured model validation for input constraints.
- Add tests around parser/evaluator changes because error paths can be subtle.
- For performance-sensitive mapper/model paths, avoid broad refactors and measure with focused tests or representative examples.
