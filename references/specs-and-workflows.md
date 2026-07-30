# Specs And Workflows

Use this reference for AccelForge YAML/Python specs, examples, and modeling workflows.

## Main Workflow

AccelForge models energy, area, and latency for accelerator architectures running tensor algebra workloads:

1. Define an architecture: hardware components, hierarchy/branches, spatial fanouts, actions, energy/area/leak/throughput, tensor-holding behavior, constraints, and HWComponents model data.
2. Define a workload: a cascade of Einsums with tensor projections, outputs, persistent tensors, rank sizes, and optional renames.
3. Map the workload to the architecture with the Fast and Fusiest Mapper (`FFM`) or provide an explicit LoopTree mapping.
4. Evaluate or inspect results: component costs, mapping metrics, plots, roofline/ski-slope visualizations, and mapping renderings.

## Where To Look

- `docs/source/guide/spec.rst`: top-level `Spec` overview and parsing docs links.
- `docs/source/guide/spec/architecture.rst`: architecture hierarchy, flattening, forks, spatial fanouts, tensor holders.
- `docs/source/guide/spec/workload.rst`: workload/Einsum syntax, tensor projections, renames, concise notation.
- `docs/source/guide/spec/mapping.rst`: LoopTree mapping concepts and YAML tags.
- `docs/source/guide/parsing/evaluation.rst`: expression evaluation rules.
- `docs/source/guide/parsing/yaml_parsing.rst`: extended YAML syntax.
- `docs/source/guide/modeling/*.rst`: component costs, accelerator metrics, assumptions, mapper behavior.
- `examples/arches/`, `examples/workloads/`, `examples/mappings/`: canonical examples.
- `notebooks/tutorials/`: user-facing workflows.

## Top-Level Spec

`accelforge.frontend.spec.Spec` includes:

- `arch`: hardware architecture.
- `workload`: program/cascade of Einsums.
- `mapping`: explicit mapping. Leave empty if mapper should generate mappings.
- `variables`: spec-level variables referenced by expressions.
- `renames`: aliases for tensors/rank variables, often canonical names like `input`, `output`, `weight`.
- `config`: component models and custom expression functions.
- `mapper`: `FFM` mapper configuration.
- `model`: metrics to evaluate.

Common Python flow:

```python
from accelforge import Spec

spec = Spec.from_yaml("arch.yaml", "workload.yaml")
spec = spec.calculate_component_costs()
mappings = spec.map_workload_to_arch()
# or, with explicit mapping in the spec:
results = spec.evaluate_mapping()
```

Confirm exact return shapes in the local source/tests before writing new user-facing code.

## Architecture Specs

Architectures are trees. A flattened architecture is a hierarchy ending in a `!Compute`.

Supported component families include:

- `!Memory`: stores/reuses tensors.
- `!Toll`: charges for non-compute data movement/transforms such as quantization or transfer.
- `!Compute`: performs Einsum computation.
- `!Container`: groups spatial fanout without being a concrete memory/compute component.
- `!Fork`: creates alternate compute paths.

Architecture names must be unique where flattening/modeling requires lookup by name. Branches are flattened into root-to-compute paths, and each Einsum runs on one compute node at a time.

Use `throughput` and `throughput_scale` on actions and components. `latency` and `latency_scale` are deprecated compatibility fields.

## Workload Specs

Workloads are cascades of Einsums. A tensor access has:

- `name`
- `projection`: list form implies uppercase rank names from lowercase rank variables, e.g. `[m, n]` means `{M: m, N: n}`.
- `output`
- `persistent`
- `backing_storage_size_scale`
- `bits_per_value`

Use dictionary projections for nontrivial rank names or expressions, e.g. `{M: m, N2: n2, C: a+b}`. List projections must use lowercase rank-variable identifiers.

Concise notation is supported for common cases:

```yaml
workload:
  einsums:
  - Y[m, n] = A[m, k] * B[k, n]
  - "QK[b, m, p, h] = Q[b, m, h, e] * K[b, M: p, h, e]"
```

Use the `einsum:` key when concise notation needs extra fields:

```yaml
- einsum: I[b, m, d] = I_in[b, m, d]
  is_copy_operation: True
  tensor_accesses: [{name: I_in, bits_per_value: 16}]
  renames: {input: I_in, output: I}
```

## Renames

Renames let specs and set expressions refer to canonical names rather than workload-specific tensor/rank names. They may be dictionaries or list entries with `name`, `source`, and optional `expected_count`.

Top-level renames can include `default`, which applies to every Einsum unless individual rename names are overridden.

## Mapping Specs

Mappings use LoopTree notation:

- Loop nodes: nested loops over rank variables.
- Storage nodes: tensor tile storage/reuse.
- Compute nodes: Einsum compute on a component.
- Splits: sequential branches for fused workloads.
- `!Nested`: YAML helper for lists of subsequent nodes.

Key interpretation rule from the docs: moving storage nodes lower in the LoopTree keeps tile size, reuse, and lifetime the same or decreases them.

## Mapper Configuration

The mapper config class is `accelforge.frontend.mapper.FFM`. Important knobs include:

- `metrics` and `info_metrics`
- `force_memory_hierarchy_order`
- `tiling_coarseness`
- `max_fused_loops_per_rank_variable`, `max_fused_loops`, `max_loops`
- `explore_loop_orders`
- `explore_imperfect_spatial_loops`
- `explore_imperfect_temporal_loops`
- `objective_tolerance`
- `resource_usage_tolerance`
- memory/time limits

Changing defaults can alter runtime, memory usage, optimality, and tests. Prefer explicit per-spec settings for examples or experiments.
