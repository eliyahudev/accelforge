"""Python model of the channel-parallel output-stationary accelerator.

The physical data paths are intentionally strict::

    AOSRAM -> ActivationBuffer -> MAC
    WeightSRAM -> WeightBuffer -> MAC
    MAC -> OutputAccumulator -> PostProcess -> OLine -> AOSRAM

Architecture nodes are listed outer-memory to compute. Outputs traverse that
hierarchy in the opposite direction and bypass holders for other tensor roles.
All capacities are bits and all throughputs are actions/second.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import accelforge as af


@dataclass(frozen=True)
class MemoryCost:
    """Per-instance costs for a memory-like component."""

    read_energy: float = 1.0
    write_energy: float = 1.0
    area: float = 0.0
    leak_power: float = 0.0


@dataclass(frozen=True)
class ComputeCost:
    """Per-instance costs for a compute or processing-stage component."""

    energy: float = 1.0
    area: float = 0.0
    leak_power: float = 0.0


@dataclass(frozen=True)
class OutputStationaryArchConfig:
    """Configuration for :func:`build_arch`.

    ``ao_sram_bytes`` is the complete physical ping-pong SRAM capacity. In
    contrast, ``activation_buffer_bytes_per_bank`` is the usable capacity of
    one activation-buffer bank; both banks exist physically.
    """

    nof_containers: int
    nof_arrays: int
    dram_read_bytes_per_us: float
    dram_write_bytes_per_us: float

    clock_hz: float = 1e9
    value_bits: int = 8
    accumulator_bits: int = 32
    post_param_bits: int = 32
    lanes: int = 240
    post_lanes: int = 12
    window_width: int = 3
    weight_buffer_bytes_per_array: int = 240
    activation_buffer_bytes_per_bank: int = 240
    output_line_bytes_per_container: int = 240
    post_params_per_channel: int = 4
    ao_buffer_sets: int = 2
    activation_buffer_sets: int = 2

    dram_bytes: int = 500_000_000
    dram_startup_cycles: int = 100
    total_global_sram_bytes: int = 512 * 1024
    ao_sram_bytes: int = 256 * 1024
    weight_sram_bytes: int = 240 * 1024
    constant_sram_bytes: int = 16 * 1024

    ao_sram_banks: Optional[int] = None
    weight_sram_banks: Optional[int] = None
    constant_sram_banks: Optional[int] = None

    ao_read_values_per_cycle: float = 1.0
    ao_write_values_per_cycle_per_container: float = 12.0
    weight_read_values_per_cycle_per_array: float = 1.0
    constant_read_values_per_cycle_per_array: float = 1.0

    dram_cost: MemoryCost = field(default_factory=MemoryCost)
    global_sram_cost: MemoryCost = field(default_factory=MemoryCost)
    local_buffer_cost: MemoryCost = field(default_factory=MemoryCost)
    mac_cost: ComputeCost = field(default_factory=ComputeCost)
    post_cost: ComputeCost = field(default_factory=ComputeCost)

    def __post_init__(self) -> None:
        positive_ints = {
            "nof_containers": self.nof_containers,
            "nof_arrays": self.nof_arrays,
            "value_bits": self.value_bits,
            "accumulator_bits": self.accumulator_bits,
            "post_param_bits": self.post_param_bits,
            "lanes": self.lanes,
            "post_lanes": self.post_lanes,
            "window_width": self.window_width,
            "weight_buffer_bytes_per_array": (
                self.weight_buffer_bytes_per_array
            ),
            "activation_buffer_bytes_per_bank": (
                self.activation_buffer_bytes_per_bank
            ),
            "output_line_bytes_per_container": (
                self.output_line_bytes_per_container
            ),
            "post_params_per_channel": self.post_params_per_channel,
            "ao_buffer_sets": self.ao_buffer_sets,
            "activation_buffer_sets": self.activation_buffer_sets,
            "dram_bytes": self.dram_bytes,
            "total_global_sram_bytes": self.total_global_sram_bytes,
            "ao_sram_bytes": self.ao_sram_bytes,
            "weight_sram_bytes": self.weight_sram_bytes,
            "constant_sram_bytes": self.constant_sram_bytes,
        }
        for name, value in positive_ints.items():
            is_positive_int = (
                isinstance(value, int)
                and not isinstance(value, bool)
                and value > 0
            )
            if not is_positive_int:
                raise ValueError(
                    f"{name} must be a positive integer, got {value!r}"
                )

        positive_numbers = {
            "clock_hz": self.clock_hz,
            "dram_read_bytes_per_us": self.dram_read_bytes_per_us,
            "dram_write_bytes_per_us": self.dram_write_bytes_per_us,
            "ao_read_values_per_cycle": self.ao_read_values_per_cycle,
            "ao_write_values_per_cycle_per_container": (
                self.ao_write_values_per_cycle_per_container
            ),
            "weight_read_values_per_cycle_per_array": (
                self.weight_read_values_per_cycle_per_array
            ),
            "constant_read_values_per_cycle_per_array": (
                self.constant_read_values_per_cycle_per_array
            ),
        }
        for name, value in positive_numbers.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value!r}")

        partitioned = (
            self.ao_sram_bytes
            + self.weight_sram_bytes
            + self.constant_sram_bytes
        )
        if partitioned != self.total_global_sram_bytes:
            raise ValueError(
                "A/O, weight, and constant SRAM partitions must sum to "
                f"total_global_sram_bytes ({partitioned} != "
                f"{self.total_global_sram_bytes})"
            )
        if self.ao_sram_bytes * 8 % self.ao_buffer_sets:
            raise ValueError(
                "AOSRAM capacity must divide evenly into its banks"
            )
        if self.post_lanes > self.lanes or self.lanes % self.post_lanes != 0:
            raise ValueError("post_lanes must be a positive divisor of lanes")
        if self.dram_startup_cycles < 0:
            raise ValueError("dram_startup_cycles cannot be negative")
        if self.activation_values_per_bank < self.window_width:
            raise ValueError("ActivationBuffer cannot hold one kernel window")
        for name in (
            "ao_sram_banks",
            "weight_sram_banks",
            "constant_sram_banks",
        ):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when specified")

    @property
    def simultaneous_output_channels(self) -> int:
        """Number of output channels processed by all arrays at once."""

        return self.nof_containers * self.nof_arrays

    @property
    def activation_values_per_bank(self) -> int:
        """Number of input values in one activation line."""

        return self.activation_buffer_bytes_per_bank * 8 // self.value_bits

    @property
    def output_line_values(self) -> int:
        """Post-processed values held by one container's OLine."""

        return self.output_line_bytes_per_container * 8 // self.value_bits

    @property
    def max_conv_output_positions(self) -> int:
        """Maximum horizontal positions in a legal per-array tile."""

        activation_limit = (
            self.activation_values_per_bank - self.window_width + 1
        )
        output_limit = self.output_line_values // self.nof_arrays
        return min(self.lanes, activation_limit, output_limit)

    def validate_conv_output_tile(self, output_positions: int) -> None:
        """Reject a tile that cannot fit its input window or output line."""

        if output_positions <= 0:
            raise ValueError("output_positions must be positive")
        if output_positions > self.max_conv_output_positions:
            raise ValueError(
                f"output tile {output_positions} exceeds the per-array limit "
                f"of {self.max_conv_output_positions}"
            )


def _actions(cost, read_throughput, write_throughput, tensor_expression):
    values = {tensor_expression: 1}
    return [
        {
            "name": "read",
            "energy": cost.read_energy,
            "throughput": read_throughput,
            "values_per_action": values,
        },
        {
            "name": "write",
            "energy": cost.write_energy,
            "throughput": write_throughput,
            "values_per_action": values,
        },
    ]


def _memory(
    *,
    name,
    size_bits,
    tensors,
    tensor_expression,
    cost,
    read_throughput,
    write_throughput,
    bits_per_value=None,
    extra_attributes=None,
    total_latency=None,
):
    kwargs = {}
    if total_latency is not None:
        kwargs["total_latency"] = total_latency
    return af.arch.Memory(
        name=name,
        size=size_bits,
        tensors=tensors,
        bits_per_value=bits_per_value or {},
        actions=_actions(
            cost, read_throughput, write_throughput, tensor_expression
        ),
        area=cost.area,
        leak_power=cost.leak_power,
        extra_attributes_for_component_model=extra_attributes or {},
        **kwargs,
    )


def _ping_pong_memory(
    *,
    name,
    bank_size_bits,
    buffer_sets,
    tensors,
    tensor_expression,
    cost,
    read_throughput,
    write_throughput,
    extra_attributes=None,
    sharing_scale=1.0,
):
    """Expose one active bank while accounting for all physical banks."""

    attributes = {
        "n_buffer_sets": buffer_sets,
        "physical_size_bits": bank_size_bits * buffer_sets,
    }
    attributes.update(extra_attributes or {})
    memory = _memory(
        name=name,
        size_bits=bank_size_bits,
        tensors=tensors,
        tensor_expression=tensor_expression,
        cost=cost,
        read_throughput=read_throughput,
        write_throughput=write_throughput,
        extra_attributes=attributes,
    )
    memory.area_scale = buffer_sets * sharing_scale
    memory.leak_power_scale = buffer_sets * sharing_scale
    memory.energy_scale = sharing_scale
    return memory


def build_arch(config: OutputStationaryArchConfig) -> af.Arch:
    """Build the output-stationary AccelForge architecture."""

    c = config
    clock = c.clock_hz
    channel_arrays = c.simultaneous_output_channels
    auxiliary = "~(input | output | weight)"
    ao_banks = c.ao_sram_banks or c.nof_containers
    weight_banks = c.weight_sram_banks or channel_arrays
    constant_banks = c.constant_sram_banks or c.nof_containers

    dram_actions = [
        {
            "name": "read",
            "energy": c.dram_cost.read_energy,
            "throughput": c.dram_read_bytes_per_us * 1e6,
            "bits_per_action": 8,
        },
        {
            "name": "write",
            "energy": c.dram_cost.write_energy,
            "throughput": c.dram_write_bytes_per_us * 1e6,
            "bits_per_action": 8,
        },
    ]
    startup_seconds = c.dram_startup_cycles / clock
    dram_latency = (
        "max(a.n_calls / a.throughput for a in actions) + "
        f"{startup_seconds!r}"
    )
    ao_read_throughput = c.ao_read_values_per_cycle * clock
    ao_write_throughput = (
        c.ao_write_values_per_cycle_per_container * c.nof_containers * clock
    )
    weight_throughput = (
        c.weight_read_values_per_cycle_per_array * channel_arrays * clock
    )
    constant_throughput = (
        c.constant_read_values_per_cycle_per_array * channel_arrays * clock
    )

    nodes = [
        af.arch.Memory(
            name="DRAM",
            size=c.dram_bytes * 8,
            tensors={"keep": "~Intermediates", "may_keep": "All"},
            actions=dram_actions,
            total_latency=dram_latency,
            area=c.dram_cost.area,
            leak_power=c.dram_cost.leak_power,
        ),
        _ping_pong_memory(
            name="AOSRAM",
            bank_size_bits=c.ao_sram_bytes * 8 // c.ao_buffer_sets,
            buffer_sets=c.ao_buffer_sets,
            tensors=af.frontend.arch.Tensors(
                keep="input | output",
                # back="Intermidiate",
            ),
            tensor_expression="input | output",
            cost=c.global_sram_cost,
            read_throughput=ao_read_throughput,
            write_throughput=ao_write_throughput,
            extra_attributes={"n_banks": ao_banks},
        ),
        _memory(
            name="WeightSRAM",
            size_bits=c.weight_sram_bytes * 8,
            tensors={"keep": "weight"},
            tensor_expression="weight",
            cost=c.global_sram_cost,
            read_throughput=weight_throughput,
            write_throughput=weight_throughput,
            extra_attributes={"n_banks": weight_banks},
        ),
        _memory(
            name="ConstantSRAM",
            size_bits=c.constant_sram_bytes * 8,
            tensors={"keep": auxiliary},
            tensor_expression=auxiliary,
            cost=c.global_sram_cost,
            read_throughput=constant_throughput,
            write_throughput=constant_throughput,
            extra_attributes={"n_banks": constant_banks},
        ),
        af.arch.Container(
            name="ComputeContainers",
            spatial=[{
                "name": "K_CONTAINER",
                "fanout": c.nof_containers,
                # "reuse": "input",
                # "may_reuse": "input",
                "min_usage": 1,
                "power_gateable": True,
            }],
        ),
        _memory(
            name="OLine",
            size_bits=c.output_line_bytes_per_container * 8,
            tensors={"keep": "output"},
            tensor_expression="output",
            cost=c.local_buffer_cost,
            read_throughput=(
                c.ao_write_values_per_cycle_per_container * clock
            ),
            write_throughput=c.post_lanes * c.nof_arrays * clock,
            extra_attributes={
                "line_bytes": c.output_line_bytes_per_container,
                "physical_instances_per_container": 1,
            },
            total_latency="max(a.n_calls / a.throughput for a in actions)",
        ),

        _memory(
            name="PostParamBuffer",
            size_bits=c.post_params_per_channel * c.post_param_bits,
            tensors={"keep": auxiliary},
            tensor_expression=auxiliary,
            cost=c.local_buffer_cost,
            read_throughput=clock,
            write_throughput=clock,
        ),
        af.arch.Toll(
            name="PostProcess",
            tensors={"keep": "output"},
            direction="up",
            actions=[{
                "name": "read",
                "energy": c.post_cost.energy,
                "throughput": clock,
                "values_per_action": {"output": 1},
            }],
            # n_parallel_instances=c.post_lanes,
            n_parallel_instances=1,
            area=c.post_cost.area,
            leak_power=c.post_cost.leak_power,
        ),

        _ping_pong_memory(
            name="ActivationBuffer",
            bank_size_bits=c.activation_buffer_bytes_per_bank * 8,
            buffer_sets=c.activation_buffer_sets,
            tensors={"keep": "input"},
            tensor_expression="input",
            cost=c.local_buffer_cost,
            read_throughput=c.lanes * clock,
            write_throughput=clock,
            extra_attributes={
                "shared_across_spatial": af.LiteralString("K_ARRAY"),
                "physical_instances_per_container": 1,
            },
            # The logical view is below each array so FFM can refill a line
            # inside reduction loops. All views share one physical buffer.
            sharing_scale=1 / c.nof_arrays,
        ),

        af.arch.Container(
            name="ChannelArrays",
            spatial=[{
                "name": "K_ARRAY",
                "fanout": c.nof_arrays,
                "reuse": "input",
                "may_reuse": "input",
                "min_usage": c.nof_arrays,
                "power_gateable": True,
            }],
        ),
        _memory(
            name="WeightBuffer",
            size_bits=c.weight_buffer_bytes_per_array * 8,
            tensors={"keep": "weight"},
            tensor_expression="weight",
            cost=c.local_buffer_cost,
            read_throughput=clock,
            write_throughput=clock,
        ),


        _memory(
            name="OutputAccumulator",
            size_bits=c.lanes * c.accumulator_bits,
            tensors={"keep": "output"},
            tensor_expression="output",
            cost=c.local_buffer_cost,
            read_throughput=c.lanes * clock,
            write_throughput=c.lanes * clock,
            bits_per_value={"output": c.accumulator_bits},
        ),
        # af.arch.Container(
        #     name="MACLanes",
        #     spatial=[{
        #         "name": "OUTPUT_POSITION",
        #         "fanout": c.lanes,
        #         # "reuse": "weight",
        #         # "may_reuse": "input | weight",
        #         "power_gateable": True,
        #     }],
        # ),
        af.arch.Compute(
            name="MAC",
            actions=[{
                "name": "compute",
                "energy": c.mac_cost.energy,
                "throughput": clock,
            }],
            area=c.mac_cost.area,
            leak_power=c.mac_cost.leak_power,
        ),
    ]
    return af.Arch(nodes=nodes)


__all__ = [
    "ComputeCost",
    "MemoryCost",
    "OutputStationaryArchConfig",
    "build_arch",
]
