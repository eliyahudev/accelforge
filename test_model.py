import unittest

from accelforge.frontend.arch import (
    Arch,
    Compute as ArchCompute,
    Memory,
    Spatial as ArchSpatial,
)
from accelforge.frontend.arch.constraints import Comparison
from accelforge.frontend.spec import Spec
from accelforge.frontend.workload import Workload
from accelforge.mapper import Metrics
from accelforge.model.main import evaluate_mapping, InvalidMappingError
from accelforge.util.parallel import set_n_parallel_jobs

try:
    from .paths import EXAMPLES_DIR
except ImportError:
    from paths import EXAMPLES_DIR

set_n_parallel_jobs(1)


class TestInvalidMapping(unittest.TestCase):
    def test_matmul_to_simple(self):
        M = 64
        KN = 64
        spec = Spec.from_yaml(
            EXAMPLES_DIR / "arches" / "simple.yaml",
            EXAMPLES_DIR / "workloads" / "matmuls.yaml",
            "tests/input_files/mapping/invalid_matmul_to_simple.yaml",
            jinja_parse_data={"N_EINSUMS": 1, "M": M, "KN": KN},
        )
        self.assertRaises(InvalidMappingError, lambda: evaluate_mapping(spec))


class TestModel(unittest.TestCase):
    def test_conv3x3_16mac_output_pixels(self):
        spec = Spec(
            workload=Workload(
                iteration_space_shape={
                    "output_x": "0 <= output_x < 64",
                    "output_y": "0 <= output_y < 1",
                    "w_x": "0 <= w_x < 3",
                    "w_y": "0 <= w_y < 3",
                    "in_channel": "0 <= in_channel < 1",
                    "out_channel": "0 <= out_channel < 1",
                },
                bits_per_value={"All": 8},
                einsums=[
                    {
                        "name": "Conv3x3",
                        "tensor_accesses": [
                            {
                                "name": "I",
                                "projection": {
                                    "C": "in_channel",
                                    "X": "output_x + w_x",
                                    "Y": "output_y + w_y",
                                },
                            },
                            {
                                "name": "W",
                                "projection": [
                                    "out_channel",
                                    "in_channel",
                                    "w_x",
                                    "w_y",
                                ],
                            },
                            {
                                "name": "O",
                                "projection": ["out_channel", "output_x", "output_y"],
                                "output": True,
                            },
                        ],
                        "renames": {"input": "I", "weight": "W", "output": "O"},
                    }
                ],
            ),
            arch=Arch(
                nodes=[
                    Memory(
                        name="DRAM",
                        size="inf",
                        leak_power=0,
                        area=0,
                        tensors={"keep": "~Intermediates", "may_keep": "All"},
                        actions=[
                            {"name": "read", "energy": 1, "latency": 0},
                            {"name": "write", "energy": 1, "latency": 0},
                        ],
                    ),
                    Memory(
                        name="SRAM",
                        size="inf",
                        leak_power=0,
                        area=0,
                        tensors={"keep": "All"},
                        actions=[
                            {"name": "read", "energy": 0, "latency": 0},
                            {"name": "write", "energy": 0, "latency": 0},
                        ],
                    ),
                    ArchCompute(
                        name="MAC",
                        spatial=[
                            ArchSpatial(
                                name="mac_lane",
                                fanout=16,
                                min_usage=1,
                                loop_bounds=[
                                    Comparison(
                                        expression="~(Outputs.rank_variables)",
                                        operator="==",
                                        value=1,
                                    ),
                                ],
                            )
                        ],
                        leak_power=0,
                        area=0,
                        actions=[{"name": "compute", "energy": 0, "latency": 1}],
                    ),
                ],
            ),
        )
        spec.mapper.metrics = Metrics.LATENCY

        result = spec.map_workload_to_arch()

        self.assertEqual(result.latency(per_einsum=True)["Conv3x3"], 36)
        action_columns = set(result.data.columns)
        for memory_name in ("DRAM", "SRAM"):
            for action_name in ("read", "write"):
                self.assertTrue(
                    any(
                        col.startswith(
                            f"Conv3x3<SEP>action<SEP>{memory_name}<SEP>"
                        )
                        and col.endswith(f"<SEP>{action_name}")
                        for col in action_columns
                    ),
                    f"{memory_name} {action_name} not found in {action_columns}",
                )

    def test_one_matmul(self):
        M = 64
        KN = 32
        BITS_PER_VALUE = 8
        spec = Spec.from_yaml(
            EXAMPLES_DIR / "arches" / "simple.yaml",
            EXAMPLES_DIR / "workloads" / "matmuls.yaml",
            EXAMPLES_DIR / "mappings" / "unfused_matmuls_to_simple.yaml",
            jinja_parse_data={"N_EINSUMS": 1, "M": M, "KN": KN},
        )

        result = evaluate_mapping(spec)
        energy_breakdown = result.energy(per_component=True, per_tensor=True)
        self.assertAlmostEqual(
            energy_breakdown[("MainMemory", "T0")], M * KN * BITS_PER_VALUE
        )
        self.assertAlmostEqual(
            energy_breakdown[("MainMemory", "T1")], M * KN * BITS_PER_VALUE
        )
        self.assertAlmostEqual(
            energy_breakdown[("MainMemory", "W0")], KN**2 * BITS_PER_VALUE
        )

    def test_bits_per_value_directly_sets(self):
        """bits_per_value on a memory directly sets the bpv for specific tensors."""
        M = 64
        KN = 32
        WORKLOAD_BPV = 8
        OVERRIDE_BPV = 4
        spec = Spec.from_yaml(
            EXAMPLES_DIR / "arches" / "simple.yaml",
            EXAMPLES_DIR / "workloads" / "matmuls.yaml",
            EXAMPLES_DIR / "mappings" / "unfused_matmuls_to_simple.yaml",
            jinja_parse_data={"N_EINSUMS": 1, "M": M, "KN": KN},
        )

        # Set bits_per_value on MainMemory to override T0's bpv to 4
        spec.arch["MainMemory"].bits_per_value = {"T0": OVERRIDE_BPV}

        result = evaluate_mapping(spec)
        energy_breakdown = result.energy(per_component=True, per_tensor=True)
        # T0 should use OVERRIDE_BPV (4) instead of WORKLOAD_BPV (8)
        self.assertAlmostEqual(
            energy_breakdown[("MainMemory", "T0")], M * KN * OVERRIDE_BPV
        )
        # T1 and W0 should still use the workload's bits_per_value (8)
        self.assertAlmostEqual(
            energy_breakdown[("MainMemory", "T1")], M * KN * WORKLOAD_BPV
        )
        self.assertAlmostEqual(
            energy_breakdown[("MainMemory", "W0")], KN**2 * WORKLOAD_BPV
        )

    def test_bits_per_value_all_tensors(self):
        """bits_per_value with All sets bpv for every tensor."""
        M = 64
        KN = 32
        OVERRIDE_BPV = 16
        spec = Spec.from_yaml(
            EXAMPLES_DIR / "arches" / "simple.yaml",
            EXAMPLES_DIR / "workloads" / "matmuls.yaml",
            EXAMPLES_DIR / "mappings" / "unfused_matmuls_to_simple.yaml",
            jinja_parse_data={"N_EINSUMS": 1, "M": M, "KN": KN},
        )

        spec.arch["MainMemory"].bits_per_value = {"All": OVERRIDE_BPV}

        result = evaluate_mapping(spec)
        energy_breakdown = result.energy(per_component=True, per_tensor=True)
        self.assertAlmostEqual(
            energy_breakdown[("MainMemory", "T0")], M * KN * OVERRIDE_BPV
        )
        self.assertAlmostEqual(
            energy_breakdown[("MainMemory", "T1")], M * KN * OVERRIDE_BPV
        )
        self.assertAlmostEqual(
            energy_breakdown[("MainMemory", "W0")], KN**2 * OVERRIDE_BPV
        )

    def test_skip_initial_output_write_false(self):
        M = 64
        KN = 32
        BITS_PER_VALUE = 8
        spec = Spec.from_yaml(
            EXAMPLES_DIR / "arches" / "simple.yaml",
            EXAMPLES_DIR / "workloads" / "matmuls.yaml",
            EXAMPLES_DIR / "mappings" / "unfused_matmuls_to_simple.yaml",
            jinja_parse_data={"N_EINSUMS": 1, "M": M, "KN": KN},
        )

        spec.arch["MainMemory"].skip_initial_output_write = False

        result = evaluate_mapping(spec)
        energy_breakdown = result.energy(per_component=True, per_tensor=True)
        # Input and weight energy unchanged
        self.assertAlmostEqual(
            energy_breakdown[("MainMemory", "T0")], M * KN * BITS_PER_VALUE
        )
        self.assertAlmostEqual(
            energy_breakdown[("MainMemory", "W0")], KN**2 * BITS_PER_VALUE
        )
        # Output energy doubles: the initial fill from MainMemory is no longer
        # skipped, so MainMemory sees both the fill read and the writeback write.
        self.assertAlmostEqual(
            energy_breakdown[("MainMemory", "T1")], 2 * M * KN * BITS_PER_VALUE
        )

    def test_two_matmuls(self):
        spec = Spec.from_yaml(
            EXAMPLES_DIR / "arches" / "simple.yaml",
            EXAMPLES_DIR / "workloads" / "matmuls.yaml",
            EXAMPLES_DIR / "mappings" / "unfused_matmuls_to_simple.yaml",
            jinja_parse_data={"N_EINSUMS": 2, "M": 64, "KN": 64},
        )

        result = evaluate_mapping(spec)


if __name__ == "__main__":
    unittest.main()
