from accelforge.frontend.workload import (
    Workload,
    Einsum,
    TensorAccess,
)

workload = Workload(
    bits_per_value={
        "All": 8,
    },

    # =====================================================
    # TENSOR RANK SIZES
    # =====================================================
    rank_sizes={
        # Batch
        "N": 1,

        # Original input
        "X0": 256,
        "Y0": 256,
        "C0": 3,

        # Conv1 output / Conv2 input
        "X1": 254,
        "Y1": 254,
        "C1": 8,

        # Conv2 output
        "X2": 252,
        "Y2": 252,
        "C2": 16,

        # Kernel
        "RX": 3,
        "RY": 3,
    },

    einsums=[
        # =================================================
        # CONV 1
        #
        # O1[n, x, y, co] +=
        #     A[n, x+rx, y+ry, ci]
        #     * W1[co, rx, ry, ci]
        # =================================================
        Einsum(
            name="Conv1",

            tensor_accesses=[
                TensorAccess(
                    name="Input",
                    projection={
                        "N": "n",
                        "X0": "x + rx",
                        "Y0": "y + ry",
                        "C0": "ci",
                    },
                    output=False,
                ),

                TensorAccess(
                    name="Weight1",
                    projection={
                        "C1": "co",
                        "RX": "rx",
                        "RY": "ry",
                        "C0": "ci",
                    },
                    output=False,
                ),

                TensorAccess(
                    name="Conv1Out",
                    projection={
                        "N": "n",
                        "X1": "x",
                        "Y1": "y",
                        "C1": "co",
                    },
                    output=True,
                ),
            ],

            renames={
                "input": "Input",
                "weight": "Weight1",
                "output": "Conv1Out",
            },
        ),

        # =================================================
        # CONV 2
        #
        # O2[n, x, y, co] +=
        #     O1[n, x+rx, y+ry, ci]
        #     * W2[co, rx, ry, ci]
        # =================================================
        Einsum(
            name="Conv2",

            tensor_accesses=[
                TensorAccess(
                    name="Conv1Out",
                    projection={
                        "N": "n",
                        "X1": "x + rx",
                        "Y1": "y + ry",
                        "C1": "ci",
                    },
                    output=False,
                ),

                TensorAccess(
                    name="Weight2",
                    projection={
                        "C2": "co",
                        "RX": "rx",
                        "RY": "ry",
                        "C1": "ci",
                    },
                    output=False,
                ),

                TensorAccess(
                    name="Conv2Out",
                    projection={
                        "N": "n",
                        "X2": "x",
                        "Y2": "y",
                        "C2": "co",
                    },
                    output=True,
                ),
            ],

            renames={
                "input": "Conv1Out",
                "weight": "Weight2",
                "output": "Conv2Out",
            },
        ),
    ],
)
