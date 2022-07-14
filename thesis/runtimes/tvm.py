import os
import contextlib

import numpy as np
import onnx

try:
    import tvm
    import tvm.relay as relay
    import tvm.autotvm
    from tvm.contrib import graph_executor
    from tvm.topi.sparse.utils import (
        random_sparse_dense_params,
        random_sparse_conv2d_params,
    )
    from tvm.relay import data_dep_optimization as ddo
except ImportError:
    # This is ok if we're not planning to use the module.
    import warnings
    warnings.warn("Couldn't import TVM")

from . import ONNXRuntime
from thesis import util


class TVM(ONNXRuntime):
    def __init__(
        self,
        quantization_mode,
        unsigned_activations=False,
        unsigned_weights=False,
        sparsity=0.0,
        tuning_records_path=None,
    ):
        """

        :param quantization_mode:
        :param unsigned_activations:
        :param unsigned_weights:
        :param sparsity:
        :param tuning_records_path: a .json of tuning records to use with the model:
            see https://tvm.apache.org/docs/tutorial/autotvm_relay_x86.html
        """
        super().__init__(quantization_mode, unsigned_activations, unsigned_weights)
        self.sparsity = sparsity
        self.tuning_records_path = tuning_records_path

    def convert(self, orig_model, get_batch_fn=None):
        # https://tvm.apache.org/docs/how_to/deploy_models/deploy_prequantized.html
        # "Set the environment variable TVM_NUM_THREADS to the number of physical cores"
        os.environ["TVM_NUM_THREADS"] = str(util.get_n_cpus_available())

        super().convert(orig_model, get_batch_fn)

        input_shape = list([1] + orig_model.input.shape[1:])
        self.output_shape = list([1] + orig_model.output.shape[1:])

        onnx_model = onnx.load(self.save_path)

        assert (
            len(onnx_model.graph.input) == 1
        ), "Expected exactly one input to the onnx model"
        self.input_name = onnx_model.graph.input[0].name
        shape_dict = {self.input_name: input_shape}

        self.mod, self.params = relay.frontend.from_onnx(onnx_model, shape_dict)

        target = "llvm -mcpu=skylake-avx512"

        if self.sparsity > 0:
            self.prune()

        if self.tuning_records_path:
            print(f"Using tuning records from {self.tuning_records_path}")
            context = tvm.autotvm.apply_history_best(self.tuning_records_path)
        else:
            # This enables us to use the context manager conditionally
            context = contextlib.nullcontext()

        with context:
            # opt_level=3 is the maximum
            with tvm.transform.PassContext(opt_level=3):
                lib = relay.build(self.mod, target=target, params=self.params)

        # create random input
        device = tvm.device(target, 0)
        # data = np.random.uniform(-1, 1, size=input_shape).astype("float32")
        # create module
        self.module = graph_executor.GraphModule(lib["default"](device))

    def prune(self):
        # Numer of elements in each array
        param_sizes = {k: np.product(v.shape) for k, v in self.params.items()}

        self.mod, self.params = convert_model_dense_to_sparse(
            self.mod,
            self.params,
            # bs_r=4, bs_c=1 is used in an earlier version of a TVM tutorial:
            # https://github.com/yuchaoli/tvm/blob/main/tutorials/auto_scheduler/tune_network_x86.py
            bs_r=4,
            sparsity=self.sparsity,
            random_params=True,  # Watch out!
            layout="NCHW",
        )

        # Check that some pruning actually happened
        contains_sparsified = False
        for k, v in self.params.items():
            # Under TVM's sparse representation, the dense parameters are replaced with
            # .data, .indices and .indptr suffices, e.g.
            # "sequential_1/sequential/conv2d_1/Conv2D/ReadVariableOp:0.indptr"
            if k.endswith(".indptr"):
                contains_sparsified = True
                orig_key = k[: -len(".indptr")]
                size_new = np.product(self.params[orig_key + ".data"].shape)

                if orig_key not in param_sizes:
                    orig_key = k[: -len(".T.indptr")]
                    assert (
                        orig_key in param_sizes
                    ), f"Original key of parameter {k} not found!"

                size_old = param_sizes[orig_key]

                actual_sparsity = 1 - size_new / size_old

                assert actual_sparsity >= 0.9 * self.sparsity, (
                    f"Parameter {k} sparsified to {actual_sparsity:.3f},"
                    f" expected {self.sparsity:.3f}"
                )

        assert (
            contains_sparsified
        ), "No sparsified parameters found in converted TVM model!"

    def run(self, data):
        # set input and parameters
        self.module.set_input(self.input_name, data)
        self.module.run()
        # get output

        out = self.module.get_output(0, tvm.nd.empty(tuple(self.output_shape))).numpy()

        return out


def convert_model_dense_to_sparse(
    mod, params, random_params=False, bs_r=1, bs_c=1, sparsity=0.85, layout="NHWC"
):
    """Convert a dense model to sparse model.

    This is like the tvm.topi.sparse.utils.convert_model_dense_to_sparse function,
    but the sparsity threshold is set to 0.0

    Parameters
    ----------
    mod : tvm.Module
        The dense model.
    params : Dict[Srting, tvm.nd.array]
        Parameters of the dense model.
    random_params : Bool = False
        True to replace the parameters of the dense model with some random sparse tensors.
        This is mainly used for testing.
    bs_r : int
        The row of BSR matrix block.
    bs_c : int
        The column of BSR matrix block.
    sparsity : float
        The sparsity of the random sparse parameters.
    layout : str
        layout of network

    Returns
    -------
    tvm.Module
        The updated sparse model.
    Dict[Srting, tvm.nd.array]
        The updated parameters.
    """

    mod, params = ddo.simplify_fc_transpose.convert(mod["main"], params)
    if random_params:
        # Manually replace the parameters of dense to sparse tensors
        params = random_sparse_dense_params(
            mod, params, bs_r=bs_r, bs_c=bs_c, density=1 - sparsity
        )
        # Manually replace the parameters of conv2d to sparse tensors
        params = random_sparse_conv2d_params(
            mod, params, bs_r=bs_r, bs_c=bs_c, density=1 - sparsity, layout=layout
        )
    # convert dense matmul to sparse matmul
    mod, params = ddo.bsr_dense.convert(
        mod, params, (bs_r, bs_c), sparsity_threshold=0.0
    )
    # convert dense conv2d to sparse conv2d
    mod, params = ddo.bsr_conv2d.convert(
        mod, params, (bs_r, bs_c), sparsity_threshold=0.0, layout=layout
    )

    return tvm.IRModule.from_expr(mod), params
