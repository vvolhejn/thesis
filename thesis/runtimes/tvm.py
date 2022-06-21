import os

import tvm.relay as relay
import tvm
from tvm.contrib import graph_executor
import onnx

from . import ONNXRuntime
from thesis import util


class TVM(ONNXRuntime):
    def convert(self, orig_model, get_batch_fn=None):
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

        target = "llvm -mcpu=haswell"

        # opt_level=3 is the maximum
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(self.mod, target=target, params=self.params)

        # create random input
        device = tvm.device(target, 0)
        # data = np.random.uniform(-1, 1, size=input_shape).astype("float32")
        # create module
        self.module = graph_executor.GraphModule(lib["default"](device))

    def run(self, data):
        # set input and parameters
        self.module.set_input(self.input_name, data)
        self.module.run()
        # get output

        out = self.module.get_output(0, tvm.nd.empty(tuple(self.output_shape))).numpy()

        return out
