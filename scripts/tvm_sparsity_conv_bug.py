import onnx
import onnx.shape_inference
import tvm
from tvm import relay

# onnx_model = onnx.load("/home/vaclav/scripts/inverted-bottleneck.onnx")
# onnx_model = onnx.load("/home/vaclav/scripts/dilated-conv-stack-ib.onnx")
onnx_model = onnx.load("/home/vaclav/scripts/dilated-conv-stack.onnx")

input_shape = [
    d.dim_value for d in onnx_model.graph.input[0].type.tensor_type.shape.dim
]

input_name = onnx_model.graph.input[0].name
shape_dict = {input_name: input_shape}

mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

bs_r, bs_c = 4, 1
sparsity = 0.9
layout = "NCHW"

from tvm.relay import data_dep_optimization as ddo

mod, params = ddo.simplify_fc_transpose.convert(mod["main"], params)

# # from tvm.topi.sparse.utils import convert_model_dense_to_sparse
# from thesis.runtimes.tvm import convert_model_dense_to_sparse

from tvm.topi.sparse.utils import random_sparse_conv2d_params

# Manually replace the parameters of conv2d to sparse tensors
params = random_sparse_conv2d_params(
    mod, params, bs_r=bs_r, bs_c=bs_c, density=1 - sparsity, layout=layout
)

# convert dense conv2d to sparse conv2d
mod, params = ddo.bsr_conv2d.convert(
    mod, params, (bs_r, bs_c), sparsity_threshold=0.0, layout=layout, kernel_size=1,
)

sparsified = False

for k in params.keys():
    # The new parameters should have some keys ending with .indices, .data and .indptr
    if k.endswith(".indices"):
        sparsified = True

if not sparsified:
    print("Didn't crash but didn't sparsify!")