import glob
import os
import subprocess
import time

import tensorflow as tf
import onnxruntime as ort
import numpy as np
import onnx

# path = os.path.join(os.path.dirname(__file__), "../cpp/data/ONNXRuntime_dense_768.onnx")
CPP_BINARY_PATH = "/cluster/home/vvolhejn/thesis/cpp/build/src/inference"


def measure_latency_python(path, n_iterations):
    onnx_model = onnx.load(path)
    onnx.checker.check_model(onnx_model)

    # We need to set intra_op_num_threads:
    # see https://github.com/microsoft/onnxruntime/issues/10113
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        path,
        # providers=[
        #     "CPUExecutionProvider"
        # ],
        sess_options=sess_options,
    )

    # qb.TEMP_DIR = "/cluster/home/vvolhejn/tmp"
    # rt = qb.ONNXRuntime("off")
    # rt.convert(dense_keras)

    input_names = [inp.name for inp in session.get_inputs()]
    assert len(input_names) == 1, "Expected only one input to ONNX model"
    output_names = [output.name for output in session.get_outputs()]

    data = np.random.rand(*session.get_inputs()[0].shape).astype(np.float32)

    session.run(output_names, {input_names[0]: data})

    t1 = time.perf_counter()
    for i in range(n_iterations):
        outputs = session.run(output_names, {input_names[0]: data})
    t2 = time.perf_counter()

    time_ms = (t2 - t1) / n_iterations * 1000
    return time_ms


def measure_latency_cpp(path, n_iterations):
    output = subprocess.run(
        [CPP_BINARY_PATH, path, str(n_iterations)], capture_output=True
    )
    time_ms = float(output.stdout)
    return time_ms


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("glob")
    parser.add_argument("--out-file")
    parser.add_argument("--ms-per-model", type=int, default=1000)
    args = parser.parse_args()

    paths = glob.glob(args.glob)

    ms_per_model = args.ms_per_model
    assert paths, f"No files matched glob {args.glob}"

    rows = []

    for path in paths:
        session = ort.InferenceSession(path, sess_options=ort.SessionOptions())
        input_shape = session.get_inputs()[0].shape

        ms_preliminary = measure_latency_python(path, n_iterations=10)
        n_iterations = max(10, int(ms_per_model / ms_preliminary) + 1)

        ms_python = measure_latency_python(path, n_iterations)
        ms_cpp = measure_latency_cpp(path, n_iterations)

        print(
            f"Python: {ms_python:.3f}ms\tC++: {ms_cpp:.3f}ms"
            f"\tRatio: {ms_cpp / ms_python:.3f}\t{os.path.basename(path)}"
        )

        rows.append(
            dict(
                path=path,
                ms_preliminary=ms_preliminary,
                n_iterations=n_iterations,
                ms_python=ms_python,
                ms_cpp=ms_cpp,
                cpp_to_python_ratio=ms_cpp / ms_python,
                input_shape=input_shape,
            )
        )

    if args.out_file:
        import pandas as pd
        pd.DataFrame(rows).to_csv(args.out_file)

    # print(outputs)
