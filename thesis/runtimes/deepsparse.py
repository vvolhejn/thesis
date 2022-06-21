import deepsparse

from . import ONNXRuntime


class DeepSparse(ONNXRuntime):
    def convert(self, orig_model, get_batch_fn=None):
        super().convert(orig_model, get_batch_fn)
        # onnx_filepath = self.save_path
        onnx_filepath = self.optimized_model_path
        self.engine = deepsparse.compile_model(
            onnx_filepath,
            batch_size=1,
            num_cores=2,
            # util.get_n_cpus_available()
        )

    def run(self, data):
        return self.engine.run([data])
