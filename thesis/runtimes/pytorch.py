import torch

from . import Runtime, NeedsPyTorchModel


class PyTorch(Runtime, NeedsPyTorchModel):
    def __init__(self, quantization_mode, use_torchscript=False):
        super().__init__()

        modes = {"off", "dynamic", "static"}
        assert quantization_mode in modes, f"quantization_mode must be one of {modes}"
        self.quantization_mode = quantization_mode
        self.use_torchscript = use_torchscript

    def convert(self, orig_model, get_batch_fn=None):
        super().convert(orig_model)

        if self.quantization_mode == "off":
            self.model = orig_model
        elif self.quantization_mode == "dynamic":
            self.model = torch.quantization.quantize_dynamic(
                orig_model,
                # {torch.nn.Linear},
                dtype=torch.qint8,
            )
        else:
            assert self.quantization_mode == "static"

            model = PyTorchQuantizationWrapper(orig_model)
            model.eval()
            model.qconfig = torch.quantization.get_default_qconfig("fbgemm")

            # Note: we could also fuse layers using torch.quantization.fuse_modules

            # Prepare for calibration
            model = torch.quantization.prepare(model)

            for i in range(100):
                data = torch.from_numpy(get_batch_fn())
                model(data)

            self.model = torch.quantization.convert(model)

        if self.use_torchscript:
            data = torch.from_numpy(get_batch_fn())
            self.model = torch.jit.trace(self.model, data)

    def run(self, data):
        super().run(data)

        data = torch.from_numpy(data)

        output = self.model(data)

        return output.detach().numpy()

    def run_timed(self, data, timer):
        """Run in a way that doesn't include the operations around"""

        data = torch.from_numpy(data)
        # Torch needs NCHW instead of TensorFlow's NHWC
        # data = torch.permute(data, (0, 3, 1, 2))

        with timer:
            output = self.model(data)

        return output.detach().numpy()

    def get_name(self):
        d = {
            "off": "",
            "dynamic": "_quant_dynamic",
            "static": "_quant_static",
        }

        prefix = "TorchScript" if self.use_torchscript else "PyTorch"

        return f"{prefix}{d[self.quantization_mode]}"


class PyTorchQuantizationWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

        # QuantStub converts tensors from floating point to quantized
        self.quant = torch.quantization.QuantStub()
        # DeQuantStub converts tensors from quantized to floating point
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x
