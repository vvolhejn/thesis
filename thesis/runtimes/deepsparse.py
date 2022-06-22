import os

import deepsparse
from sparseml.pytorch.optim import ScheduledModifierManager
from sparseml.pytorch.utils import ModuleExporter, get_prunable_layers, tensor_sparsity

import torch

from .runtime import TEMP_DIR
from . import ONNXRuntime, Runtime, NeedsPyTorchModel


class DeepSparse(Runtime, NeedsPyTorchModel):
    def __init__(self, quantization_mode, sparsity=0.0):
        super().__init__()

        modes = {"off", "static"}
        assert quantization_mode in modes, f"quantization_mode must be one of {modes}"
        self.quantization_mode = quantization_mode

        assert 0.0 <= sparsity < 1.0
        self.sparsity = sparsity

    def convert(self, orig_model, get_batch_fn=None):
        super().convert(orig_model, get_batch_fn)

        recipe_path = self.create_recipe(orig_model)
        manager = ScheduledModifierManager.from_yaml(recipe_path)

        learning_rate = 0.001
        steps_per_epoch = 1
        n_epochs = 5

        optimizer = torch.optim.SGD(orig_model.parameters(), lr=learning_rate)
        optimizer = manager.modify(
            orig_model, optimizer, steps_per_epoch=steps_per_epoch
        )
        loss_fn = torch.nn.MSELoss()

        for i in range(steps_per_epoch * n_epochs):
            data = torch.from_numpy(get_batch_fn())
            pred = orig_model(data)
            loss = loss_fn(pred, torch.randn_like(pred))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        manager.finalize(orig_model)

        if self.sparsity > 0:
            for (name, layer) in get_prunable_layers(orig_model):
                # print(f"{name}.weight: {tensor_sparsity(layer.weight).item():.4f}")
                layer_sparsity = tensor_sparsity(layer.weight).item()
                assert (
                    layer_sparsity >= 0.9 * self.sparsity
                ), f"Layer {name} has sparsity {layer_sparsity}, expected ~{self.sparsity}"

        save_dir = TEMP_DIR
        quant_onnx_graph_name = f"{self.get_id()}.onnx"
        self.save_path = os.path.join(save_dir, quant_onnx_graph_name)

        exporter = ModuleExporter(orig_model, output_dir=save_dir)
        exporter.export_onnx(
            torch.from_numpy(get_batch_fn()),
            name=quant_onnx_graph_name,
            convert_qat=True,
        )

        self.engine = deepsparse.compile_model(
            self.save_path,
            batch_size=1,
            # num_cores=thesis.util.get_n_cpus_available(),
        )

    def create_recipe(self, model):
        path = f"/tmp/{self.get_id()}_recipe.yaml"

        layers_to_prune = []
        block_size = (1, 4)

        # We cannot prune layers whose weights' shapes are not divided evenly
        # by the block size
        for (name, layer) in get_prunable_layers(model):
            weight_shape = layer.weight.shape

            compatible = True
            for weight_dim, block_dim in zip(weight_shape, block_size):
                if weight_dim % block_dim != 0:
                    # compatible = False
                    pass

            if compatible:
                # print(weight_shape, "ok")
                layers_to_prune.append(f"{name}.weight")

        assert layers_to_prune, "Didn't find any prunable layers"

        template_args = {
            "sparsity": self.sparsity,
            "layers_to_prune": layers_to_prune,
        }

        with open(path, "w") as f:
            f.write(base_recipe_template.format(**template_args))

            if self.sparsity > 0:
                f.write(pruning_template.format(**template_args))

            if self.quantization_mode != "off":
                f.write(quantization_template.format(**template_args))

        print(f"Wrote recipe to {path}")

        return path

    def run(self, data):
        return self.engine.run([data])

    def get_name(self):
        d = {
            "off": "",
            "static": "_quant_static",
        }
        res = "DeepSparse" + d[self.quantization_mode]

        if self.sparsity > 0:
            res += f"_{self.sparsity:.2f}_sparse"

        return res


base_recipe_template = """
version: 0.1.0
modifiers:
    - !EpochRangeModifier
        start_epoch: 0.0
        end_epoch: 2

    - !LearningRateModifier
        start_epoch: 0
        end_epoch: 2
        init_lr: 0.005
        lr_class:  ExponentialLR
        lr_kwargs:
            gamma: 0.9
"""

pruning_template = """
    - !GMPruningModifier
        start_epoch: 0
        end_epoch: 1
        update_frequency: 1.0
        init_sparsity: 0.05
        final_sparsity: {sparsity}
#        mask_type: [1, 4]
        mask_type: block4
#        params: ['foo.weight', 'bar.weight']
#        params:  __ALL__
        params: {layers_to_prune}
"""

quantization_template = """
# Quantization "docs": https://github.com/neuralmagic/sparseml/blob/main/src/sparseml/pytorch/sparsification/quantization/modifier_quantization.py
    - !QuantizationModifier
        start_epoch: 0.0
"""
