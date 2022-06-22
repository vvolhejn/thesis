import codetiming
import gin

# import tensorflow as tf
# import tensorflow.keras.layers as tfkl
# from keras import backend
# import tensorflow_addons as tfa
import torch

import ddsp.training


def InvertedBottleneckBlock(
    filters,
    kernel_size,
    dilation_rate,
    expansion_rate=4,
    in_filters=None,
    **kwargs,  # For `transpose` and `stride`
):
    channels = filters
    expanded_channels = filters * expansion_rate
    in_filters = in_filters or filters

    return torch.nn.Sequential(
        torch.nn.Conv2d(
            in_channels=in_filters,
            out_channels=expanded_channels,
            kernel_size=1,
            bias=False,
        ),
        torch.nn.BatchNorm2d(
            num_features=expanded_channels, eps=1e-3, momentum=1 - 0.999
        ),
        torch.nn.ReLU6(),
        torch.nn.Conv2d(
            in_channels=expanded_channels,
            out_channels=expanded_channels,
            groups=expanded_channels,  # Make it depthwise
            kernel_size=(kernel_size, 1),
            # Torch quantization doesn't recognize padding="same", so we need to
            # explicitly pass a tuple
            padding=(dilation_rate, 0),
            dilation=(dilation_rate, 1),
            bias=False,
        ),
        torch.nn.BatchNorm2d(
            num_features=expanded_channels, eps=1e-3, momentum=1 - 0.999
        ),
        torch.nn.ReLU6(),
        torch.nn.Conv2d(
            in_channels=expanded_channels,
            out_channels=channels,
            kernel_size=1,
            bias=False,
        ),
        torch.nn.BatchNorm2d(num_features=channels, eps=1e-3, momentum=1 - 0.999),
    )


def BasicBlock(
    filters,
    kernel_size,
    dilation_rate,
    stride=1,
    transpose=False,
    in_filters=None,
):
    layer_class = torch.nn.ConvTranspose2d if transpose else torch.nn.Conv2d

    channels = filters
    in_filters = in_filters or filters

    return torch.nn.Sequential(
        layer_class(
            in_channels=in_filters,
            out_channels=channels,
            kernel_size=(kernel_size, 1),
            dilation=(dilation_rate, 1),
            stride=(stride, 1),
            padding=(dilation_rate - stride + 1, 0),
            bias=False,
        ),
        torch.nn.BatchNorm2d(num_features=channels, eps=1e-3, momentum=1 - 0.999),
        torch.nn.ReLU6(),
    )


@gin.register
class DilatedConvStack(torch.nn.Module):
    """Stack of dilated 1-D convolutions, optional conditioning at each layer."""

    def __init__(
        self,
        ch=256,
        layers_per_stack=5,
        stacks=2,
        kernel_size=3,
        dilation=2,
        resample_type=None,
        resample_stride=1,
        stacks_per_resample=1,
        resample_after_convolve=True,
        use_inverted_bottleneck=False,
        in_ch=None,
    ):
        """
        Please refer to dilated_conv.py (the TensorFlow version of this)
        for documentation.
        """
        super().__init__()
        self.resample_after_convolve = resample_after_convolve

        def conv(ch, k, stride=1, dilation=1, in_ch=None, transpose=False):
            """Make a convolution block."""

            if stride == 1 and use_inverted_bottleneck:
                block_class = InvertedBottleneckBlock
            else:
                block_class = BasicBlock

            b = block_class(
                ch,
                kernel_size=k,
                stride=stride,
                dilation_rate=dilation,
                transpose=transpose,
                in_filters=in_ch,
            )

            return b

        def resample_layer():
            """Generates a resampling layer."""
            if resample_type == "downsample":
                return conv(ch, resample_stride, resample_stride)
            elif resample_type == "upsample":
                return conv(ch, resample_stride * 2, resample_stride, transpose=True)
            else:
                raise ValueError(
                    f"invalid resample type: {resample_type}, "
                    "must be either `upsample` or `downsample`."
                )

        # Layers.
        self.conv_in = conv(ch, kernel_size, in_ch=in_ch)
        self.layers = []
        self.norms = []
        self.resample_layers = []

        # Stacks.
        for i in range(stacks):
            # Option: Resample before convolve.
            if (
                resample_type
                and not self.resample_after_convolve
                and i % stacks_per_resample == 0
            ):
                layer = resample_layer()
                self.resample_layers.append(layer)
                self.add_module(f"resample-{i}", layer)

            # Convolve.
            for j in range(layers_per_stack):
                # Convolution.
                dilation_rate = int(dilation**j)
                layer = conv(ch, kernel_size, stride=1, dilation=dilation_rate)

                # Add to the stack.
                self.layers.append(layer)
                self.add_module(f"block-{i}-{j}", layer)

            # Option: Resample after convolve.
            if (
                resample_type
                and self.resample_after_convolve
                and (i + 1) % stacks_per_resample == 0
            ):
                layer = resample_layer()
                self.resample_layers.append(layer)
                self.add_module(f"resample-{i}", layer)

        # For forward pass, calculate layers per a resample.
        if self.resample_layers:
            self.layers_per_resample = len(self.layers) // len(self.resample_layers)
        else:
            self.layers_per_resample = 0

    def forward(self, inputs):
        """Forward pass."""
        # Get inputs.
        x = inputs
        x = ensure_4d(x)

        # Run them through the network.
        x = self.conv_in(x)
        # print(x.shape)
        # print(torch.all(x[0,:,:100,0] == 0, dim=1))

        # return x[:, :, :, 0]  # Convert back to 3-D.

        # Stacks.
        for i, layer in enumerate(self.layers):
            with codetiming.Timer(f"dilated_cnn.layer_{i+1}", logger=None):
                # Optional: Resample before conv.
                if (
                    self.resample_layers
                    and not self.resample_after_convolve
                    and i % self.layers_per_resample == 0
                ):
                    x = self.resample_layers[i // self.layers_per_resample](x)

                x += layer(x)
                # print(x.sum())
                # print(torch.any(x[0, :, :100, 0] == 0, dim=1))

                # return x[:, :, :, 0]  # Convert back to 3-D.

                # Optional: Resample after conv.
                if (
                    self.resample_layers
                    and self.resample_after_convolve
                    and (i + 1) % self.layers_per_resample == 0
                ):
                    x = self.resample_layers[i // self.layers_per_resample](x)

        return x[:, :, :, 0]  # Convert back to 3-D.


def ensure_4d(x):
    """Add extra dimensions to make sure tensor has height and width."""
    if len(x.shape) == 2:
        return x[:, None, :, None]
    elif len(x.shape) == 3:
        return x[:, :, :, None]
    else:
        return x
