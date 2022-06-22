import codetiming
import gin
import tensorflow as tf
import tensorflow.keras.layers as tfkl
from keras import backend

import ddsp.training


def InvertedBottleneckBlock(
    filters,
    kernel_size,
    dilation_rate,
    expansion_rate=4,
    transpose=False,
    stride=1,  # **kwargs
):
    """
    Expects input of shape [batch, time, 1, in_channels]
    and returns [batch, time, 1, out_channels]
    """
    assert not transpose, (
        "The `transpose` argument of InvertedBottleneckBlock "
        "is only included for signature compatibility and cannot be used."
    )

    assert stride == 1, (
        "The `stride` argument of InvertedBottleneckBlock "
        "is only included for signature compatibility and cannot be used."
    )

    channel_axis = 1 if backend.image_data_format() == "channels_first" else -1

    return tf.keras.Sequential(
        [
            # Expand with a pointwise 1x1 convolution.
            tfkl.Conv2D(
                expansion_rate * filters,
                kernel_size=1,
                padding="same",
                use_bias=False,
                # **kwargs,
            ),
            tfkl.BatchNormalization(
                axis=channel_axis,
                epsilon=1e-3,
                momentum=0.999,
            ),
            tfkl.Activation(tf.nn.relu6),
            # Depthwise 3x3 convolution.
            tfkl.DepthwiseConv2D(
                # no channels argument since this is a depthwise conv
                kernel_size=(kernel_size, 1),  # should probably be (3, 1)
                padding="same",
                use_bias=False,
                dilation_rate=(dilation_rate, 1),
                # **kwargs,
            ),
            tfkl.BatchNormalization(
                axis=channel_axis,
                epsilon=1e-3,
                momentum=0.999,
            ),
            tfkl.Activation(tf.nn.relu6),
            # Project with a pointwise 1x1 convolution.
            tfkl.Conv2D(
                filters,
                kernel_size=1,
                padding="same",
                use_bias=False,
                activation=None,
                # **kwargs,
            ),
            tfkl.BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999),
        ]
    )


def BasicBlock(
    filters,
    kernel_size,
    dilation_rate,
    stride=1,
    transpose=False,
    **kwargs,
):
    channel_axis = 1 if backend.image_data_format() == "channels_first" else -1

    layer_class = tfkl.Conv2DTranspose if transpose else tfkl.Conv2D

    return tf.keras.Sequential(
        [
            layer_class(
                filters,
                kernel_size=(kernel_size, 1),
                padding="same",
                use_bias=False,
                dilation_rate=(dilation_rate, 1),
                strides=(stride, 1),
                **kwargs,
            ),
            tfkl.BatchNormalization(
                axis=channel_axis,
                epsilon=1e-3,
                momentum=0.999,
            ),
            tfkl.Activation(tf.nn.relu6),
        ]
    )


@gin.register
class DilatedConvStack(tfkl.Layer):
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
        **kwargs,
    ):
        """Constructor.

        Args:
          ch: Number of channels in each convolution layer.
          layers_per_stack: Convolution layers in each 'stack'. Dilation increases
            exponentially with layer depth inside a stack.
          stacks: Number of convolutions stacks.
          kernel_size: Size of convolution kernel.
          dilation: Exponent base of dilation factor within a stack.
          resample_type: Whether to 'upsample' or 'downsample' the signal. None
            performs no resampling.
          resample_stride: Stride for upsample or downsample layers.
          stacks_per_resample: Number of stacks per a resample layer.
          resample_after_convolve: Ordering of convolution and resampling. If True,
            apply `stacks_per_resample` stacks of convolution then a resampling
            layer. If False, apply the opposite order.
          **kwargs: Other keras kwargs.

        Returns:
          Convolved and resampled signal. If inputs shape is [batch, time, ch_in],
          output shape is [batch, time_out, ch], where `ch` is the class kwarg, and
          `time_out` is (stacks // stacks_per_resample) * resample_stride times
          smaller or larger than `time` depending on whether `resample_type` is
          upsampling or downsampling.
        """
        super().__init__(**kwargs)
        self.resample_after_convolve = resample_after_convolve

        self.config_dict = {
            "ch": ch,
            "layers_per_stack": layers_per_stack,
            "stacks": stacks,
            "kernel_size": kernel_size,
            "dilation": dilation,
            "resample_type": resample_type,
            "resample_stride": resample_stride,
            "stacks_per_resample": stacks_per_resample,
            "resample_after_convolve": resample_after_convolve,
        }

        def conv(ch, k, stride=1, dilation=1, transpose=False):
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
        self.conv_in = conv(ch, kernel_size)
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
                self.resample_layers.append(resample_layer())

            # Convolve.
            for j in range(layers_per_stack):
                # Convolution.
                dilation_rate = int(dilation**j)
                layer = conv(ch, kernel_size, stride=1, dilation=dilation_rate)

                # Add to the stack.
                self.layers.append(layer)

            # Option: Resample after convolve.
            if (
                resample_type
                and self.resample_after_convolve
                and (i + 1) % stacks_per_resample == 0
            ):
                self.resample_layers.append(resample_layer())

        # For forward pass, calculate layers per a resample.
        if self.resample_layers:
            self.layers_per_resample = len(self.layers) // len(self.resample_layers)
        else:
            self.layers_per_resample = 0

    def get_config(self):
        return self.config_dict

    def call(self, inputs):
        """Forward pass."""
        # Get inputs.
        x = inputs
        x = ddsp.training.nn.ensure_4d(x)

        # Run them through the network.
        # print("SHAPE", x.shape)
        x = self.conv_in(x)

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

                # Optional: Resample after conv.
                if (
                    self.resample_layers
                    and self.resample_after_convolve
                    and (i + 1) % self.layers_per_resample == 0
                ):
                    x = self.resample_layers[i // self.layers_per_resample](x)

        return x[:, :, 0, :]  # Convert back to 3-D.

    def total_padding(self):
        """
        How many zeroes does TensorFlow implicitly add to the input to get an output of the
        same size? I found this formula by disabling padding and residual connections
        (replacing `x += norm(layer(x))` with `x = norm(layer(x))`). Then I tried different
        parameter combinations and found the formula by trial and error. The "total padding"
        is then the difference in length between the input and output tensors in this setup.
        """
        kernel_size = self.config_dict["kernel_size"]
        stacks = self.config_dict["stacks"]
        dilation = self.config_dict["dilation"]
        layers_per_stack = self.config_dict["layers_per_stack"]

        stacks_correction = (kernel_size - 1) * (stacks - 1)
        res = (kernel_size - 1) * (
            stacks * dilation**layers_per_stack
        ) - stacks_correction
        return res
