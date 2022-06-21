import math

import tensorflow as tf
import torch

import ddsp.training


def dense_models(n_sizes=8, n_layers=1):
    base_size = 256
    expansion = 4

    for i in range(1, n_sizes + 1):
        hidden_size = base_size * i

        layers_keras = [tf.keras.layers.Input((hidden_size,))]
        layers_torch = []

        for l in range(n_layers):
            if l > 0:
                layers_keras.append(tf.keras.layers.Activation(tf.nn.relu))
                layers_torch.append(torch.nn.ReLU())

            layers_keras += [
                tf.keras.layers.Dense(hidden_size * expansion),
                tf.keras.layers.Activation(tf.nn.relu),
                tf.keras.layers.Dense(hidden_size),
            ]

            layers_torch += [
                torch.nn.Linear(
                    in_features=hidden_size, out_features=hidden_size * expansion
                ),
                torch.nn.ReLU(),
                torch.nn.Linear(
                    in_features=hidden_size * expansion, out_features=hidden_size
                ),
            ]

        dense_keras = tf.keras.Sequential(layers_keras)
        dense_torch = torch.nn.Sequential(*layers_torch)

        yield dense_keras, dense_torch, {"base_size": base_size, "expansion": expansion}


def cnn_models(n_sizes=10, n_layers=1):
    for i in range(n_sizes):
        size = 8 * 2**i
        n_channels = 64

        conv_layers_keras = []
        conv_layers_torch = []

        for _ in range(n_layers):
            conv_layers_keras.append(
                tf.keras.layers.Conv2D(
                    filters=n_channels,
                    kernel_size=3,
                    activation=tf.nn.relu,
                    padding="same",
                )
            )

            conv_layers_torch += [
                torch.nn.Conv2d(
                    in_channels=n_channels,
                    out_channels=n_channels,
                    kernel_size=3,
                    padding=[1, 1],
                ),
                torch.nn.ReLU(),
            ]

        cnn_keras = tf.keras.Sequential(
            [
                tf.keras.layers.Input((size, size, n_channels)),
                *conv_layers_keras,
                tf.keras.layers.MaxPooling2D(size),
            ]
        )

        cnn_torch = torch.nn.Sequential(
            *conv_layers_torch,
            torch.nn.MaxPool2d(size),
        )

        yield cnn_keras, cnn_torch, {"n_channels": n_channels, "n_layers": n_layers}


def keras_inverted_bottleneck(channels, expansion, resolution):
    """
    Note that batchnorm is omitted since this is meant for inference only,
    where BN would get folded into the conv layers anyway.
    """
    return tf.keras.Sequential(
        [
            tf.keras.Input(shape=(resolution, resolution, channels)),
            # Expand with a pointwise 1x1 convolution.
            tf.keras.layers.Conv2D(
                expansion * channels,
                kernel_size=1,
                padding="same",
                use_bias=False,
                activation="relu6",
            ),
            # Depthwise 3x3 convolution.
            tf.keras.layers.DepthwiseConv2D(
                # no channels argument since this is a depthwise conv
                kernel_size=3,
                padding="same",
                use_bias=False,
                activation="relu6",
            ),
            # Project with a pointwise 1x1 convolution.
            tf.keras.layers.Conv2D(
                channels,
                kernel_size=1,
                padding="same",
                use_bias=False,
                activation=None,
            ),
        ]
    )


def torch_inverted_bottleneck(channels, expansion):
    expanded_channels = round(channels * expansion)

    return torch.nn.Sequential(
        torch.nn.Conv2d(
            in_channels=channels,
            out_channels=expanded_channels,
            kernel_size=1,
            bias=False,
        ),
        torch.nn.ReLU6(),
        torch.nn.Conv2d(
            in_channels=expanded_channels,
            out_channels=expanded_channels,
            groups=expanded_channels,  # Make it depthwise
            kernel_size=3,
            # Torch quantization doesn't recognize padding="same", so we need to
            # explicitly pass a tuple
            padding=[1, 1],
            bias=False,
        ),
        torch.nn.ReLU6(),
        torch.nn.Conv2d(
            in_channels=expanded_channels,
            out_channels=channels,
            kernel_size=1,
            bias=False,
        ),
    )


def inverted_bottleneck_models(n_sizes=10, expansion=6):
    """
    The MobileNet inverted bottleneck block.
    Keras implementation for reference:
    https://github.com/keras-team/keras/blob/07e13740fd181fc3ddec7d9a594d8a08666645f6/keras/applications/mobilenet_v2.py#L431
    Torch implementation for reference:
    https://github.com/pytorch/vision/blob/d4a03fc02d0566ec97341046de58160370a35bd2/torchvision/models/mobilenetv2.py#L39

    We omit batch norm as well as the residual connection.
    The first would be folded into the convolutions during inference and the second
    caused problems with PyTorch's static quantization for some reason. It's a negligible
    part of the computation anyway.
    """

    # Coefficient values from EfficientNet: https://arxiv.org/pdf/1905.11946.pdf
    resolution_coef = 1.15
    channels_coef = 1.1
    resolution_base = 14
    channels_base = 160

    for i in range(n_sizes):
        channels = math.ceil(channels_base * channels_coef**i)
        resolution = math.ceil(resolution_base * resolution_coef**i)

        model_keras = keras_inverted_bottleneck(channels, expansion, resolution)

        model_torch = torch_inverted_bottleneck(channels, expansion)

        yield model_keras, model_torch, {
            "channels": channels,
            "resolution": resolution,
            "expansion": expansion,
        }


def dilated_conv_models(n_sizes, n_layers):

    for size in range(n_sizes):
        channels = 64 * 2**size
        model = ddsp.training.nn.DilatedConvStack(
            ch=channels,
            layers_per_stack=n_layers,
            stacks=1,
            kernel_size=3,
            dilation=2,
        )

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input((64000, 1, 1)),
                model,
            ]
        )

        yield model, None, {"n_channels": channels, "n_layers": n_layers}


def get_models(kind, n_sizes, n_layers=1):
    if kind == "dense":
        models = dense_models(n_sizes=n_sizes, n_layers=n_layers)
    elif kind == "inverted_bottleneck":
        assert n_layers == 1, "n_layers is not used for inverted bottleneck models"
        models = inverted_bottleneck_models(n_sizes=n_sizes)
    elif kind == "cnn":
        models = cnn_models(n_sizes=n_sizes, n_layers=n_layers)
    elif kind == "dilated_cnn":
        models = dilated_conv_models(n_sizes=n_sizes, n_layers=n_layers)
    else:
        raise ValueError("Unknown kind")

    return models
