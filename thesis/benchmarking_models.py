import math

import tensorflow as tf
import torch

import ddsp.training
import thesis.dilated_conv
import thesis.dilated_conv_torch


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

        yield dense_keras, dense_torch, {
            "hidden_size": hidden_size,
            "expansion": expansion,
            "n_layers": n_layers,
        }


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

        yield cnn_keras, cnn_torch, {
            "n_channels": n_channels,
            "n_layers": n_layers,
            "resolution": size,
        }


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

    block_size = 4

    for i in range(n_sizes):
        channels = math.ceil(channels_base * channels_coef**i)
        # For block sparsity, TVM requires that the number of channels be a multiple
        # of the block size
        channels = math.ceil(channels / block_size) * block_size

        resolution = math.ceil(resolution_base * resolution_coef**i)

        model_keras = tf.keras.Sequential(
            [
                tf.keras.Input(shape=(resolution, resolution, channels)),
                thesis.dilated_conv.InvertedBottleneckBlock(
                    filters=channels,
                    expansion_rate=expansion,
                    batch_norm=False,
                    is_1d=False,
                ),
            ]
        )

        model_torch = thesis.dilated_conv_torch.InvertedBottleneckBlock(
            filters=channels, expansion_rate=expansion, batch_norm=False, is_1d=False
        )

        yield model_keras, model_torch, {
            "channels": channels,
            "resolution": resolution,
            "expansion": expansion,
        }


def dilated_conv_models(n_sizes, n_layers, use_inverted_bottleneck=False):
    kernel_size = 3
    dilation = 3
    input_size = 32  # 2048

    for size in range(n_sizes):
        channels = 64 * (size + 1)

        if use_inverted_bottleneck:
            channels //= 2

        model_keras = thesis.dilated_conv.DilatedConvStack(
            ch=channels,
            layers_per_stack=n_layers,
            stacks=1,
            kernel_size=kernel_size,
            dilation=dilation,
            use_inverted_bottleneck=use_inverted_bottleneck,
        )

        model_keras = tf.keras.Sequential(
            [
                tf.keras.layers.Input((input_size, 1, 1)),
                model_keras,
            ]
        )

        model_torch = thesis.dilated_conv_torch.DilatedConvStack(
            ch=channels,
            layers_per_stack=n_layers,
            stacks=1,
            kernel_size=kernel_size,
            dilation=dilation,
            use_inverted_bottleneck=use_inverted_bottleneck,
            in_ch=1,  # Need to manually specify #input channels for Torch
        )

        yield model_keras, model_torch, {
            "n_channels": channels,
            "n_layers": n_layers,
            "use_inverted_bottleneck": use_inverted_bottleneck,
            "kernel_size": kernel_size,
            "dilation": dilation,
            "input_size": input_size,
        }


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
    elif kind == "dilated_cnn_ib":
        models = dilated_conv_models(
            n_sizes=n_sizes, n_layers=n_layers, use_inverted_bottleneck=True
        )
    else:
        raise ValueError("Unknown kind")

    return models
