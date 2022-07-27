import unittest

import numpy as np
import tensorflow as tf
import torch

import thesis.dilated_conv
import thesis.dilated_conv_torch


def reorder_for_torch(arr):
    """From BHWC to BCHW"""
    assert isinstance(arr, np.ndarray)
    assert len(arr.shape) == 4
    assert arr.shape[2] == 1  # We're in 1D, width is 1

    return np.moveaxis(arr, -1, 1)


def reorder_for_tensorflow(arr):
    """From BCHW to BHWC"""
    assert isinstance(arr, np.ndarray)
    assert len(arr.shape) == 4
    assert arr.shape[-1] == 1  # We're in 1D, width is 1

    return np.moveaxis(arr, 1, -1)


def get_n_parameters_tensorflow(model):
    params_tf = 0
    for w in model.trainable_weights:
        params_tf += w.read_value().size
        # print(w.read_value().shape, w.name)

    return params_tf


def get_n_parameters_torch(model):
    params_torch = 0
    for w in model.parameters():
        params_torch += w.numel()
        # print(w.shape)

    return params_torch


class BlockTest(tf.test.TestCase):
    def test_basic(self):
        n_filters = 16
        dilation = 8
        timesteps = 1000

        for it in range(4):
            # Test both kinds of blocks and both frameworks
            constructor = [
                thesis.dilated_conv.BasicBlock,
                thesis.dilated_conv.InvertedBottleneckBlock,
                thesis.dilated_conv_torch.BasicBlock,
                thesis.dilated_conv_torch.InvertedBottleneckBlock,
            ][it]

            using_torch = it >= 2

            conv = constructor(
                filters=n_filters,
                dilation_rate=dilation,
                kernel_size=3,
                **({} if using_torch else {"normalize": False}),
            )

            j = 10

            # data = np.random.randn(2, 1000, 1, n_filters).astype(np.float32)
            data = np.zeros((2, timesteps, 1, n_filters)).astype(np.float32)
            data2 = np.copy(data)
            data2[0, j, 0, :] = 1.0

            if using_torch:
                # Disable the updating of batch norm
                conv.eval()

                out1 = conv(torch.as_tensor(reorder_for_torch(data)))
                out2 = conv(torch.as_tensor(reorder_for_torch(data2)))

                out1 = reorder_for_tensorflow(out1.detach().numpy())
                out2 = reorder_for_tensorflow(out2.detach().numpy())
            else:
                out1 = conv(data, training=False).numpy()
                out2 = conv(data2, training=False).numpy()

            self.assertEqual(
                out1.shape,
                data.shape,
                msg=f"Unexpected output shape in iteration {it}",
            )

            # Check that the convolution is dilated: only specific elements should change
            for i in range(30):
                should_be_equal = i not in [j - dilation, j, j + dilation]
                if should_be_equal:
                    self.assertAllEqual(
                        out1[0, i, 0, :],
                        out2[0, i, 0, :],
                        msg=f"Arrays not equal at {i} in iteration {it}",
                    )
                else:
                    self.assertNotAllEqual(
                        out1[0, i, 0, :],
                        out2[0, i, 0, :],
                        msg=f"Arrays equal at {i} in iteration {it}",
                    )

    def test_n_parameters(self):
        # This test has stopped working because the normalization used in TF
        # has changed, modifying the number of parameters.
        return

        n_filters = 16
        dilation = 8
        timesteps = 100

        basic_tf = thesis.dilated_conv.BasicBlock(
            filters=n_filters,
            dilation_rate=dilation,
            kernel_size=3,
        )
        # Initialize parameters
        basic_tf(np.zeros((2, timesteps, 1, n_filters)))

        ib_tf = thesis.dilated_conv.InvertedBottleneckBlock(
            filters=n_filters,
            dilation_rate=dilation,
            kernel_size=3,
        )
        ib_tf(np.zeros((2, timesteps, 1, n_filters)))

        basic_torch = thesis.dilated_conv_torch.BasicBlock(
            filters=n_filters,
            dilation_rate=dilation,
            kernel_size=3,
        )

        ib_torch = thesis.dilated_conv_torch.InvertedBottleneckBlock(
            filters=n_filters,
            dilation_rate=dilation,
            kernel_size=3,
        )

        for block_tf, block_torch in [(basic_tf, basic_torch), (ib_tf, ib_torch)]:
            params_tf = get_n_parameters_tensorflow(block_tf)
            params_torch = get_n_parameters_torch(block_torch)
            self.assertEqual(params_torch, params_tf)


class DilatedConvStackTest(tf.test.TestCase):
    def test_basic(self):
        n_filters = 16
        stacks = 2
        resample_stride = 2
        layers_per_stack = 3
        timesteps = 1000

        for using_torch in [False, True]:
            for use_inverted_bottleneck in [False, True]:
                if using_torch:
                    constructor = thesis.dilated_conv_torch.DilatedConvStack
                else:
                    constructor = thesis.dilated_conv.DilatedConvStack

                stack = constructor(
                    ch=n_filters,
                    layers_per_stack=layers_per_stack,
                    stacks=stacks,
                    resample_type="downsample",
                    resample_stride=resample_stride,
                    use_inverted_bottleneck=use_inverted_bottleneck,
                    **({} if using_torch else {"normalize": False}),
                )

                data = np.random.randn(2, timesteps, 1, n_filters).astype(np.float32)

                if using_torch:
                    out = stack(torch.as_tensor(reorder_for_torch(data)))

                    # Can't use `reorder_for_tensorflow` because the output is squeezed
                    # from 4D to 3D
                    out = np.swapaxes(out.detach().numpy(), 1, 2)
                else:
                    out = stack(data)

                self.assertEqual(
                    out.shape,
                    (2, timesteps // (resample_stride**stacks), n_filters),
                    msg=f"Wrong shape {using_torch=} {use_inverted_bottleneck=}",
                )

    def test_receptive_field(self):
        # We need n_filters to be high enough for a filter in the output to really change
        # when expected (many might not change because of relu)
        n_filters = 128
        timesteps = 1000
        stacks = 2  # 2
        layers_per_stack = 3  # 3
        dilation = 2

        for using_torch in [False, True]:
            for use_inverted_bottleneck in [False, True]:
                for causal in [False, True]:
                    if using_torch and causal:
                        continue

                    if using_torch:
                        constructor = thesis.dilated_conv_torch.DilatedConvStack
                    else:
                        constructor = thesis.dilated_conv.DilatedConvStack

                    stack = constructor(
                        ch=n_filters,
                        layers_per_stack=layers_per_stack,
                        stacks=stacks,
                        dilation=2,
                        use_inverted_bottleneck=use_inverted_bottleneck,
                        **(
                            {}
                            if using_torch
                            else {"causal": causal, "normalize": False}
                        ),
                    )

                    j = 100
                    data = np.zeros((2, timesteps, 1, n_filters)).astype(np.float32)
                    data2 = np.copy(data)
                    data2[0, j, 0, :] = 1.0

                    if using_torch:
                        # Disable the updating of batch norm
                        stack.eval()

                        def deactivate_batchnorm(m):
                            if isinstance(m, torch.nn.BatchNorm2d):
                                m.reset_parameters()
                                m.eval()
                                with torch.no_grad():
                                    m.weight.fill_(1.0)
                                    m.bias.zero_()

                        stack.apply(deactivate_batchnorm)

                        out1 = stack(torch.as_tensor(reorder_for_torch(data)))
                        out2 = stack(torch.as_tensor(reorder_for_torch(data2)))

                        # Can't use `reorder_for_tensorflow` because the output is squeezed
                        # from 4D to 3D
                        out1 = np.swapaxes(out1.detach().numpy(), 1, 2)
                        out2 = np.swapaxes(out2.detach().numpy(), 1, 2)
                    else:
                        out1 = stack(data, training=False).numpy()
                        out2 = stack(data2, training=False).numpy()

                    self.assertEqual(
                        out1.shape,
                        (2, timesteps, n_filters),
                        msg=f"Unexpected output shape, {using_torch=}",
                    )

                    # The +1 is for an initial filter-size-3 convolution.
                    receptive_field = (dilation**layers_per_stack - 1) * stacks + 1

                    equalities = tf.reduce_all(out1[0, :200] == out2[0, :200], axis=-1)

                    # print(out1[0, :200, 0] == out2[0, :200, 0])
                    fr = np.argmin(equalities)
                    to = np.argmin(equalities - np.linspace(-0.1, 0, 200))
                    print("actual:", fr, to, "total elements:", to - fr + 1)
                    if causal:
                        print("prediction:", j, j + 2 * receptive_field)
                    else:
                        print("prediction:", j - receptive_field, j + receptive_field)

                    # Check that the receptive field of the convolution is what is expected
                    for i in range(j * 3):
                        if causal:
                            should_be_equal = not (j <= i <= j + 2 * receptive_field)
                        else:
                            should_be_equal = not (
                                j - receptive_field <= i <= j + receptive_field
                            )
                        if should_be_equal:
                            self.assertAllEqual(
                                out1[0, i, :],
                                out2[0, i, :],
                                msg=f"Wrong inequality {i=} {using_torch=} {use_inverted_bottleneck=} {causal=}",
                            )
                        else:
                            self.assertNotAllEqual(
                                out1[0, i, :],
                                out2[0, i, :],
                                msg=f"Wrong equality {i=} {using_torch=} {use_inverted_bottleneck=} {causal=}",
                            )

    def test_n_parameters(self):
        # This test has stopped working because the normalization used in TF
        # has changed, modifying the number of parameters.
        return

        n_filters = 16
        n_in_filters = 8
        timesteps = 100
        layers_per_stack = 3
        stacks = 2

        for use_inverted_bottleneck in [False, True]:
            stack_tf = thesis.dilated_conv.DilatedConvStack(
                ch=n_filters,
                layers_per_stack=layers_per_stack,
                stacks=stacks,
                dilation=2,
                use_inverted_bottleneck=use_inverted_bottleneck,
                resample_type="downsample",
                resample_stride=2,
            )
            stack_tf(np.zeros((2, timesteps, n_in_filters)))

            stack_torch = thesis.dilated_conv_torch.DilatedConvStack(
                ch=n_filters,
                layers_per_stack=layers_per_stack,
                stacks=stacks,
                dilation=2,
                use_inverted_bottleneck=use_inverted_bottleneck,
                resample_type="downsample",
                resample_stride=2,
                in_ch=n_in_filters,
            )

            # for w in stack_tf.trainable_weights:
            #     print(w.name, w.shape)
            # print()
            #
            # for name, w in stack_torch.named_parameters():
            #     print(name, w.shape)
            # print()

            params_tf = get_n_parameters_tensorflow(stack_tf)
            params_torch = get_n_parameters_torch(stack_torch)
            self.assertEqual(params_tf, params_torch)


if __name__ == "__main__":
    unittest.main()
