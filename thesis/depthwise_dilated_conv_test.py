import unittest

import numpy as np
import tensorflow as tf

import thesis.depthwise_dilated_conv


class DepthwiseDilatedConvTest(tf.test.TestCase):
    def test_basic(self):
        n_filters = 16
        dilation = 8
        conv = thesis.depthwise_dilated_conv.DepthwiseDilatedConv1D(
            filters=n_filters, dilation_rate=dilation
        )

        data = np.random.randn(2, 1000, 1, n_filters).astype(np.float32)
        # data2 = tf.random.normal((2, 1000, n_filters))

        out1 = conv(data)

        self.assertEqual(
            out1.shape,
            (2, 1000, 1, n_filters),
            msg="Unexpected output shape",
        )

        j = 10
        data[0, j, 0, 0] = -3.0
        out2 = conv(data)

        # Check that the convolution is dilated: only specific elements should change
        for i in range(30):
            should_be_equal = i not in [j - dilation, j, j + dilation]
            if should_be_equal:
                self.assertEqual(out1[0, i, 0, 0], out2[0, i, 0, 0])
            else:
                self.assertNotEqual(out1[0, i, 0, 0], out2[0, i, 0, 0])


class DepthwiseDilatedConvStackTest(tf.test.TestCase):
    def test_basic(self):
        n_filters = 16
        stacks = 2
        resample_stride = 2
        layers_per_stack = 3

        stack = thesis.depthwise_dilated_conv.DepthwiseDilatedConvStack(
            ch=n_filters,
            layers_per_stack=layers_per_stack,
            stacks=stacks,
            resample_type="downsample",
            resample_stride=resample_stride,
        )

        data = np.random.randn(2, 1000, n_filters).astype(np.float32)
        out = stack(data)

        self.assertEqual(
            out.shape,
            (2, 1000 // (resample_stride**stacks), n_filters),
        )

    def test_receptive_field(self):
        n_filters = 16
        stacks = 2
        layers_per_stack = 3
        dilation = 2

        stack = thesis.depthwise_dilated_conv.DepthwiseDilatedConvStack(
            ch=n_filters,
            layers_per_stack=layers_per_stack,
            stacks=stacks,
            dilation=2,
        )

        data = np.random.randn(2, 1000, n_filters).astype(np.float32)
        out1 = stack(data)

        receptive_field = (dilation**layers_per_stack - 1) * stacks

        j = 3 * receptive_field
        data[0, j, 0] = -3.0
        out2 = stack(data)

        # print(out1[0, :200, 0] == out2[0, :200, 0])
        # fr = np.argmin(out1[0, :200, 0] == out2[0, :200, 0])
        # to = np.argmin(
        #     (out1[0, :200, 0] == out2[0, :200, 0]) - np.linspace(-0.1, 0, 200)
        # )
        # print("actual:", fr, to, "total elements:", to - fr + 1)
        # print("prediction:", j - receptive_field, j + receptive_field)

        # Check that the receptive field of the convolution is what is expected
        for i in range(j * 3):
            # The receptive field is one lower to the left than what I would expect.
            # Honestly, I'm not sure where this off-by-one is coming from.
            should_be_equal = i not in range(
                j - receptive_field + 1, j + receptive_field + 1
            )
            if should_be_equal:
                self.assertEqual(out1[0, i, 0], out2[0, i, 0], msg=str(i))
            else:
                self.assertNotEqual(out1[0, i, 0], out2[0, i, 0], msg=str(i))


if __name__ == "__main__":
    unittest.main()
