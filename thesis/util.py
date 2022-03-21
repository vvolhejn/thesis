import tensorflow as tf


def resample(x, output_size):
    """
    Takes a tensor of shape [batch_size, time, channels]
    and stretches it (linear interpolation) to shape [batch_size, output_size, channels].
    """

    # tf.image.resize expects the shape [batch_size, w, h, channels] so we need to add
    # and then remove an extra dimension.
    y = tf.image.resize(tf.expand_dims(x, 1), [1, output_size])
    y = tf.squeeze(y, axis=1)

    tf.debugging.assert_shapes(
        [
            (x, ("batch_size", "time", "channels")),
            (y, ("batch_size", "output_size", "channels")),
        ]
    )

    return y
