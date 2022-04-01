import datetime

import gin
import tensorflow as tf
from rich.pretty import pprint

import ddsp.training


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


def print_tensors_concisely(x):
    x = ddsp.training.train_util.summarize_tensors(x)
    pprint(x)


@gin.register
def summarize_generic(outputs, step):
    audios_with_labels = [
        (outputs["audio"][0], "Original"),
        (outputs["audio_synth"][0], "Synthesized"),
    ]

    ddsp.training.summaries.spectrogram_array_summary(
        audios_with_labels, name="spectrograms", step=step
    )


@gin.register
def summarize_ddspae(outputs, step):
    audios_with_labels = [
        (outputs["audio"][0], "Original"),
        (outputs["audio_synth"][0], "Synthesized"),
        (outputs["filtered_noise"]["signal"][0], "Filtered noise"),
        (outputs["harmonic"]["signal"][0], "Harmonic synth"),
    ]

    ddsp.training.summaries.spectrogram_array_summary(
        audios_with_labels, name="spectrograms", step=step
    )


def get_today_string():
    """ 0331, 0611 etc. """
    today = datetime.date.today()
    date_s = today.strftime("%m%d")
    return date_s