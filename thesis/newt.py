import math

import gin
import tensorflow as tf
import ddsp
import ddsp.training
import einops

tfkl = tf.keras.layers

from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()


# See https://github.com/ben-hayes/neural-waveshaping-synthesis/blob/main/neural_waveshaping_synthesis/models/modules/generators.py
@gin.configurable
class NEWTHarmonic(ddsp.processors.Processor):
    """
    Generates harmonics above a fundamental frequency.
    Has `n_outputs` output channels, where each one has a mix of harmonics determined
    by a linear layer
    """

    def __init__(
        self, n_harmonics, n_outputs, n_samples, sample_rate, name="newt_harmonic"
    ):
        super().__init__(name=name)
        self.sample_rate = sample_rate
        self.n_harmonics = n_harmonics
        self.harmonic_axis = self._create_harmonic_axis(n_harmonics).reshape((1, 1, -1))

        self.n_outputs = n_outputs
        self.n_samples = n_samples

        self.harmonic_mixer = tfkl.Dense(self.n_outputs)

    def _create_harmonic_axis(self, n_harmonics):
        return tf.range(1, n_harmonics + 1, dtype=tf.float32)

    def _create_antialias_mask(self, f0):
        freqs = tf.expand_dims(f0, axis=2) * self.harmonic_axis
        return freqs < (self.sample_rate / 2)

    @staticmethod
    def _create_phase_shift(n_harmonics):
        # TODO: why is this important?
        shift = tf.random.uniform((n_harmonics,), minval=-math.pi, maxval=math.pi)
        return shift

    def get_controls(self, f0):
        f0_upsampled = resample(f0, self.n_samples)
        f0_upsampled = tf.squeeze(f0_upsampled, axis=2)

        return {"f0_upsampled": f0_upsampled}

    def get_signal(self, f0_upsampled) -> tf.Tensor:
        phase = math.tau * tf.cumsum(f0_upsampled, axis=-1) / self.sample_rate

        harmonic_phase = self.harmonic_axis * tf.expand_dims(phase, axis=2)
        harmonic_phase = harmonic_phase + self._create_phase_shift(
            self.n_harmonics
        ).reshape((1, 1, -1))

        antialias_mask = self._create_antialias_mask(f0_upsampled)

        harmonics = tf.sin(harmonic_phase) * tf.cast(antialias_mask, dtype=tf.float32)
        mixed = self.harmonic_mixer(harmonics)

        tf.debugging.assert_shapes(
            [
                (f0_upsampled, ("batch_size", "n_samples")),
                (mixed, ("batch_size", "n_samples", "n_outputs")),
            ]
        )

        return mixed


class NEWTFc(tf.keras.Sequential):
    """
    A fully connected layer, optionally with layer normalization and an activation.
    """

    def __init__(self, ch=128, nonlinearity="leaky_relu", layer_norm=False, **kwargs):
        layers = [tfkl.Dense(ch)]
        if layer_norm:
            layers.append(tfkl.LayerNormalization())

        if nonlinearity is not None:
            layers.append(
                tfkl.Activation(ddsp.training.nn.get_nonlinearity(nonlinearity))
            )

        super().__init__(layers, **kwargs)


class NEWTFcStack(tf.keras.Sequential):
    """
    Stack Dense -> LayerNorm -> Leaky ReLU layers.
    Like the FcStack class from DDSP, but has an output layer with a different size
    and without a nonlinearity applied
    """

    def __init__(
        self,
        layers=2,
        hidden_size=256,
        out_size=256,
        nonlinearity="leaky_relu",
        out_nonlinearity=tf.identity,
        layer_norm=True,
        out_layer_norm=False,
        **kwargs
    ):
        assert layers >= 2, "Depth must be at least 2"

        layers_list = []
        for i in range(layers):
            if i < layers - 1:
                layers_list.append(NEWTFc(hidden_size, nonlinearity, layer_norm))
            else:
                layers_list.append(NEWTFc(out_size, out_nonlinearity, out_layer_norm))

        super().__init__(layers_list, **kwargs)


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


@gin.configurable
class NEWTWaveshaper(ddsp.processors.Processor):
    """
    Applies waveshapers to exciter signals.
    """

    def __init__(
        self,
        n_waveshapers: int,
        control_embedding_size: int,
        shaping_fn_hidden_size: int = 16,
        name="newt_waveshaper",
    ):
        super().__init__(name=name)

        self.n_waveshapers = n_waveshapers

        self.mlp = NEWTFcStack(
            hidden_size=control_embedding_size, out_size=n_waveshapers * 4, layers=4
        )

        self.input_scale = tf.Variable(
            tf.random.normal(shape=[1, n_waveshapers, 1], stddev=10),
            trainable=True,
        )

        self.shaping_fn = NEWTFcStack(
            hidden_size=shaping_fn_hidden_size,
            out_size=1,
            nonlinearity=tf.sin,
            out_nonlinearity=tf.sin,
            layer_norm=False,
            out_layer_norm=False,
        )

        self.mixer = tfkl.Dense(1)

    def get_controls(self, exciter, control_embedding):
        return {"exciter": exciter, "control_embedding": control_embedding}

    def get_signal(self, exciter, control_embedding):
        """
        :param exciter: tensor of shape [batch_size, n_samples, n_waveshapers]
        :param control_embedding: tensor of shape
            [batch_size, n_control_samples, control_embedding_size].
            n_samples should be a multiple of n_control_samples.
        :return: the exciters modulated with the waveshapers and mixed down.
            A tensor of shape [batch_size, n_samples].
        """
        tf.debugging.assert_shapes(
            [
                (exciter, ("batch_size", "n_samples", "n_waveshapers")),
                (control_embedding, ("batch_size", "n_control_samples", "n_control")),
            ]
        )

        film_params = self.mlp(control_embedding)
        film_params = resample(film_params, exciter.shape[1])

        gamma_index, beta_index, gamma_norm, beta_norm = tf.split(
            film_params, 4, axis=2
        )

        x = exciter * gamma_index + beta_index

        # We add an axis so that [b, t, waveshaper] all act as batch dimensions
        # and we apply a function f: R->R to each element separately
        x = einops.rearrange(x, "b t waveshaper -> b t waveshaper 1")
        x = self.shaping_fn(x)

        tf.debugging.assert_shapes(
            [
                (exciter, ("batch_size", "n_samples", "n_waveshapers")),
                (x, ("batch_size", "n_samples", "n_waveshapers", "1")),
            ]
        )
        x = einops.rearrange(x, "b t waveshaper 1 -> b t waveshaper")

        x = x * gamma_norm + beta_norm

        return tf.squeeze(self.mixer(x), axis=2)
