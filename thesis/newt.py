import math

import gin
import tensorflow as tf
import ddsp
import ddsp.training
import einops

tfkl = tf.keras.layers

from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()

from thesis.util import resample

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


class ScalingLayer(tf.keras.layers.Layer):
    def __init__(self, n_waveshapers, stddev=10):
        super().__init__()
        self.stddev = stddev
        self.scale = tf.Variable(
            tf.random.normal(shape=[1, 1, n_waveshapers], stddev=stddev),
            trainable=True,
        )

    def call(self, inputs):
        return inputs * self.scale


class TrainableShapingFunctions(tf.keras.Sequential):
    """
    Stack Dense -> LayerNorm -> Leaky ReLU layers.
    Like the FcStack class from DDSP, but has an output layer with a different size
    and without a nonlinearity applied
    """

    def __init__(
        self,
        layers=2,
        hidden_size=8,
        n_waveshapers=64,
        nonlinearity=tf.sin,
        out_nonlinearity=tf.sin,
        **kwargs
    ):
        super().__init__()
        self.n_waveshapers = n_waveshapers

        assert layers >= 2, "Depth must be at least 2"

        layers_list = [
            # tfkl.Lambda(lambda x: x * self.input_scale)
            ScalingLayer(n_waveshapers)
        ]

        for i in range(layers):

            if i < layers - 1:
                filters = n_waveshapers * hidden_size
                cur_nonlinearity = nonlinearity
            else:
                filters = n_waveshapers
                cur_nonlinearity = out_nonlinearity

            layers_list.append(
                tfkl.Conv1D(
                    filters=filters,
                    kernel_size=1,
                    padding="same",
                    groups=n_waveshapers,
                    use_bias=True,
                )
            )

            layers_list.append(
                tfkl.Activation(ddsp.training.nn.get_nonlinearity(cur_nonlinearity))
            )
            # TODO: test if this works

        super().__init__(layers_list, **kwargs)


class CachedShapingFunctions(tf.keras.layers.Layer):
    def __init__(
        self,
        waveshapers: TrainableShapingFunctions,
        min_value=-3,
        max_value=3,
        n_buckets=4096,
    ):
        """
        Precomputes the values of a `TranableWaveshapers` instance and returns a linear
        interpolation when called.
        """
        super().__init__()

        self.min_value = min_value
        self.max_value = max_value
        self.n_buckets = n_buckets
        self.n_waveshapers = waveshapers.n_waveshapers

        x = tf.linspace(min_value, max_value, n_buckets)
        x = einops.repeat(
            x,
            "n_samples -> 1 n_samples n_waveshapers",
            n_waveshapers=waveshapers.n_waveshapers,
        )
        self.lookup_table = waveshapers(x)[0]

    def lookup(self, indices):
        tf.ensure_shape(indices, [None, None, self.n_waveshapers])

        # The indices we need to know which waveshaper we want for each value.
        waveshaper_indices = einops.repeat(
            tf.range(self.n_waveshapers),
            "n_waveshapers -> b t n_waveshapers",
            b=indices.shape[0],
            t=indices.shape[1],
        )
        indices = tf.stack([indices, waveshaper_indices], axis=-1)

        return tf.gather_nd(params=self.lookup_table, indices=indices)

    def call(self, inputs):
        tf.ensure_shape(inputs, [None, None, self.n_waveshapers])
        indices_continuous = (
            (self.n_buckets - 1)
            * (inputs - self.min_value)
            / (self.max_value - self.min_value)
        )

        indices_lower = tf.clip_by_value(
            tf.floor(indices_continuous).astype(tf.int32), 0, self.n_buckets - 1
        )
        indices_upper = tf.clip_by_value(
            tf.floor(indices_continuous).astype(tf.int32) + 1, 0, self.n_buckets - 1
        )

        values_lower = self.lookup(indices_lower)
        values_upper = self.lookup(indices_upper)

        upper_fraction = indices_continuous - indices_lower

        # Do linear interpolation. Beyond the bounds of the lookup table,
        # the min/max value is returned.
        return values_lower + (values_upper - values_lower) * upper_fraction


@gin.configurable
class NEWTWaveshaper(ddsp.processors.Processor):
    """
    Applies waveshapers to exciter signals.
    """

    def __init__(
        self,
        n_waveshapers: int,
        control_embedding_size: int,
        shaping_fn_hidden_size: int = 8,
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

        self.shaping_fn = TrainableShapingFunctions(
            n_waveshapers=n_waveshapers, hidden_size=shaping_fn_hidden_size
        )

        self.cached_shaping_fn = None

        self.mixer = tfkl.Dense(1)
        self.lookup_table = None

    def cache_shaping_fn(self):
        self.cached_shaping_fn = CachedShapingFunctions(self.shaping_fn)

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

        if self.cached_shaping_fn:
            x = self.cached_shaping_fn(x)
        else:
            x = self.shaping_fn(x)

        tf.ensure_shape(x, exciter.shape)

        x = x * gamma_norm + beta_norm

        return tf.squeeze(self.mixer(x), axis=2)


@gin.register
def summarize_newt(outputs, step):
    audios_with_labels = [
        (outputs["audio"][0], "Original"),
        (outputs["audio_synth"][0], "Synthesized"),
        (outputs["filtered_noise"]["signal"][0], "Filtered noise"),
        (outputs["newt_waveshaper"]["signal"][0], "Waveshaper"),
    ]

    ddsp.training.summaries.spectrogram_array_summary(
        audios_with_labels, name="spectrograms", step=step
    )
