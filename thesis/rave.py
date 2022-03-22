import gin
import tensorflow as tf
import tensorflow.keras.layers as tfkl
import ddsp
import ddsp.training
import einops

from thesis.util import resample

# https://github.com/caillonantoine/RAVE/blob/1.0/rave/model.py#L236
@gin.configurable
class RAVEWaveformGenerator(ddsp.processors.Processor):
    """
    Takes a control embedding at (multiband) sample rate
    and produces a multiband waveform.
    """

    def __init__(self, n_samples, n_bands, name="rave_waveform_generator"):
        super().__init__(name=name)
        self.n_samples = n_samples
        self.n_bands = n_bands

        self.wave_gen = tfkl.Conv1D(
            filters=self.n_bands,
            kernel_size=7,
            padding="same",
        )

        self.loud_gen = tfkl.Conv1D(
            filters=1,
            kernel_size=3,
            padding="same",
        )

    def get_controls(self, control_embedding):
        # [batch, n_samples_multiband, ch]
        # control_embedding = resample(control_embedding, self.n_samples // self.n_bands)
        tf.ensure_shape(control_embedding, [None, self.n_samples // self.n_bands, None])
        return {"control_embedding": control_embedding}

    def get_signal(self, control_embedding) -> tf.Tensor:
        waveform = self.wave_gen(control_embedding)
        loudness = self.loud_gen(control_embedding)

        tf.debugging.assert_shapes(
            [
                (waveform, ("batch_size", "n_samples", "channels")),
                (loudness, ("batch_size", "n_samples", 1)),
            ]
        )

        waveform = tf.tanh(waveform) * ddsp.core.exp_sigmoid(loudness)
        tf.ensure_shape(waveform, [None, self.n_samples // self.n_bands, self.n_bands])

        return waveform


# https://github.com/caillonantoine/RAVE/blob/1.0/rave/model.py#L133
@gin.configurable
class RAVEDownsamplingCNN(ddsp.processors.Processor):
    def __init__(
        self,
        ch,
        n_layers,
        downsample_per_layer,
        n_bands,
        n_noise_bands,
        name="rave_downsampling_cnn",
    ):
        """
        A convolution stack that downsamples in each layer. Used for the noise generator.

        :param ch: Number of hidden channels in each layer.
        :param n_layers: Number of layers.
        :param downsample_per_layer: Stride in each layer. In total, downsamples by
            `downsample_per_layer ** n_layers`.
        :param n_bands: How many bands are used in multiband decomposition.
        :param n_filter_banks: How many bands in the filtered noise.
        """
        super().__init__(name=name)
        self.n_bands = n_bands
        self.n_noise_bands = n_noise_bands

        layers = []
        for i in range(n_layers):
            if i < n_layers - 1:
                activation = tfkl.Activation(tf.nn.leaky_relu)
                filters = ch
            else:
                activation = None
                filters = n_bands * n_noise_bands

            layers.append(
                tfkl.Conv1D(
                    filters=filters,
                    kernel_size=3,
                    strides=downsample_per_layer,
                    padding="same",
                    activation=activation,
                )
            )

        self.net = tf.keras.Sequential(layers)

    def get_controls(self, control_embedding):
        # [batch, time, channels]
        tf.ensure_shape(control_embedding, [None, None, None])
        return {"control_embedding": control_embedding}

    def get_signal(self, control_embedding):
        y = self.net(control_embedding)
        y = einops.rearrange(
            y,
            "b t (noise_bands bands) -> b t noise_bands bands",
            bands=self.n_bands,
            noise_bands=self.n_noise_bands,
        )
        return y


@gin.register
class MultibandFilteredNoise(ddsp.synths.FilteredNoise):
    """
    A version of FilteredNoise that operates on multiband audio, meaning the function
    now has shapes
    [batch, n_frames, n_filter_banks, n_bands] -> [batch, n_samples, n_bands]
    rather than
    [batch, n_frames, n_filter_banks] -> [batch, n_samples]
    """

    def __init__(self, n_samples, n_bands, name="multiband_filtered_noise", **kwargs):
        super().__init__(name=name, n_samples=n_samples // n_bands, **kwargs)
        self.n_samples_single_band = n_samples
        self.n_bands = n_bands

    def get_controls(self, magnitudes):
        tf.ensure_shape(magnitudes, [None, None, None, self.n_bands])

        magnitudes = einops.rearrange(
            magnitudes, "batch t filters bands -> (batch bands) t filters"
        )
        return super().get_controls(magnitudes)

    def get_signal(self, magnitudes):
        signal = super().get_signal(magnitudes)
        signal = einops.rearrange(
            signal, "(batch bands) t -> batch t bands", bands=self.n_bands
        )
        # tf.ensure_shape(signal, [None, 4000, 16])  # TODO: un-hardcode

        return signal


@gin.register
class RAVECNNEncoder(ddsp.training.nn.DictLayer):
    def __init__(self, input_keys, capacity, latent_size, ratios, bias=False):
        """
        An encoder for a VAE, it computes the means and unnormalized stds of z.
        """
        super().__init__(
            input_keys=input_keys,
            output_keys=["z_mean", "z_std_raw"],
        )
        self.latent_size = latent_size

        layers = [
            tfkl.Conv1D(filters=capacity, kernel_size=7, padding="same", use_bias=bias)
        ]

        for i, r in enumerate(ratios):
            out_dim = 2 ** (i + 1) * capacity

            layers.append(tfkl.BatchNormalization())
            layers.append(tfkl.Activation(tf.nn.leaky_relu))
            layers.append(
                tfkl.Conv1D(
                    filters=out_dim,
                    kernel_size=2 * r + 1,
                    padding="same",
                    strides=r,
                    use_bias=bias,
                )
            )

        layers.append(tfkl.Activation(tf.nn.leaky_relu))
        layers.append(
            tfkl.Conv1D(
                filters=2 * latent_size,
                kernel_size=5,
                padding="same",
                groups=2,
                use_bias=bias,
            )
        )

        self.net = tf.keras.Sequential(layers)

    def call(self, audio):
        z = self.net(audio)
        res = ddsp.training.nn.split_to_dict(
            z, (("z_mean", self.latent_size), ("z_std_raw", self.latent_size))
        )

        return res
