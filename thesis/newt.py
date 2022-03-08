import math

import gin
import tensorflow as tf
import ddsp

tfkl = tf.keras.layers

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()


# See https://github.com/ben-hayes/neural-waveshaping-synthesis/blob/main/neural_waveshaping_synthesis/models/modules/generators.py
# @gin.configurable
class NEWTHarmonic(ddsp.processors.Processor):
    """
    Generates harmonics above a fundamental frequency.
    Has `n_outputs` output channels, where each one has a mix of harmonics determined
    by a linear layer
    """

    def __init__(self, n_harmonics, n_outputs, sample_rate, name="newt_harmonic"):
        super().__init__(name=name)
        self.sample_rate = sample_rate
        self.n_harmonics = n_harmonics
        self.harmonic_axis = self._create_harmonic_axis(n_harmonics).reshape((1, 1, -1))

        self.n_outputs = n_outputs

        # NEWT implements this as a 1D conv though it's just a linear layer,
        # but distributed along the time axis
        self.harmonic_mixer = tfkl.Conv1D(self.n_outputs, kernel_size=1)

    def _create_harmonic_axis(self, n_harmonics):
        return tf.range(1, n_harmonics + 1, dtype=tf.float32)

    def _create_antialias_mask(self, f0):
        freqs = tf.expand_dims(f0, axis=2) * self.harmonic_axis
        return freqs < (self.sample_rate / 2)

    def _create_phase_shift(self, n_harmonics):
        # TODO: why is this important?
        shift = tf.random.uniform((n_harmonics, ), minval=-math.pi, maxval=math.pi)
        return shift

    def get_controls(self, f0):
        return {"f0": f0}

    def get_signal(self, f0) -> tf.Tensor:
        phase = math.tau * tf.cumsum(f0, axis=-1) / self.sample_rate

        harmonic_phase = self.harmonic_axis * tf.expand_dims(phase, axis=2)
        harmonic_phase = harmonic_phase + self._create_phase_shift(self.n_harmonics).reshape((1, 1, -1))

        antialias_mask = self._create_antialias_mask(f0)

        harmonics = tf.sin(harmonic_phase) * tf.cast(antialias_mask, dtype=tf.float32)
        mixed = self.harmonic_mixer(harmonics)

        return mixed


@gin.register
class Harmonic(ddsp.processors.Processor):
    """Synthesize audio with a bank of harmonic sinusoidal oscillators."""

    def __init__(
        self,
        n_samples=64000,
        sample_rate=16000,
        scale_fn=ddsp.core.exp_sigmoid,
        normalize_below_nyquist=True,
        amp_resample_method="window",
        use_angular_cumsum=False,
        name="harmonic",
    ):
        """Constructor.
        Args:
          n_samples: Fixed length of output audio.
          sample_rate: Samples per a second.
          scale_fn: Scale function for amplitude and harmonic distribution inputs.
          normalize_below_nyquist: Remove harmonics above the nyquist frequency
            and normalize the remaining harmonic distribution to sum to 1.0.
          amp_resample_method: Mode with which to resample amplitude envelopes.
            Must be in ['nearest', 'linear', 'cubic', 'window']. 'window' uses
            overlapping windows (only for upsampling) which is smoother
            for amplitude envelopes with large frame sizes.
          use_angular_cumsum: Use angular cumulative sum on accumulating phase
            instead of tf.cumsum. If synthesized examples are longer than ~100k
            audio samples, consider use_angular_cumsum to avoid accumulating
            noticible phase errors due to the limited precision of tf.cumsum.
            However, using angular cumulative sum is slower on accelerators.
          name: Synth name.
        """
        super().__init__(name=name)
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.scale_fn = scale_fn
        self.normalize_below_nyquist = normalize_below_nyquist
        self.amp_resample_method = amp_resample_method
        self.use_angular_cumsum = use_angular_cumsum

    def get_controls(self, amplitudes, harmonic_distribution, f0_hz):
        """Convert network output tensors into a dictionary of synthesizer controls.
        Args:
          amplitudes: 3-D Tensor of synthesizer controls, of shape
            [batch, time, 1].
          harmonic_distribution: 3-D Tensor of synthesizer controls, of shape
            [batch, time, n_harmonics].
          f0_hz: Fundamental frequencies in hertz. Shape [batch, time, 1].
        Returns:
          controls: Dictionary of tensors of synthesizer controls.
        """
        # Scale the amplitudes.
        if self.scale_fn is not None:
            amplitudes = self.scale_fn(amplitudes)
            harmonic_distribution = self.scale_fn(harmonic_distribution)

        harmonic_distribution = ddsp.core.normalize_harmonics(
            harmonic_distribution,
            f0_hz,
            self.sample_rate if self.normalize_below_nyquist else None,
        )

        return {
            "amplitudes": amplitudes,
            "harmonic_distribution": harmonic_distribution,
            "f0_hz": f0_hz,
        }

    def get_signal(self, amplitudes, harmonic_distribution, f0_hz):
        """Synthesize audio with harmonic synthesizer from controls.
        Args:
          amplitudes: Amplitude tensor of shape [batch, n_frames, 1]. Expects
            float32 that is strictly positive.
          harmonic_distribution: Tensor of shape [batch, n_frames, n_harmonics].
            Expects float32 that is strictly positive and normalized in the last
            dimension.
          f0_hz: The fundamental frequency in Hertz. Tensor of shape [batch,
            n_frames, 1].
        Returns:
          signal: A tensor of harmonic waves of shape [batch, n_samples].
        """
        signal = ddsp.core.harmonic_synthesis(
            frequencies=f0_hz,
            amplitudes=amplitudes,
            harmonic_distribution=harmonic_distribution,
            n_samples=self.n_samples,
            sample_rate=self.sample_rate,
            amp_resample_method=self.amp_resample_method,
            use_angular_cumsum=self.use_angular_cumsum,
        )
        return signal
