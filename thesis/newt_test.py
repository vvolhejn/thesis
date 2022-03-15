import tensorflow as tf

from thesis import newt


class HarmonicTest(tf.test.TestCase):
    def test_output_shape_is_correct(self):
        n_outputs = 2
        n_samples = 16000

        harm = newt.NEWTHarmonic(
            n_harmonics=3, n_outputs=n_outputs, sample_rate=16000, n_samples=16000
        )

        batch_size = 3
        f0 = tf.zeros((batch_size, n_samples, 1), dtype=tf.float32) + 440

        output = harm(f0)

        self.assertAllEqual([batch_size, n_samples, n_outputs], output.shape.as_list())

    def test_simple_sine(self):
        n_samples = 16000
        batch_size = 1
        freq = 440
        f0 = tf.zeros((batch_size, n_samples, 1), dtype=tf.float32) + freq

        harm = newt.NEWTHarmonic(
            n_harmonics=1, n_outputs=1, sample_rate=16000, n_samples=n_samples
        )
        x = harm(f0)[0, :, 0]

        peaks_mask = (x[1:-1] > x[:-2]) & (x[1:-1] > x[2:])
        n_peaks = sum(peaks_mask)

        self.assertBetween(n_peaks, freq - 1, freq + 1)


class NEWTWaveshaperTest(tf.test.TestCase):
    def test_output_shape_is_correct(self):
        batch_size = 2
        n_waveshapers = 3
        control_embedding_size = 8
        n_samples = 16000

        # control time axis is upsampled to n_samples
        control = tf.random.normal((batch_size, 100, control_embedding_size))
        exciter = tf.random.normal((batch_size, n_samples, n_waveshapers))

        shaper = newt.NEWTWaveshaper(
            n_waveshapers=n_waveshapers,
            control_embedding_size=control_embedding_size,
            shaping_fn_hidden_size=16,
        )

        output = shaper(exciter, control)

        self.assertAllEqual([batch_size, n_samples], output.shape.as_list())
