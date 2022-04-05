import einops
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

        n_samples2 = 100
        exciter2 = tf.linspace(-5, 5, n_samples2)
        exciter2 = einops.repeat(
            exciter2,
            "x -> b x n_waveshapers",
            n_waveshapers=shaper.n_waveshapers,
            b=batch_size,
        )
        unmixed_output = shaper.shaping_fn(exciter2)

        self.assertAllEqual(
            [batch_size, n_samples2, shaper.n_waveshapers],
            unmixed_output.shape.as_list(),
        )

        self.assertNotAllClose(
            unmixed_output[0, :, 0],
            unmixed_output[0, :, 1],
            msg="The output for different waveshapers should be different.",
        )

        self.assertAllClose(
            unmixed_output[0, :, 0],
            unmixed_output[1, :, 0],
            msg="The output for different batches should be the same.",
        )


class CachedWaveshapersTest(tf.test.TestCase):
    def test_matches_original_values(self):
        n_waveshapers = 3
        max_value = 3
        n_samples = 100
        batch_size = 2

        waveshapers = newt.TrainableShapingFunctions(n_waveshapers=n_waveshapers)

        cached_waveshapers = newt.CachedShapingFunctions(
            waveshapers, min_value=-max_value, max_value=max_value
        )

        exciter = tf.linspace(-max_value, max_value, n_samples)
        exciter = einops.repeat(
            exciter,
            "x -> b x n_waveshapers",
            n_waveshapers=n_waveshapers,
            b=batch_size,
        )

        true_values = waveshapers(exciter)
        cached_values = cached_waveshapers(exciter)

        # Checks that absolute(a - b) <= (atol + rtol * absolute(b))
        self.assertAllClose(
            true_values,
            cached_values,
            atol=1e-5,
            msg="Cached values do not match the original",
        )

        # Check behavior for out-of-bounds elements.
        exciter_oob = tf.convert_to_tensor(
            [-max_value - 1, -max_value, max_value, max_value + 1]
        )
        exciter_oob = einops.repeat(
            exciter_oob, "x -> 1 x n_waveshapers", n_waveshapers=n_waveshapers
        )

        oob_values = cached_waveshapers(exciter_oob)

        self.assertAllClose(
            oob_values[0, 0, :],
            oob_values[0, 1, :],
            msg="Beyond the cached bounds, should return the value at the border (lower)",
        )
        self.assertAllClose(
            oob_values[0, 2, :],
            oob_values[0, 3, :],
            msg="Beyond the cached bounds, should return the value at the border (upper)",
        )
