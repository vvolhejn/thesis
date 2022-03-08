from ddsp import core
from ddsp import synths
import numpy as np
import tensorflow as tf

from thesis import newt

class HarmonicTest(tf.test.TestCase):
    def test_output_shape_is_correct(self):
        n_outputs = 2
        harm = newt.NEWTHarmonic(n_harmonics=3, n_outputs=n_outputs, sample_rate=16000)

        n_samples = 16000
        batch_size = 3
        f0 = tf.zeros((batch_size, n_samples), dtype=tf.float32) + 440

        output = harm(f0)

        self.assertAllEqual([batch_size, n_samples, n_outputs], output.shape.as_list())

    def test_simple_sine(self):
        harm = newt.NEWTHarmonic(n_harmonics=1, n_outputs=1, sample_rate=16000)

        n_samples = 16000
        batch_size = 1
        freq = 440
        f0 = tf.zeros((batch_size, n_samples), dtype=tf.float32) + freq

        x = harm(f0)[0,:,0]

        peaks_mask = (x[1:-1] > x[:-2]) & (x[1:-1] > x[2:])
        n_peaks = sum(peaks_mask)

        self.assertBetween(n_peaks, freq - 1, freq + 1)
