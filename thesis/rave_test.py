import tensorflow as tf

from thesis import rave


class FilteredNoiseTest(tf.test.TestCase):
    def test_downsampler_output_shape_is_correct(self):
        n_layers = 3
        downsample_per_layer = 2
        n_bands = 8
        n_filter_banks = 4
        batch_size = 2
        n_samples = 16000
        embedding_size = 32

        net = rave.RAVEDownsamplingCNN(
            ch=8,
            n_layers=n_layers,
            downsample_per_layer=downsample_per_layer,
            n_bands=n_bands,
            n_filter_banks=n_filter_banks,
        )

        x = tf.random.normal((batch_size, n_samples, embedding_size))
        y = net(x)

        self.assertEqual(
            y.shape.as_list(),
            [
                batch_size,
                n_samples // (downsample_per_layer**n_layers),
                n_bands * n_filter_banks,
            ],
        )
