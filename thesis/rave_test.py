import tensorflow as tf

from thesis import rave, pqmf


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


class EncoderTest(tf.test.TestCase):
    def test_output_shape_is_correct(self):
        latent_size = 16
        batch_size = 2
        n_samples = 16000
        n_bands = 8

        net = rave.RAVECNNEncoder(
            input_keys=["audio"],
            capacity=8,
            latent_size=latent_size,
            ratios=[4, 2],
        )

        # we want multiband audio
        x = tf.random.normal((batch_size, n_samples, n_bands))
        outputs = net(x)

        self.assertEqual(
            outputs["z_mean"].shape.as_list(),
            [
                batch_size,
                n_samples // (4 * 2),  # Downsampling given by `ratios`
                latent_size,
            ],
        )
        self.assertEqual(
            outputs["z_std_raw"].shape.as_list(),
            [
                batch_size,
                n_samples // (4 * 2),
                latent_size,
            ],
        )


class VAETest(tf.test.TestCase):
    def test_encoder(self):
        latent_size = 16
        batch_size = 2
        n_samples = 16000
        n_bands = 8

        preprocessor = pqmf.PQMFAnalysis(
            pqmf.PQMFBank(attenuation=100, n_bands=n_bands)
        )

        encoder = rave.RAVECNNEncoder(
            input_keys=["audio_multiband"],
            capacity=8,
            latent_size=latent_size,
            ratios=[4, 2],
        )

        model = rave.VariationalAutoencoder(preprocessor=preprocessor, encoder=encoder)

        # Single-band audio because we want the PQMF preprocessor to analyze the signal
        x = tf.random.normal((batch_size, n_samples, 1))

        features = model.encode({"audio": x})
        z1 = features["z"]

        self.assertEqual(
            features["audio_multiband"].shape.as_list(),
            [batch_size, n_samples // n_bands, n_bands],
            "Audio is not converted to multiband properly",
        )
        self.assertEqual(
            features["z"].shape.as_list(),
            [batch_size, n_samples // (n_bands * 4 * 2), latent_size],
            "Audio is not converted to multiband properly",
        )

        for key in ["z", "z_mean", "z_std_raw", "audio_multiband"]:
            self.assertIn(key, features, "Features are missing a key")

        self.assertIn("kl_loss", model._losses_dict, "KL loss is not saved")
        self.assertGreater(model._losses_dict["kl_loss"], 0, "KL loss is not positive")

        # Encode again with different randomness. Make sure encodings differ.
        features2 = model.encode({"audio": x})
        z2 = features2["z"]

        self.assertNotAllClose(z1, z2, msg="Encodings are deterministic")