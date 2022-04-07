import tensorflow as tf

from thesis import adversarial


class SingleDiscriminatorTest(tf.test.TestCase):
    def test_output_shape_is_correct(self):
        dis = adversarial.Discriminator()
        batch_size = 2
        n_samples = 16000

        batch = tf.random.normal((batch_size, n_samples, 1))
        features = dis(batch)

        self.assertIsInstance(features, list)

        for i, layer_features in enumerate(features):
            tf.ensure_shape(
                features[i], [batch_size, None, None if i != len(features) - 1 else 1]
            )


class MultiScaleDiscriminatorTest(tf.test.TestCase):
    def test_output_shape_is_correct(self):
        n = 3
        multi_dis = adversarial.MultiScaleDiscriminators(n=n)
        batch_size = 2
        n_samples = 64000

        batch = tf.random.normal((batch_size, n_samples, 1))
        multiscale_features = multi_dis(batch)

        for i in range(n - 1):
            self.assertEqual(
                multiscale_features[i][0].shape[1],
                multiscale_features[i + 1][0].shape[1] * 2,
                msg="Features should be downscaled between consecutive discriminators",
            )


class DiscriminatorStepTest(tf.test.TestCase):
    def test_output_is_correct(self):
        n = 3
        multi_dis = adversarial.MultiScaleDiscriminators(n=n)
        batch_size = 2
        n_samples = 1000

        batch_real = tf.random.normal((batch_size, n_samples, 1))
        batch_fake = tf.random.normal((batch_size, n_samples, 1))

        out = adversarial.discriminator_step(multi_dis, batch_real, batch_fake)

        keys = [
            "loss_gen",
            "loss_dis",
            "loss_feature_matching",
            "score_real",
            "score_fake",
        ]

        for k in keys:
            self.assertIn(k, out.keys())
            self.assertEqual(out[k].shape.as_list(), [])
