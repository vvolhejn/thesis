import gin

import ddsp.training
import tensorflow as tf
from codetiming import Timer


@gin.register
class VAE(ddsp.training.models.Autoencoder):
    """Variational autoencoder."""

    def __init__(self, kl_loss_weight=1.0, **kwargs):
        super().__init__(**kwargs)
        assert self.encoder is not None

        self.kl_loss_weight = kl_loss_weight

    def encode(self, features, training=True):
        """
        Get conditioning by preprocessing then encoding.
        Here is the "variational" part: the latents are drawn from a distribution
        parametrized by the encoder.
        It must return a dict with the keys "z_mean" and "z_std_raw".
        """

        with Timer("Autoencoder.preprocessor", logger=None):
            if self.preprocessor is not None:
                features.update(self.preprocessor(features, training=training))

        with Timer("Autoencoder.encoder", logger=None):
            if self.encoder is not None:
                z_dict = self.encoder(features)

                assert "z_mean" in z_dict
                assert "z_std_raw" in z_dict
                tf.debugging.assert_shapes(
                    [
                        (z_dict["z_mean"], ("batch_size", "time", "channels")),
                        (z_dict["z_std_raw"], ("batch_size", "time", "channels")),
                    ]
                )

                mean = z_dict["z_mean"]
                std = tf.math.softplus(z_dict["z_std_raw"]) + 1e-4
                var = std * std

                # Not clear if TF can differentiate through this
                # (without the reparametrization trick)
                # z = tf.random.normal(mean.shape, mean=mean, stddev=std)
                z = tf.random.normal(mean.shape) * mean + std

                kl = tf.reduce_mean(
                    tf.reduce_sum((mean * mean + var - tf.math.log(var) - 1), axis=1)
                )

                kl_loss = kl * self.kl_loss_weight

                self._losses_dict.update({"kl_loss": kl_loss})

                features.update(z_dict)
                features.update({"z": z, "kl": kl})

        return features
