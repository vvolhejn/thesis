""" Based on RAVE, which in turn is based on MelGAN, based on conditional GANs... """
import einops
import gin
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.layers as tfkl

import ddsp.training


class Discriminator(tf.keras.Model):

    # Default params are from RAVE
    def __init__(self, capacity=16, multiplier=2, n_layers=4):
        super().__init__()

        layers = [
            tfa.layers.WeightNormalization(
                tfkl.Conv1D(
                    filters=capacity,
                    kernel_size=15,
                    padding="same",
                ),
            ),
            tfkl.Activation(tf.nn.leaky_relu),
        ]

        for i in range(n_layers):
            layers.append(
                # tfa.layers.WeightNormalization(
                tfkl.Conv1D(
                    filters=min(1024, capacity * multiplier ** (i + 1)),
                    kernel_size=41,
                    strides=multiplier,
                    padding="same",
                    # groups=multiplier ** (i + 1),
                )
                # )
            )
            layers.append(tfkl.Activation(tf.nn.leaky_relu))

        layers.append(
            # tfa.layers.WeightNormalization(
            tfkl.Conv1D(
                filters=min(1024, capacity * multiplier**n_layers),
                kernel_size=5,
                padding="same",
            )
            # )
        )

        layers.append(tfkl.Activation(tf.nn.leaky_relu))
        layers.append(
            tfa.layers.WeightNormalization(
                tfkl.Conv1D(
                    filters=1,
                    kernel_size=1,
                    padding="same",
                )
            )
        )

        # `layers` is a built-in name so we can't use that.
        self.net = layers

    def call(self, inputs, training=None, mask=None):
        """
        Returns the features from all conv layers in the network.
        The final layer
        """
        tf.ensure_shape(inputs, [None, None, 1])  # batch, time, 1

        features = []

        x = inputs

        for layer in self.net:
            tf.print(x.shape, layer, layer.name)
            # if isinstance(layer, tfkl.Conv1D):
            # print(layer.filters, layer.groups)

            x = layer(x)

            # The conv layers are wrapped in weight norm.
            if isinstance(layer, tfa.layers.WeightNormalization):
                features.append(x)

        # The final layer has a single score value for each
        # (downsampled) chunk of the audio
        tf.ensure_shape(features[-1], [x.shape[0], None, 1])

        return features


class MultiScaleDiscriminators(tf.keras.Model):
    def __init__(self, n, *args, **kwargs):
        """
        Take `n` `Discriminator`s that operate on more and more downsampled audio.
        """
        super().__init__()
        self.discriminators = [Discriminator(*args, **kwargs) for i in range(n)]

    def call(self, x, training=None, mask=None):
        """
        Returns a list of `Discriminator` outputs, so features_per_dis[i][j] is the output
        of the j-th layer of the i-th discriminator, a tensor of shape
        [batch, time, channels]
        """
        multiscale_features = []

        for dis in self.discriminators:
            multiscale_features.append(dis(x))

            # Halve the sampling rate for the next discriminator.
            x = tf.keras.layers.AveragePooling1D(pool_size=2, padding="valid")(x)

        return multiscale_features


def discriminator_step(
    discriminators: MultiScaleDiscriminators, x_real, x_fake, loss_mode="hinge"
):
    multiscale_features_real = discriminators(x_real)
    multiscale_features_fake = discriminators(x_fake)

    loss_dis, loss_gen = 0, 0
    scores_real, scores_fake = [], []

    loss_feature_matching = 0

    for features_real, features_fake in zip(
        multiscale_features_real, multiscale_features_fake
    ):
        loss_feature_matching += sum(
            map(
                lambda x, y: tf.reduce_mean(abs(x - y)),
                features_real,
                features_fake,
            )
        ) / len(features_real)

        loss_gen_cur, loss_dis_cur = get_adversarial_loss(
            features_real[-1],
            features_fake[-1],
            loss_mode=loss_mode,
        )

        scores_real.append(tf.reduce_mean(features_real[-1]))
        scores_fake.append(tf.reduce_mean(features_fake[-1]))

        loss_gen += loss_gen_cur
        loss_dis += loss_dis_cur

    return {
        "loss_gen": loss_gen,
        "loss_dis": loss_dis,
        "loss_feature_matching": loss_feature_matching,
        "score_real": tf.reduce_mean(scores_real),
        "score_fake": tf.reduce_mean(scores_fake),
    }


def get_adversarial_loss(score_real, score_fake, loss_mode="hinge"):
    """
    Get the losses for the generator and discriminator based on the latter's outputs.
    A high score means the discriminator thinks the output is real.

    :param score_real: The discriminator's output for the real input.
    :param score_fake: The discriminator's output for the resynthesized input.
    :param loss_mode: Which type of GAN loss. to use
    :return: Tuple (generator loss, discriminator loss)
    """

    if loss_mode == "hinge":
        loss_dis = tf.nn.relu(1 - score_real) + tf.nn.relu(1 + score_fake)
        loss_dis = tf.reduce_mean(loss_dis)
        loss_gen = -tf.reduce_mean(score_fake)
    elif loss_mode == "square":
        loss_dis = (score_real - 1).pow(2) + score_fake.pow(2)
        loss_dis = tf.reduce_mean(loss_dis)
        loss_gen = tf.reduce_mean((score_fake - 1).pow(2))
    elif loss_mode == "nonsaturating":
        score_real = tf.clip_by_value(tf.nn.sigmoid(score_real), 1e-7, 1 - 1e-7)
        score_fake = tf.clip_by_value(tf.nn.sigmoid(score_fake), 1e-7, 1 - 1e-7)
        loss_dis = -tf.reduce_mean(
            tf.math.log(score_real) + tf.math.log(1 - score_fake)
        )
        loss_gen = -tf.reduce_mean(tf.math.log(score_fake))
    else:
        raise NotImplementedError
    return loss_gen, loss_dis


@gin.configurable
class AdversarialAutoencoder(ddsp.training.models.Autoencoder):
    """
    A version of Autoencoder that also supports adversarial training (can be disabled).
    """

    def __init__(self, adversarial_training=True, **kwargs):
        super().__init__(**kwargs)

        self.adversarial_training = adversarial_training
        if adversarial_training:
            self.discriminators = MultiScaleDiscriminators(n=3)

    def call(self, features, training=False):
        outputs = super().call(features, training)

        # TODO: only train from a certain step
        # TODO: freeze encoder
        if training and self.adversarial_training:
            dis_info = discriminator_step(
                self.discriminators,
                x_real=einops.rearrange(outputs["audio"], "b t -> b t 1"),
                x_fake=einops.rearrange(
                    self.get_audio_from_outputs(outputs), "b t -> b t 1"
                ),
            )

            self._losses_dict.update(
                {
                    "adv_loss_gen": dis_info["loss_gen"],
                    "adv_loss_dis": dis_info["loss_dis"],
                    "loss_feature_matching": dis_info["loss_feature_matching"],
                }
            )

        tf.print(self._losses_dict)

        return outputs
