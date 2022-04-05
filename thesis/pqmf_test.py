import os

import einops
import tensorflow as tf
from ddsp.losses import SpectralLoss
from ddsp.colab.colab_utils import audio_bytes_to_np

from thesis.pqmf import PQMFBank


def get_loss(target_audio, audio):
    loss_class = SpectralLoss()
    loss = loss_class(tf.convert_to_tensor(target_audio), tf.convert_to_tensor(audio))
    return float(loss)


def get_audio_fixture():
    input_f = open(
        os.path.join(os.path.dirname(__file__), "../fixtures/violin_1s.wav"), "rb"
    )
    wav_bytes = input_f.read()
    audio = audio_bytes_to_np(wav_bytes)
    return audio


class PQMFTest(tf.test.TestCase):
    def test_pqmf(self):
        audio = get_audio_fixture()
        audio = tf.convert_to_tensor(audio)

        # Split into two halves to have a batch size of more than 1
        batch_size = 2
        audio = einops.rearrange(audio, "(b t) -> b t 1", b=batch_size)
        n_samples = audio.shape[1]

        bands = 4
        attenuation = 100

        pqmf = PQMFBank(attenuation=attenuation, n_bands=bands)

        analyzed = pqmf.analysis(audio)
        self.assertEqual(
            analyzed.shape.as_list(), [batch_size, n_samples // bands, bands]
        )

        synthesized = pqmf.synthesis(analyzed)
        self.assertEqual(synthesized.shape.as_list(), [batch_size, n_samples, 1])

        # The reconstructed audio should be close enough to the original.
        # The actual loss value should be <=0.1, so the threshold 1.0 is lenient.
        self.assertLess(get_loss(audio, synthesized), 1.0)
