import os

import einops
import tensorflow as tf
from ddsp.losses import SpectralLoss
# from ddsp.colab.colab_utils import audio_bytes_to_np
import note_seq

from thesis.pqmf import PQMFBank


def audio_bytes_to_np(wav_data,
                      sample_rate=16000,
                      normalize_db=0.1,
                      mono=True):
  """Convert audio file data (in bytes) into a numpy array using Pydub.

  Args:
    wav_data: A byte stream of audio data.
    sample_rate: Resample recorded audio to this sample rate.
    normalize_db: Normalize the audio to this many decibels. Set to None to skip
      normalization step.
    mono: Force stereo signals to single channel. If false, output can one or
      two channels depending on the source signal.

  Returns:
    An array of the recorded audio at sample_rate, shape [channels, time].
  """
  return note_seq.audio_io.wav_data_to_samples_pydub(
      wav_data=wav_data, sample_rate=sample_rate, normalize_db=normalize_db,
      num_channels=1 if mono else None)



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
