import numpy as np
import IPython.display
import note_seq

from ddsp.training.plotting import specplot


def play_audio(audio, sample_rate=16000, normalize=False):
    audio = np.array(audio)
    audio = np.squeeze(audio)
    IPython.display.display(IPython.display.Audio(audio, rate=sample_rate, normalize=normalize))


def audio_bytes_to_np(wav_data,
                      sample_rate=16000,
                      normalize_db=0.1,
                      mono=True):
  """Convert audio file data (in bytes) into a numpy array using Pydub.

  From DDSP's Colab utils.

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