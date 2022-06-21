import numpy as np
import IPython.display

from ddsp.training.plotting import specplot


def play_audio(audio, sample_rate=16000):
    audio = np.array(audio)
    audio = np.squeeze(audio)
    IPython.display.display(IPython.display.Audio(audio, rate=sample_rate))
