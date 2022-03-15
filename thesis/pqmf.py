"""
Pseudo Quadrature Mirror Filter bank for multi-band decomposition of a signal.
As seen in RAVE and elsewhere.

Based on Antoine Caillon's implementation from RAVE:
    https://github.com/caillonantoine/RAVE/blob/master/rave/pqmf.py
and on the implementation from TensorFlowTTS:
    https://github.com/TensorSpeech/TensorFlowTTS/blob/master/tensorflow_tts/models/mb_melgan.py
TODO: check what their respective licenses require as a notice
TODO: implement the faster version from RAVE if needed

The TensorFlowTTS implementation yielded a reconstruction with audible artifacts
so the code is based more on the RAVE implementation.
"""

import math

import numpy as np
import tensorflow as tf
from scipy.signal import kaiser, kaiserord, firwin
from scipy.optimize import fmin
from einops import rearrange

from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()


def reverse_half(x):
    mask = np.ones_like(x)
    mask[..., ::2, 1::2] = -1

    return x * mask


def center_pad_next_pow_2(x):
    """
    Pad a Tensor of shape [a, b] to shape [a, b']
    where b' is the smallest power of two not smaller than b.
    """
    next_2 = 2 ** math.ceil(math.log2(x.shape[-1]))
    pad = next_2 - x.shape[-1]
    # return nn.functional.pad(x, (pad // 2, pad // 2 + int(pad % 2)))
    return tf.pad(x, [(0, 0), (pad // 2, pad // 2 + int(pad % 2))])


def get_qmf_bank(h, n_band):
    """
    Modulates an input protoype filter into a bank of
    cosine modulated filters
    Parameters
    ----------
    h: torch.Tensor
        prototype filter
    n_band: int
        number of sub-bands
    """
    k = tf.range(n_band).reshape(-1, 1)
    N = h.shape[-1]
    t = tf.range(-(N // 2), N // 2 + 1)

    p = (-1) ** k * math.pi / 4

    mod = tf.cos((2 * k + 1) * math.pi / (2 * n_band) * t + p)
    hk = 2 * h * mod

    return hk.astype(tf.float32)


def kaiser_filter(wc, atten, N=None):
    """
    Computes a kaiser lowpass filter
    Parameters
    ----------
    wc: float
        Angular frequency

    atten: float
        Attenuation (dB, positive)
    """
    N_, beta = kaiserord(atten, wc / np.pi)

    N_ = 2 * (N_ // 2) + 1
    N = N if N is not None else N_
    h = firwin(N, wc, window=("kaiser", beta), scale=False, nyq=np.pi)
    return h


def loss_wc(wc, atten, M, N):
    """
    Computes the objective described in https://ieeexplore.ieee.org/document/681427
    """
    h = kaiser_filter(wc, atten, N)
    g = np.convolve(h, h[::-1], "full")
    g = abs(g[g.shape[-1] // 2 :: 2 * M][1:])
    return np.max(g)


def get_prototype(atten, M, N=None):
    """
    Given an attenuation objective and the number of bands
    returns the corresponding lowpass filter
    """
    wc = fmin(lambda w: loss_wc(w, atten, M, N), 1 / M, disp=0)[0]
    return kaiser_filter(wc, atten, N)


class PQMF(tf.keras.layers.Layer):
    """
    Pseudo Quadrature Mirror Filter multiband decomposition / reconstruction
    Parameters
    ----------
    attenuation: int
        Attenuation of the rejected bands (dB, 80 - 120)
    n_bands: int
        Number of bands, must be a power of 2 if the polyphase implementation
        is needed (currently not implemented)
    """

    def __init__(self, attenuation, n_bands):
        super().__init__()
        h = get_prototype(attenuation, n_bands)

        self.h = tf.convert_to_tensor(h).astype(tf.float32)
        self.hk = get_qmf_bank(h, n_bands)
        self.hk = center_pad_next_pow_2(self.hk)
        self.hk = rearrange(self.hk, "filters taps -> taps 1 filters")

        self.n_bands = n_bands

        updown_filter = np.zeros((n_bands, n_bands, n_bands), dtype=np.float32)
        for k in range(n_bands):
            updown_filter[0, k, k] = 1.0
        self.updown_filter = updown_filter.astype(np.float32)

        self.taps = int(self.h.shape[0])

    def analysis(self, x):
        """
        PQMF bank analysis.
        :param x: a Tensor of shape [batch, n_samples, 1]
        :return: a Tensor of shape [batch, n_samples / n_bands, n_bands]
        """
        tf.ensure_shape(x, [None, None, 1])

        if self.n_bands == 1:
            return x

        x = tf.nn.conv1d(x, self.hk, stride=self.hk.shape[2], padding="SAME")
        x = reverse_half(x)

        tf.ensure_shape(x, [None, None, self.n_bands])
        return x

    def synthesis(self, x):
        """
        PQMF bank synthesis.
        :param x: a Tensor of shape [batch, n_samples / n_bands, n_bands]
            produced by self.analysis(x)
        :return: a Tensor of shape [batch, n_samples, 1],
            a reconstruction of the original x
        """
        tf.ensure_shape(x, [None, None, self.n_bands])

        if self.n_bands == 1:
            return x

        x = reverse_half(x)

        # First, upsample x to the original n_samples by adding zeroes so that each value
        # in `x` is `n_bands` positions apart, e.g. for n_bands=4 we'd have
        # x[i,:,j] <- [x[i,0,j] 0 0 0 x[i,1,j] 0 0 0...]
        x = tf.nn.conv1d_transpose(
            x,
            self.updown_filter * self.n_bands,
            strides=self.n_bands,
            output_shape=(
                tf.shape(x)[0],
                tf.shape(x)[1] * self.n_bands,
                self.n_bands,
            ),
        )

        # We need to transpose since the convolution is now going
        # from n_bands channels to one
        hk = rearrange(self.hk, "taps 1 filters -> taps filters 1")

        hk = tf.reverse(hk, [0])

        taps = hk.shape[0]

        x = tf.pad(x, [[0, 0], [taps // 2, taps // 2], [0, 0]])
        y = tf.nn.conv1d(x, hk, stride=1, padding="VALID")[..., 1:, :]

        tf.ensure_shape(y, [None, None, 1])
        return y
