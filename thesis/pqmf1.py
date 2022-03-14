import math

import numpy as np
import tensorflow as tf
from scipy.signal import kaiser, kaiserord, firwin
from scipy.optimize import fmin
from einops import rearrange
import ddsp

from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()


# class Conv1d(nn.Conv1d):
#     def __init__(self, *args, **kwargs):
#         self._pad = kwargs.get("padding", (0, 0))
#         kwargs["padding"] = 0
#         super().__init__(*args, **kwargs)
#         self.future_compensation = 0
#
#     def forward(self, x):
#         x = nn.functional.pad(x, self._pad)
#         return nn.functional.conv1d(
#             x,
#             self.weight,
#             self.bias,
#             self.stride,
#             self.padding,
#             self.dilation,
#             self.groups,
#         )


def reverse_half(x):
    mask = np.ones_like(x)
    # mask[..., 1::2, ::2] = -1
    mask[..., ::2, 1::2] = -1

    return x * mask


def center_pad_next_pow_2(x):
    next_2 = 2 ** math.ceil(math.log2(x.shape[-1]))
    pad = next_2 - x.shape[-1]
    # return nn.functional.pad(x, (pad // 2, pad // 2 + int(pad % 2)))
    return tf.pad(x, [(0, 0), (pad // 2, pad // 2 + int(pad % 2))])


def make_odd(x):
    if not x.shape[-1] % 2:
        x = nn.functional.pad(x, (0, 1))
    return x


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


def polyphase_forward(x, hk, rearrange_filter=True):
    """
    Polyphase implementation of the analysis process (fast)
    Parameters
    ----------
    x: torch.Tensor
        signal to analyse ( B x 1 x T )

    hk: torch.Tensor
        filter bank ( M x T )
    """
    # print(x)
    x = rearrange(x, "b c (t m) -> b t (c m)", m=hk.shape[0])
    # print(x)
    if rearrange_filter:
        hk = rearrange(hk, "c (t m) -> c t m", m=hk.shape[0])
    # x = nn.functional.conv1d(x, hk, padding=hk.shape[-1] // 2)[..., :-1]

    print(x.shape)
    print(hk.shape)
    x = tf.nn.conv1d(x, hk, stride=1, padding="SAME")
    return x


def polyphase_inverse(x, hk, rearrange_filter=True):
    """
    Polyphase implementation of the synthesis process (fast)
    Parameters
    ----------
    x: torch.Tensor
        signal to synthesize from ( B x 1 x T )

    hk: torch.Tensor
        filter bank ( M x T )
    """

    m = hk.shape[0]

    if rearrange_filter:
        hk = hk.flip(-1)
        hk = rearrange(hk, "c (t m) -> m c t", m=m)  # polyphase

    pad = hk.shape[-1] // 2 + 1
    # x = nn.functional.conv1d(x, hk, padding=int(pad))[..., :-1] * m
    x = tf.nn.conv1d(x, hk, padding=int(pad))[..., :-1] * m

    x = x.flip(1)
    x = rearrange(x, "b (c m) t -> b c (t m)", m=m)
    x = x[..., 2 * hk.shape[1] :]
    return x


def classic_forward(x, hk):
    """
    Naive implementation of the analysis process (slow)
    Parameters
    ----------
    x: torch.Tensor
        signal to analyse ( B x 1 x T )

    hk: torch.Tensor
        filter bank ( M x T )
    """
    # x = nn.functional.conv1d(
    #     x,
    #     hk.unsqueeze(1),
    #     stride=hk.shape[0],
    #     padding=hk.shape[-1] // 2,
    # )[..., :-1]
    print(x.shape)
    print(hk.shape)
    x = tf.nn.conv1d(x, hk, stride=hk.shape[2], padding="SAME")
    return x


class PQMF(tf.keras.layers.Layer):
    """
    Pseudo Quadrature Mirror Filter multiband decomposition / reconstruction
    Parameters
    ----------
    attenuation: int
        Attenuation of the rejected bands (dB, 80 - 120)
    n_band: int
        Number of bands, must be a power of 2 if the polyphase implementation
        is needed
    """

    def __init__(self, attenuation, n_band, polyphase=True, name="pqmf"):
        super().__init__()
        h = get_prototype(attenuation, n_band)

        if polyphase:
            power = math.log2(n_band)
            assert power == math.floor(
                power
            ), "when using the polyphase algorithm, n_band must be a power of 2"

        self.h = tf.convert_to_tensor(h).astype(tf.float32)
        self.hk = get_qmf_bank(h, n_band)
        # print("hk", self.hk)
        self.hk = center_pad_next_pow_2(self.hk)
        self.hk = rearrange(self.hk, "filters taps -> taps 1 filters")
        # print("h", self.h)
        # print("hk", self.hk)

        # self.register_buffer("hk", hk)
        # self.register_buffer("h", h)
        self.n_band = n_band
        self.polyphase = polyphase

        updown_filter = np.zeros((n_band, n_band, n_band), dtype=np.float32)
        for k in range(n_band):
            updown_filter[0, k, k] = 1.0
        self.updown_filter = updown_filter.astype(np.float32)

        self.taps = int(self.h.shape[0])

    def analysis(self, x):
        if self.n_band == 1:
            return x
        elif self.polyphase:
            x = polyphase_forward(x, self.hk)
        else:
            x = classic_forward(x, self.hk)

        x = reverse_half(x)

        return x

    def synthesis(self, x):
        if self.n_band == 1:
            return x

        x = reverse_half(x)

        if self.polyphase:
            return polyphase_inverse(x, self.hk)
        else:
            return self.classic_inverse2(x)

    def classic_inverse(self, x):
        x = tf.nn.conv1d_transpose(
            x,
            self.updown_filter * self.n_band,
            strides=self.n_band,
            output_shape=(
                tf.shape(x)[0],
                tf.shape(x)[1] * self.n_band,
                self.n_band,
            ),
        )
        print(x.shape)
        print(np.array(tf.transpose(x, perm=[0, 2, 1])[:, :, :50]))

        hk = rearrange(self.hk, "taps 1 filters -> taps filters 1")
        taps = hk.shape[0]
        x = tf.pad(x, [[0, 0], [taps // 2, taps // 2 - 1], [0, 0]])
        return tf.nn.conv1d(x, hk, stride=1, padding="VALID")

    def classic_inverse2(self, x):
        x = tf.nn.conv1d_transpose(
            x,
            self.updown_filter * self.n_band,
            strides=self.n_band,
            output_shape=(
                tf.shape(x)[0],
                tf.shape(x)[1] * self.n_band,
                self.n_band,
            ),
        )

        hk = rearrange(self.hk, "taps 1 filters -> taps filters 1")
        hk = tf.reverse(hk, [0])

        taps = hk.shape[0]
        print(hk.shape)
        print(tf.reduce_sum(x))

        x = tf.pad(x, [[0, 0], [taps // 2, taps // 2], [0, 0]])
        print(x.shape)
        print(hk.shape)

        # y = nn.functional.conv1d(
        #     y,
        #     hk.unsqueeze(0),
        #     padding=hk.shape[-1] // 2,
        # )[..., 1:]

        y = tf.nn.conv1d(x, hk, stride=1, padding="VALID")[..., 1:, :]

        return y
