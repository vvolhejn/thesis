"""
Mostly adapted from DDSP's timbre transfer notebook:
https://github.com/magenta/ddsp/blob/main/ddsp/colab/demos/timbre_transfer.ipynb
"""
import pickle

from absl import logging
import librosa
import numpy as np
import ddsp.training
import tensorflow as tf


def adjust_batch(
    audio_features,
    dataset_stats,
    autotune_factor=0.0,
    suppress_off_notes_db=20,
    note_detection_threshold=1.0,
    lufs_normalization=None,
):
    """
    :param audio_features: a dictionary of audio features (a batch).
        Must contain the keys {"loudness_db", "f0_hz", "f0_confidence"}, each of which
        should have the shape (batch_size, n_frames). The batch dimension may be ommited
        if it is 1.
    :param dataset_stats: a dictionary
    :param autotune_factor: Force pitch to nearest note (amount between 0 and 1)
    :param suppress_off_notes_db: Make parts without notes detected quieter by this much.
    :param note_detection_threshold: A higher threshold turns more parts off.
        Between 0 and 2 is reasonable.
    :param lufs_normalization: If sets, normalizes the audio volume to the given dB LUFS
        instead of using quantile transforms.
        A reasonable value is -14, which is what Spotify uses.
    :return: An adjusted version of `audio_features`.
    """
    assert {"loudness_db", "f0_hz", "f0_confidence"}.issubset(audio_features.keys())

    def cast(v):
        v = np.array(v)

        # Make two-dimensional.
        if len(v.shape) == 1:
            v = v[np.newaxis, :]

        return v

    audio_features = {k: cast(v) for k, v in audio_features.items()}
    audio_features_mod = {k: cast(v) for k, v in audio_features.items()}

    batch_size = audio_features["loudness_db"].shape[0]

    for k, v in audio_features.items():
        assert len(v.shape) == 2
        assert (
            v.shape[0] == batch_size
        ), f"Batch size for {k} is {v.shape[0]} instead of {batch_size}"

    audio_features_mod["mask_on"] = np.zeros_like(audio_features["loudness_db"]).astype(
        bool
    )

    for i in range(batch_size):
        audio_features_one = {k: v[i] for k, v in audio_features.items()}

        audio_features_one_mod = adjust_one(
            audio_features_one,
            dataset_stats,
            autotune_factor,
            suppress_off_notes_db,
            note_detection_threshold,
            lufs_normalization,
        )

        for k, v in audio_features_one_mod.items():
            audio_features_mod[k][i, :] = v

    return audio_features_mod


def adjust_one(
    audio_features,
    dataset_stats,
    autotune_factor=0.0,
    suppress_off_notes_db=20,
    note_detection_threshold=1.0,
    lufs_normalization=None,
):
    """An unbatched variant of `adjust_batch`."""
    audio_features_mod = {k: v.copy() for k, v in audio_features.items()}

    # Detect sections that are "on".
    mask_on, note_on_value = ddsp.training.postprocessing.detect_notes(
        audio_features["loudness_db"],
        audio_features["f0_confidence"],
        note_detection_threshold,
    )

    if not np.any(mask_on):
        logging.warning("No notes detected by adjust_one(), not adjusting")
        return audio_features_mod

    audio_features_mod["mask_on"] = mask_on

    # Shift the pitch register.
    target_mean_pitch = dataset_stats["mean_pitch"]
    pitch = ddsp.core.hz_to_midi(audio_features["f0_hz"])
    mean_pitch = np.mean(pitch[mask_on])
    p_diff = target_mean_pitch - mean_pitch
    p_diff_octave = p_diff / 12.0
    round_fn = np.floor if p_diff_octave > 1.5 else np.ceil
    p_diff_octave = round_fn(p_diff_octave)
    audio_features_mod = shift_f0(audio_features_mod, p_diff_octave)

    if lufs_normalization is None:
        # Quantile shift the note_on parts.
        _, loudness_norm = ddsp.training.postprocessing.fit_quantile_transform(
            audio_features["loudness_db"],
            mask_on,
            inv_quantile=dataset_stats["quantile_transform"],
        )

        # Turn down the note_off parts.
        mask_off = np.logical_not(mask_on)
        loudness_norm[mask_off] -= suppress_off_notes_db * (
            1.0 - note_on_value[mask_off][:, np.newaxis]
        )
        loudness_norm = np.reshape(loudness_norm, audio_features["loudness_db"].shape)
        audio_features_mod["loudness_db"] = loudness_norm
    else:
        assert lufs_normalization <= 0

        actual_lufs = get_lufs(audio_features["audio_16k"], sample_rate=16000)
        target_lufs = lufs_normalization
        audio_features_mod["loudness_db"] += target_lufs - actual_lufs

        normalized_audio = normalize_lufs(
            audio_features["audio_16k"],
            16000,
            target_lufs=target_lufs,
            actual_lufs=actual_lufs,
        )
        audio_features_mod["audio_16k"] = tf.convert_to_tensor(normalized_audio)
        # This will not work for sampling rates other than 16000
        audio_features_mod["audio"] = tf.convert_to_tensor(normalized_audio)

        logging.info(
            f"Loudness adjusted from {actual_lufs:.1f} dB LUFS to {target_lufs:.1f}."
        )

    # Auto-tune.
    if autotune_factor:
        from ddsp.colab.colab_utils import auto_tune, get_tuning_factor

        f0_midi = np.array(ddsp.core.hz_to_midi(audio_features_mod["f0_hz"]))
        tuning_factor = get_tuning_factor(
            f0_midi, audio_features_mod["f0_confidence"], mask_on
        )
        f0_midi_at = auto_tune(f0_midi, tuning_factor, mask_on, amount=autotune_factor)
        audio_features_mod["f0_hz"] = ddsp.core.midi_to_hz(f0_midi_at)

    return audio_features_mod


def shift_ld(audio_features, ld_shift=0.0):
    """Shift loudness by a number of octaves."""
    audio_features["loudness_db"] += ld_shift
    return audio_features


def shift_f0(audio_features, pitch_shift=0.0):
    """Shift f0 by a number of octaves."""
    audio_features["f0_hz"] *= 2.0 ** (pitch_shift)
    audio_features["f0_hz"] = np.clip(
        audio_features["f0_hz"], 0.0, librosa.midi_to_hz(110.0)
    )
    return audio_features


def get_lufs(audio, sample_rate=16000):
    """Get Loudness Units relative to Full Scale, as per ITU-R BS.1770"""
    import pyloudnorm

    if not isinstance(audio, np.ndarray):
        audio = np.array(audio)

    meter = pyloudnorm.Meter(rate=sample_rate)
    return meter.integrated_loudness(audio)


def normalize_lufs(audio, sample_rate=16000, target_lufs=-14.0, actual_lufs=None):
    """-14 LUFS is what Spotify uses."""
    import pyloudnorm

    assert target_lufs <= 0

    if not isinstance(audio, np.ndarray):
        audio = np.array(audio)

    if actual_lufs is None:
        actual_lufs = get_lufs(audio, sample_rate)

    return pyloudnorm.normalize.loudness(audio, actual_lufs, target_lufs)


def load_dataset_statistics(path):
    try:
        with open(path, "rb") as f:
            dataset_stats = pickle.load(f)
            return dataset_stats
    except Exception as err:
        raise RuntimeError(
            f"Loading dataset statistics from pickle file {path} failed."
        ) from err
