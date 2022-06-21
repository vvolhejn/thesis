import os

import note_seq
import tensorflow as tf
from codetiming import Timer
import tqdm

import ddsp
import ddsp.training
import numpy as np

import gin

from ddsp.losses import SpectralLoss
from ddsp.training.ddsp_export import get_representative_dataset


def play_audio(audio):
    pass
    # audio = np.array(audio)
    # audio = np.squeeze(audio)
    # IPython.display.display(IPython.display.Audio(audio, rate=16000))


def main(quantized):
    print("Running")

    model_dir = "/Volumes/euler/"

    with gin.unlock_config():
        operative_config = ddsp.training.train_util.get_latest_operative_config(
            model_dir
        )
        gin.parse_config_file(operative_config, skip_unknown=True)
        print(gin.config.config_str())

    representative_dataset = get_representative_dataset(
        "/Users/vaclav/prog/thesis/data/violin2/violin2.tfrecord*",
        include_f0_hz=True,
    )

    print(f"Loading model ({'quantized' if quantized else 'unquantized'})")
    model = ddsp.training.models.get_model()
    checkpoint_path = tf.train.latest_checkpoint(model_dir, latest_filename=None)
    # checkpoint_path = (
    #     "/Users/vaclav/prog/thesis/data/models/0323-halfrave-1/ckpt-100000"
    # )

    assert checkpoint_path is not None, f"No checkpoint found in {model_dir}"
    # print(model.processor_group.processors[3].)
    model.restore(checkpoint_path, verbose=False)

    tflite_file_path = (
        "model_quantized.tflite" if quantized else "model_unquantized.tflite"
    )
    tflite_file_path = os.path.join(model_dir, "export/tflite", tflite_file_path)
    interpreter = tf.lite.Interpreter(tflite_file_path)
    my_signature = interpreter.get_signature_runner()

    loss = SpectralLoss(logmag_weight=1.0)
    print(
        "Logmag weight (should be 1.0 to match operative config):", loss.logmag_weight
    )

    distances = [[], [], []]

    print("Running model")

    for i, batch in enumerate(tqdm.tqdm(representative_dataset())):
        # print(x)

        # TFLITE_FILE_PATH = "/cluster/scratch/vvolhejn/models/0503-ddspae-vst-cnn-2/export/tflite/model.tflite"

        # audio = tf.cast(tf.reshape(tf.sin(tf.linspace(0, 2000, 64000) + (tf.linspace(0, 1, 64000) ** 2) * 2000), [64000]), tf.float32)
        # audio = tf.reshape(x[0], [64000])

        inputs_scaled = [None, None]
        for input_i, input_data in enumerate([batch[0], batch[1]]):
            input_details = interpreter.get_input_details()[input_i]
            if input_details["dtype"] == np.int8:
                print("SCALING")
                input_scale, input_zero_point = input_details["quantization"]
                # TODO: fix this
                input_scaled = input_data / input_scale + input_zero_point
                input_scaled = np.clip(input_scaled, -128, 127).astype(
                    input_details["dtype"]
                )
                # print(input_scale, input_zero_point)
                inputs_scaled[input_i] = input_scaled
            else:
                inputs_scaled[input_i] = input_data

        with Timer("Autoencoder.QuantizedDecoder", logger=None):
            features = my_signature(
                # f0_scaled=tf.constant([400] * n_frames, shape=(n_frames,), dtype=tf.float32),
                # pw_scaled=tf.constant([0.4] * n_frames, shape=(n_frames,), dtype=tf.float32),
                # f0_scaled=batch[0],
                # pw_scaled=batch[1],
                f0_scaled=inputs_scaled[0],
                pw_scaled=inputs_scaled[1],
            )
        features["f0_hz"] = batch[2]

        pg_out = model.processor_group(features, return_outputs_dict=True)

        # print(pg_out["signal"].keys())
        # print(pg_out["controls"]["add"]["signal"].numpy())
        # print("IR:", pg_out["controls"]["reverb"]["controls"]["ir"].numpy().sum())

        output_unquantized = model(
            {"audio": batch[3], "f0_hz": batch[2], "f0_confidence": batch[4]},
            training=False,
        )

        distances[0].append(loss(batch[3], pg_out["signal"]))
        distances[1].append(loss(batch[3], output_unquantized["audio_synth"]))
        distances[2].append(loss(output_unquantized["audio_synth"], pg_out["signal"]))

        # print(Timer.timers._timings["Autoencoder.decoder"])
        # print(Timer.timers._timings["Autoencoder.QuantizedDecoder"])

        if i == 0:
            path = "/tmp/synthesized.wav"

            with open(path, "wb") as f:
                # data = pg_out["signal"].numpy()[0]
                data = np.concatenate(
                    [
                        batch[3][0],
                        pg_out["signal"].numpy()[0],
                        output_unquantized["audio_synth"][0],
                    ]
                )

                wav_data = note_seq.audio_io.samples_to_wav_data(
                    data, sample_rate=16000
                )
                f.write(wav_data)

            print("Saved to", path)

        if i == 10:
            break

    print(
        "Normal runtime:",
        np.array(Timer.timers._timings["Autoencoder.decoder"])[1:].mean(),
    )
    print(
        "Quantized runtime:",
        np.array(Timer.timers._timings["Autoencoder.QuantizedDecoder"])[1:].mean(),
    )
    for i, name in enumerate(
        ["Ground truth - TFLite", "Ground truth - TF", "TFLite - TF"]
    ):
        print(f"{name}:", np.array(distances[i]).mean())

    print(f"Reminder: model is {'quantized' if quantized else 'unquantized'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--quantized", action="store_true")
    parser.add_argument("--no-quantized", action="store_false", dest="quantized")
    parser.set_defaults(quantized=True)

    args = parser.parse_args()

    main(quantized=args.quantized)
