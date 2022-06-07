# There was an issue where the Keras model wasn't being pruned:
# https://github.com/tensorflow/model-optimization/issues/973

import tempfile

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot


def main(batch_size):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(28, 28)),
            tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10),
        ]
    )

    # model.compile(
    #     loss=tf.keras.losses.MeanSquaredError(), optimizer="adam", metrics=["accuracy"]
    # )

    model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(
        tf.keras.models.clone_model(model)
    )

    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
    ]

    model_for_pruning.compile(
        loss=tf.keras.losses.MeanSquaredError(), optimizer="adam", metrics=["accuracy"]
    )

    model_for_pruning.fit(
        np.random.randn(100, 28, 28).astype(np.float32),
        np.random.randn(100, 10).astype(np.float32),
        callbacks=callbacks,
        epochs=2,
        batch_size=batch_size,
        # validation_split=0.1,
        verbose=0,
    )

    weights = model_for_pruning.get_weights()[1]
    print(f"(Checking weights of shape {weights.shape})")
    print(
        f"Sparsity with batch size {batch_size}:",
        (weights == 0).mean(),
    )

    weights = model.get_weights()[0]
    print(f"(Checking weights of shape {weights.shape})")
    print(
        f"Sparsity with batch size {batch_size}:",
        (weights == 0).mean(),
    )
    print()


main(batch_size=1)
main(batch_size=2)
main(batch_size=3)
main(batch_size=32)
