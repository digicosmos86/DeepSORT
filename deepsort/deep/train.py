import re
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import (
    RandomFlip,
    RandomRotation,
    RandomZoom,
)

from model import DeepAppearanceDescriptor
from utils import CustomModelSaver, parse_args

resize_scale_rotate = tf.keras.Sequential(
    [
        RandomRotation((-0.2, 0.2)),
        RandomZoom(0.2, 0.2),
        RandomFlip(mode="horizontal"),
    ]
)


def process_image(img_path):
    img = tf.io.read_file(img_path)
    img = (
        tf.cast(
            tf.image.resize(tf.image.decode_jpeg(img, channels=3), size=(64, 128)),
            tf.float32,
        )
        / 255.0
    )
    img = resize_scale_rotate(img)

    label = tf.strings.to_number(
        tf.strings.split(tf.strings.split(img_path, sep="\\")[-1], sep="_")[0],
        out_type=tf.int32,
    )
    return img, label


ds_train = (
    tf.data.Dataset.list_files("../data/train/*.jpg", shuffle=True)
    .map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(5000)
    .batch(32, num_parallel_calls=tf.data.AUTOTUNE)
)

ds_val = (
    tf.data.Dataset.list_files("../data/val/*.jpg", shuffle=True)
    .map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(5000)
    .batch(128, num_parallel_calls=tf.data.AUTOTUNE)
)

if __name__ == "__main__":

    ARGS = parse_args()

    time_now = datetime.now()
    timestamp = time_now.strftime("%m%d%y-%H%M%S")

    checkpoint_dir = Path("./checkpoints/deep_appearance_descriptor" + "-" + timestamp)

    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir()

    model = DeepAppearanceDescriptor()

    callbacks_list = [
        tf.keras.callbacks.TensorBoard(
            log_dir="./logs/deep_appearance_descriptor",
            histogram_freq=0,
        ),
        CustomModelSaver(checkpoint_dir, 5),
    ]

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
    )

    model(tf.keras.Input(shape=(64, 128, 3)))
    model.summary()

    if ARGS.load_checkpoint is not None:
        ARGS.load_checkpoint = Path(ARGS.load_checkpoint)
        model.load_weights(ARGS.load_checkpoint)

    model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=200,
        callbacks=callbacks_list,
    )
