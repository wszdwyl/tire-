# coding: utf-8
"""Train the improved ordinal-aware triplet network."""

import csv
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
)
from tensorflow.keras.optimizers import Adam

import main_model_v2
import DataLoader_v2


SEED = 42
BATCH_SIZE = 12
EPOCHS = 50
TRAIN_TRIPLETS = 2400
VAL_TRIPLETS = 600
BEST_TRIPLET_WEIGHTS = "triplet_best.weights.h5"
BEST_ENCODER_WEIGHTS = "encoder_best.weights.h5"


def set_reproducible(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def configure_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            print(f"[WARN] GPU memory-growth setting failed: {exc}")


def ordinal_adaptive_triplet_loss(y_true, y_pred):
    """Triplet loss with sample-wise margin supplied by the ordinal data loader."""
    d = main_model_v2.EMBED_DIM
    anchor = y_pred[:, :d]
    positive = y_pred[:, d:2 * d]
    negative = y_pred[:, 2 * d:3 * d]

    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
    margin = tf.reshape(tf.cast(y_true[:, 0], y_pred.dtype), (-1,))
    return tf.reduce_mean(tf.nn.relu(pos_dist - neg_dist + margin))


def mean_positive_distance(y_true, y_pred):
    d = main_model_v2.EMBED_DIM
    return tf.reduce_mean(tf.reduce_sum(tf.square(y_pred[:, :d] - y_pred[:, d:2 * d]), axis=1))


def mean_negative_distance(y_true, y_pred):
    d = main_model_v2.EMBED_DIM
    return tf.reduce_mean(tf.reduce_sum(tf.square(y_pred[:, :d] - y_pred[:, 2 * d:3 * d]), axis=1))


def active_triplet_rate(y_true, y_pred):
    d = main_model_v2.EMBED_DIM
    pos_dist = tf.reduce_sum(tf.square(y_pred[:, :d] - y_pred[:, d:2 * d]), axis=1)
    neg_dist = tf.reduce_sum(tf.square(y_pred[:, :d] - y_pred[:, 2 * d:3 * d]), axis=1)
    margin = tf.reshape(tf.cast(y_true[:, 0], y_pred.dtype), (-1,))
    return tf.reduce_mean(tf.cast(pos_dist - neg_dist + margin > 0.0, tf.float32))


def main():
    set_reproducible()
    configure_gpu()

    train_seq, val_seq, train_paths, val_paths = DataLoader_v2.build_sequences(
        val_ratio=0.2,
        batch_size=BATCH_SIZE,
        train_triplets=TRAIN_TRIPLETS,
        val_triplets=VAL_TRIPLETS,
        seed=SEED,
    )
    print("[INFO] train class counts:", [len(x) for x in train_paths])
    print("[INFO] val class counts:", [len(x) for x in val_paths])

    model = main_model_v2.TripletModel()
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=ordinal_adaptive_triplet_loss,
        metrics=[mean_positive_distance, mean_negative_distance, active_triplet_rate],
    )

    callbacks = [
        ModelCheckpoint(
            BEST_TRIPLET_WEIGHTS,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        CSVLogger("triplet_training_history.csv"),
    ]

    model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    if os.path.exists(BEST_TRIPLET_WEIGHTS):
        model.load_weights(BEST_TRIPLET_WEIGHTS)
    encoder = model.get_layer("tire_encoder")
    encoder.save_weights(BEST_ENCODER_WEIGHTS)
    print(f"[INFO] encoder weights saved: {BEST_ENCODER_WEIGHTS}")

    with open("training_config.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerows([
            ["seed", SEED],
            ["batch_size", BATCH_SIZE],
            ["epochs_max", EPOCHS],
            ["train_triplets_per_epoch", TRAIN_TRIPLETS],
            ["val_triplets_per_epoch", VAL_TRIPLETS],
            ["embedding_dim", main_model_v2.EMBED_DIM],
        ])


if __name__ == "__main__":
    main()
