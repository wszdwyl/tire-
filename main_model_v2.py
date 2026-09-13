# coding: utf-8
"""Improved tire-wear metric-learning network.

Paper-oriented design:
1) lightweight multi-scale feature extraction;
2) channel attention (SE) to emphasize wear-related texture channels;
3) 128-D L2-normalized embedding for ordinal triplet metric learning.
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, SeparableConv2D, BatchNormalization, ReLU,
    MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout,
    Concatenate, Reshape, Multiply, Add
)
from tensorflow.keras.models import Model


EMBED_DIM = 128
INPUT_SHAPE = (295, 295, 3)


class L2Normalize(tf.keras.layers.Layer):
    """Serializable L2 normalization layer."""

    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=-1)

    def get_config(self):
        return super().get_config()


def _conv_bn_relu(x, filters, kernel_size, strides=1, name=None):
    x = Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        name=None if name is None else f"{name}_conv",
    )(x)
    x = BatchNormalization(name=None if name is None else f"{name}_bn")(x)
    return ReLU(name=None if name is None else f"{name}_relu")(x)


def _se_block(x, reduction=8, name="se"):
    """Squeeze-and-Excitation channel attention."""
    channels = int(x.shape[-1])
    squeeze = GlobalAveragePooling2D(name=f"{name}_gap")(x)
    excite = Dense(
        max(channels // reduction, 8), activation="relu", use_bias=False,
        name=f"{name}_fc1"
    )(squeeze)
    excite = Dense(channels, activation="sigmoid", use_bias=False, name=f"{name}_fc2")(excite)
    excite = Reshape((1, 1, channels), name=f"{name}_reshape")(excite)
    return Multiply(name=f"{name}_scale")([x, excite])


def _multiscale_attention_block(x, filters, name):
    """Parallel local/dilated depthwise-separable branches + SE attention + residual."""
    shortcut = x

    # Local texture branch: tread edge / fine crack / local pattern.
    b1 = SeparableConv2D(
        filters // 2, 3, padding="same", use_bias=False,
        depthwise_initializer="he_normal", pointwise_initializer="he_normal",
        name=f"{name}_local_sep"
    )(x)
    b1 = BatchNormalization(name=f"{name}_local_bn")(b1)
    b1 = ReLU(name=f"{name}_local_relu")(b1)

    # Dilated branch: larger receptive field for groove and global wear pattern.
    b2 = SeparableConv2D(
        filters // 2, 3, padding="same", dilation_rate=2, use_bias=False,
        depthwise_initializer="he_normal", pointwise_initializer="he_normal",
        name=f"{name}_dilated_sep"
    )(x)
    b2 = BatchNormalization(name=f"{name}_dilated_bn")(b2)
    b2 = ReLU(name=f"{name}_dilated_relu")(b2)

    x = Concatenate(name=f"{name}_concat")([b1, b2])
    x = Conv2D(filters, 1, padding="same", use_bias=False, name=f"{name}_fuse_conv")(x)
    x = BatchNormalization(name=f"{name}_fuse_bn")(x)
    x = _se_block(x, reduction=8, name=f"{name}_se")

    # Residual projection when channel counts differ.
    if int(shortcut.shape[-1]) != filters:
        shortcut = Conv2D(filters, 1, padding="same", use_bias=False, name=f"{name}_shortcut_conv")(shortcut)
        shortcut = BatchNormalization(name=f"{name}_shortcut_bn")(shortcut)

    x = Add(name=f"{name}_add")([x, shortcut])
    return ReLU(name=f"{name}_out_relu")(x)


def build_base_network(input_shape=INPUT_SHAPE, embedding_dim=EMBED_DIM):
    """Build the lightweight multi-scale attention encoder."""
    inputs = Input(shape=input_shape, name="image")

    x = _conv_bn_relu(inputs, 32, 5, strides=2, name="stem")
    x = MaxPooling2D(2, name="stem_pool")(x)

    x = _multiscale_attention_block(x, 64, name="msa1")
    x = MaxPooling2D(2, name="pool1")(x)

    x = _multiscale_attention_block(x, 128, name="msa2")
    x = MaxPooling2D(2, name="pool2")(x)

    x = _multiscale_attention_block(x, 192, name="msa3")
    x = GlobalAveragePooling2D(name="global_pool")(x)
    x = Dense(256, activation="relu", kernel_initializer="he_normal", name="embed_fc1")(x)
    x = Dropout(0.25, name="embed_dropout")(x)
    x = Dense(embedding_dim, activation=None, name="embedding_raw")(x)
    outputs = L2Normalize(name="embedding")(x)

    return Model(inputs, outputs, name="tire_encoder")


def TripletModel(input_shape=INPUT_SHAPE, embedding_dim=EMBED_DIM):
    """Shared-weight triplet model: anchor, positive, negative -> concatenated embeddings."""
    anc_input = Input(shape=input_shape, name="anc_input")
    pos_input = Input(shape=input_shape, name="pos_input")
    neg_input = Input(shape=input_shape, name="neg_input")

    encoder = build_base_network(input_shape, embedding_dim)
    anc_embed = encoder(anc_input)
    pos_embed = encoder(pos_input)
    neg_embed = encoder(neg_input)

    merged = Concatenate(axis=-1, name="triplet_embeddings")(
        [anc_embed, pos_embed, neg_embed]
    )
    return Model([anc_input, pos_input, neg_input], merged, name="tire_triplet_model")


if __name__ == "__main__":
    model = TripletModel()
    model.summary()
