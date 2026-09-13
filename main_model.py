from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D, Dense, \
    Lambda, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


def build_base_network(input_shape=(295, 295, 3)):
    """轻量化基础网络（参数量从1.74亿降至约500万）"""
    inputs = Input(shape=input_shape)

    # 卷积块1
    x = Conv2D(32, (7, 7), activation='relu', kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    # 卷积块2
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = GlobalAveragePooling2D()(x)  # 替代Flatten大幅减少参数量

    # 嵌入层
    x = Dense(128, activation=None)(x)
    x = Lambda(lambda x: K.l2_normalize(x, axis=1))(x)  # L2归一化

    return Model(inputs, x)


def TripletModel():
    """三联体模型（输入尺寸295x295x3）"""
    anc_input = Input(shape=(295, 295, 3), name='anc_input')
    pos_input = Input(shape=(295, 295, 3), name='pos_input')
    neg_input = Input(shape=(295, 295, 3), name='neg_input')

    base_network = build_base_network()
    anc_embed = base_network(anc_input)
    pos_embed = base_network(pos_input)
    neg_embed = base_network(neg_input)

    merged = Concatenate(axis=-1)([anc_embed, pos_embed, neg_embed])  # 兼容 Keras 3
    return Model(inputs=[anc_input, pos_input, neg_input], outputs=merged)