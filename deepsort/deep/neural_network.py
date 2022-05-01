import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    MaxPooling2D,
    Layer,
    Dense,
    Flatten,
    Dropout,
)
from tensorflow.keras.initializers import (
    TruncatedNormal,
    Zeros,
)
from tensorflow.keras.regularizers import L2
from tensorflow.keras.activations import elu


class ResidualBlock(Layer):
    def __init__(self, n, is_first=False, increase_dim=False, **kwargs):

        super(ResidualBlock, self).__init__(**kwargs)

        self.n = n * 2 if increase_dim else n
        self.stride = 2 if increase_dim else 1
        self.is_first = is_first

        weight_init = TruncatedNormal(stddev=1e-3)
        weight_regularizer = L2(1e-8)
        bias_init = Zeros()

        if not self.is_first:
            self.bn1 = BatchNormalization()

        self.conv1 = Conv2D(
            filters=self.n,
            kernel_size=(3, 3),
            strides=self.stride,
            padding="same",
            activation="elu",
            kernel_initializer=weight_init,
            bias_initializer=bias_init,
            kernel_regularizer=weight_regularizer,
        )

        self.bn2 = BatchNormalization()

        self.conv2 = Conv2D(
            filters=self.n,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            activation=None,
            kernel_initializer=weight_init,
            bias_initializer=bias_init,
            kernel_regularizer=weight_regularizer,
        )

    def call(self, x):

        if self.is_first:
            x1 = x
        else:
            x1 = self.bn1(x)
            x1 = elu(x1, alpha=1.0)

        x1 = self.conv1(x1)
        x1 = self.bn2(x1)
        x1 = self.conv2(x1)

        pre_block_dim = x.shape[-1]
        post_block_dim = x1.shape[-1]

        if pre_block_dim != post_block_dim:
            assert post_block_dim == pre_block_dim * 2
            self.projection = Conv2D(
                filters=post_block_dim,
                kernel_size=(1, 1),
                strides=(2, 2),
                padding="same",
                activation=None,
                kernel_initializer=TruncatedNormal(stddev=1e-3),
                bias_initializer=Zeros(),
                kernel_regularizer=L2(1e-8),
            )

            x = self.projection(x)

        return x + x1


class DeepAppearanceDescriptor(tf.keras.models.Model):
    def __init__(self):

        super(DeepAppearanceDescriptor, self).__init__()

        weight_init = TruncatedNormal(stddev=1e-3)
        weight_regularizer = L2(1e-8)
        bias_init = Zeros()

        self.conv1 = Conv2D(
            filters=32,
            kernel_size=(3, 3),
            padding="same",
            activation="elu",
            kernel_initializer=weight_init,
            bias_initializer=bias_init,
            kernel_regularizer=weight_regularizer,
        )

        self.batch_norm1 = BatchNormalization()

        self.conv2 = Conv2D(
            filters=32,
            kernel_size=(3, 3),
            padding="same",
            activation="elu",
            kernel_initializer=weight_init,
            bias_initializer=bias_init,
            kernel_regularizer=weight_regularizer,
        )

        self.batch_norm2 = BatchNormalization()

        self.maxpool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")

        self.residual_block1 = ResidualBlock(n=32, is_first=True)
        self.residual_block2 = ResidualBlock(n=32)
        self.residual_block3 = ResidualBlock(n=32, increase_dim=True)
        self.residual_block4 = ResidualBlock(n=64)
        self.residual_block5 = ResidualBlock(n=64, increase_dim=True)
        self.residual_block6 = ResidualBlock(n=128)

        self.flatten = Flatten()
        self.dropout = Dropout(0.6)
        self.dense = Dense(
            128,
            activation="elu",
            kernel_initializer=weight_init,
            bias_initializer=bias_init,
            kernel_regularizer=weight_regularizer,
        )
        self.batch_norm3 = BatchNormalization()

    def call(self, inputs):

        x = self.conv1(inputs)
        x = self.batch_norm1(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.maxpool1(x)
        print(x.shape)
        x = self.residual_block1(x)
        print(x.shape)
        x = self.residual_block2(x)
        print(x.shape)
        x = self.residual_block3(x)
        print(x.shape)
        x = self.residual_block4(x)
        print(x.shape)
        x = self.residual_block5(x)
        print(x.shape)
        x = self.residual_block6(x)
        print(x.shape)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.dense(x)
        x = self.batch_norm3(x)
        x = tf.nn.l2_normalize(x, axis=1)

        return x


if __name__ == "__main__":
    import numpy as np

    model = DeepAppearanceDescriptor()
    model.build((1, 128, 64, 3))
    model.summary()

    x = np.random.randn(1, 128, 64, 3)
    y = model(x)
    print(y.shape)
