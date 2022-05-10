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

L2_reg_strength = 1e-8


class ResidualBlock(Layer):
    def __init__(self, n, is_first=False, increase_dim=False, **kwargs):

        super(ResidualBlock, self).__init__(**kwargs)

        self.n = n * 2 if increase_dim else n
        self.stride = 2 if increase_dim else 1
        self.is_first = is_first

        weight_init = TruncatedNormal(stddev=1e-3)
        weight_regularizer = L2(L2_reg_strength)
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

        self.projection = Conv2D(
            filters=self.n,
            kernel_size=(1, 1),
            strides=(2, 2),
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

            x = self.projection(x)

        return x + x1


class CosineSimilarity(Layer):
    def __init__(self, units, **kwargs):
        super(CosineSimilarity, self).__init__(**kwargs)

        self.units = units

    def build(self, input_shape):

        w_init = TruncatedNormal(1e-3)
        self.layer_weights = tf.Variable(
            initial_value=w_init(
                shape=(input_shape[-1], self.units),
                dtype=tf.float32,
            ),
            trainable=True,
            name="my_weights",
        )

        scale_init = tf.constant_initializer(0.1)
        self.scale = tf.Variable(
            initial_value=scale_init(shape=(1)),
            trainable=True,
            name="k",
        )

        self.regularizer = L2(1e-1)

    def call(self, x):
        scale = tf.nn.softplus(self.regularizer(self.scale))
        weights_norm = tf.nn.l2_normalize(self.layer_weights, axis=0)
        logits = scale * tf.matmul(x, weights_norm)

        return logits


class DeepAppearanceDescriptor(tf.keras.models.Model):
    def __init__(self):

        super(DeepAppearanceDescriptor, self).__init__()

        weight_init = TruncatedNormal(stddev=1e-3)
        weight_regularizer = L2(L2_reg_strength)
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
        self.dropout1 = Dropout(0.6)
        self.dense = Dense(
            128,
            activation="elu",
            kernel_initializer=weight_init,
            bias_initializer=bias_init,
            kernel_regularizer=weight_regularizer,
        )
        self.batch_norm3 = BatchNormalization()
        self.classifier = CosineSimilarity(units=1502, name="classification_layer")

    def call(self, inputs, train=True):

        x = self.conv1(inputs)
        x = self.batch_norm1(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.maxpool1(x)
        x = self.residual_block1(x)
        x = self.residual_block2(x)
        x = self.residual_block3(x)
        x = self.residual_block4(x)
        x = self.residual_block5(x)
        x = self.residual_block6(x)
        x = self.flatten(x)
        x = self.dropout1(x)
        x = self.dense(x)
        x = self.batch_norm3(x)
        x = tf.nn.l2_normalize(x, axis=1)

        if not train:
            return x

        x = self.classifier(x)
        return x

    @tf.function
    def train_step(self, data):

        images, labels = data

        with tf.GradientTape() as tape:
            outputs = self(images, train=True)
            loss = self.compiled_loss(labels, outputs)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.compiled_metrics.update_state(labels, outputs)

        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, data):

        images, labels = data
        outputs = self(images, train=True)
        self.compiled_loss(labels, outputs)

        self.compiled_metrics.update_state(labels, outputs)
        return {m.name: m.result() for m in self.metrics}


if __name__ == "__main__":
    import numpy as np

    model = DeepAppearanceDescriptor()
    model(tf.keras.Input(shape=(64, 128, 3)))
    model.summary()
