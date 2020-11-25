# import necessary modules
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Dense, Input


class DenseBlock(tf.keras.Model):
    def __init__(self, layers, layer_width, bn=False, name="encoder"):
        """
        Creates a sequence of equal width dense layers
        :param layers: number of dense layer repetitions
        :param layer_width: number of neurons per dense layer
        """
        # supercharge
        super(DenseBlock, self).__init__()

        # operations
        self.dense = self._make_dense(layers, layer_width, bn, name)

    @staticmethod
    def _make_dense(layers, layer_width, bn, name, init="glorot_uniform"):
        input_shape = Input(shape=(1,), name="input")
        x = input_shape

        for i in range(layers):
            x = Dense(
                layer_width,
                activation="tanh",
                kernel_initializer=init,
                name="dense_{}".format(i + 1),
            )(x)
            if bn:
                x = BatchNormalization(name="bn_{}".format(i + 1))(x)

        y1 = Dense(1, kernel_initializer=init, name="y1")(x)
        y2 = Dense(1, kernel_initializer=init, name="y2")(x)
        y3 = Dense(1, kernel_initializer=init, name="y3")(x)

        return tf.keras.Model(inputs=input_shape, outputs=[y1, y2, y3], name=name)

    def call(self, x):
        return self.dense(x)
