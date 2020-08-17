from tensorflow.keras.layers import Conv2D, Concatenate, UpSampling2D, Conv2DTranspose
import tensorflow as tf
import odl.contrib.tensorflow


class UNetUpsampling:

    def __init__(self, operator, pseudoinverse):
        self._operator = operator
        self._psuedoinverse = pseudoinverse
        self._operator_tensorflow = odl.contrib.tensorflow.as_tensorflow_layer(operator, 'OperatorUpsampling')
        self._pseudoinverse_tensorflow = odl.contrib.tensorflow.as_tensorflow_layer(pseudoinverse, 'PseudoUpsampling')
        self._out_shape = operator.range.shape

    def input_layer(self, inp_shape):
        # Return input layer with appropriate input size
        inp = tf.placeholder(tf.float32, shape=(None, ) + inp_shape, name='input_upsample')
        upsampling_factor = self._out_shape[0]//inp_shape[0]
        out = UpSampling2D(size=(upsampling_factor, 1), interpolation='bilinear')(inp)
        return inp, out

    def network(self, inp_shape, filters=32, kernel_size=(3, 3)):
        inp, out = self.input_layer(inp_shape)

        out = Conv2D(filters, kernel_size=kernel_size, activation='relu', padding='same')(out)
        out1 = Conv2D(filters, kernel_size=kernel_size, activation='relu', padding='same')(out)
        out = Conv2D(filters, kernel_size=(2, 2), strides=(2, 2), activation='relu')(out1)

        out = Conv2D(2*filters, kernel_size=kernel_size, activation='relu', padding='same')(out)
        out2 = Conv2D(2*filters, kernel_size=kernel_size, activation='relu', padding='same')(out)

        out = Conv2D(2*filters, kernel_size=(2, 2), strides=(2, 2), activation='relu')(out2)

        out = Conv2D(4 * filters, kernel_size=kernel_size, activation='relu', padding='same')(out)
        out3 = Conv2D(4 * filters, kernel_size=kernel_size, activation='relu', padding='same')(out)

        out = Conv2D(4 * filters, kernel_size=(2, 2), strides=(2, 2), activation='relu')(out3)

        out = Conv2D(8 * filters, kernel_size=kernel_size, activation='relu', padding='same')(out)
        out = Conv2D(8 * filters, kernel_size=kernel_size, activation='relu', padding='same')(out)

        out = Conv2DTranspose(4 * filters, kernel_size=kernel_size, strides=(2, 2), activation='relu', padding='same')(out)
        out = Concatenate()([out, out3])

        out = Conv2D(4 * filters, kernel_size=kernel_size, activation='relu', padding='same')(out)
        out = Conv2D(4 * filters, kernel_size=kernel_size, activation='relu', padding='same')(out)

        out = Conv2DTranspose(2 * filters, kernel_size=kernel_size, strides=(2, 2), activation='relu', padding='same')(out)
        out = Concatenate()([out, out2])

        out = Conv2D(2 * filters, kernel_size=kernel_size, activation='relu', padding='same')(out)
        out = Conv2D(2 * filters, kernel_size=kernel_size, activation='relu', padding='same')(out)

        out = Conv2DTranspose(filters, kernel_size=kernel_size, strides=(2, 2), activation='relu', padding='same')(out)
        out = Concatenate()([out, out1])

        out = Conv2D(filters, kernel_size=kernel_size, activation='relu', padding='same')(out)
        out = Conv2D(filters, kernel_size=kernel_size, activation='relu', padding='same')(out)

        out = Conv2D(1, (1, 1))(out)

        out = self._pseudoinverse_tensorflow(out)

        out = Conv2D(64, (10, 10), padding='same')(out)
        out = Conv2D(1, (1, 1), padding='same')(out)

        out = self._operator_tensorflow(out)

        out = tf.identity(out, name='output_upsample')
        return inp, out
