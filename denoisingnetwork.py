from tensorflow.keras.layers import Conv2D, PReLU, Add
import tensorflow as tf
import odl.contrib.tensorflow


class DenoisingNetwork:
    def __init__(self, operator, pseudoinverse):
        self._operator = operator
        self._psuedoinverse = pseudoinverse
        self._operator_tensorflow = odl.contrib.tensorflow.as_tensorflow_layer(operator, 'OperatorDenoising')
        self._pseudoinverse_tensorflow = odl.contrib.tensorflow.as_tensorflow_layer(pseudoinverse, 'PseudoDenoising')

    def network(self):
        inp_shape = self._operator.range.shape + (1,)

        inp = tf.placeholder(tf.float32, shape=(None,) + inp_shape, name='input_denoising')

        out = Conv2D(64, (3, 3), padding='same')(inp)
        out = PReLU()(out)

        out = Conv2D(64, (3, 3), padding='same')(out)
        out = PReLU()(out)

        out = Conv2D(64, (3, 3), padding='same')(out)
        out = PReLU()(out)

        out = Conv2D(1, (1, 1), padding='same')(out)
        out = Add()([out, inp])

        # Make output operator consistent
        out = self._pseudoinverse_tensorflow(out)

        out = Conv2D(64, (10, 10), padding='same')(out)
        out = Conv2D(1, (1, 1), padding='same')(out)

        out = self._operator_tensorflow(out)
        out = tf.identity(out, name='output_denoising')
        return inp, out
