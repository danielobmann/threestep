from tensorflow.keras.layers import Conv2D, PReLU, BatchNormalization, UpSampling2D, Concatenate, ReLU, Input
import tensorflow as tf
import odl.contrib.tensorflow


class OperatorConsistentNetwork:

    def __init__(self, operator, pseudoinverse):
        self._operator = operator
        self._psuedoinverse = pseudoinverse
        self._operator_tensorflow = odl.contrib.tensorflow.as_tensorflow_layer(operator, 'OperatorUpsampling')
        self._pseudoinverse_tensorflow = odl.contrib.tensorflow.as_tensorflow_layer(pseudoinverse, 'PseudoUpsampling')

    @staticmethod
    def _residual_dense_block(inp, c=3, g=64):
        out_concatenate = inp
        for _ in range(c):
            out = Conv2D(g, (3, 3), padding='same')(out_concatenate)
            out = ReLU()(out)
            out_concatenate = Concatenate()([out, out_concatenate])
        out = Conv2D(g, (1, 1), padding='same')(out_concatenate)
        out = inp + out
        return out

    def _periodic_shuffling(self, inp, r=10):
        N = int(inp.shape[1])
        temp = tf.unstack(inp, axis=-1)
        temp = [tf.unstack(t, axis=1) for t in temp]
        temp = [tf.stack([temp[j][i] for j in range(r)], axis=1) for i in range(N)]
        temp = tf.concat(temp, axis=1)
        return tf.expand_dims(temp, axis=-1)

    def network(self, shape, r=10, g=64, n_rdb=3, c=3):
        inp = Input(shape=shape, name='sr_input')
        #inp = tf.placeholder(tf.float32, shape=shape, name='sr_input')

        # Feature extraction on low resolution sinogram
        out_low = Conv2D(g, (3, 3), padding='same')(inp)
        out = Conv2D(g, (3, 3), padding='same')(out_low)
        rdb_out = []

        for _ in range(n_rdb):
            out = self._residual_dense_block(out, g=g, c=c)
            rdb_out.append(out)

        rdb_conc = Concatenate()(rdb_out)
        out = Conv2D(g, (1, 1), padding='same')(rdb_conc)
        out = Conv2D(g, (3, 3), padding='same')(out)

        out = out + out_low

        # Upsampling

        out = Conv2D(r, (3, 3), padding='same')(out)
        out = self._periodic_shuffling(out, r=r)
        out = Conv2D(1, (3, 3), padding='same')(out)

        # Operator consistency

        out = self._pseudoinverse_tensorflow(out)
        out = Conv2D(g, (10, 10), padding='same')(out)
        out = Conv2D(1, (1, 1), padding='same')(out)

        out = self._operator_tensorflow(out)

        out = tf.identity(out, name='sr_output')
        return inp, out
