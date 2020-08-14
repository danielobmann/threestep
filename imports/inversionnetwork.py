from tensorflow.keras.layers import Conv2D, PReLU
import tensorflow as tf
import odl.contrib.tensorflow


class InversionNetwork:
    def __init__(self, operator):
        self._operator = operator
        self._adjoint = operator.adjoint
        self._operator_tensorflow = odl.contrib.tensorflow.as_tensorflow_layer(operator, 'OperatorInversion')
        self._adjoint_tensorflow = odl.contrib.tensorflow.as_tensorflow_layer(operator.adjoint, 'AdjointInversion')

    @staticmethod
    def _apply_conv(inputs, filters=32, act=True):
        outputs = Conv2D(filters, kernel_size=(3, 3), padding='same')(inputs)
        if act:
            outputs = PReLU()(outputs)
        return outputs

    def network(self, n_primal=5, n_dual=5, n_iter=5):
        input_y = tf.placeholder(tf.float32, shape=(None, ) + self._operator.range.shape + (1,), name='input_inversion_y')
        input_x = tf.placeholder(tf.float32, shape=(None, ) + self._adjoint.range.shape + (1,), name='input_inversion_x')

        with tf.name_scope('tomography'):
            with tf.name_scope('initial_values'):
                primal = tf.concat([tf.zeros_like(input_x)] * n_primal, axis=-1)
                dual = tf.concat([tf.zeros_like(input_y)] * n_dual, axis=-1)

            for i in range(n_iter):
                with tf.variable_scope('dual_iterate_{}'.format(i)):
                    evalpt = primal[..., 1:2]
                    evalop = self._operator_tensorflow(evalpt)
                    update = tf.concat([dual, evalop, input_y], axis=-1)

                    update = self._apply_conv(update)
                    update = self._apply_conv(update)
                    update = self._apply_conv(update, filters=n_dual, act=False)
                    dual = dual + update

                with tf.variable_scope('primal_iterate_{}'.format(i)):
                    evalpt_fwd = primal[..., 0:1]
                    evalop_fwd = self._operator_tensorflow(evalpt_fwd)

                    evalpt = dual[..., 0:1]
                    evalop = self._adjoint_tensorflow(evalop_fwd * evalpt)
                    update = tf.concat([primal, evalop], axis=-1)

                    update = self._apply_conv(update)
                    update = self._apply_conv(update)
                    update = self._apply_conv(update, filters=n_primal, act=False)
                    primal = primal + update

            x_result = primal[..., 0:1]

        output = tf.identity(x_result, name='output_inversion')
        return input_y, input_x, output
