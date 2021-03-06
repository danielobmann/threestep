from tensorflow.keras.layers import Conv2D, PReLU, BatchNormalization, UpSampling2D, Concatenate
import tensorflow as tf
import odl.contrib.tensorflow
import numpy as np


class DataConsistentNetwork:

    def __init__(self, operator, pseudoinverse):
        self._operator = operator
        self._psuedoinverse = pseudoinverse
        self._operator_tensorflow = odl.contrib.tensorflow.as_tensorflow_layer(operator, 'OperatorUpsampling')
        self._pseudoinverse_tensorflow = odl.contrib.tensorflow.as_tensorflow_layer(pseudoinverse, 'PseudoUpsampling')
        self._out_shape = operator.range.shape
        self._mask = np.zeros(self._out_shape)

    def input_layer(self, inp_shape):
        # Return input layer with appropriate input size
        inp = tf.placeholder(tf.float32, shape=(None, ) + inp_shape, name='input_upsample')
        upsampling_factor = self._out_shape[0]//inp_shape[0]

        # Simple upsampling, e.g. by filling with zeros
        for i in range(inp_shape[0]):
            self._mask[i*upsampling_factor, ...] = 1

        self._mask = tf.constant(self._mask[None, ..., None], dtype=tf.float32, name='mask')
        out = UpSampling2D(size=(upsampling_factor, 1), interpolation='bilinear')(inp)
        y0 = out*self._mask
        return inp, out, y0

    @staticmethod
    def _convolution_block(out, global_step, steps=2, filters=32, kernel_size=(3, 3), batch=False, act=True):

        for step in range(steps):
            out = Conv2D(filters, kernel_size, padding='same', name='conv_dcs_' + str(global_step) + '_' + str(step))(out)
            if batch:
                out = BatchNormalization(name='batch_dcs_' + str(global_step) + '_' + str(step))(out)
            if act:
                out = PReLU(name='prelu_dcs_' + str(global_step) + '_' + str(step))(out)

        return out

    @staticmethod
    def _convolution_block_residual(out, global_step, steps=2, filters=32, kernel_size=(3, 3), act=True):
        out = Conv2D(1, (1, 1), padding='same', name='conv_block_'+str(global_step))(out)
        for step in range(steps):
            out = Conv2D(filters, kernel_size, padding='same', name='conv_dcs_' + str(global_step) + '_' + str(step))(out)
            out = BatchNormalization(name='batch_dcs_' + str(global_step) + '_' + str(step))(out)

            if act:
                out = PReLU(name='prelu_dcs_' + str(global_step) + '_' + str(step), shared_axes=[1, 2])(out)

        return out

    def _data_consistency_block(self, inp, y0, global_step, filters=32, kernel_size=(3, 3), batch=False):

        out = self._convolution_block(inp, global_step=global_step, filters=filters, kernel_size=kernel_size, batch=batch)
        out = Conv2D(1, (1, 1), padding='same', name='linear_combination_' + str(global_step))(out)

        # Enforce consistency with data
        out = out*(1-self._mask) + y0

        return out

    def _data_consistency_block_residual(self, u0, out, y0, global_step, filters=32, kernel_size=(3, 3)):

        out = Concatenate()([u0, out, y0])
        out = self._convolution_block_residual(out, global_step=global_step, filters=filters, kernel_size=kernel_size)
        out = Conv2D(1, (1, 1), padding='same', name='linear_combination_dcs' + str(global_step) + '_2')(out)

        # Enforce consistency with data
        out = out*(1-self._mask) + y0
        return u0, out, y0

    def _operator_consistency_block(self, inp, global_step, filters=32, kernel_size=(3, 3), batch=False):

        out = self._convolution_block(inp, global_step=global_step, filters=filters, kernel_size=kernel_size, batch=batch)
        out = Conv2D(1, (1, 1), padding='same', name='linear_combination_' + str(global_step) + '_1')(out)

        # Enforce operator consistency
        out = self._pseudoinverse_tensorflow(out)

        out = self._convolution_block(out, global_step=global_step+100, filters=64, kernel_size=(10, 10), batch=batch, act=False)
        out = Conv2D(1, (1, 1), padding='same', name='linear_combination_' + str(global_step) + '_2')(out)

        out = self._operator_tensorflow(out)

        return out

    def _operator_consistency_block_residual(self, u0, out, y0, global_step, filters=32, kernel_size=(3, 3)):

        out = Concatenate()([u0, out, y0])
        out = self._convolution_block_residual(out, global_step=global_step, filters=filters, kernel_size=kernel_size)
        out = Conv2D(1, (1, 1), padding='same', name='linear_combination_operator' + str(global_step) + '_1')(out)

        # Enforce operator consistency
        out = self._pseudoinverse_tensorflow(out)

        out = Conv2D(64, (10, 10), name='x_space_convolution' + str(global_step), padding='same')(out)
        out = Conv2D(1, (1, 1), padding='same', name='linear_combination_operator' + str(global_step) + '_2')(out)

        out = self._operator_tensorflow(out)
        return u0, out, y0

    def network(self, inp_shape, steps=5, filters=16, kernel_size=(3, 3), batch=False):
        inp, out, y0 = self.input_layer(inp_shape)
        # Go in steps of 2 to have a unique global step
        for i in range(0, 2*steps, 2):
            out = self._data_consistency_block(out, y0, global_step=i, filters=filters, kernel_size=kernel_size, batch=batch)
            out = self._operator_consistency_block(out, global_step=i+1, filters=filters, kernel_size=kernel_size, batch=batch)
        out = tf.identity(out, name='output_upsample')
        return inp, out

    def network_residual(self, inp_shape, steps=3, filters=32, kernel_size=(3, 3), consistency='operator'):
        inp, u0, y0 = self.input_layer(inp_shape)
        out = u0

        # Go in steps of 2 to have a unique global step
        for i in range(0, 2*steps, 2):
            if consistency == 'operator':
                u0, out, y0 = self._data_consistency_block_residual(u0, out, y0, global_step=i, filters=filters,
                                                                    kernel_size=kernel_size)
                u0, out, y0 = self._operator_consistency_block_residual(u0, out, y0, global_step=i+1, filters=filters,
                                                                        kernel_size=kernel_size)
            elif consistency == 'data':
                u0, out, y0 = self._operator_consistency_block_residual(u0, out, y0, global_step=i + 1, filters=filters,
                                                                        kernel_size=kernel_size)
                u0, out, y0 = self._data_consistency_block_residual(u0, out, y0, global_step=i, filters=filters,
                                                                    kernel_size=kernel_size)
            else:
                ValueError('Unknown consistency type.')
        out = tf.identity(out, name='output_upsample')
        return inp, out
