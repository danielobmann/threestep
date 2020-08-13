import sys
sys.path.append('..')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.layers import Conv2D, BatchNormalization, PReLU, UpSampling2D, Add
import numpy as np
import tensorflow as tf
import odl
import odl.contrib.tensorflow
import matplotlib.pyplot as plt
from datanetwork import *

sess = tf.Session()

# ---------------------------
# Specify parameters
epochs = 21
batch_size = 1
n_training_samples = 1709
n_validation_samples = 458
n_batches = n_training_samples//batch_size
n_batches_val = n_validation_samples//batch_size

initial_lr = 1e-3

size = 512
n_theta = 32
upsampling_factor = 23
n_s = 768

# ---------------------------
# Set up tomography operator

space = odl.uniform_discr([-128, -128], [128, 128], [size, size], dtype='float32', weighting=1.0)
angle_partition = odl.uniform_partition(0, np.pi, n_theta)
angle_partition_up = odl.uniform_partition(0, np.pi, n_theta*upsampling_factor)
detector_partition = odl.uniform_partition(-360, 360, n_s)
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)
geometry_up = odl.tomo.Parallel2dGeometry(angle_partition_up, detector_partition)

operator = odl.tomo.RayTransform(space, geometry)
pseudoinverse = odl.tomo.fbp_op(operator)

Radon = odl.tomo.RayTransform(space, geometry_up)
FBP = odl.tomo.fbp_op(Radon)

operator /= odl.operator.power_method_opnorm(operator)
Radon /= odl.operator.power_method_opnorm(Radon)

# Create tensorflow layer from odl operator
odl_op_layer = odl.contrib.tensorflow.as_tensorflow_layer(operator, 'RayTransform')
odl_op_layer_pseudo = odl.contrib.tensorflow.as_tensorflow_layer(pseudoinverse, 'RayTransformPseudo')

radon_layer = odl.contrib.tensorflow.as_tensorflow_layer(Radon, 'Radon')
radon_adjoint_layer = odl.contrib.tensorflow.as_tensorflow_layer(Radon.adjoint, 'RadonAdjoint')
fbp_layer = odl.contrib.tensorflow.as_tensorflow_layer(FBP, 'FBP')

# ---------------------------
# Set up denoising network
inp_shape = operator.range.shape + (1, )

inp = tf.placeholder(tf.float32, shape=(None,) + inp_shape, name='input_denoising')

out = Conv2D(64, (3, 3), padding='same')(inp)
out = BatchNormalization()(out)
out = PReLU()(out)

out = Conv2D(64, (3, 3), padding='same')(out)
out = BatchNormalization()(out)
out = PReLU()(out)

out = Conv2D(64, (3, 3), padding='same')(out)
out = BatchNormalization()(out)
out = PReLU()(out)

out = Conv2D(64, (3, 3), padding='same')(out)
out = BatchNormalization()(out)
out = PReLU()(out)

out = Conv2D(1, (3, 3), padding='same')(out)
out = PReLU()(out)

out = Add()([out, inp])

# Make output operator consistent
out = odl_op_layer_pseudo(out)
out = Conv2D(64, (10, 10), padding='same')(out)
out = BatchNormalization()(out)
out = PReLU()(out)
out = Conv2D(1, (1, 1), padding='same')(out)
out = odl_op_layer(out)
out = tf.identity(out, name='output_denoising')

# ---------------------------
# Define upsampling network
DCS = DataConsistentNetwork(Radon, FBP)
inp_up, out_up = DCS.network(inp_shape, steps=5, filters=16)

sess.run(tf.global_variables_initializer())

# Restore graph from trained model
restore_path = "models/upsampling/"
if 1:
    new_saver = tf.train.import_meta_graph(restore_path + '-0.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint(restore_path))

graph = tf.get_default_graph()


# ---------------------------
# Define inputs and outputs of denoising/upsampling network
inp_denois = graph.get_tensor_by_name("input_denoising:0")
out_denois = graph.get_tensor_by_name("output_denoising:0")

inp_up_new = graph.get_tensor_by_name("input_upsample:0")
out_up_new = graph.get_tensor_by_name("output_upsample:0")


# ---------------------------
# Define generator for data
data_path = "../data/mayoclinic/data/full3mm/"
sigma = 0.2


def data_generator_inversion(batch_size=32, mode='train', rescale=1000.):
    p = data_path + mode
    files = np.random.choice(os.listdir(p), size=batch_size, replace=False)
    X = [np.load(p + '/' + file) for file in files]
    y_n = [operator(x / rescale) for x in X]

    x_t = np.stack(X)[..., None]/rescale

    # Get low resolution sinograms

    y_n = [y + np.random.normal(0, 1, y.shape) * sigma for y in y_n]
    y_n = np.stack(y_n)[..., None]
    y_t = y_n

    # Process images
    y_n = sess.run(out_denois, feed_dict={inp_denois: y_n})
    y_up = sess.run(out_up, feed_dict={inp_up: y_n})

    return y_up, x_t, y_n, y_t


# ---------------------------
# Input example

y_example, x_example, y_noisy, y_t = data_generator_inversion(1)
y_example1 = sess.run(out_up_new, feed_dict={inp_up_new: y_noisy})

print(np.mean(y_example == y_example1))

plt.subplot(221)
plt.imshow(y_example[0, ..., 0], cmap='bone')
plt.colorbar()

plt.subplot(222)
plt.imshow(FBP(y_example[0, ..., 0]), cmap='bone')

plt.subplot(223)
plt.imshow(Radon(x_example[0, ..., 0]), cmap='bone')
plt.colorbar()

plt.subplot(224)
plt.imshow(x_example[0, ..., 0], cmap='bone')

plt.savefig("images/inversion_input_example.pdf", format='pdf')
plt.clf()


fig, axs = plt.subplots(nrows=2, ncols=2)
im = axs[0, 0].imshow(y_noisy[0, ..., 0], cmap='bone')
axs[0, 0].set_aspect(n_s / n_theta)
axs[0, 0].axis('off')
axs[0, 0].set_title('Denoised')
fig.colorbar(im, ax=axs[0, 0])

im = axs[0, 1].imshow(y_t[0, ..., 0], cmap='bone')
axs[0, 1].set_aspect(n_s / n_theta)
axs[0, 1].axis('off')
axs[0, 1].set_title('Input')
fig.colorbar(im, ax=axs[0, 1])

plt.savefig("images/inversion_denoised_input.pdf", format='pdf')
