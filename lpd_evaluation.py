import sys
sys.path.append('..')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.layers import Conv2D, PReLU
import numpy as np
import tensorflow as tf
import odl
import odl.contrib.tensorflow
import matplotlib.pyplot as plt


sess = tf.Session()

# ---------------------------
# Specify parameters
size = 512
n_theta = 32
n_s = 768

# ---------------------------
# Set up tomography operator

space = odl.uniform_discr([-128, -128], [128, 128], [size, size], dtype='float32', weighting=1.0)
angle_partition = odl.uniform_partition(0, np.pi, n_theta)
detector_partition = odl.uniform_partition(-360, 360, n_s)
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

operator = odl.tomo.RayTransform(space, geometry)
pseudoinverse = odl.tomo.fbp_op(operator)
FBP = odl.tomo.fbp_op(operator, frequency_scaling=0.2)

operator /= odl.operator.power_method_opnorm(operator)

# Create tensorflow layer from odl operator
odl_op_layer = odl.contrib.tensorflow.as_tensorflow_layer(operator, 'RayTransform')
odl_op_layer_adjoint = odl.contrib.tensorflow.as_tensorflow_layer(operator.adjoint, 'RayTransformAdjoint')

# ---------------------------
# Define inversion network

n_primal, n_dual, n_iter = 5, 5, 5

input_y = tf.placeholder(tf.float32, shape=(None, n_theta, n_s, 1), name='input_y')
input_x = tf.placeholder(tf.float32, shape=(None, size, size, 1), name='input_x')


def apply_conv(inputs, filters=32, act=True):
    outputs = Conv2D(filters, kernel_size=(3, 3), padding='same')(inputs)
    if act:
        outputs = PReLU()(outputs)
    return outputs


with tf.name_scope('tomography'):
    with tf.name_scope('initial_values'):
        primal = tf.concat([tf.zeros_like(input_x)] * n_primal, axis=-1)
        dual = tf.concat([tf.zeros_like(input_y)] * n_dual, axis=-1)

    for i in range(n_iter):
        with tf.variable_scope('dual_iterate_{}'.format(i)):
            evalpt = primal[..., 1:2]
            evalop = odl_op_layer(evalpt)
            update = tf.concat([dual, evalop, input_y], axis=-1)

            update = apply_conv(update)
            update = apply_conv(update)
            update = apply_conv(update, filters=n_dual, act=False)
            dual = dual + update

        with tf.variable_scope('primal_iterate_{}'.format(i)):
            evalpt_fwd = primal[..., 0:1]
            evalop_fwd = odl_op_layer(evalpt_fwd)

            evalpt = dual[..., 0:1]
            evalop = odl_op_layer_adjoint(evalop_fwd * evalpt)
            update = tf.concat([primal, evalop], axis=-1)

            update = apply_conv(update)
            update = apply_conv(update)
            update = apply_conv(update, filters=n_primal, act=False)
            primal = primal + update

    x_result = primal[..., 0:1]

output = tf.identity(x_result, name='output_inversion')

sess.run(tf.global_variables_initializer())

# Restore graph from trained model
restore_path = "models/"
if 1:
    new_saver = tf.train.import_meta_graph(restore_path + 'lpd_network-10.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint(restore_path))

graph = tf.get_default_graph()

# my_list = [n.name for n in tf.get_default_graph().as_graph_def().node]
#
# with open('your_file.txt', 'w') as f:
#     for item in my_list:
#         f.write("%s\n" % item)

inp_inversion = graph.get_tensor_by_name("input_y_1:0")
inp_x = graph.get_tensor_by_name("input_x_1:0")
out_inversion = graph.get_tensor_by_name("output_inversion_1:0")

# ---------------------------
# Define metrics for evaluation


def nmse(x_true, x_pred):
    d = np.mean((x_true - x_pred)**2)
    n = np.mean(x_true**2)
    return d/n


def psnr(x_true, x_pred):
    ran = np.amax(x_true) - np.amin(x_true)
    return 20*np.log(ran) - 10*np.log(np.mean((x_true - x_pred)**2))


# ---------------------------
# Evaluate on test-data
path = "../data/mayoclinic/data/full3mm/test"
sigma = 0.2
rescale = 1000.


# ------------------------
# Plot one specific example for paper
np.random.seed(1)
file = np.random.choice(os.listdir(path))
x = np.load(path + '/' + file)/rescale
y_n = operator(x)
y_n += np.random.normal(0, 1, y_n.shape)*sigma
y_n = y_n[None, ..., None]

x_rec = sess.run(out_inversion, feed_dict={inp_inversion: y_n, inp_x: np.zeros((1, size, size, 1))})
x_rec1 = sess.run(output, feed_dict={input_y: y_n, input_x: np.zeros((1, size, size, 1))})

print(np.mean(x_rec == x_rec1))

x_rec = x_rec[0, ..., 0]

plt.imshow(y_n[0, ..., 0], cmap='bone')
plt.colorbar()
plt.savefig("images/reconstruction_data.pdf", format='pdf')
plt.clf()

plt.imshow(x_rec, cmap='bone')
plt.axis('off')
plt.savefig('images/reconstruction_lpd.pdf', format='pdf')

plt.imshow(x, cmap='bone')
plt.axis('off')
plt.savefig('images/reconstruction_gt.pdf', format='pdf')

plt.imshow(FBP(y_n[0, ..., 0]), cmap='bone')
plt.axis('off')
plt.savefig('images/reconstruction_fbp.pdf', format='pdf')


NMSE = []
PSNR = []


for file in os.listdir(path):
    x = np.load(path + '/' + file)/rescale
    y_n = operator(x)
    y_n += np.random.normal(0, 1, y_n.shape)*sigma
    y_n = y_n[None, ..., None]

    x_rec = sess.run(out_inversion, feed_dict={inp_inversion: y_n, inp_x: np.zeros((1, size, size, 1))})
    x_rec = x_rec[0, ..., 0]

    NMSE.append(nmse(x, x_rec))
    PSNR.append(psnr(x, x_rec))

    plt.subplot(221)
    plt.imshow(x, cmap='bone')
    plt.axis('off')
    plt.title('True')

    plt.subplot(222)
    plt.imshow(pseudoinverse(y_n[0, ..., 0]), cmap='bone')
    plt.axis('off')
    plt.title('FBP')

    plt.subplot(223)
    plt.imshow(x_rec, cmap='bone')
    plt.axis('off')
    plt.title('Prediction')

    plt.subplot(224)
    plt.imshow(np.abs(x - x_rec), cmap='bone')
    plt.axis('off')
    plt.colorbar()
    plt.title('Difference')

    plt.savefig("images/test_lpd/" + file[0:-8] + '.pdf', format='pdf')
    plt.clf()


print(NMSE)
print(PSNR)



