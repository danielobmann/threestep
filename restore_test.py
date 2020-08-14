import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib.pyplot as plt

from imports.operators import *
from imports.misc import *
from imports.denoisingnetwork import *
from imports.upsamplingnetwork import *
from imports.inversionnetwork import *

sess = tf.Session()

restore_path = "models/upsampling/"

# ---------------------------
# Set up networks
inp_denoising, out_denoising = DenoisingNetwork(RadonSparse, FBPSparse).network()
inp_up, out_up = DataConsistentNetwork(Radon, FBP).network((n_theta, n_s, 1), steps=5, filters=16)
inp_y, inp_x, out_inversion = InversionNetwork(Radon).network(n_primal=5, n_dual=5, n_iter=5)

sess.run(tf.global_variables_initializer())

saver = tf.train.import_meta_graph(restore_path + 'upsampling_network-0.meta')
saver.restore(sess, tf.train.latest_checkpoint(restore_path))

DG = DataGenerator(operator=RadonSparse, operator_up=Radon)

# ---------------------------
# Input example

y_noisy, y_true, x_true = DG.get_batch_upsampling(1)
y_denoised = sess.run(out_denoising, feed_dict={inp_denoising: y_noisy})
y_upsample = sess.run(out_up, feed_dict={inp_up: y_denoised})

plt.subplot(221)
plt.imshow(y_upsample[0, ..., 0], cmap='bone')
plt.colorbar()

plt.subplot(222)
plt.imshow(FBP(y_upsample[0, ..., 0]), cmap='bone')
plt.colorbar()

plt.subplot(223)
plt.imshow(y_true[0, ..., 0], cmap='bone')
plt.colorbar()

plt.subplot(224)
plt.imshow(x_true[0, ..., 0], cmap='bone')
plt.colorbar()

plt.savefig("images/inversion_input_example.pdf", format='pdf')
plt.clf()


fig, axs = plt.subplots(nrows=2, ncols=2)
im = axs[0, 0].imshow(y_noisy[0, ..., 0], cmap='bone')
axs[0, 0].set_aspect(n_s / n_theta)
axs[0, 0].axis('off')
axs[0, 0].set_title('Noisy')
fig.colorbar(im, ax=axs[0, 0])

im = axs[0, 1].imshow(y_denoised[0, ..., 0], cmap='bone')
axs[0, 1].set_aspect(n_s / n_theta)
axs[0, 1].axis('off')
axs[0, 1].set_title('Input/Denoised')
fig.colorbar(im, ax=axs[0, 1])

plt.savefig("images/inversion_denoised_input.pdf", format='pdf')
