import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib.pyplot as plt

from imports.operators import *
from imports.misc import *
from imports.denoisingnetwork import *
from imports.upsamplingunet import *
from imports.inversionnetwork import *

sess = tf.Session()

# ---------------------------
# Specify parameters
epochs = 5
batch_size = 4
batch_size_val = 16
n_training_samples = 1709
n_validation_samples = 458
n_batches = n_training_samples//batch_size
n_batches_val = n_validation_samples//batch_size_val

initial_lr = 1e-3

restore_path = "models/denoising/"

# ---------------------------
# Define networks
inp_denoising, out_denoising = DenoisingNetwork(RadonSparse, FBPSparse).network()
inp_up, out_up = UNetUpsampling(Radon, FBP).network((n_theta, n_s, 1))
inp_y, inp_x, out_inversion = InversionNetwork(Radon).network(n_primal=5, n_dual=5, n_iter=5)

# Upsampling

y_true = tf.placeholder(shape=(None, n_theta*upsampling_factor, n_s, 1), dtype=tf.float32)
loss = tf.reduce_mean(tf.squared_difference(out_up, y_true))

learning_rate = tf.placeholder(dtype=tf.float32, name='lr_upsampling')
opt = tf.train.AdamOptimizer(learning_rate=learning_rate, name='adam_upsampling')

train_op = opt.minimize(loss)

# Inversion
x_true = tf.placeholder(shape=(None, size, size, 1), dtype=tf.float32)
loss_inv = tf.reduce_mean(tf.squared_difference(out_inversion, x_true))

learning_rate_inv = tf.placeholder(dtype=tf.float32, name='lr_inversion')
opt_inv = tf.train.AdamOptimizer(learning_rate=1e-3, name='adam_inversion')

train_op_inv = opt_inv.minimize(loss_inv)

sess.run(tf.global_variables_initializer())

# ----------------------------
# Restore models

saver = tf.train.import_meta_graph(restore_path + 'denoising_network-20.meta')
saver.restore(sess, tf.train.latest_checkpoint(restore_path))

# ---------------------------
# Set up various functions


def plot_validation(y_in, y_pred, y_true, epoch=10):
    n_images = y_in.shape[0]
    for i in range(n_images):
        fig, axs = plt.subplots(nrows=2, ncols=2)
        im = axs[0, 0].imshow(y_true[i, ..., 0], cmap='bone')
        axs[0, 0].axis('off')
        axs[0, 0].set_title('True')
        fig.colorbar(im, ax=axs[0, 0])

        im = axs[0, 1].imshow(y_in[i, ..., 0], cmap='bone')
        axs[0, 1].set_aspect(n_s / n_theta)
        axs[0, 1].axis('off')
        axs[0, 1].set_title('Input')
        fig.colorbar(im, ax=axs[0, 1])

        im = axs[1, 0].imshow(y_pred[i, ..., 0], cmap='bone')
        axs[1, 0].axis('off')
        axs[1, 0].set_title('Prediction')
        fig.colorbar(im, ax=axs[1, 0])

        im = axs[1, 1].imshow(np.abs(y_true[i, ..., 0] - y_pred[i, ..., 0]), cmap='bone')
        axs[1, 1].axis('off')
        axs[1, 1].set_title('Difference')
        fig.colorbar(im, ax=axs[1, 1])

        fig.savefig('images/UNET_ValidationImage_Epoch' + str(epoch) + '_' + str(i) + '.pdf', format='pdf')
        fig.clf()

        plt.subplot(221)
        plt.imshow(FBP(y_true[i, ..., 0]), cmap='bone')
        plt.axis('off')
        plt.title('True')

        plt.subplot(222)
        plt.imshow(FBPSparse(y_in[i, ..., 0]), cmap='bone')
        plt.axis('off')
        plt.title('Input')

        plt.subplot(223)
        plt.imshow(FBP(y_pred[i, ..., 0]), cmap='bone')
        plt.axis('off')
        plt.title('Prediction')

        plt.subplot(224)
        plt.imshow(FBP(y_true[i, ..., 0] - y_pred[i, ..., 0]), cmap='bone')
        plt.axis('off')
        plt.title('Difference')
        plt.colorbar()

        fig.savefig('images/UNET_ReconImage_Epoch' + str(epoch) + '_' + str(i) + '.pdf', format='pdf')
        fig.clf()
    pass


nmse = NMSE(out_up, y_true)
psnr = PSNR(out_up, y_true)

DG = DataGenerator(operator=RadonSparse, operator_up=Radon)


# ---------------------------
# Test input preprocessing

y_n, y_t, _ = DG.get_batch_upsampling(1)
y_d = sess.run(out_denoising, feed_dict={inp_denoising: y_n})

fig, axs = plt.subplots(nrows=2, ncols=2)
im = axs[0, 0].imshow(y_t[0, ..., 0], cmap='bone')
axs[0, 0].axis('off')
axs[0, 0].set_title('True')
fig.colorbar(im, ax=axs[0, 0])

im = axs[0, 1].imshow(y_n[0, ..., 0], cmap='bone')
axs[0, 1].set_aspect(n_s/n_theta)
axs[0, 1].axis('off')
axs[0, 1].set_title('Noisy')
fig.colorbar(im, ax=axs[0, 1])

im = axs[1, 1].imshow(y_d[0, ..., 0], cmap='bone')
axs[1, 1].set_aspect(n_s/n_theta)
axs[1, 1].axis('off')
axs[1, 1].set_title('Denoised')
fig.colorbar(im, ax=axs[1, 1])

im = axs[1, 0].imshow(y_d[0, ..., 0], cmap='bone')
axs[1, 0].set_aspect(n_s/n_theta)
axs[1, 0].axis('off')
axs[1, 0].set_title('Denoised')
fig.colorbar(im, ax=axs[1, 0])

plt.savefig("images/UNET_InputTest.pdf", format='pdf')
plt.clf()

# ---------------------------
save_path = "models/unet/unet"
n_save = 5
n_val = 1
n_plot = 1

print("Initialization successful. Starting training...", flush=True)

for epoch in range(epochs):
    ERR = []
    NMSE_train = []
    PSNR_train = []

    print("### Epoch %d/%d ###" % (epoch + 1, epochs))
    for j in range(n_batches):
        y_input, y_output, _ = DG.get_batch_upsampling(batch_size=batch_size, mode='train')
        y_denois = sess.run(out_denoising, feed_dict={inp_denoising: y_input})

        fd = {inp_up: y_denois,
              y_true: y_output,
              learning_rate: cosine_decay(epoch, epochs)}

        c, nm, ps, _ = sess.run([loss, nmse, psnr, train_op], feed_dict=fd)
        NMSE_train.append(nm)
        PSNR_train.append(ps)
        ERR.append(c)

        print("Progress %f, Loss %f" % ((j + 1) / n_batches, np.mean(ERR)), end='\r', flush=True)

    print("Training: Loss %f NMSE %f PSNR %f" % (np.mean(ERR), np.mean(NMSE_train), np.mean(PSNR_train)), end='\r')

    # Validate model performance
    if epoch % n_val == 0:
        ERR_VAL = []
        NMSE_VAL = []
        PSN_VAL = []
        for j in range(n_batches_val):
            y_input, y_output, _ = DG.get_batch_upsampling(batch_size=batch_size_val, mode='val')
            y_denois = sess.run(out_denoising, feed_dict={inp_denoising: y_input})

            fd = {inp_up: y_denois,
                  y_true: y_output}

            c, nm, ps = sess.run([loss, nmse, psnr], feed_dict=fd)
            ERR_VAL.append(c)
            PSN_VAL.append(ps)
            NMSE_VAL.append(nm)
        print(" ")
        print("Validation: Loss %f NMSE %f PSNR %f" % (np.mean(ERR_VAL), np.mean(NMSE_VAL), np.mean(PSN_VAL)))
        print(" ", flush=True)

    if (epoch % n_plot) == 0:
        y_input, y_output, _ = DG.get_batch_upsampling(batch_size=batch_size, mode='val')
        y_denois = sess.run(out_denoising, feed_dict={inp_denoising: y_input})

        fd = {inp_up: y_denois,
              y_true: y_output}

        y_pred = sess.run(out_up, feed_dict=fd)
        plot_validation(y_input, y_pred, y_output, epoch=epoch)

    # Save model every
    if (epoch % n_save) == 0:
        saver.save(sess, save_path, global_step=epoch)

# -----------------------------------
# ###################################
# -----------------------------------
# Start inversion training

batch_size = 1
n_batches = n_training_samples//batch_size
n_batches_val = n_validation_samples//batch_size

initial_lr = 1e-3

# ----------------------------
# Restore models

nmse = NMSE(out_inversion, x_true)
psnr = PSNR(out_inversion, x_true)

DG = DataGenerator(operator=RadonSparse, operator_up=Radon)


def plot_validation(x_pred, x_true, epoch=10):
    n_images = x_pred.shape[0]
    for i in range(n_images):
        plt.subplot(131)
        plt.imshow(x_true[i, ..., 0], cmap='bone')
        plt.axis('off')
        plt.title('True')

        plt.subplot(132)
        plt.imshow(x_pred[i, ..., 0], cmap='bone')
        plt.axis('off')
        plt.title('Output')

        plt.subplot(133)
        plt.imshow(np.abs(x_pred[i, ..., 0] - x_true[i, ..., 0]), cmap='bone')
        plt.axis('off')
        plt.title('Difference')

        plt.savefig('images/UNET_InversionValidationImage_Epoch' + str(epoch) + '_' + str(i) + '.pdf', format='pdf')
        plt.clf()
    pass


# ---------------------------
# Input example

y_noisy, y_true, x_t = DG.get_batch_upsampling(1)
y_denoised = sess.run(out_denoising, feed_dict={inp_denoising: y_noisy})
y_upsample = sess.run(out_up, feed_dict={inp_up: y_denoised})

plt.subplot(221)
plt.imshow(y_upsample[0, ..., 0], cmap='bone')
plt.colorbar()

plt.subplot(222)
plt.imshow(FBP(y_upsample[0, ..., 0]), cmap='bone')

plt.subplot(223)
plt.imshow(y_true[0, ..., 0], cmap='bone')
plt.colorbar()

plt.subplot(224)
plt.imshow(x_t[0, ..., 0], cmap='bone')

plt.savefig("images/UNET_inversion_input_example.pdf", format='pdf')
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

plt.savefig("images/UNET_inversion_denoised_input.pdf", format='pdf')


# ---------------------------
save_path = "models/unet_inversion/UNET_inversion_network"
n_save = 1
n_val = 1
n_plot = 1

print("Initialization successful. Starting training...", flush=True)

for epoch in range(epochs):
    ERR = []
    NMSE_train = []
    PSNR_train = []

    print("### Epoch %d/%d ###" % (epoch + 1, epochs))
    for j in range(n_batches):

        y_noisy, _, x_t = DG.get_batch(batch_size=batch_size, mode='train')
        y_denoised = sess.run(out_denoising, feed_dict={inp_denoising: y_noisy})
        y_upsample = sess.run(out_up, feed_dict={inp_up: y_denoised})

        fd = {inp_y: y_upsample,
              inp_x: np.zeros((batch_size, size, size, 1)),
              x_true: x_t, learning_rate: cosine_decay(epoch, epochs)}

        c, nm, ps, _ = sess.run([loss_inv, nmse, psnr, train_op_inv], feed_dict=fd)
        NMSE_train.append(nm)
        PSNR_train.append(ps)
        ERR.append(c)
        print("Progress %f, Loss %f" % ((j + 1) / n_batches, np.mean(ERR)), end='\r', flush=True)

    print("Training: Loss %f NMSE %f PSNR %f" % (np.mean(ERR), np.mean(NMSE_train), np.mean(PSNR_train)), end='\r')

    # Validate model performance
    if (epoch % n_val) == 0:
        ERR_VAL = []
        NMSE_VAL = []
        PSNR_VAL = []
        for j in range(n_batches_val):
            y_noisy, _, x_t = DG.get_batch(batch_size=batch_size, mode='val')
            y_denoised = sess.run(out_denoising, feed_dict={inp_denoising: y_noisy})
            y_upsample = sess.run(out_up, feed_dict={inp_up: y_denoised})

            fd = {inp_y: y_upsample,
                  inp_x: np.zeros((batch_size, size, size, 1)),
                  x_true: x_t}

            c, nm, ps = sess.run([loss_inv, nmse, psnr], feed_dict=fd)
            ERR_VAL.append(c)
            NMSE_VAL.append(nm)
            PSNR_VAL.append(ps)
        print(" ")
        print("Validation: Loss %f NMSE %f PSNR %f" % (np.mean(ERR_VAL), np.mean(NMSE_VAL), np.mean(PSNR_VAL)))
        print(" ", flush=True)

    if (epoch % n_plot) == 0:
        y_noisy, _, x_t = DG.get_batch(batch_size=batch_size, mode='val')
        y_denoised = sess.run(out_denoising, feed_dict={inp_denoising: y_noisy})
        y_upsample = sess.run(out_up, feed_dict={inp_up: y_denoised})

        fd = {inp_y: y_upsample,
              inp_x: np.zeros((batch_size, size, size, 1)),
              x_true: x_t}

        x_pred = sess.run(out_inversion, feed_dict=fd)
        plot_validation(x_pred, x_t, epoch=epoch)

    # Save model every
    if (epoch % n_save) == 0:
        saver.save(sess, save_path, global_step=epoch)


# ------------------------
# Plot one specific example for paper
rescale = 1000.
sigma = 0.2

np.random.seed(1)
path = "../data/mayoclinic/data/full3mm/test"
file = np.random.choice(os.listdir(path))
x = np.load(path + '/' + file)/rescale
y_n = RadonSparse(x)
y_n += np.random.normal(0, 1, y_n.shape)*sigma
y_n = y_n[None, ..., None]

y_n = sess.run(out_denoising, feed_dict={inp_denoising: y_n})
y_up = sess.run(out_up, feed_dict={inp_up: y_n})

x_rec = sess.run(out_inversion, feed_dict={inp_y: y_up, inp_x: np.zeros((1, size, size, 1))})
x_rec = x_rec[0, ..., 0]


plt.imshow(x_rec, cmap='bone')
plt.axis('off')
plt.savefig('images/UNET_reconstruction_racoon.pdf', format='pdf')
