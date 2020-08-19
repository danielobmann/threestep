import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib.pyplot as plt

from imports.operators import *
from imports.misc import *
from imports.denoisingnetwork import *
from imports.upsamplingnetwork import *
from imports.inversionnetwork import *

sess = tf.Session()

# ---------------------------
# Specify parameters
epochs = 5
batch_size = 2
batch_size_val = 16
n_training_samples = 1709
n_validation_samples = 458
n_batches = n_training_samples//batch_size
n_batches_val = n_validation_samples//batch_size_val

initial_lr = 1e-3

# ---------------------------
# Define networks
inp_denoising, out_denoising = DenoisingNetwork(RadonSparse, FBPSparse).network()
inp_up, out_up = DataConsistentNetwork(Radon, FBP).network_residual((n_theta, n_s, 1), steps=3, filters=32)
inp_y, inp_x, out_inversion = InversionNetwork(Radon).network(n_primal=5, n_dual=5, n_iter=5)

y_true = tf.placeholder(shape=(None, n_theta, n_s, 1), dtype=tf.float32)
loss = tf.reduce_mean(tf.squared_difference(out_denoising, y_true))

learning_rate = tf.placeholder(dtype=tf.float32, name='lr_denoising')
opt = tf.train.AdamOptimizer(learning_rate=learning_rate, name='adam_denoising')

train_op = opt.minimize(loss)

# ---------------------------
# Set up various functions


def plot_validation(y_in, y_pred, y_true, epoch=10):
    n_images = y_in.shape[0]
    for i in range(n_images):
        fig, axs = plt.subplots(nrows=2, ncols=2)
        im = axs[0, 0].imshow(y_true[i, ..., 0], cmap='bone')
        axs[0, 0].set_aspect(n_s / n_theta)
        axs[0, 0].axis('off')
        axs[0, 0].set_title('True')
        fig.colorbar(im, ax=axs[0, 0])

        im = axs[0, 1].imshow(y_in[i, ..., 0], cmap='bone')
        axs[0, 1].set_aspect(n_s / n_theta)
        axs[0, 1].axis('off')
        axs[0, 1].set_title('Input')
        fig.colorbar(im, ax=axs[0, 1])

        im = axs[1, 0].imshow(y_pred[i, ..., 0], cmap='bone')
        axs[1, 0].set_aspect(n_s / n_theta)
        axs[1, 0].axis('off')
        axs[1, 0].set_title('Prediction')
        fig.colorbar(im, ax=axs[1, 0])

        im = axs[1, 1].imshow(np.abs(y_true[i, ..., 0] - y_pred[i, ..., 0]), cmap='bone')
        axs[1, 1].set_aspect(n_s / n_theta)
        axs[1, 1].axis('off')
        axs[1, 1].set_title('Difference')
        fig.colorbar(im, ax=axs[1, 1])

        fig.savefig('images/DenoisingResidualValidationImage_Epoch' + str(epoch) + '_' + str(i) + '.pdf', format='pdf')
        fig.clf()
    pass


nmse = NMSE(out_denoising, y_true)
psnr = PSNR(out_denoising, y_true)

sess.run(tf.global_variables_initializer())

DG = DataGenerator(RadonSparse)

# ---------------------------
save_path = "models/denoising_residual/denoising_network"
n_save = 1
n_val = 1
n_plot = 1

print("Initialization successful. Starting training...", flush=True)
saver = tf.train.Saver()


for epoch in range(epochs):
    ERR = []
    NMSE_train = []
    PSNR_train = []

    print("### Epoch %d/%d ###" % (epoch + 1, epochs))
    for j in range(n_batches):
        y_input, y_output, _ = DG.get_batch(batch_size=batch_size, mode='train')

        fd = {inp_denoising: y_input,
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
            y_input, y_output, _ = DG.get_batch(batch_size=batch_size, mode='val')

            fd = {inp_denoising: y_input,
                  y_true: y_output}

            c, nm, ps = sess.run([loss, nmse, psnr], feed_dict=fd)
            ERR_VAL.append(c)
            PSN_VAL.append(ps)
            NMSE_VAL.append(nm)
        print(" ")
        print("   Validation: Loss %f NMSE %f PSNR %f" % (np.mean(ERR_VAL), np.mean(NMSE_VAL), np.mean(PSN_VAL)))
        print(" ", flush=True)

    if (epoch % n_plot) == 0:
        y_input, y_output, _ = DG.get_batch(batch_size=batch_size, mode='val')

        fd = {inp_denoising: y_input,
              y_true: y_output}

        y_pred = sess.run(out_denoising, feed_dict=fd)
        plot_validation(y_input, y_pred, y_output, epoch=epoch)

    # Save model every
    if (epoch % n_save) == 0:
        saver.save(sess, save_path, global_step=epoch)
