import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib.pyplot as plt

from imports.operators import *
from imports.misc import *
from imports.denoisingnetwork import *
from imports.upsamplingnetwork import *

sess = tf.Session()

# ---------------------------
# Specify parameters
epochs = 21
batch_size = 2
n_training_samples = 1709
n_validation_samples = 458
n_batches = n_training_samples//batch_size
n_batches_val = n_validation_samples//batch_size

initial_lr = 1e-3

# ---------------------------
# Set up denoising network
inp_denoising, out_denoising = DenoisingNetwork(RadonSparse, FBPSparse).network()

# ---------------------------
# Define upsampling network

inp_up, out_up = DataConsistentNetwork(Radon, Radon.adjoint).network((n_theta, n_s, 1), steps=5, filters=16)

y_true = tf.placeholder(shape=(None, n_theta*upsampling_factor, n_s, 1), dtype=tf.float32)
loss = tf.reduce_mean(tf.squared_difference(out_up, y_true))

learning_rate = tf.placeholder(dtype=tf.float32, name='lr_upsample')
opt = tf.train.AdamOptimizer(learning_rate=learning_rate, name='adam_upsample')

train_op = opt.minimize(loss)

# ----------------------------
# Restore denoising model
sess.run(tf.global_variables_initializer())

# Restore graph from trained model
restore_path = "models/denoising/"
if 1:
    new_saver = tf.train.import_meta_graph(restore_path + 'denoising_network-20.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint(restore_path))

graph = tf.get_default_graph()

# inp_denois = graph.get_tensor_by_name("input_denoising:0")
# out_denois = graph.get_tensor_by_name("output_denoising:0")

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

        fig.savefig('images/UpsampleValidationImage_Epoch' + str(epoch) + '_' + str(i) + '.pdf', format='pdf')
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

        fig.savefig('images/UpsampleReconImage_Epoch' + str(epoch) + '_' + str(i) + '.pdf', format='pdf')
        fig.clf()
    pass


nmse = NMSE(y_true, out_up)
psnr = PSNR(y_true, out_up)

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
axs[1, 0].set_title('True')
fig.colorbar(im, ax=axs[1, 0])

plt.savefig("images/UpsamplingInputTest.pdf", format='pdf')
plt.clf()

# ---------------------------
save_path = "models/upsampling/upsampling_network"
n_save = 10
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
            y_input, y_output, _ = DG.get_batch_upsampling(batch_size=batch_size, mode='val')
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
        new_saver.save(sess, save_path, global_step=epoch)

