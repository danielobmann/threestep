import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib.pyplot as plt

from imports.operators import *
from imports.misc import *
from imports.inversionnetwork import *

sess = tf.Session()

# ---------------------------
# Specify parameters
epochs = 11
batch_size = 1
n_training_samples = 1709
n_validation_samples = 458
n_batches = n_training_samples//batch_size
n_batches_val = n_validation_samples//batch_size

initial_lr = 1e-3


# ---------------------------
# Set up network
inp_y, inp_x, out_inversion = InversionNetwork(Radon).network(n_primal=5, n_dual=5, n_iter=5)

x_true = tf.placeholder(shape=(None, size, size, 1), dtype=tf.float32)
loss = tf.reduce_mean(tf.squared_difference(out_inversion, x_true))

learning_rate = tf.placeholder(dtype=tf.float32)
opt = tf.train.AdamOptimizer(learning_rate=1e-3)

train_op = opt.minimize(loss)

sess.run(tf.global_variables_initializer())

nmse = NMSE(out_inversion, x_true)
psnr = PSNR(out_inversion, x_true)

DG = DataGenerator(operator=RadonSparse)


def plot_validation(x_pred, x_true, epoch=10):
    n_images = x_pred.shape[0]
    for i in range(n_images):
        plt.subplot(131)
        plt.imshow(x_true[i, ..., 0], cmap='bone')
        plt.axis('off')
        plt.title('True')
        plt.colorbar()

        plt.subplot(132)
        plt.imshow(x_pred[i, ..., 0], cmap='bone')
        plt.axis('off')
        plt.title('Output')
        plt.colorbar()

        plt.subplot(133)
        plt.imshow(np.abs(x_pred[i, ..., 0] - x_true[i, ..., 0]), cmap='bone')
        plt.axis('off')
        plt.title('Difference')
        plt.colorbar()

        plt.savefig('images/LPDValidationImage_Epoch' + str(epoch) + '_' + str(i) + '.pdf', format='pdf')
        plt.clf()
    pass


# ---------------------------
save_path = "models/lpd/"
saver = tf.train.Saver()
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

        y_noisy, _, x_t = DG.get_batch(batch_size=batch_size, mode='train')

        fd = {inp_y: y_noisy,
              inp_x: np.zeros((batch_size, size, size, 1)),
              x_true: x_t, learning_rate: cosine_decay(epoch, epochs)}

        c, nm, ps, _ = sess.run([loss, nmse, psnr, train_op], feed_dict=fd)
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

            fd = {inp_y: y_noisy,
                  inp_x: np.zeros((batch_size, size, size, 1)),
                  x_true: x_t}

            c, nm, ps = sess.run([loss, nmse, psnr], feed_dict=fd)
            ERR_VAL.append(c)
            NMSE_VAL.append(nm)
            PSNR_VAL.append(ps)
        print(" ")
        print("Validation: Loss %f NMSE %f PSNR %f" % (np.mean(ERR_VAL), np.mean(NMSE_VAL), np.mean(PSNR_VAL)))
        print(" ", flush=True)

    if (epoch % n_plot) == 0:
        y_noisy, _, x_t = DG.get_batch(batch_size=batch_size, mode='val')

        fd = {inp_y: y_noisy,
              inp_x: np.zeros((batch_size, size, size, 1)),
              x_true: x_t}

        x_pred = sess.run(out_inversion, feed_dict=fd)
        plot_validation(x_pred, x_t, epoch=epoch)

    # Save model every
    if (epoch % n_save) == 0:
        saver.save(sess, save_path, global_step=epoch)


# ------------------------
# Plot one specific example for paper

path = "../data/mayoclinic/data/full3mm/test"
sigma = 0.2
rescale = 1000.


np.random.seed(1)
file = np.random.choice(os.listdir(path))
x = np.load(path + '/' + file)/rescale
y_n = RadonSparse(x)
y_n += np.random.normal(0, 1, y_n.shape)*sigma
y_n = y_n[None, ..., None]

x_rec = sess.run(out_inversion, feed_dict={inp_y: y_n, inp_x: np.zeros((1, size, size, 1))})
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
