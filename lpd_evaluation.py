import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib.pyplot as plt

from imports.operators import *
from imports.misc import *
from imports.inversionnetwork import *

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

restore_path = "models/lpd/"
inp_y, inp_x, out_inversion = InversionNetwork(RadonSparse).network(n_primal=5, n_dual=5, n_iter=5)

# ----------------------------
# Restore models

saver = tf.train.import_meta_graph(restore_path + 'lpd_network-20.meta')
saver.restore(sess, tf.train.latest_checkpoint(restore_path))

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

plt.imshow(FBPSparse(y_n[0, ..., 0]), cmap='bone')
plt.axis('off')
plt.savefig('images/reconstruction_fbp.pdf', format='pdf')


NMSE = []
PSNR = []


for file in os.listdir(path):
    x = np.load(path + '/' + file)/rescale
    y_n = RadonSparse(x)
    y_n += np.random.normal(0, 1, y_n.shape)*sigma
    y_n = y_n[None, ..., None]

    x_rec = sess.run(out_inversion, feed_dict={inp_y: y_n, inp_x: np.zeros((1, size, size, 1))})
    x_rec = x_rec[0, ..., 0]

    NMSE.append(NMSE_numpy(x_rec, x))
    PSNR.append(PSRN_numpy(x_rec, x))

    plt.subplot(221)
    plt.imshow(x, cmap='bone')
    plt.axis('off')
    plt.title('True')

    plt.subplot(222)
    plt.imshow(FBPSparse(y_n[0, ..., 0]), cmap='bone')
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
