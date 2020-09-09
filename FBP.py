import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from imports.operators import *
from imports.misc import *

import matplotlib.pyplot as plt

# ----------------------------
# Plot examples

path = "../../data/mayoclinic/data/full3mm/test"
sigma = 0.02
rescale = 1000.

NMSE = []
PSNR = []

for file in os.listdir(path):
    x = np.load(path + '/' + file)/rescale
    y_n = RadonSparse(x)
    y_n += np.random.normal(0, 1, y_n.shape)*sigma
    x_rec = FBPSparse(y_n)*c

    plt.subplot(121)
    plt.imshow(x, cmap='bone')
    plt.axis('off')
    plt.colorbar()

    plt.subplot(122)
    plt.imshow(x_rec, cmap='bone')
    plt.axis('off')
    plt.colorbar()

    plt.savefig("images/fbp/" + file + ".pdf")
    plt.clf()

    NMSE.append(NMSE_numpy(x, x_rec))
    PSNR.append(PSRN_numpy(x, x_rec))


with open("results/fbp.txt", "w") as f:
    f.writelines(str(NMSE))
    f.write("\n")
    f.writelines(str(PSNR))
