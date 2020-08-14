import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from imports.operators import *
from imports.misc import *

# ----------------------------
# Plot examples

path = "../data/mayoclinic/data/full3mm/test"
sigma = 0.2
rescale = 1000.

NMSE = []
PSNR = []

for file in os.listdir(path):
    x = np.load(path + '/' + file)/rescale
    y_n = RadonSparse(x)
    y_n += np.random.normal(0, 1, y_n.shape)*sigma
    x_rec = FBPSparse(y_n)

    NMSE.append(NMSE_numpy(x_rec, x))
    PSNR.append(PSRN_numpy(x_rec, x))

print(NMSE)
print(PSNR)
