import os
import numpy as np
import odl
import odl.contrib.tensorflow
import matplotlib.pyplot as plt

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
FBP = odl.tomo.fbp_op(operator, frequency_scaling=0.2)

operator /= odl.operator.power_method_opnorm(operator)

# ----------------------------
# Plot examples


path = "../data/mayoclinic/data/full3mm/test"
sigma = 0.2
rescale = 1000.


def nmse(x_true, x_pred):
    d = np.mean((x_true - x_pred)**2)
    n = np.mean(x_true**2)
    return d/n


def psnr(x_true, x_pred):
    ran = np.amax(x_true) - np.amin(x_true)
    return 20*np.log(ran) - 10*np.log(np.mean((x_true - x_pred)**2))


NMSE = []
PSNR = []


for file in os.listdir(path):
    x = np.load(path + '/' + file)/rescale
    y_n = operator(x)
    y_n += np.random.normal(0, 1, y_n.shape)*sigma
    x_rec = FBP(y_n)

    NMSE.append(nmse(x, x_rec))
    PSNR.append(psnr(x, x_rec))

print(NMSE)
print(PSNR)


