import odl
import numpy as np

size = 512
n_theta = 60
upsampling_factor = 10
n_s = 768

# ---------------------------
# Set up tomography operator

space = odl.uniform_discr([-128, -128], [128, 128], [size, size], dtype='float32', weighting=1.0)
angle_partition = odl.uniform_partition(0, np.pi, n_theta)
angle_partition_up = odl.uniform_partition(0, np.pi, n_theta*upsampling_factor)
detector_partition = odl.uniform_partition(-360, 360, n_s)
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)
geometry_up = odl.tomo.Parallel2dGeometry(angle_partition_up, detector_partition)

RadonSparse = odl.tomo.RayTransform(space, geometry)
FBPSparse = odl.tomo.fbp_op(RadonSparse)

Radon = odl.tomo.RayTransform(space, geometry_up)
FBP = odl.tomo.fbp_op(Radon)

c = odl.operator.power_method_opnorm(RadonSparse)
RadonSparse /= c
Radon /= c
