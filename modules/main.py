from random import seed

from modules.smc import Particle, Cloud
from modules.measurements import ReducedMatrixMeasurement
from modules.gradient import *

np.random.seed(42)
seed(42)
n_spins = 3
beta = 0.3

# Create random basis angles
theta = np.random.rand(n_spins) * np.pi
phi = np.random.rand(n_spins) * np.pi
angles = np.array([theta, phi]).T


# Create target hamiltonian
x_t = np.random.rand(n_spins) * 2 - 1
z_t = np.random.rand(n_spins) * 2 - 1

target = Hamiltonian(n_spins, beta, x=x_t, z=z_t)
target.set_density_mat()

# And measure it
singles_t, correlators_t = target.measure(angles)


# Create a pull of random hamiltonians (call it particle according to Sequential Monte Carlo terminology)
n_particles = 30
g_cloud = Cloud(n_particles, n_spins, beta, fields=["x", "y"])
# print(g_cloud.particles_list[3].x)

# Update weight according to particle's distance to target hamiltonian
g_cloud.list_weight_update(angles, singles_t, correlators_t)

# Make a recycling wheel to get rid of far particles
g_cloud.resampling_wheel()

# weighted sum of all particles -- our current best result
g = g_cloud.weighted_sum()












