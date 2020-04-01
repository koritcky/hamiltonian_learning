from random import seed

from modules.smc import Particle, Cloud
from modules.gradient import ProbabilityDerivative, Gradient
from modules.hamiltonian import *
from modules.measurements import *

# np.random.seed(43)
# seed(43)


def main_cycle(n_cycles, n_particles, lr, gradient_iterations):
    # Some initial parameters
    n_spins = 3
    beta = 0.3

    # Create initial random basis angles
    theta = np.random.rand(n_spins) * np.pi
    phi = np.random.rand(n_spins) * 2 * np.pi
    # theta = np.array([0.])
    # phi = np.array([0.])

    angles = np.array([theta, phi]).T

    # Create target hamiltonian
    x_t = np.random.rand(n_spins) * 2 - 1
    z_t = np.random.rand(n_spins) * 2 - 1
    # z_t = [-1, 0.3, 0.8]
    # # x_t = [0]

    hamiltonian_t = Hamiltonian(n_spins, beta, z=z_t)
    hamiltonian_t.set_density_mat()
    print('Target density matrix')
    print(hamiltonian_t.density_mat)

    # Create an initial pull of random hamiltonians
    # (we call it "particle" according to Sequential Monte Carlo terminology)
    g_cloud = Cloud(n_particles, n_spins, beta, fields=["z"])
    for i in range(n_cycles):
        # Measure target hamiltonian
        singles_t = hamiltonian_t.measure(angles)

        # Update weight according to particle's distance to target hamiltonian
        g_cloud.list_weight_update(angles, singles_t)

        # Make a recycling wheel to get rid of far particles
        g_cloud.resampling_wheel()

        # Weighted sum of all particles -- our current best result
        hamiltonian_g = g_cloud.weighted_sum()

        mse = hamiltonian_difference(hamiltonian_t, hamiltonian_g)
        print(f"iteration {i}")
        print(f"mse {mse}")
        print(f"hamiltonian_g.z {hamiltonian_g.z}")
        singles_g = hamiltonian_g.measure(angles)
        print(f"distance {distance_by_measurements(singles_g, singles_t)}")
        print(f"theta {angles[0, 0] / np.pi}")
        print(f"phi {angles[0, 1] / np.pi}")
        print('')

        # Make a gradient descent to determine new angles
        grad = Gradient(hamiltonian_t, hamiltonian_g, angles)
        angles = grad.gradient_descent(lr=lr, num_iterations=gradient_iterations)
    print(singles_t)
    print(singles_g)

if __name__ == '__main__':
    main_cycle(n_cycles=300, n_particles=500, lr=0.1, gradient_iterations=300)


