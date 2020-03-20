"""This is Sequential Monte Carlo"""
import numpy as np
from modules.matrices import Generator
import itertools


class Particle(Generator):
    def __init__(self, weight, n_spins, beta=0.3, **kwargs):
        super().__init__(n_spins, beta, **kwargs)
        self.weight = weight

    def set_weight(self, weight):
        self.weight = weight

    @staticmethod
    def initial_generation(n_particles, n_spins, beta, fields=None, correlators=None):
        """Creates an initial list of 'particles' -- hamiltonians with random coefficients and equal weights"""
        particles_list = []
        weight = 1 / n_particles
        for i in range(n_particles):
            particle = Particle(weight=weight, n_spins=n_spins, beta=beta)

            # This line adds coefficients to hamiltonian
            if fields:
                particle.__dict__.update({field: np.random.rand(n_spins) * 2 - 1 for field in fields})
            if correlators:
                particle.__dict__.update({corr: np.random.rand(n_spins - 1) * 2 - 1 for corr in correlators})
            particle.density_mat()
            particles_list.append(particle)
        return np.array(particles_list)


# particles_list = Particle.initial_generation(n_particles=5, n_spins=3, beta=0.3, fields=['x', 'z'])
# print(len(particles_list))
# print(np.trace(particles_list[0].density_mat@particles_list[0].density_mat))

