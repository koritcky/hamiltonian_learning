"""This is Sequential Monte Carlo"""
import numpy as np
from scipy.stats import norm
from modules.matrices import Hamiltonian
import modules.measurements as measurements
import itertools


class Particle(Hamiltonian):
    def __init__(self, weight, n_spins, beta=0.3, **kwargs):
        super().__init__(n_spins, beta, **kwargs)
        self.weight = weight

    def set_weight(self, weight):
        self.weight = weight

    def weight_update(self, angles, singles_t, correlators_t):
        singles_g, correlators_g = self.measure(angles)
        distance = measurements.distance_by_measurements(singles_g, singles_t, correlators_g, correlators_t)
        weight = np.exp(- distance ** 2 / 2) / np.sqrt(2 * np.pi)
        self.set_weight(weight)

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

    @staticmethod
    def list_weight_update(particles_list, angles, singles_t, correlators_t):
        total_weight = 0
        for particle in particles_list:
            particle.weight_update(angles, singles_t, correlators_t)
            total_weight += particle.weight
        particles_list = particles_list/total_weight

        return particles_list

    @staticmethod
    def resampling_wheel(particles_list, weights):
        n_particles = len(particles_list)
        max_weight = 0
        for particle in particles_list:
            if particle.weight > max_weight:
                max_weight = particle.weight

        new_particles_list = []
        index = np.random.randint(n_particles)
        beta = 0
        total_weight = 0
        for _ in range(n_particles):
            beta += np.random.uniform(0, 2 * max_weight)
            while particles_list[index].weight < beta:
                beta -= particles_list[index].weight
                index += 1
            new_particles_list.append(particles_list[index])
            total_weight += particles_list[index]
        new_particles_list = np.array(new_particles_list) / total_weight
        return new_particles_list

    

# particles_list = Particle.initial_generation(n_particles=5, n_spins=3, beta=0.3, fields=['x', 'z'])
# print(len(particles_list))
# print(np.trace(particles_list[0].density_mat@particles_list[0].density_mat))

