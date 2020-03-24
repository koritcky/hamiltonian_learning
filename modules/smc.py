"""This is Sequential Monte Carlo"""
import numpy as np
from modules.hamiltonian import Hamiltonian
import modules.measurements as measurements


class Particle(Hamiltonian):
    def __init__(self, weight, n_spins, beta=0.3, **kwargs):
        super().__init__(n_spins, beta, **kwargs)
        self.weight = weight

    def set_weight(self, weight):
        self.weight = weight

    def weight_update(self, angles, singles_t, correlators_t):
        """Update weight of particle according to it's distance to target hamiltonian"""
        singles_g, correlators_g = self.measure(angles)
        distance = measurements.distance_by_measurements(singles_g, singles_t, correlators_g, correlators_t)
        weight = np.exp(- distance ** 2 / 2) / np.sqrt(2 * np.pi)
        self.set_weight(weight)

class Cloud:
    """Essentially, pull of particles"""

    def __init__(self, n_particles, n_spins, beta=0.3, fields=None, correlators=None):
        """Creates an initial list of 'particles' -- hamiltonians with random coefficients and equal weights"""

        self.particles_list = []
        self.n_particles = n_particles
        self.n_spins = n_spins
        if fields:
            self.fields = fields
        if correlators:
            self.correlators = correlators

        weight = 1 / n_particles
        self.total_weight = 1
        for i in range(self.n_particles):
            particle = Particle(weight=weight, n_spins=n_spins, beta=beta)

            # This line adds random coefficients to hamiltonian
            if hasattr(self, 'fields'):
                particle.__dict__.update({field: np.random.rand(self.n_spins) * 2 - 1 for field in self.fields})
            if hasattr(self, 'correlators'):
                particle.__dict__.update({corr: np.random.rand(self.n_spins - 1) * 2 - 1 for corr in self.correlators})

            particle.set_density_mat()
            self.particles_list.append(particle)

    def weight_normalization(self):
        """Normalizes weight of all particles such that the sum = 1"""
        for i in range(len(self.particles_list)):
            self.particles_list[i].weight /= self.total_weight

        self.total_weight = 1

        return self.particles_list

    def list_weight_update(self, angles, singles_t, correlators_t):
        """Update weight of particle according to it's distance to target hamiltonian"""
        self.total_weight = 0

        for particle in self.particles_list:
            particle.weight_update(angles, singles_t, correlators_t)
            self.total_weight += particle.weight

        # normalize weights
        self.weight_normalization()

        return self.particles_list

    def resampling_wheel(self):
        # sry, not my function

        max_weight = 0
        for particle in self.particles_list:
            if particle.weight > max_weight:
                max_weight = particle.weight

        new_particles_list = []
        index = np.random.randint(self.n_particles)
        beta = 0
        total_weight = 0
        for _ in range(self.n_particles):
            beta += np.random.uniform(0, 2 * max_weight)
            while self.particles_list[index].weight < beta:
                beta -= self.particles_list[index].weight
                index = (index + 1) % self.n_particles
            new_particles_list.append(self.particles_list[index])
            total_weight += self.particles_list[index].weight

        self.particles_list = new_particles_list
        self.weight_normalization()
        return self.particles_list

    # @staticmethod
    # def average_particle(particles_list, fields=None, correlators=None):
    #     particle = Hamiltonian()
    #     for particle in particles_list:
    #         print(particle)

# particles_list = Particle.initial_generation(n_particles=5, n_spins=3, beta=0.3, fields=['x', 'z'])
# print(len(particles_list))
# print(np.trace(particles_list[0].density_mat@particles_list[0].density_mat))

