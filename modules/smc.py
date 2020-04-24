"""This is Sequential Monte Carlo"""
import numpy as np
from modules.hamiltonian import Hamiltonian
import modules.measurements as measurements
import copy


class Particle(Hamiltonian):
    def __init__(self, weight, n_spins, beta=0.3, **kwargs):
        super().__init__(n_spins, beta, **kwargs)
        self.weight = weight

    def set_weight(self, weight):
        self.weight = weight

    def scale_weight(self, scalar):
        # to be sure
        self.set_weight(self.weight * scalar)

    def weight_update(self, mean, sd):
        """Update weight of particle according to it's distance to target hamiltonian
        mean is distance
        sd is its standard deviation
        """
         # ?? this parameter must be adjusted and justified
        weight = np.exp(- ((mean / sd) ** 2) * (1 / 2)) / np.sqrt(2 * np.pi * sd)
        # weight = 1/distance
        self.set_weight(weight)


class Cloud:
    """Essentially, pull of particles"""

    def __init__(self, n_particles, n_spins, beta=0.3, fields=None, couplings=None):
        """Creates an initial list of 'particles' -- hamiltonians with random coefficients and equal weights"""

        self.particles_list = []
        self.n_particles = n_particles
        self.n_spins = n_spins
        self.beta = beta

        if fields:
            self.fields = fields
        if couplings:
            self.couplings = couplings

        self.weights_list = np.array([1 / n_particles for i in range(n_particles)])
        self.total_weight = sum(self.weights_list)

        weight = 1 / n_particles
        for i in range(self.n_particles):
            particle = Particle(weight=weight, n_spins=n_spins, beta=beta)

            # This line adds random coefficients to hamiltonian
            if hasattr(self, 'fields'):
                # particle.__dict__.update({field: np.random.randint(2, size=self.n_spins) * 2 - 1 for field in self.fields})  # Either +1 or -1
                particle.__dict__.update({field: np.random.rand(n_spins) * 2 - 1 for field in
                                          self.fields})
            if hasattr(self, 'couplings'):
                particle.__dict__.update({coupling: np.random.rand(self.n_spins - 1) * 2 - 1 for coupling in self.couplings})

            particle.set_density_mat()
            self.particles_list.append(particle)

    def set_weights_list(self):
        weights = []
        for particle in self.particles_list:
            weights.append(particle.weight)
        self.weights_list = np.array(weights)
        return self.weights_list

    def weight_normalization(self):
        """Normalizes weight of all particles such that the sum of all weights = 1"""
        self.total_weight = sum(self.set_weights_list())
        for i in range(self.n_particles):
            self.particles_list[i].scale_weight(1 / self.total_weight)
        self.set_weights_list()

    def list_weight_update(self, angles, singles_t, correlators_t=None):
        """Update weight of particle according to it's distance to target hamiltonian"""
        self.total_weight = 0
        mean = []

        # find means and sd
        for particle in self.particles_list:
            singles_g, correlators_g = particle.measure(angles)
            mean.append(measurements.distance_by_measurements(singles_g, singles_t, correlators_g, correlators_t))
        sd = np.sqrt(1 / (self.n_particles - 1) * np.sum(np.array(mean) ** 2))

        for i in range(len(self.particles_list)):
            self.particles_list[i].weight = np.exp(- ((mean[i] / sd) ** 2) * (1 / 2)) / np.sqrt(2 * np.pi * sd)

        # normalize weights
        self.weight_normalization()

    def resampling_wheel(self):
        # sry, not my function

        max_weight = 0
        for particle in self.particles_list:
            if particle.weight > max_weight:
                max_weight = particle.weight

        new_particles_list = []
        index = np.random.randint(self.n_particles)
        beta = 0

        for _ in range(self.n_particles):
            beta += np.random.uniform(0, 2 * max_weight)
            while self.particles_list[index].weight < beta:
                beta -= self.particles_list[index].weight
                index = (index + 1) % self.n_particles

            new_particle = copy.deepcopy(self.particles_list[index])  # It's required to create a copy. Probably not the best variant to use "deepcopy"
            new_particles_list.append(new_particle)

        self.particles_list = np.array(new_particles_list)
        self.weight_normalization()

    def weighted_sum(self):
        """weighted sum of all particles"""
        resulting_particle = Hamiltonian(self.n_spins, self.beta)

        if hasattr(self, 'fields'):
            for field in self.fields:
                f = np.zeros(self.n_spins)
                for particle in self.particles_list:
                    f += particle.__dict__[field] * particle.weight

                resulting_particle.__dict__.update({field: f})

        if hasattr(self, 'couplings'):
            for coupling in self.couplings:
                c = np.zeros(self.n_spins - 1)
                for particle in self.particles_list:
                    c += particle.__dict__[coupling] * particle.weight

                resulting_particle.__dict__.update({coupling: c})

        resulting_particle.set_density_mat()
        self.resulting_particle = resulting_particle

        return self.resulting_particle





# particles_list = Particle.initial_generation(n_particles=5, n_spins=3, beta=0.3, fields=['x', 'z'])
# print(len(particles_list))
# print(np.trace(particles_list[0].density_mat@particles_list[0].density_mat))

