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

    # def weight_update(self, mean, sd):
    #     """Update weight of particle according to it's distance to target hamiltonian
    #     mean is distance
    #     sd is its standard deviation
    #     """
    #     # TODO this parameter must be adjusted and justified
    #     weight = np.exp(- ((mean / sd) ** 2) * (1 / 2)) / np.sqrt(2 * np.pi * sd)
    #     # weight = 1/distance
    #     self.set_weight(weight)

    @staticmethod
    def vectorize(dic: dict):
        vec = []
        for field in ['x', 'y', 'z']:
            if field in dic.keys():
                vec += list(dic[field])

        for coupling in ['xx', 'yy', 'zz']:
            if coupling in dic.keys():
                vec += list(dic[coupling])
        return vec

    @staticmethod
    def dictorize(v: list, n_spins, coeffs_types):
        dic = {}
        i = 0

        for field in ['x', 'y', 'z']:
            if field in coeffs_types:
                dic[field] = v[i: i + n_spins]
                i += n_spins
        for coupling in ['xx', 'yy', 'zz']:
            if coupling in coeffs_types:
                dic[coupling] = v[i: i + n_spins - 1]
                i += n_spins - 1

        return dic


class Cloud:
    """Essentially, pull of particles"""

    def __init__(self, n_particles, n_spins, beta, coeffs_types):
        """Creates an initial list of 'particles' -- hamiltonians with random coefficients and equal weights"""

        self.particles_list = []
        self.n_particles = n_particles
        self.n_spins = n_spins
        self.beta = beta

        self.coeffs_types = coeffs_types
        self.coeffs_len = n_spins * len(coeffs_types)

        for type in coeffs_types:
            if len(type) == 2:
                self.coeffs_len -= 1

        self.weights_list = np.array([1 / n_particles for _ in range(n_particles)])
        self.total_weight = sum(self.weights_list)

        weight = 1 / n_particles
        for i in range(self.n_particles):

            rand_coeff = np.random.rand(self.coeffs_len) * 2 - 1
            rand_coeff = Particle.dictorize(rand_coeff, self.n_spins, self.coeffs_types)

            particle = Particle(weight=weight, n_spins=n_spins, beta=beta, **rand_coeff)
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
            self.particles_list[i].weight /= self.total_weight
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

        for i in range(self.n_particles):
            self.particles_list[i].weight = np.exp(- ((mean[i] / sd) ** 2) * (1 / 2)) / np.sqrt(2 * np.pi * sd)

        # normalize weights
        self.weight_normalization()

    def resampling(self, alpha=0.98):
        """Process of removing far away particles and adding gauss noise"""
        X = self.get_X()
        _, mean = self.mean_vec(X)
        cov = self.cov(X) * (1 - alpha ** 2)

        new_particles_ind = np.random.choice(self.n_particles, self.n_particles, replace=True, p=self.weights_list)
        new_particles_list = []
        weight = 1 / self.n_particles
        for i in new_particles_ind:
            mean_i = alpha * X[i] + (1 - alpha) * mean
            x = np.random.multivariate_normal(mean_i,  cov)
            coefs = Particle.dictorize(x, self.n_spins, coeffs_types=self.coeffs_types)
            particle = Particle(weight, self.n_spins, self.beta, **coefs)
            particle.set_density_mat()
            new_particles_list.append(particle)

        self.particles_list = np.array(new_particles_list)
        self.weight_normalization()

    def get_X(self):
        X = []
        for particle in self.particles_list:
            x = Particle.vectorize({coeff: particle.__dict__[coeff] for coeff in self.coeffs_types})
            X.append(x)
        return np.array(X)


    def mean_vec(self, X):
        """weighted sum of all particles"""

        mean_vec = self.weights_list @ X
        mean_dic = Particle.dictorize(mean_vec, self.n_spins, self.coeffs_types)
        mean_particle = Hamiltonian(self.n_spins, self.beta, **mean_dic)
        mean_particle.set_density_mat()

        return mean_particle, mean_vec

    def cov(self, X):
        _, mean_vec = self.mean_vec(X)
        return np.einsum('k, ki, kj-> ij', self.weights_list, X, X) - np.outer(mean_vec, mean_vec)

