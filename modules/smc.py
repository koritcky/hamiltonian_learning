"""This is Sequential Monte Carlo"""
import numpy as np
from modules.matrices import Generator

class Particle(Generator):

    #TODO: make it nice
    # def __init__(self,
    #              n_spins,
    #              weight,
    #              hx=None,
    #              hy=None,
    #              hz=None,
    #              Jxx=None,
    #              Jyy=None,
    #              Jzz=None):

    def __init__(self, weight, **kwargs):
        self.__dict__.update(kwargs)
        self.weight = weight

    def set_weight(self, weight):
        self.weight = weight

    @staticmethod
    def initial_generation(n_particles, **kwargs):
        for i in range(n_particles):
            np.array([Particle])
        return np.array


particle = Particle(0.5, hx=np.ran, hy=[])
