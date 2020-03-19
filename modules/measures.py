import numpy as np
import math

def measure(rho, n_measurements):
    """
    By given density matrix returns list of distributions of single
    spin measurements and paired correlators
    :param rho: density matrix
    :param n_measurements: number of measurements
    :return: singles, correlators
    """

    # Get probabilities from diagonal elements
    prob = np.diag(rho).real

    # Determine number of spins in system
    dim = len(prob)
    n_spins = int(math.log2(dim))

    # Get the list of numbers of each outcome
    try:
        samples = np.random.choice(dim, n_measurements, p=prob)  # in decimal
    except ValueError:
        prob = prob/sum(prob)
        samples = np.random.choice(dim, n_measurements, p=prob)

    # Convert it into bitstrings corresponding to each outocme
    bitstrings = np.zeros((n_measurements, n_spins))
    for i, sample in enumerate(samples):
        bitstrings[i] = np.array([int(i) for i in bin(sample)[2:].zfill(n_spins)])

    # Get list of numbers of each outcome
    singles = np.zeros((n_spins, 2))  # single spins
    correlators = np.zeros((n_spins, 4))  # correlators distribution
    for throw in range(n_measurements):
        for spin in range(n_spins):
            value = int(bitstrings[throw, spin])
            neighbor_value = int(bitstrings[throw, (spin + 1) % n_spins])
            singles[spin, value] += 1
            correlators[spin, int(str(value) + str(neighbor_value), 2)] += 1
    singles /= n_measurements
    correlators /= n_measurements
    return singles, correlators


def matrix_distance(rho_1, rho_2, n_measurements):
    """Find the distance between 2 matrices according to singles and correlators measurements"""
    singles_1, correlators_1 = measure(rho_1, n_measurements)
    singles_2, correlators_2 = measure(rho_2, n_measurements)

    return sum(sum(abs(singles_1 - singles_2))) + sum(sum(abs(correlators_2 - correlators_1)))