import numpy as np
import math


class ReducedMatrixMeasurement:

    @staticmethod
    def split(density_mat, left_mat_dim):
        """
        General density matrix split funciton.
        Split big density matrix of the whole system by two reduced matrices such that matrix of "left" system
        has the dimension of left_mat_dim.
        """

        #Set dimensions of splitted matrices
        dim_1 = left_mat_dim
        if density_mat.shape[0] % dim_1 != 0:
            raise Warning("Wrong dimensions of splitted matrices")
        dim_2 = int(density_mat.shape[0] / dim_1)

        # Create matrix of block matrices of size dim_2 x dim_2
        B = []
        for i in range(dim_1):
            B_row = []
            for j in range(dim_1):
                submat = density_mat[j * dim_2:(j + 1) * dim_2, i * dim_2:(i + 1) * dim_2]
                B_row.append(submat)
            B.append(B_row)
        B = np.array(B)

        # Create matrices after splitting according to reduced matrix rule
        submat_1 = np.zeros((dim_1, dim_1), dtype='complex128')
        submat_2 = np.zeros((dim_2, dim_2), dtype='complex128')
        for i in range(dim_1):
            submat_2 += B[i, i]
            for j in range(dim_1):
                submat_1[i, j] = np.trace(B[i, j])

        return submat_1, submat_2

    @staticmethod
    def reduced_matrix(density_mat, left_spin, right_spin):
        """
        By given a density matrix of general spin system and 2 spins -- adjacent on the same -- return density matrix
        of subsystem of this 2 spins.
        """
        resulted_system = right_spin - left_spin + 1
        resulted_dim = 2 ** resulted_system
        if resulted_system != 2 and resulted_system != 1:
            raise Warning("You can calculate only reduced matrix of adjacent or single spin using this function")

        left_part_dim = 2 ** (right_spin + 1)
        left_part, _ = ReducedMatrixMeasurement.split(density_mat, left_part_dim)

        if left_part_dim % resulted_dim != 0:
            raise Warning("Something wrong with dimensions")

        left_part_dim = int(left_part_dim / resulted_dim)
        _, reduced_mat = ReducedMatrixMeasurement.split(left_part, left_part_dim)

        return reduced_mat

    @staticmethod
    def reduced_matrix_measurements(density_mat):
        """
        By given density mat return all singles and correlators
        """
        dim = density_mat.shape[0]
        n_spins = int(math.log2(dim))

        singles = np.zeros((n_spins, 2))  # single spins
        # correlators = np.zeros((n_spins - 1, 4))  # correlators distribution
        for i in range(n_spins):
            single_rho = ReducedMatrixMeasurement.reduced_matrix(density_mat, i, i)
            singles[i] = np.diag(single_rho).real
            # if i + 1 < n_spins:
            #     correlator_rho = ReducedMatrixMeasurement.reduced_matrix(density_mat, i, i+1)
            #     correlators[i] = np.diag(correlator_rho).real

        return singles


class SamplingMeasurement:
    @staticmethod
    def sampling_measurements(rho, n_measurements):
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
        correlators = correlators[:-1]
        return singles, correlators


### This funciton is universal ###
def distance_by_measurements(singles_1, singles_2):
    """Find the distance between 2 matrices according to singles and correlators measurements"""
    p0_1 = singles_1[:, 0]
    p0_2 = singles_2[:, 0]
    return ((p0_1 - p0_2) ** 2).mean()


# n = 3
# m = np.random.rand(2 ** 3, 2 ** 3)
# rho = m + np.conjugate(m.T)
# rho = rho/np.trace(rho)
#
# singles_1, correlators_1 = reduced_matrix_measurements(rho)
# singles_2, correlators_2 = sampling_measurements(rho, 10 ** 6)
# d = distance_by_measurements(singles_1, singles_1, correlators_1, correlators_2)
# print(d)

