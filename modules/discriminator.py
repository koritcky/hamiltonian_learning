import numpy as np
from collections import Counter
from scipy.stats import entropy
from sklearn.metrics import mean_squared_error as mse


class Discriminator:
    def __init__(self, angles):
        self.angles = angles
        self.umat = self.angles_to_umat(self.angles)

    def train(self, new_angles: dict):
        self.angles = new_angles
        self.umat = self.angles_to_umat(self.angles)

    @staticmethod
    def angles_to_umat(angles):
        """Calculates rotation matrices"""

        theta = angles[0, 0]
        phi = angles[0, 1]

        U = u_mat(theta, phi)

        for i in range(angles.shape[0] - 1):
            theta = angles[i + 1, 0]
            phi = angles[i + 1, 1]

            U = np.kron(U, u_mat(theta, phi)).astype('complex64')

        return U

    @staticmethod
    def rotate(rho: np.array, u_mat):
        """ Return density matrix in new basis"""
        try:
            return u_mat @ rho @ u_mat.T.conj()
        except ValueError:
            print('Havent rotated rotate matrix')
            return rho

    @staticmethod
    def loss_function(angles: list, *args):
        N_spins = len(angles)//2

        angles = np.reshape(angles, (N_spins, 2))
        rho_g_z = args[0]
        rho_t_z = args[1]

        # We use this function that return u_mats for number of different choices of basises. But we pass just one basis
        umat = Discriminator.angles_to_umat(angles)

        # Get rho in new basis
        rho_g_new = Discriminator.rotate(rho_g_z, umat)
        rho_t_new = Discriminator.rotate(rho_t_z, umat)

        # Get distribution
        distr_g = np.diag(rho_g_new).real
        distr_t = np.diag(rho_t_new).real

        # Distance between distributions
        ent = entropy(distr_g, distr_t)
        return -ent


def u_mat(theta, phi):
    """ Generates U (rotation matrix) by given angles of Bloch sphere"""

    u11 = np.cos(theta / 2) * np.exp(phi / 2. * 1j)
    u12 = np.sin(theta / 2) * np.exp(- phi / 2. * 1j)
    u21 = -np.sin(theta / 2) * np.exp(phi / 2. * 1j)
    u22 = np.cos(theta / 2) * np.exp(- phi / 2. * 1j)

    u = np.array([[u11, u12], [u21, u22]])

    return u


def rho_to_distr(rho, N=50000):
    """Return distribution list by given density matrix rho of N 'experiments' """
    probs = np.diag(rho)
    cumsum = np.cumsum(probs)

    result = np.zeros(N)
    for n in range(N):
        r = np.random.uniform(0, 1)
        for i in range(len(cumsum)):
            if r < cumsum[i]:
                break
        result[n] = i
    result = Counter(result)

    dim = len(probs)
    distr = [result[float(i)] / N for i in range(dim)]

    return distr

if __name__ == '__main__':
    rho = np.array([[0.3, 0.4, 0.4],
                    [0.4, 0.399, 0.3],
                    [0.9, 0.4, 0.3]])
    probs = np.diag(rho)
    d1 = rho_to_distr(rho, 1000)
    d2 = rho_to_distr(rho, 10000)
    d3 = rho_to_distr(rho, 10000)
    print(mse(probs, d1))
    print(mse(probs, d2))
    print(mse(probs, d3))
    print(entropy(probs, probs))
    print(type(probs))
    print(type(d1))

