import numpy as np


class Discriminator:
    def __init__(self, angles):
        self.angles = angles
        self.umat = self.angles_to_umat(self.angles)

    def train(self, new_angles: dict):
        self.angles = new_angles
        self.umat = self.angles_to_umat(self.angles)

    @staticmethod
    def angles_to_umat(angles):
        # TODO: realize
        u_mat = np.array([])
        return u_mat

    @staticmethod
    def rotate(rho: np.array, umat):
        """ Return density matrix in """
        try:
            return umat @ rho @ umat.T.conj()
        except ValueError:
            print('Didnt rotate matrix')
            return rho


def u_mat(theta, phi):
    """ Generates U (rotation matrix) by given angles of Bloch sphere"""

    u11 = np.cos(theta / 2) * np.exp(phi / 2. * 1j)
    u12 = np.sin(theta / 2) * np.exp(- phi / 2. * 1j)
    u21 = -np.sin(theta / 2) * np.exp(phi / 2. * 1j)
    u22 = np.cos(theta / 2) * np.exp(- phi / 2. * 1j)

    u = np.array([[u11, u12], [u21, u22]])

    return u

if __name__ == '__main__':
    d = Discriminator()