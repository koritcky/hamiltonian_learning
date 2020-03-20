# Here lies function of matrices manipulations

import numpy as np
import scipy as sp

from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d

import warnings
warnings.filterwarnings("error")

import modules.measurements as measurements

class Hamiltonian:
    def __init__(self, n_spins, beta=0.3, **kwargs):
        """
        kwargs consist of fields and couplings:
        x, y, z,
        xx, yy, zz
        """
        self.beta = beta
        self.__dict__.update(kwargs)
        self.n_spins = n_spins
        self.density_mat = None


    def get_density_mat(self):
        """Builds Gibbs density matrix based on exchange coeffs and fields
        params: x,y,z - fields
        xx, yy, zz - couplings

        """

        no_checks = {"check_herm": False,
                     "check_pcon": False,
                     "check_symm": False}


        static = []

        for field in ['x', 'y', 'z']:
            if field in self.__dict__.keys():
                fields_list = self.__dict__[field]
                fields_list = [[fields_list[i], i] for i in range(self.n_spins)]

                static.append([field, fields_list])

        for coupling in ['xx', 'yy', 'zz']:
            if coupling in self.__dict__.keys():
                couplings_list = self.__dict__[coupling]
                couplings_list = [[couplings_list[i], i, (i + 1) % self.n_spins] for i in range(self.n_spins - 1)]
                static.append([coupling, couplings_list])

        basis = spin_basis_1d(self.n_spins)
        dynamic = []

        # generating h_xamiltonian
        H = hamiltonian(static, dynamic, basis=basis, **no_checks)
        H = H.toarray()

        # normalization constant
        Z = np.trace(sp.linalg.expm(-self.beta * H))

        # density matrix
        try:
            rho = sp.linalg.expm(-self.beta * H) / Z
        except RuntimeWarning:
            print(f"Z={Z}")
            print(f"Static={static}")
            print(f"H={H}")

        self.density_mat = rho
        return self.density_mat

    def rotation(self, angles):
        """Rotates state
        params: [[theta1, phi1], [theta2, phi2],  ..., [thetaN, phiN]]"""

        umat = angles_to_umat(angles)
        density_mat = rotate(self.density_mat, umat)

        return density_mat

    def measure(self, angles):
        density_mat = self.rotation(angles)
        singles, correlators = measurements.reduced_matrix_measurements(density_mat)
        return singles, correlators


def u_mat(theta, phi):
    """ Generates U (rotation matrix) by given angles of Bloch sphere"""

    u11 = np.cos(theta / 2)
    u12 = np.sin(theta / 2) * np.exp(- phi * 1j)
    u21 = -np.sin(theta / 2)
    u22 = np.cos(theta / 2) * np.exp(- phi * 1j)

    u = np.array([[u11, u12], [u21, u22]])

    return u


def angles_to_umat(angles):
    """Calculates rotation matrices
    angles: np.array, columns are different spins, rows are theta and phi
    """
    # First matrix
    theta, phi = angles[0]
    U = u_mat(theta, phi)

    # If number of spins is more than 1
    for i in range(1, angles.shape[0]):
        theta, phi = angles[i]

        U = np.kron(U, u_mat(theta, phi)).astype('complex64')

    return U


def rotate(rho: np.array, u_mat):
    """ Return density matrix in new basis"""
    try:
        return u_mat @ rho @ u_mat.T.conj()
    except ValueError:
        print('Didnt rotate matrix')
        return rho


def spher_to_cartesian(params_spher):
    """Transform spherical angles to cartesian components"""
    params_cartesian = []
    for theta_h, phi_h in params_spher.T:
        hx = np.sin(theta_h)*np.cos(phi_h)
        hy = np.sin(theta_h)*np.sin(phi_h)
        hz = np.cos(theta_h)
        params_cartesian.append([hx, hy, hz])

    return np.array(params_cartesian).T

# g = Generator(3, 0.3, x=[0.5, 1, 0.7], z = [0.3, 1, 1])
#
# g.density_mat()
# print(np.trace(g.density_mat@g.density_mat))
