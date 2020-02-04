# Here lies function of matrices manipulations

import numpy as np
import scipy as sp

from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d


def density_matr(params, beta=0.3, ising=False, **kwargs):
    """Builds Gibbs density matrix based on exchange coeffs and fields
    params: [[hz1, ..., hzN],
             [Jzz1, Jzz2, ..., JzzN]]
             where hz - field,
                   Jzz - interaction integral

    """

    if 'no_checks' in kwargs:
        no_checks = kwargs['no_checks']
    else:
        no_checks = {"check_herm": False,
                     "check_pcon": False,
                     "check_symm": False}

    if ising:
        hz, Jzz = params
        N_spins = len(hz)

        hz = [[hz[i], i] for i in range(N_spins)]
        Jzz = [[Jzz[i], i, (i + 1) % N_spins] for i in range(N_spins)]

        static = [
            ['z', hz],
            ['zz', Jzz]]
    else:
        theta_h, phi_h, Jxx, Jyy, Jzz = params
        hx, hy, hz = spher_to_cartesian(params[0:2])
        N_spins = len(hz)

        hx = [[hx[i], i] for i in range(N_spins)]
        hy = [[hy[i], i] for i in range(N_spins)]
        hz = [[hz[i], i] for i in range(N_spins)]

        Jxx = [[Jxx[i], i, (i + 1) % N_spins] for i in range(N_spins)]
        Jyy = [[Jyy[i], i, (i + 1) % N_spins] for i in range(N_spins)]
        Jzz = [[Jzz[i], i, (i + 1) % N_spins] for i in range(N_spins)]

        static = [
            ['x', hx],
            ['y', hy],
            ['z', hz],
            ['xx', Jxx],
            ['yy', Jyy],
            ['zz', Jzz]]

    basis = spin_basis_1d(N_spins)
    dynamic = []

    # generating h_xamiltonian
    H = hamiltonian(static, dynamic, basis=basis, **no_checks)
    H = H.toarray()

    # normalization constant
    Z = np.trace(sp.linalg.expm(-beta * H))

    # density matrix
    rho = sp.linalg.expm(-beta * H) / Z

    return rho


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
    angles = np.array(angles).T
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
