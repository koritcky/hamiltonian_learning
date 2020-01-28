# Here lies function of matrices manipulations

import numpy as np
import scipy as sp

from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d


def density_matr(params, beta=1, **kwargs):
    """Builds Gibbs density matrix based on exchange coeffs and fields
    params: [[theta_h1, theta_h2, ..., theta_hN],
             [phi_h1, phi_h2, ..., phi_hN],
             [theta_J1, theta_J2, ..., theta_JN],
             [phi_J1, phi_J2, ..., phi_JN]]
             where theta_hi,phi_hi determine field direction on ith spin
                   theta_Ji, phi_Ji determine interaction (Jxx, Jyy, Jzz) between i and i+1 spins

    """
    hx, hy, hz, Jxx, Jyy, Jzz = spher_to_cartesian(params)
    N_spins = len(hx)

    if 'no_checks' in kwargs:
        no_checks = kwargs['no_checks']
    else:
        no_checks = {"check_herm": False,
                     "check_pcon": False,
                     "check_symm": False}

    basis = spin_basis_1d(N_spins)

    # make coupling and field types correct for used library
    hx = [[hx[i], i] for i in range(N_spins)]
    hy = [[hy[i], i] for i in range(N_spins)]
    hz = [[hz[i], i] for i in range(N_spins)]

    Jxx = [[Jxx[i], i, (i + 1) % N_spins] for i in range(N_spins)]
    Jyy = [[Jyy[i], i, (i + 1) % N_spins] for i in range(N_spins)]
    Jzz = [[Jzz[i], i, (i + 1) % N_spins] for i in range(N_spins)]

    static = [['x', hx],
              ['y', hy],
              ['z', hz],
              ['xx', Jxx],
              ['yy', Jyy],
              ['zz', Jzz]]
    # static = [['x', hx],
    #           ['y', hy],
    #           ['z', hz]]

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
    for angles in params_spher.T:
        theta_h, phi_h, theta_J, phi_J = angles
        hx = np.sin(theta_h)*np.cos(phi_h)
        hy = np.sin(theta_h)*np.sin(phi_h)
        hz = np.cos(theta_h)
        Jxx = np.sin(theta_J) * np.cos(phi_J)
        Jyy = np.sin(theta_J) * np.sin(phi_J)
        Jzz = np.cos(theta_J)
        params_cartesian.append([hx, hy, hz, Jxx, Jyy, Jzz])
    return np.array(params_cartesian).T