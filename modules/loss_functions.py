import numpy as np
from modules.matrices import *


def trace_distance(d1, d2):
    return np.linalg.norm((d1 - d2), ord=1)


def g_loss_func(params, *args):
    """
    Generator loss function.
    Estimates how far density matrix from generated hamiltonian (params)
    in compatison with given density matrix rho_t_z.

    params: [theta1, theta2, ..., thetaN, phi1, phi2, ..., phiN]
             where thetaI,phiI determine field direction on Ith spin
    """
    rho_t_z, angles = args

    N_spins = np.shape(angles)[1]
    params = params.reshape(2, N_spins)


    # Generator manipulate rho_g to get closer to rho_t_z
    rho_g_z = density_matr(params)
    umat = angles_to_umat(angles)

    # Get rho in new basis
    rho_g_new = rotate(rho_g_z, umat)
    rho_t_new = rotate(rho_t_z, umat)

    # Get distribution
    # (in current version just get diagonal elements)
    prob_g = np.diag(rho_g_new).real
    prob_t = np.diag(rho_t_new).real
    # distr_g = np.random.choice([0, 1], size=10000, p=prob_g)
    # distr_t = np.random.choice([0, 1], size=10000, p=prob_t)

    return trace_distance(prob_g, prob_t)


def d_loss_func(angles: np.array, *args):
    # Discriminator loss function
    N_spins = len(angles)//2

    angles = angles.reshape(2, N_spins)
    rho_g_z, rho_t_z = args

    # We use this function that return u_mats for number of different choices of basises. But we pass just one basis
    umat = angles_to_umat(angles)

    # Get rho in new basis
    rho_g_new = rotate(rho_g_z, umat)
    rho_t_new = rotate(rho_t_z, umat)

    # Get distribution
    # (in current version just get diagonal elements)
    prob_g = np.diag(rho_g_new).real
    prob_t = np.diag(rho_t_new).real

    return -trace_distance(prob_g, prob_t)


def spher_to_cartesian(params_spher):
    params_cartesian = []
    for angles in params_spher.T:
        theta, phi = angles
        hx = np.sin(theta)*np.cos(phi)
        hy = np.sin(theta)*np.sin(phi)
        hz = np.cos(theta)
        params_cartesian.append([hx, hy, hz])
    return np.array(params_cartesian).T