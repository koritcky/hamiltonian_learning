import numpy as np
from modules.measures import *
from modules.matrices import *




def generate_params(fields, interactions, N_spins):
    if fields > 3 or interactions > 3:
        raise Warning('Too many components. Must be <= 3 for fields and interactions')
    h = np.random.randn(fields, N_spins)
    J = np.random.randn(interactions, N_spins - 1)
    return h, J

def flat_to_params(flat_params, fields, interactions, N_spins):
    h_flat, J_flat = np.split(flat_params, [fields * N_spins])
    h = h_flat.reshape(fields, N_spins)
    J = J_flat.reshape(interactions, N_spins - 1)
    return h, J



def g_loss_func(flat_params, *args):
    """
    Generator loss function.
    Estimates how far density matrix from generated hamiltonian (params)
    in compatison with given density matrix rho_t_z.

    params: [theta1, theta2, ..., thetaN, phi1, phi2, ..., phiN, Jxx1, ..., JxxN, ..., Jzz1, ..., JzzN]
             where thetaI,phiI determine field direction on Ith spin
    """
    rho_t_z, angles, beta, model, fields, interactions, N_spins = args

    h, J = flat_to_params(flat_params,fields, interactions, N_spins )


    # Generator manipulate rho_g to get closer to rho_t_z
    rho_g_z = density_matr(h, J, beta, model)
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

    return matrix_distance(rho_g_new, rho_t_new, 10 ** 3)


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

    return -matrix_distance(rho_g_new, rho_t_new, 10 ** 3)

