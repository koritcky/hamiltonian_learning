import numpy as np
from modules.smc import Particle


def conjugated_angles(original_angles):
    """
    Here we create new series of angles for gradient measurements. We use here the notation of bloch sphere basis.
    2 letters notation is exhaustively because basis choice is repeated for the following spines.
    For example "YZ" means "YZYZYZ...". Bloch notation means:
        - Z: theta=theta, phi=phi
        - X: theta=theta+pi/2, phi=phi
        - Y: thetha=theta+pi/2, phi=phi+pi/2

    """
    n_spins = len(original_angles)

    ZZ = np.copy(original_angles)
    XZ = np.copy(original_angles)
    YZ = np.copy(original_angles)
    ZX = np.copy(original_angles)
    ZY = np.copy(original_angles)
    for i in range(n_spins):
        if i % 2 == 0:
            XZ[i, 0] += np.pi / 2
            YZ[i] += np.pi / 2  # Both angles are increases by pi/2
        else:
            ZX[i, 0] += np.pi / 2
            ZY[i] += np.pi / 2  # Both angles are increases by pi/2
    return ZZ, XZ, YZ, ZX, ZY


def measurements_for_gradient(particle: Particle, original_angles):
    # TODO: make correlators
    """

    It's a key function of our work, where measurements are performed.
    There are 2 types of basis choices:
        - Original (O) with theta, phi from our primary basis choice. Label it as "/"
        - Conjugated (C) with theta + pi/2, phi. Label it as "\"
    We perform 3 types of measurement:
        - All original (orig). With labels it looks like : "//////" for 6 spins
        - Even spins in conjugated basis and odd in original (even): "\/\/\/" (spins are numerated from 0)
        - Even spins in original basis and odd in conjugated (odd) : "/\/\/\"
    Then we take singles and correlators from this measurements. We name it like, for example:
        - "singles_C" for singles in conjugated basis: \
        - "correlators_OC" for correlators for first spin in original basis and second in conjugated: /\
    """
    n_spins = particle.n_spins

    # Generate angles for measurements
    (ZZ, XZ, YZ, ZX, ZY) = conjugated_angles(original_angles)

    # Perform measurements
    raw_singles_ZZ, raw_correlators_ZZ = particle.measure(ZZ)
    raw_singles_XZ, raw_correlators_XZ = particle.measure(XZ)
    raw_singles_YZ, raw_correlators_YZ = particle.measure(YZ)
    raw_singles_ZX, raw_correlators_ZX = particle.measure(ZX)
    raw_singles_ZY, raw_correlators_ZY = particle.measure(ZY)

    # In Z basis we immediately got answers, so just for consistency:
    singles_Z = raw_singles_ZZ
    correlators_ZZ = raw_correlators_ZZ

    # Create blanks for other singles and correlators
    singles_X = np.zeros((n_spins, 2))
    singles_Y = np.zeros((n_spins, 2))
    correlators_XZ = np.zeros((n_spins - 1, 4))
    correlators_YZ = np.zeros((n_spins - 1, 4))

    for i in range(n_spins):
        if i % 2 == 0:
            singles_X[i] = raw_singles_XZ[i]
            singles_Y[i] = raw_singles_YZ[i]
            if i + 1 < n_spins:
                correlators_XZ[i] = raw_correlators_XZ[i]
                correlators_YZ[i] = raw_correlators_YZ[i]
        else:
            singles_X[i] = raw_singles_ZX[i]
            singles_Y[i] = raw_singles_ZY[i]
            if i + 1 < n_spins:
                correlators_XZ[i] = raw_correlators_ZX[i]
                correlators_YZ[i] = raw_correlators_ZY[i]

    return singles_Z, singles_X, singles_Y


def single_probability_gradient(coefs, angles):
    """
    By given coefficients [A, B, C] and angles [theta, phi] returns corresponding single derivative
        A = 1/2 - rho_00
        B = Re(rho_01 * e^{i*phi}
        C = Im(rho_01 * e^{i*phi}
    """
    A, B, C = coefs
    theta = angles[:, 0] # we need only thetas

    d_theta = A * np.cos(theta) + B * np.sin(theta)
    d_phi = - C * np.sin(theta)

    d_angles = np.array([d_theta, d_phi]).T

    return d_angles


def get_ABC(particle:Particle, angles):

    theta = angles[:, 0]  # we need only thetas

    singles_Z, singles_X, singles_Y = measurements_for_gradient(particle, angles)
    p_Z, p_X, p_Y = singles_Z[:, 0], singles_X[:, 0], singles_Y[:, 0]  # Probabilities of getting 0 in each basis


    V1 = p_Z - ((np.sin(theta / 2)) ** 2)
    V2 = p_X - (1 / 2) * (np.sin(theta) + 1)

    A = np.cos(theta) * V1 - np.sin(theta) * V2
    B = np.sin(theta) * V1 + np.cos(theta) * V2
    C = (1 / 2) - p_Y

    return A, B, C

n_spins = 5
x = np.random.rand(n_spins) * 2 - 1
y = np.random.rand(n_spins) * 2 - 1
xx = np.random.rand(n_spins - 1) * 2 - 1
particle = Particle(weight=0.5, n_spins=5, beta=0.3, x=x, y=y, xx=xx)

particle.set_density_mat()

orig_angles = np.random.rand(n_spins, 2)

A, B, C = get_ABC(particle, orig_angles)

print(single_probability_gradient([A, B, C], orig_angles))
