import numpy as np
from modules.hamiltonian import Hamiltonian


class ProbabilityDerivative:
    """Perform measurements and calculates derivatives of probabilites for hamiltonian"""

    def __init__(self, hamiltonian: Hamiltonian, original_angles):
        self.hamiltonian = hamiltonian
        self.original_angles = original_angles
        self.n_spins = self.hamiltonian.n_spins

    def conjugated_angles(self):
        """
        Here we create new series of angles for gradient measurements. We use here the notation of bloch sphere basis.
        2 letters notation is exhaustively because basis choice is repeated for the following spines.
        For example "YZ" means "YZYZYZ...". Bloch notation means:
            - Z: theta=theta, phi=phi
            - X: theta=theta+pi/2, phi=phi
            - Y: thetha=theta+pi/2, phi=phi+pi/2

        """
        original_angles = self.original_angles

        ZZ = np.copy(original_angles)
        XZ = np.copy(original_angles)
        YZ = np.copy(original_angles)
        ZX = np.copy(original_angles)
        ZY = np.copy(original_angles)
        for i in range(self.n_spins):
            if i % 2 == 0:
                XZ[i, 0] += np.pi / 2  # theta
                YZ[i, 0] = np.pi / 2  # theta
                YZ[i, 1] += np.pi / 2  # phi
            else:
                ZX[i, 0] += np.pi / 2  # theta
                ZY[i, 0] = np.pi / 2  # theta
                ZY[i, 1] += np.pi / 2  # phi

        return ZZ, XZ, YZ, ZX, ZY

    def measurements_for_gradient(self):
        # TODO: make correlators
        """
        It's a key function of our work, where measurements are performed.
        There are 2 types of basis choices:
            - Original (O) with theta, phi from our primary basis choice. Label it as "/"
            - Conjugated (C) with theta + pi/2, phi. Label it as "\"
        We perform 3 types of measurement:
            - All original (orig). With labels it looks like : "//////" for 6 spins
            - Even spins in conjugated basis and odd in original (even): (spins are numerated from 0)
            - Even spins in original basis and odd in conjugated (odd) :
        Then we take singles and correlators from this measurements. We name it like, for example:
            - "singles_C" for singles in conjugated basis: \
            - "correlators_OC" for correlators for first spin in original basis and second in conjugated: /\
        """
        n_spins = self.n_spins
        hamiltonian = self.hamiltonian
        # Generate angles for measurements
        (ZZ, XZ, YZ, ZX, ZY) = self.conjugated_angles()

        # Perform measurements
        # raw_singles_ZZ, raw_correlators_ZZ = hamiltonian.measure(ZZ)
        # raw_singles_XZ, raw_correlators_XZ = hamiltonian.measure(XZ)
        # raw_singles_YZ, raw_correlators_YZ = hamiltonian.measure(YZ)
        # raw_singles_ZX, raw_correlators_ZX = hamiltonian.measure(ZX)
        # raw_singles_ZY, raw_correlators_ZY = hamiltonian.measure(ZY)
        raw_singles_ZZ = hamiltonian.measure(ZZ)
        raw_singles_XZ = hamiltonian.measure(XZ)
        raw_singles_YZ = hamiltonian.measure(YZ)
        raw_singles_ZX = hamiltonian.measure(ZX)
        raw_singles_ZY = hamiltonian.measure(ZY)

        # In Z basis we immediately got answers, so just for consistency:
        singles_Z = raw_singles_ZZ
        # correlators_ZZ = raw_correlators_ZZ

        # Create blanks for other singles and correlators
        singles_X = np.zeros((n_spins, 2))
        singles_Y = np.zeros((n_spins, 2))
        # correlators_XZ = np.zeros((n_spins - 1, 4))
        # correlators_YZ = np.zeros((n_spins - 1, 4))

        for i in range(n_spins):
            if i % 2 == 0:
                singles_X[i] = raw_singles_XZ[i]
                singles_Y[i] = raw_singles_YZ[i]
                # if i + 1 < n_spins:
                #     correlators_XZ[i] = raw_correlators_XZ[i]
                #     correlators_YZ[i] = raw_correlators_YZ[i]
            else:
                singles_X[i] = raw_singles_ZX[i]
                singles_Y[i] = raw_singles_ZY[i]
                # if i + 1 < n_spins:
                #     correlators_XZ[i] = raw_correlators_ZX[i]
                #     correlators_YZ[i] = raw_correlators_ZY[i]

        self.singles_Z = singles_Z
        self.singles_X = singles_X
        self.singles_Y = singles_Y

    def get_ABC(self):

        theta = self.original_angles[:, 0]  # we need only thetas

        self.measurements_for_gradient()

        # Probabilities of getting 0 in each basis
        p_Z, p_X, p_Y = self.singles_Z[:, 0], self.singles_X[:, 0], self.singles_Y[:, 0]

        # Auxiliary vectors
        V1 = p_Z - (1 / 2)
        V2 = p_X - (1 / 2)

        A = - np.cos(theta) * V1 + np.sin(theta) * V2
        B = np.sin(theta) * V1 + np.cos(theta) * V2
        C = (1 / 2) - p_Y

        self.coefs = np.array([A, B, C])
        return self.coefs

    def d_prob(self):
        """
        By given coefficients [A, B, C] and angles [theta, phi] returns corresponding single derivative
            A = 1/2 - rho_00
            B = Re(rho_01 * e^{i*phi}
            C = Im(rho_01 * e^{i*phi}
        """
        A, B, C = self.get_ABC()
        theta = self.original_angles[:, 0]  # we need only thetas

        # Actually, this is derivatives of probabilities by d_theta and d_phi
        d_theta = A * np.sin(theta) + B * np.cos(theta)  # \frac{d_prob}{d_theta}
        d_phi = C * np.sin(theta)  # \frac{d_prob}{d_phi}

        d_prob = np.array([d_theta, d_phi]).T
        self.d_prob = d_prob

        return self.d_prob


class Gradient:
    def __init__(self, hamiltonian_t: Hamiltonian, hamiltonian_g, original_angles):
        self.hamiltonian_t = hamiltonian_t
        self.hamiltonian_g = hamiltonian_g
        self.original_angles = original_angles

        self.prob_der_t = ProbabilityDerivative(hamiltonian_t, original_angles)
        self.prob_der_g = ProbabilityDerivative(hamiltonian_g, original_angles)
        self.d_prob_t = self.prob_der_t.d_prob()
        self.d_prob_g = self.prob_der_g.d_prob()

    def gradient(self):
        """calculates gradient vector"""
        d_angles = 2 * (self.prob_der_g.singles_Z - self.prob_der_t.singles_Z) * (self.d_prob_g - self.d_prob_t)
        self.d_angles = d_angles

        return d_angles

    def gradient_descent(self, lr=0.01, num_iterations=10):
        """Perform gradient descent"""
        for i in range(num_iterations):
            self.original_angles += lr * self.gradient()

        return self.original_angles


# n_spins = 5
# x = np.random.rand(n_spins) * 2 - 1
# y = np.random.rand(n_spins) * 2 - 1
# xx = np.random.rand(n_spins - 1) * 2 - 1
# hamiltonian = Hamiltonian(n_spins=5, beta=0.3, x=x, y=y, xx=xx)
#
# hamiltonian.set_density_mat()
#
# orig_angles = np.random.rand(n_spins, 2)
#
# A, B, C = get_ABC(hamiltonian, orig_angles)
#
# print(gradient_descent(hamiltonian, orig_angles))
