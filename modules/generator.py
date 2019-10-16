import numpy as np
from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d
import scipy as sp
from modules.discriminator import Discriminator, u_mat, rho_to_distr
from scipy.stats import entropy



class Generator:
    def __init__(self, couplings, beta=None, **kwargs):
        if beta is None:
            beta = 1
        self.couplings = couplings
        self.beta = beta
        self.rho = self.density_matr(self.couplings, self.beta, **kwargs)

    def generate_mat(self):
        return self.rho

    def train(self):
        # TODO
        pass

    @staticmethod
    def couplings_unpack(couplings):
        """Returns Jxx, Jyy, Jzz, hx couplings from it concatenated form"""
        N_spins = (len(couplings) + 3) // 4
        Jxx, Jyy, Jzz = (couplings[i: i + (N_spins - 1)] for i in range(0, len(couplings) - N_spins, N_spins - 1))
        hx = couplings[-N_spins:]
        return Jxx, Jyy, Jzz, hx

    @staticmethod
    def loss_function(couplings: list, *args):
        rho_t_z, angles = args

        # Generator manipulate rho_g to get closer to rho_t_z
        rho_g_z = Generator.density_matr(couplings)
        umat = Discriminator.angles_to_umat(angles)

        # Get rho in new basis
        rho_g_new = Discriminator.rotate(rho_g_z, umat)
        rho_t_new = Discriminator.rotate(rho_t_z, umat)

        # Get distribution
        # (in current version just get diagonal elements)
        distr_g = np.diag(rho_g_new).real
        distr_t = np.diag(rho_t_new).real

        # Distance between distributions
        ent = entropy(distr_g, distr_t)
        return ent


    @staticmethod
    def density_matr(couplings, beta=1, **kwargs):
        """Calculates Gibbs density matrix"""
        N_spins = (len(couplings) + 3) // 4
        Jxx, Jyy, Jzz, hx = Generator.couplings_unpack(couplings)
        if 'no_checks' in kwargs:
            no_checks = kwargs['no_checks']
        else:
            no_checks = {"check_herm": False,
                         "check_pcon": False,
                         "check_symm": False}

        adj_mat = np.zeros((N_spins, N_spins))
        tmp_mat = np.zeros((N_spins - 1, N_spins - 1))
        np.fill_diagonal(tmp_mat, 1)
        adj_mat[:-1, 1:] = tmp_mat
        nz_ind = np.argwhere(adj_mat == 1).astype(object)

        basis = spin_basis_1d(N_spins)

        J_xx = np.insert(nz_ind, 0, Jxx, axis=1).tolist()
        J_yy = np.insert(nz_ind, 0, Jyy, axis=1).tolist()
        J_zz = np.insert(nz_ind, 0, Jzz, axis=1).tolist()
        # Why x, not z ?!2
        h_x = [[hx[i], i] for i in range(N_spins)]

        static = [["xx", J_xx], ["yy", J_yy], ["zz", J_zz], ["x", h_x]]

        dynamic = []

        # generating h_xamiltonian
        H = hamiltonian(static, dynamic, basis=basis,
                        **no_checks, dtype=np.float64)
        H = H.toarray()
        # normalization constant
        Z = np.trace(sp.linalg.expm(-beta * H))
        # density matrix
        rho = sp.linalg.expm(-beta * H) / Z

        return rho

