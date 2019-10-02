import numpy as np
from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d
import scipy as sp



class Generator:
    def __init__(self, Jxx, Jyy, Jzz, hx, N_spins, beta=None, **kwargs):
        """ Couplings: dict with lists of Jxx, Jyy, Jzz, hx. """
        if beta is None:
            beta = 1
        self.Jxx = Jxx
        self.Jyy = Jyy
        self.Jzz = Jzz
        self.hx = hx
        self.N_spins = N_spins
        self.beta = beta
        self.rho = density_matr(self.Jxx, self.Jyy, self.Jzz, self.hx, self.beta, self.N_spins, **kwargs)

        assert len(self.Jxx) == len(self.Jyy) == len(self.Jzz) == (len(self.hx) -1) == (self.N_spins -1), 'Number of couplings is not consistent'

    def generate_mat(self):
        return self.rho

    def train(self, new_couplings):
        return density_matr(self.Nspins, self.beta,  **new_couplings)

    @staticmethod
    def loss_function(couplings: list, *args):
        N_spins = args[0]
        Jxx, Jyy, Jzz = (couplings[i: i + (N_spins - 1)] for i in range(0, len(couplings)-N_spins, N_spins-1))
        hx = couplings[-N_spins:]
        
def density_matr(Jxx, Jyy, Jzz, hx, beta, N_spins, **kwargs):
    """Calculates Gibbs density matrix"""

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
    h_x = [[hx[i], i] for i in range(N_spins)]

    static = [["xx", J_xx], ["yy", J_yy], ["zz", J_zz], ["x", h_x]]

    dynamic = []

    # generating hamiltonian
    H = hamiltonian(static, dynamic, basis=basis,
                    **no_checks, dtype=np.float64)
    H = H.toarray()
    # normalization constant
    Z = np.trace(sp.linalg.expm(-beta * H))
    # density matrix
    rho = sp.linalg.expm(-beta * H) / Z

    return rho

