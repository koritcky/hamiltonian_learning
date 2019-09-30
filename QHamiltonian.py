#!/usr/bin/env python
# coding: utf-8

# ## Quantum Hamiltonian reconstruction on nearest neighbor graph

# Reconstruction is performed for Heizenberg XYZ Hamiltonian of the following form:
# 
# $$H = \sum_{{i,j}\in G}J^{xx}_{ij}\sigma^x_i\sigma^x_j + J^{yy}_{ij}\sigma^y_i \sigma^y_j 
# + J^{zz}_{ij}\sigma^z_i \sigma^z_j + \sum_{i\in G} h^x_i$$
# 
# Further in code, XYZ can be turned into any isotropic (XXX, XXZ, etc.) form by defining equal X Y and Z arrays of couplings

# In[25]:


import matplotlib.pyplot as plt
import numpy as np
import torch
import scipy as sp
from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d
from sklearn.metrics import mean_squared_error as mse
from time import time, perf_counter
from scipy.optimize import minimize
from scipy.stats import entropy
import copy
from scipy.optimize import basinhopping
from sys import getsizeof
import pickle


# In[26]:


# Lets remove Hermiticity checks!
no_checks={"check_herm":False,
           "check_pcon":False,
           "check_symm":False}    


# # Define critical functions

# In[27]:


##Gibbs density matrix calculation##

def density_matr (N_spins, Jxx, Jyy, Jzz, hx, beta):
    
    """Calculates Gibbs density matrix

    Args:
        N (int): number of spins 
        coupl_matr (np.array): J_ij 
                    coupling matrix
        field_vec (np.array): array of
            transverse magnetic fields
        beta (float): inverse temperature

    Returns:
        np.array:

    """   
    
    adj_mat = np.zeros ((N_spins, N_spins))
    tmp_mat = np.zeros ((N_spins - 1, N_spins - 1))
    np.fill_diagonal (tmp_mat, 1)
    adj_mat [:-1, 1:] = tmp_mat
    nz_ind = np.argwhere (adj_mat == 1).astype (object)
    
    basis = spin_basis_1d (N_spins)
    
    
    J_xx = np.insert (nz_ind, 0, Jxx, axis = 1).tolist ()
    J_yy = np.insert (nz_ind, 0, Jyy, axis = 1).tolist ()
    J_zz = np.insert (nz_ind, 0, Jzz, axis = 1).tolist ()
    h_x = [[hx [i], i] for i in range (N_spins)]
    
    
    static = [["xx", J_xx], ["yy", J_yy], ["zz", J_zz], ["x", h_x]]
        
    dynamic = []

    # generating hamiltonian
    H = hamiltonian (static, dynamic, basis = basis, 
                     **no_checks, dtype=np.float64)
    H = H.toarray ()
    #normalization constant
    Z = np.trace (sp.linalg.expm (-beta*H))
    #density matrix
    rho = sp.linalg.expm (-beta*H)/Z
    
    return rho


# In[28]:


##Returns dictionary of angles for 3^N basis states#

def xyz_rot_angles (N_spins):
    
    """Returns dict with 3**N keys, corresponding to 3^N basis states
    
    Each key contains set of angles, corresponding to this basis state
    """
    
    ang_dict = dict ()
    
    #construction of x y z bases space
    bases_ternary = list ()

    for num in range (3**N_spins):
        ternary_basis = np.base_repr (num, 3)
        full_repr = '0' * (N_spins - len (ternary_basis)) + ternary_basis
        list_basis = list (full_repr)
        bases_ternary.append (list_basis)
        
    ang_array = np.array ([[np.pi/2, 0], [np.pi/2, np.pi/2], [0, 0]])  
    bases_ternary = np.array (bases_ternary)
    ang_blank = np.zeros ((N_spins, 2))
    
    for i in range (len (bases_ternary)):
        int_indices = bases_ternary [i].astype (int)
        ang_blank = ang_array [int_indices]
        key = ''.join (bases_ternary [i])
        ang_dict [key] = ang_blank
        
        
    return ang_dict


# In[29]:


##Returns dictionary of angles for a defined number of random basis states#

def rand_rot_angles (N_spins, N_bases):
    
    """Returns dict with user defined number of keys, where each key 
    correspond to a random basis state (phi and theta are randomly chosen on a Bloch sphere)
    
    N_spins - number of spins in a chain
    N_bases - number of random bases
    
    """
    
    ang_dict = dict()
    
    bases = np.arange(N_bases)
        
    cos_thetas = [2*(np.random.rand (N_spins)-.5) for i in bases]
                  
    phis = [2*(np.random.rand (N_spins)-.5) for i in bases]
    
    for i in bases:
        
        angs_arr = np.zeros ((N_spins, 2))
        z = cos_thetas [i]
        theta = np.arccos (z)
        phi = phis [i]
        
        angs_arr [:, 0] = theta
        angs_arr [:, 1] = phi
        
        ang_dict [bases [i]] = angs_arr
    
    return ang_dict


# In[30]:


##Calculates U and U_+ which will be further used to acces diagonal element of density matrix in log_like func##

def rot_matr (angles):
    
    """Calculates rotation matrices (and Hermitian conj) for the given set of {theta, phi} array

    Args:
        angles dict (from functions xyz_rot_angles or rand_rot_angles)

    Returns:
        dict(): dict['angles'],
                dict['rot_matr'],
                dict['rot_H']                
    """    
    
    upd_dict = dict()
    
    for key in list (angles.keys ()):
    
        it_ranges = angles [key].shape

        theta = angles [key] [0, 0]
        phi = angles [key] [0, 1]

        a = np.cos (theta/2) * np.exp(phi/2.*1j)
        b = np.sin (theta/2) * np.exp( - phi/2.*1j)
        c = -np.sin (theta/2) * np.exp(phi/2.*1j)
        d = np.cos (theta/2) * np.exp( - phi/2.*1j)

        U = np.array ([[a, b], [c, d]])

        for i in range (it_ranges [0] - 1):

            theta = angles [key] [i + 1, 0]
            phi = angles [key] [i + 1, 1]

            a = np.cos (theta/2) * np.exp(phi/2.*1j)
            b = np.sin (theta/2) * np.exp( - phi/2.*1j)
            c = -np.sin (theta/2) * np.exp(phi/2.*1j)
            d = np.cos (theta/2) * np.exp( - phi/2.*1j)
            

            U = np.kron (U, np.array ([[a, b], [c, d]])).astype ('complex64')
            
        matr_dict = dict()
        
        matr_dict ['angles'] = angles [key]
        
        matr_dict ['rot_matr'] = U
        
        U_asmatr = np.asmatrix (U)
        rot_H = U_asmatr.getH ()
        rot_H = rot_H.getA ()
        
        matr_dict ['rot_H'] = rot_H
        
        upd_dict [key] = matr_dict
        
    return upd_dict
    
    

# In[31]:


#Return diagonal of denisty matrix, which will be sampled for the dataset#

def get_rho_diag (rho, rot_matrices):
    
    """Calculates diagonal elements of the density matrix 
    (corresponds to projective measurements)

    Args:
        rho(np.array): density matrix
        rot_matrices(dict): 

    Returns:
        dict():  dict['angles'],
                 dict['rot_matr'],
                 dict['rot_H'],
                 dict['rho_diag'],
    """  
    
    upd_dict = dict()
    
    for key in list(rot_matrices.keys ()):
        
        basis = {}
        
        rot_matr = rot_matrices [key] ['rot_matr']
        rot_H = rot_matrices [key] ['rot_H']
        rot_rho = np.dot (np.dot (rot_matr, rho), rot_H)
    
        diag = np.real (rot_rho.diagonal ())
        
        basis ['angles'] = rot_matrices [key] ['angles']
        basis ['rot_matr'] = rot_matr
        basis ['rot_H'] = rot_H
        basis ['diag'] = diag
        
        upd_dict [key] = basis
    
    return upd_dict


# In[32]:


#From the dataset, returned by get_rho_diag, choose a batch dataset#

def get_batch_dataset (diag, N_rb, N_m):
    
    batch_ds = dict ()
    
    """
    N_rb - number of randomly chosen bases from the dataset
    N_m - number of projective measurements in each basis
    """
    
    rand_ints = np.random.choice (range (0, len (diag)), N_rb, replace = False)
    keys = np.array (list (diag.keys ()))
    selected_keys = keys [rand_ints]
    for key in selected_keys:
        basis = dict()
    
        s_arr = np.cumsum (diag [key] ['diag'])

        result = np.zeros (N_m)
        for n in range (N_m):
            r = np.random.uniform (0, 1)
            for j in range (len (s_arr)):
                if r < s_arr [j]:
                    break
            result [n] = j + 1

        basis ['measurements'] = result
        basis ['rot_matr'] = diag [key] ['rot_matr']
        basis ['rot_H'] = diag [key] ['rot_H']
        batch_ds [key] = basis
        
    return batch_ds


# In[33]:


#Return full dataset#

def get_dataset (diag, N_exp):
    
    """
    Returns dataset
    N_exp - number of measurements (samples from diagonal of density matrix)
    
    """
        
    dataset = dict()
    
    for key in list(diag.keys ()):        
        basis = dict()
    
        s_arr = np.cumsum (diag [key] ['diag'])
        result = np.zeros (N_exp)
        for n in range (N_exp):
            r = np.random.uniform (0, 1)
            for j in range (len (s_arr)):
                if r < s_arr [j]:
                    break
            result [n] = j
            
        basis ['measurements'] = result
        basis ['rot_matr'] = diag [key] ['rot_matr']
        basis ['rot_H'] = diag [key] ['rot_H']
        
        dataset [key] = basis
                
    return dataset


# In[34]:


def random_bases_choice (dataset, N):
    
    rand_ds = dict ()
    
    """
    N - number of randomly chosen bases from the dataset
    """
    
    while True:
        rand_ints = np.random.randint (1, len (dataset), N)
        if len (np.unique (rand_ints)) == len (rand_ints):
            break 
    keys = np.array (list (dataset.keys ()))
    selected_keys = keys [rand_ints]
    for key in selected_keys:
        rand_ds [key] = dataset [key]
        
    return rand_ds  


# # Prepare dataset

# In[35]:

def prepare_dataset (N_spins, N_bases, N_measurements, rho):

    #couplings, transverse field
    #XYZ hamiltonian, to create XXX use Jxx = Jyy = Jzz
    #Jyy = np.random.uniform (-1, 1, N_spins - 1)
    #Jxx = np.random.uniform (-1, 1, N_spins - 1)
    #Jzz = np.random.uniform (-1, 1, N_spins - 1)
    #hx = np.random.uniform (-1, 1, N_spins)
    #beta = 1



    #rho = density_matr (N_spins, Jxx, Jyy, Jzz, hx, beta)
    rot_matrices_xyz = rot_matr (xyz_rot_angles(N_spins)) 
    diag = get_rho_diag (rho, rot_matrices_xyz)
    batch_ds = get_batch_dataset (diag, N_bases, N_measurements)
    
    return batch_ds


# # Define (minus) Log-likelihood function
# $$L = -\sum_{\{\sigma_m\}} \log \left( |\langle \sigma_m |U_m \frac{e^{-\beta H(J)}}{Z(J)} U_m^+|\sigma_m\rangle|^2 \right)$$ where $\sigma_m$ - results of the projective measurements in the basis $m$

# In[38]:

def log_like (J_arr, N_spins, dataset, beta):
    
    """
    Returns log-likelihood function
    
    """  
    
    Jxx = J_arr [: N_spins - 1]
    Jyy = J_arr [N_spins - 1: 2*N_spins - 2]
    Jzz = J_arr [2*N_spins - 2: 3*N_spins - 3]
    hx = J_arr [3*N_spins - 3 : 4*N_spins - 3]
    
    basis = spin_basis_1d (N_spins)
    
    adj_mat = np.zeros ((N_spins, N_spins))
    tmp_mat = np.zeros ((N_spins - 1, N_spins - 1))
    np.fill_diagonal (tmp_mat, 1)
    adj_mat [:-1, 1:] = tmp_mat
    nz_ind = np.argwhere (adj_mat == 1).astype (object)
    
    
    J_xx = np.insert (nz_ind, 0, Jxx, axis = 1).tolist ()
    J_yy = np.insert (nz_ind, 0, Jyy, axis = 1).tolist ()
    J_zz = np.insert (nz_ind, 0, Jzz, axis = 1).tolist ()
    h_x = [[hx [i], i] for i in range (N_spins)]
    
    
    
    static = [["xx", J_xx], ["yy", J_yy], ["zz", J_zz], ["x", h_x]]
        
    dynamic = []

    # generating hamiltonian
    H = hamiltonian (static, dynamic, basis = basis, 
                     **no_checks, dtype=np.float64)
    H = H.toarray ()
    #normalization constant
    Z = np.trace (sp.linalg.expm (-beta*H))
    #density matrix
    rho = sp.linalg.expm (-beta*H)/Z
    
    ll = 0
    
    sign = -1
    
    keys = list(dataset.keys())
    
    t1 = perf_counter ()
    for key in keys:
        rot_m = dataset [key] ['rot_matr']
        rot_H = dataset [key] ['rot_H']  
        rot_rho = np.dot (np.dot (rot_m, rho), rot_H)
        idx = dataset [key] ['measurements'] - 1
        ll -= np.sum(np.log(np.real(np.diag(rot_rho)[idx.astype (int)])))
    t2 = perf_counter ()

    #for key in keys:
        #rot_m = dataset [key] ['rot_matr']
        #rot_H = dataset [key] ['rot_H']  
        
        
        #rot_rho = np.dot (np.dot (rot_m, rho), rot_H)
        #spin_measurements = measurements [key]
        
        #for m in spin_measurements: 
            #tmp = np.dot (np.dot (m, rot_rho), m) 
            #ll += sign * np.real (np.log (tmp))

    return ll

def optim_ll (N_spins, x0, batch_ds, beta, J_arr, ll):
    
    log = dict ()

    args = (N_spins, batch_ds, beta)
    #x0 = np.ones (len (J_arr)) * 0.2
    t1 = time ()
    res = minimize (log_like, x0, method = 'COBYLA', args = (N_spins, batch_ds, beta),
                    options = {'maxiter': 50000})
    t2 = time ()
    
    true = J_arr
    Jxx = J_arr [: N_spins - 1]
    Jyy = J_arr [N_spins - 1: 2*N_spins - 2]
    Jzz = J_arr [2*N_spins - 2: 3*N_spins - 3]
    hx = J_arr [3*N_spins - 3 : 4*N_spins - 3]
    predicted = res ['x']
    
    #np.save ('true_fullb_5sp', true)
    #np.save ('rec_fullb_5sp', predicted)
    err = np.sqrt (mse (true, predicted))
    #np.save ('QHamiltonian Learning/5 spins/rec_120_bases', predicted)
    #np.save ('QHamiltonian Learning/5 spins/rmse_120_bases', err)
    tr_J_xx = Jxx
    rec_J_xx = res ['x'] [:N_spins - 1]
    tr_J_yy = Jyy
    rec_J_yy = res ['x'] [N_spins - 1 : 2*N_spins - 2]
    tr_J_zz = Jzz
    rec_J_zz = res ['x'] [2*N_spins - 2 : 3*N_spins - 3]
    tr_h_x = hx
    rec_h_x = res ['x'] [3*N_spins - 3 :]
    print ('------')
    print ('True J_xx: ', tr_J_xx)
    print ('Recovered J_xx: ', rec_J_xx)
    print ('rmse:', np.sqrt (mse (tr_J_xx, rec_J_xx)))
    print ('------')
    print ('True J_yy: ', tr_J_yy)
    print ('Recovered J_yy: ', rec_J_yy)
    print ('rmse:', np.sqrt (mse (tr_J_yy, rec_J_yy)))
    print ('------')
    print ('True J_zz: ', tr_J_zz)
    print ('Recovered J_zz: ', rec_J_zz)
    print ('rmse:', np.sqrt (mse (tr_J_zz, rec_J_zz)))
    print ('------')
    print ('True h_x: ', tr_h_x)
    print ('Recovered h_x: ', rec_h_x)
    print ('rmse:', np.sqrt (mse (tr_h_x, rec_h_x)))
    print ('------')

    print ('Total RMSE: ', err)
    print ('------')
    
    print ('True log like maximum: ', ll)
    print ('Optimized maximum: ', res ['fun'])
    print ('------')
    
    print ('Optimization time, s: ', t2 - t1) 
    
    return {'RMSE': err, 'Time': t2 - t1}

def discriminator (angles, rho_t, rho_g, N_m):
    
    """             
    """    
    
    angles = angles.reshape ((int (len (angles) / 2), 2))
    
    it_ranges = angles.shape

    theta = angles [0, 0]
    phi = angles [0, 1]

    a = np.cos (theta/2) * np.exp(phi/2.*1j)
    b = np.sin (theta/2) * np.exp( - phi/2.*1j)
    c = -np.sin (theta/2) * np.exp(phi/2.*1j)
    d = np.cos (theta/2) * np.exp( - phi/2.*1j)

    U = np.array ([[a, b], [c, d]])

    for i in range (it_ranges [0] - 1):

        theta = angles [i + 1, 0]
        phi = angles [i + 1, 1]

        a = np.cos (theta/2) * np.exp(phi/2.*1j)
        b = np.sin (theta/2) * np.exp( - phi/2.*1j)
        c = -np.sin (theta/2) * np.exp(phi/2.*1j)
        d = np.cos (theta/2) * np.exp( - phi/2.*1j)
        
        U = np.kron (U, np.array ([[a, b], [c, d]])).astype ('complex64')
            
        
    U_asmatr = np.asmatrix (U)
    rot_H = U_asmatr.getH ()
    rot_H = rot_H.getA ()
        
    rot_rho_t = np.dot (np.dot (U, rho_t), rot_H)
    rot_rho_g = np.dot (np.dot (U, rho_g), rot_H)
    
    diag_t = np.real (rot_rho_t.diagonal ())
    diag_g = np.real (rot_rho_g.diagonal ())
    
    cs_t = np.cumsum (diag_t)
    cs_g = np.cumsum (diag_g)
    result_t = np.zeros (N_m)
    result_g = np.zeros (N_m)
    for n in range (N_m):
        r = np.random.uniform (0, 1)
        for j in range (len (cs_t)):
            if r < cs_t [j]:
                break
        result_t [n] = j
    for n in range (N_m):
        r = np.random.uniform (0, 1)
        for j in range (len (cs_g)):
            if r < cs_g [j]:
                break
        result_g [n] = j
        
    #un_t = np.unique (result_t)
    dim = len (diag_t) # frequency vector size
    un_t = np.zeros (dim)
    #freq_t = []
    for el in result_t:
        #counts = np.count_nonzero (result_t == el)
        un_t [int (el)] += 1
        #freq_t.append (counts)
        
    #un_g = np.unique (result_g)
    un_g = np.zeros (dim)
    #freq_g = []
    for el in result_g:
        #counts = np.count_nonzero (result_g == el)
        un_g [int (el)] += 1
        #freq_g.append (counts)

        
    distr_t = np.array (un_t) / N_m
    distr_g = np.array (un_g) / N_m
    print (un_t)
    print (un_g)
    
    ent = entropy (distr_g, distr_t)
    
    return ent