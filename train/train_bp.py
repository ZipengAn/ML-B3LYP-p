import numpy as np
from pyscf import dft
from pyscf.dft import numint
import torch
from torch.autograd import Variable
from torch.optim import Adam
import pandas as pd
import time
import sys

from snn_3h import SNN, load_param
from file_input import input_files, load_ccsd_t_data, logger


# basic parameters in test
k = []
hidden = [40, 40, 40]

use_cuda = False
basis_path = '/your/path/to/put/this/code/ML-B3LYP-p/'

distance_limit = 1e-6

const1 = 0
const2 = 0

n_rand = 10
n_scf = 500
n_postscf = 200

lr_bp = float(sys.argv[1])

converged_tol_list = [1e-12, 1e-11, 1e-10, 1e-9]
diis_levelshift_list = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

atm_num_tr = np.arange(18)
mol_num_tr = [0, 2, 7, 8, 9, 10, 12, 13, 14, 15, 35]
mol_num_val = [1, 3, 4, 5, 6, 11, 16, 17, 18, 19]
mol_num_pre = np.arange(148)

# fixed basic parameters
device = torch.device("cuda" if use_cuda else "cpu")
lamda = 1e-5
beta = 1.5
mindiv = 1e-10
verbose = 1
n_k = len(k)
input_dim = 6 + n_k
n_net = hidden[0] * (input_dim + 1) + hidden[1] * (hidden[0] + 1) + hidden[2] * (hidden[1] + 2) + 1

s_nn = SNN(input_dim, hidden, lamda, beta, use_cuda)
snn_param = s_nn.state_dict()
s_nn.to(device)

i_cycle = 0
i_change = 0
loss_tr_list = []
loss_val_list = []
epoch_list = []

info_file_name = "info_file_sa.log"
file_x_init_name = "sa_best_x_init.log"

atm_path = basis_path + 'training_data_set/atoms/'
atm_ene_file = basis_path + 'ene_info/ene_atom_cc_training_out.log'

mol_path = basis_path + 'training_data_set/mols/'
mol_ene_file = basis_path + 'ene_info/ene_mol_cc_training_out.log'

atm_data_load = pd.read_csv(atm_ene_file, sep = '\t', header = None, usecols = [7], names = ['e_tot'])
mol_data_load = pd.read_csv(mol_ene_file, sep = '\t', header = None, usecols = [7], names = ['e_tot'])

def electron_nuclear_potential(grid, mol, ra, rb, rho_min = 1e-10):
    coords = grid.coords
    ngrids = coords.shape[0]
    ene_nuc = np.zeros(ngrids)
    rho = ra + rb
    for ia in range(mol.natm):
        Z = mol.atom_charge(ia)
        alpha = Z / np.power(rho_min, 2)
        beta = -Z / rho_min
        coord = mol.atom_coord(ia)
        r_diff = coords - coord
        distance = np.linalg.norm(r_diff, axis=1)
        mask = distance < rho_min
        ene_nuc[mask] += (alpha * distance[mask] + beta) * rho[mask]
        ene_nuc[~mask] += -Z / distance[~mask] * rho[~mask]
    return ene_nuc

def hyb_exc_cal(r_a, phi, nu):
    '''
    '''
    r1 = 2 * r_a
    r2 = -np.einsum('ij,kl->iklj', r1, r1, optimize=True)/2
    Pr = np.einsum('ijkl,ri,rj->rkl',r2, phi, phi, optimize=True)
    e_all_coul = np.einsum('rij,rij->r',Pr,nu, optimize=True)
    e_xc = (0.5 * e_all_coul).reshape(-1, 1)

    return e_xc

def uhyb_exc_cal(r_a, r_b, phi, nu):
    '''
    '''
    r1 = r_a + r_b
    dm2aa = -np.einsum('ij,kl->iklj', r_a, r_a, optimize=True)
    dm2bb = -np.einsum('ij,kl->iklj', r_b, r_b, optimize=True)
    r2 = dm2aa + dm2bb
    Pr = np.einsum('ijkl,ri,rj->rkl',r2, phi, phi, optimize=True)
    e_all_coul = np.einsum('rij,rij->r', Pr, nu, optimize=True)
    e_xc = (0.5 * e_all_coul).reshape(-1, 1)

    return e_xc

def dft_cal(mymol, mydft, grid):
    '''
    Calculate the features and labels of delta E_XC on grids.
    
    parameters:
    *** need to add ***
    
    results:
    *** need to add ***
    
    '''
    ao_value = numint.eval_ao(mymol, grid.coords, deriv=2)
    phi = ao_value[0]
    nu = mymol.intor('int1e_grids_sph', grids=grid.coords)

    rdm1_dft = mydft.make_rdm1()
    rdm1_dft_a = rdm1_dft / 2.0

    rho_dft = np.einsum('ri,ij,rj->r',phi,rdm1_dft,phi)
    e_coul_dft = .5 * np.einsum('rij,r,ij->r',nu,rho_dft,rdm1_dft)

    rho_dft_a = numint.eval_rho(mymol, ao_value, rdm1_dft_a, xctype='meta-GGA')
    rho_dft_b = rho_dft_a
    rho_dft = rho_dft_a + rho_dft_b
    t_dft_a = rho_dft_a[5]
    t_dft_b = rho_dft_b[5]
    t_dft = t_dft_a + t_dft_b

    e_ext_dft = electron_nuclear_potential(grid, mymol, rho_dft_a[0], rho_dft_b[0], rho_min = distance_limit)

    # features on grids
    ## electron density
    rho_a = rho_dft_a[0]
    rho_b = rho_a
    ## derivatives of electron density
    gamma_a = np.power(rho_dft_a[1].reshape(-1,1),2)+np.power(rho_dft_a[2].reshape(-1,1),2)+np.power(rho_dft_a[3].reshape(-1,1),2)
    gamma_b = gamma_a
    gamma_ab = gamma_a

    # labels on grids
    e_xc_dft_kernel = dft.libxc.eval_xc('b3lyp', rho_dft)[0]
    e_xc_dft = (e_xc_dft_kernel * rho_dft[0]).flatten() + hyb_exc_cal(rdm1_dft_a, phi, nu).flatten() * 0.2
    e_dft = t_dft + e_ext_dft + e_coul_dft + e_xc_dft
    
    return e_dft, rho_a, rho_b, gamma_a, gamma_ab, gamma_b, t_dft_a, t_dft_b

def udft_cal(mymol, mydft, grid):
    '''
    Calculate the features and labels of delta E_XC on grids.
    
    parameters:
    *** need to add ***
    
    results:
    *** need to add ***
    
    '''
    ao_value = numint.eval_ao(mymol, grid.coords, deriv=2)
    phi = ao_value[0]
    nu = mymol.intor('int1e_grids_sph', grids=grid.coords)

    rdm1_dft_ao = mydft.make_rdm1()
    rdm1_dft_a = rdm1_dft_ao[0]
    rdm1_dft_b = rdm1_dft_ao[1]
    rdm1_dft = rdm1_dft_a + rdm1_dft_b

    rho_dft = np.einsum('ri,ij,rj->r',phi,rdm1_dft,phi)
    e_coul_dft = .5 * np.einsum('rij,r,ij->r',nu,rho_dft,rdm1_dft)

    rho_dft_a = numint.eval_rho(mymol, ao_value, rdm1_dft_a, xctype='meta-GGA')
    rho_dft_b = numint.eval_rho(mymol, ao_value, rdm1_dft_b, xctype='meta-GGA')
    rho_dft = rho_dft_a + rho_dft_b
    t_dft_a = rho_dft_a[5]
    t_dft_b = rho_dft_b[5]
    t_dft = t_dft_a + t_dft_b

    e_ext_dft = electron_nuclear_potential(grid, mymol, rho_dft_a[0], rho_dft_b[0], rho_min = distance_limit)

    # features on grids
    ## electron density
    rho_a = rho_dft_a[0]
    rho_b = rho_dft_b[0]
    ## derivatives of electron density
    gamma_a = np.power(rho_dft_a[1].reshape(-1,1),2)+np.power(rho_dft_a[2].reshape(-1,1),2)+np.power(rho_dft_a[3].reshape(-1,1),2)
    gamma_b = np.power(rho_dft_b[1].reshape(-1,1),2)+np.power(rho_dft_b[2].reshape(-1,1),2)+np.power(rho_dft_b[3].reshape(-1,1),2)
    gamma_ab = np.multiply(rho_dft_a[1].reshape(-1,1),rho_dft_b[1].reshape(-1,1))+np.multiply(rho_dft_a[2].reshape(-1,1),rho_dft_b[2].reshape(-1,1))+np.multiply(rho_dft_a[3].reshape(-1,1),rho_dft_b[3].reshape(-1,1))

    # labels on grids
    e_xc_dft_kernel = dft.libxc.eval_xc('b3lyp', (rho_dft_a, rho_dft_b), spin = mymol.spin)[0]
    e_xc_dft = (e_xc_dft_kernel * rho_dft[0]).flatten() + uhyb_exc_cal(rdm1_dft_a, rdm1_dft_b, phi, nu).flatten() * 0.2
    e_dft = t_dft + e_ext_dft + e_coul_dft + e_xc_dft

    return e_dft, rho_a, rho_b, gamma_a, gamma_ab, gamma_b, t_dft_a, t_dft_b

def Rm_cal(k, rhos, weight, coords):
    N = rhos.shape[0]
    weight1 = weight.reshape(N, 1)
    rhos1 = rhos.reshape(N, 1)
    Ne = np.dot(rhos, weight)   # the numble of electrons  ()= (grids,) \cdot (grids, )
    k1 = np.power((np.sum(k * coords * k * coords * rhos1 * weight1 / Ne, axis = 0) -
        np.square(np.sum(k * coords * rhos1 * weight1 / Ne, axis = 0))), -0.5)  # (3,)

    ksin1 = np.sin(np.sum(k1 * coords, axis=1)) # ksin1 is a scalar but have multiple dimension determined by grids.  (grids,) = (grids,3) * (3,)
    kcos1 = np.cos(np.sum(k1 * coords, axis=1))
    ksin1w = ksin1 * weight    # \sin(\vec{k} \cdot \vec{r})d\vec{r}  (grids,) = (grids,)(grids,)
    kcos1w = kcos1 * weight
    fac1sin = np.sum(rhos * ksin1w)   #\int \rho \sin(\vec{k} \cdot \vec{r})d\vec{r}  here,we get a number.  ()
    fac1cos = np.sum(rhos * kcos1w)
    Rm = ksin1 * fac1cos - kcos1 * fac1sin  # (grids,)

    return Rm

def dRm_cal(k, Rgra, rhos, weight, coords):
    N = rhos.shape[0]
    weight1 = weight.reshape(N, 1)
    rhos1 = rhos.reshape(N, 1)
    Rgra1 = Rgra.reshape(N,)
    Ne = np.dot(rhos, weight)

    k1 = np.power((np.sum(k * coords * k * coords * rhos1 * weight1 / Ne, axis = 0) -
        np.square(np.sum(k * coords * rhos1 * weight1 / Ne, axis = 0))), -0.5)  # (3,)
    ksin1 = np.sin(np.sum(k1 * coords, axis=1)) # ksin1 is a scalar but have multiple dimension determined by grids.  (grids,) = (grids,3) * (3,)
    kcos1 = np.cos(np.sum(k1 * coords, axis=1))
    ksin1w = ksin1 * weight    # \sin(\vec{k} \cdot \vec{r})d\vec{r}  (grids,) = (grids,)(grids,)
    kcos1w = kcos1 * weight
    
    Rgradfac1sin = np.sum(Rgra1 * ksin1w)    #     () =   (grids,)(grids,).sum= (grids,).sum
    Rgradfac1cos = np.sum(Rgra1 * kcos1w)
    dR1 = ksin1 * Rgradfac1cos - kcos1 * Rgradfac1sin   # (grids,)

    # here, we calculate variations of km from \rho
    pat1k1 = -0.5 * np.power(k1, -3/2)   # (3,)
    pat2k1 = coords*coords/Ne - 2* np.sum(coords*rhos1*weight1/Ne,axis=0)*coords/Ne  # right term = (grids,3)(grids,1)(grids,1).sum(axis=1)(grids,3)=(3,)(grids,3)=(grids,3)
    r_sin = coords*ksin1.reshape(N,1)#  \vec{r}sin(\vec{k} \cdot \vec{r})d\    (grids,3)(grids,1)=(grids,3)
    r_cos = coords*kcos1.reshape(N,1)
    r_sin_rho = r_sin * rhos1   # (grids,3)(grids,1) =(grids,3)
    r_cos_rho = r_cos * rhos1
    r_sin_rho_int = np.sum(r_sin_rho * weight1, axis=0)  #(3, )  (grids,3)(grids,1).sum(axis=0)
    r_cos_rho_int = np.sum(r_cos_rho * weight1, axis=0)
    sin_rho_int = np.sum(ksin1w * rhos)  #()=[(grids,)(grids,)].sum=(grids,).sum
    cos_rho_int = np.sum(kcos1w * rhos)
    vec1 = pat1k1 * pat2k1  #(grids,3)
    vec2 = Rgradfac1cos * r_cos_rho_int + \
           Rgradfac1sin * r_sin_rho_int - \
           np.sum(Rgra1.reshape(N, 1)*r_cos*weight1, axis=0)*cos_rho_int - \
           np.sum(Rgra1.reshape(N, 1) * r_sin * weight1, axis=0) * sin_rho_int
    dR10 = vec1 * vec2  #(grids,3)
    dR11 = np.sum(dR10, axis=1)  #(grids,)

    return dR1 + dR11

def eval_xc_ml(xc_code, rho, weight, coords, spin, relativity=0, deriv=2, verbose=None):
    if spin!=0:
        rho1 = rho[0]
        rho2 = rho[1]
        rho01, dx1, dy1, dz1, lap1, tau1 = rho1[:6]
        rho02, dx2, dy2, dz2, lap2, tau2 = rho2[:6]
        gamma1 = dx1**2+dy1**2+dz1**2
        gamma2 = dx2**2+dy2**2+dz2**2
        gamma12 = dx1*dx2+dy1*dy2+dz1*dz2
    else:
        rho0, dx, dy, dz, lap, tau = rho[:6]
        gamma1 = gamma2 = gamma12 = (dx**2+dy**2+dz**2)*0.25
        rho01 = rho02 = rho0 * 0.5
        tau1 = tau2 = tau * 0.5

    rhos = rho01 + rho02
    N_elec = np.sum(rhos * weight)
    Rm = np.zeros(shape = (weight.shape[0], n_k))
    for i_k in range(n_k):
        Rm[:, i_k] = Rm_cal(k[i_k], rhos, weight, coords)

    ml_in_ = np.concatenate((rho01.reshape((-1, 1)), rho02.reshape((-1, 1)),\
                             gamma1.reshape((-1, 1)), gamma12.reshape((-1, 1)), gamma2.reshape((-1, 1)),\
                             tau1.reshape((-1, 1)), tau2.reshape((-1, 1)), Rm),axis=1)
    ml_in = Variable(torch.Tensor(ml_in_).to(device), requires_grad = True)
    exc_ml_out = s_nn(ml_in, N_elec)

    ml_exc = exc_ml_out.data[:,0]
    exc_ml = torch.dot(exc_ml_out[:,0], ml_in[:,0] + ml_in[:,1])
    exc_ml.backward()
    grad = ml_in.grad.cpu().data.numpy()
    grad[np.isnan(grad)] = 0
    grad[np.isinf(grad)] = 0

    dRm = np.zeros(shape = (weight.shape[0], n_k))
    for i_k in range(n_k):
        dRm[:, i_k] = dRm_cal(k[i_k], grad[:, 7 + i_k], rhos, weight, coords)

    if spin!=0:
        vrho_ml = np.hstack(((grad[:,0] + np.sum(dRm, axis = 1)).reshape((-1,1)),(grad[:,1] + np.sum(dRm, axis = 1)).reshape((-1,1))))
        vgamma_ml = np.hstack((grad[:,2].reshape((-1,1)),grad[:,3].reshape((-1,1)),grad[:,4].reshape((-1,1))))
        vlap_ml = np.zeros((ml_in.shape[0], 2))
        vtau_ml = np.hstack((grad[:,5].reshape((-1,1)),grad[:,6].reshape((-1,1))))
    else:
        vrho_ml = (grad[:,0] + grad[:,1]) / 2 + np.sum(dRm, axis = 1)
        vgamma_ml = (grad[:,2] + grad[:,4] + grad[:,3]) / 4
        vlap_ml = np.zeros(ml_in.shape[0])
        vtau_ml = (grad[:,5] + grad[:,6]) / 2
    
    # Mix with existing functionals
    dft_xc = dft.libxc.eval_xc('b3lyp', rho, spin, relativity, deriv, verbose)
    dft_exc = np.array(dft_xc[0])
    dft_vrho = np.array(dft_xc[1][0])
    dft_vgamma = np.array(dft_xc[1][1])
    dft_vlap = np.array(dft_xc[1][2])
    dft_vtau = np.array(dft_xc[1][3])
    
    exc = dft_exc + ml_exc.cpu().numpy()
    vrho = dft_vrho + vrho_ml
    vgamma = dft_vgamma + vgamma_ml
    if dft_vlap == None:
        vlap = vlap_ml
    else:
        vlap = dft_vlap + vlap_ml
    if dft_vtau == None:
        vtau = vtau_ml
    else:
        vtau = dft_vtau + vtau_ml
    
    vxc = (vrho, vgamma, vlap, vtau)
    
    return exc, vxc, None, None


# begin calculation
open(info_file_name, "w").close()

logger("Program starts.", info_file_name)

# load mole model & grid data
i_a_t = 0
i_m_t = 0
i_m_v = 0
i_m_p = 0

best_loss_tr = torch.inf
best_del_ene_tr = torch.inf

atm_tr = []
mol_tr = []
mol_val = []
mol_pre = []

atm_ccsd_t_tr = []
mol_ccsd_t_tr = []
mol_ccsd_t_val = []

atm_tot_ene_tr = []
mol_tot_ene_tr = []
mol_tot_ene_val = []
mol_tot_ene_pre = []

atm_grids_tr = []
mol_grids_tr = []
mol_grids_val = []
mol_grids_pre = []

atm_nelec_tr = []
mol_nelec_tr = []
mol_nelec_val = []
mol_nelec_pre = []

atm_den_tr = []
mol_den_tr = []
mol_den_val = []

atm_ene_den_tr = []
mol_ene_den_tr = []
mol_ene_den_val = []

logger("\n\nloading atoms in training set: \n\n", info_file_name)
input_files(atm_path, atm_num_tr, atm_ene_file, atm_tr, atm_nelec_tr, atm_grids_tr, atm_tot_ene_tr, info_file_name)
load_ccsd_t_data(atm_path, atm_num_tr, atm_ccsd_t_tr, info_file_name, distance_limit)

logger("\n\nloading mols in training set: \n\n", info_file_name)
input_files(mol_path, mol_num_tr, mol_ene_file, mol_tr, mol_nelec_tr, mol_grids_tr, mol_tot_ene_tr, info_file_name)
load_ccsd_t_data(mol_path, mol_num_tr, mol_ccsd_t_tr, info_file_name, distance_limit)

logger("\n\nloading mols in validation set: \n\n", info_file_name)
input_files(mol_path, mol_num_val, mol_ene_file, mol_val, mol_nelec_val, mol_grids_val, mol_tot_ene_val, info_file_name)
load_ccsd_t_data(mol_path, mol_num_val, mol_ccsd_t_val, info_file_name, distance_limit)

logger("\n\nloading mols in predict validation set: \n\n", info_file_name)
input_files(mol_path, mol_num_pre, mol_ene_file, mol_pre, mol_nelec_pre, mol_grids_pre, mol_tot_ene_pre, info_file_name)

logger("\n\n\n\n", info_file_name)
           
N_atm_tr = len(atm_tr)
N_mol_tr = len(mol_tr)
N_mol_val = len(mol_val)
N_mol_pre = len(mol_pre)

atm_rel_ene = np.zeros((18,))
mol_rel_ene = np.zeros((N_mol_pre,))

atm_rel_ene_ml = np.zeros((18,))
mol_rel_ene_ml = np.zeros((N_mol_pre,))

for i in range(18):
    if i == 0:
        atm_rel_ene[i] = float(atm_data_load['e_tot'].values[i + 1])
    elif i < 8:
        atm_rel_ene[i + 1] = float(atm_data_load['e_tot'].values[i + 1])
    elif i < 16:
        atm_rel_ene[i + 2] = float(atm_data_load['e_tot'].values[i + 1])
    elif i == 16:
        atm_rel_ene[1] = float(atm_data_load['e_tot'].values[i + 1])
    elif i == 17:
        atm_rel_ene[9] = float(atm_data_load['e_tot'].values[i + 1])
    else:
        raise("error")
    
for i in range(N_mol_pre):
    for i_a in range(mol_pre[i].natm):
        mol_rel_ene[mol_num_pre[i]] += atm_rel_ene[mol_pre[i]._atm[i_a][0] - 1]
    mol_rel_ene[mol_num_pre[i]] -= float(mol_data_load['e_tot'].values[mol_num_pre[i] + 1])

# definition of functional #
def loss_postscf_atm():
    Loss = 0
    E_loss = 0
    D_loss = 0
    E_del_sum = 0
    D_del_sum = 0

    global atm_rel_ene_ml

    time_st = time.time()

    logger("Train: atoms calculation starts", info_file_name)
    for i in range(N_atm_tr):
        info_file = open(info_file_name, "a+")
            
        grid = atm_grids_tr[i]

        ml_input = atm_den_tr[i]
        rhos = (ml_input[:,0] + ml_input[:,1]).detach().numpy()
        gamma1 = ml_input[:,2].detach().numpy()
        gamma12 = ml_input[:,3].detach().numpy()
        gamma2 = ml_input[:,4].detach().numpy()

        e_dft = atm_ene_den_tr[i]

        Rm = np.zeros(shape = (grid.weights.shape[0], n_k))
        for i_k in range(n_k):
            Rm[:, i_k] = Rm_cal(k[i_k], rhos, grid.weights, grid.coords)

        ccsd_t_tmp = torch.Tensor(atm_ccsd_t_tr[i]).to(device)
        nelec_tmp = atm_nelec_tr[i]
        
        d_ml = ml_input[:,0] + ml_input[:,1]
        e_ml_ = s_nn(ml_input, nelec_tmp)
        e_ml = e_ml_.flatten()
        e_ml = torch.Tensor(e_ml * d_ml)
        e_cc = ccsd_t_tmp[0]
        wt_cc = ccsd_t_tmp[1]
        d_cc = ccsd_t_tmp[2] + ccsd_t_tmp[3]

        e_delta = e_cc - torch.Tensor(e_dft) - e_ml
        e_loss = torch.sum(e_delta * wt_cc) * 627.51
        E_delta = torch.sum(e_delta * wt_cc) * 627.51
        d_delta = torch.abs(d_cc - d_ml)
        d_loss = torch.sum(d_delta * wt_cc) * 1e3
        D_delta = torch.sum(d_delta * wt_cc)
        E_loss += torch.abs(e_loss)
        D_loss += d_loss
        tot_loss = torch.abs(e_loss) + const2 * d_loss
        Loss += tot_loss
        E_del_sum += E_delta
        D_del_sum += D_delta

        e_ml_tot = atm_tot_ene_tr[i] - E_delta.detach().numpy() / 627.51
        if atm_num_tr[i] == 0:
            atm_rel_ene_ml[0] = e_ml_tot
        elif atm_num_tr[i] < 8:
            atm_rel_ene_ml[atm_num_tr[i] + 1] = e_ml_tot
        elif atm_num_tr[i] < 16:
            atm_rel_ene_ml[atm_num_tr[i] + 2] = e_ml_tot
        elif atm_num_tr[i] == 16:
            atm_rel_ene_ml[1] = e_ml_tot
        elif atm_num_tr[i] == 17:
            atm_rel_ene_ml[9] = e_ml_tot
        else:
            raise("error")
        
        print("Train: Atom No.", i + 1, file = info_file)
        print("Train: E_post-SCF_loss: ", e_loss.cpu().detach().numpy(), file = info_file)
        print("Train: D_post-SCF_loss: ", d_loss.cpu().detach().numpy(), file = info_file)
        print("Train: Total_post-SCF_loss: ", tot_loss.cpu().detach().numpy(), file = info_file)
        print("Train: E_post-SCF_delta: ", E_delta.cpu().detach().numpy(), file = info_file)
        print("Train: D_post-SCF_delta: ", D_delta.cpu().detach().numpy(), file = info_file)
        print("\n\n", file = info_file)

        info_file.close()

    e_loss_avg = E_loss / N_atm_tr
    d_loss_avg = D_loss / N_atm_tr
    loss_avg = Loss / N_atm_tr
    e_del_avg = E_del_sum / N_atm_tr
    d_del_avg = D_del_sum / N_atm_tr

    info_file = open(info_file_name, "a+")

    print("\n\n", file = info_file)
    print("Train: Final_atoms_E_post-SCF_loss: ", e_loss_avg.cpu().detach().numpy(), file = info_file)
    print("Train: Final_atoms_D_post-SCF_loss: ", d_loss_avg.cpu().detach().numpy(), file = info_file)
    print("Train: Final_atoms_tot_post-SCF_loss: ", loss_avg.cpu().detach().numpy(), file = info_file)
    print("Train: Final_atoms_E_post-SCF_delta: ", e_del_avg.cpu().detach().numpy(), file = info_file)
    print("Train: Final_atoms_D_post-SCF_delta: ", d_del_avg.cpu().detach().numpy(), file = info_file)
    print("\n\n", file = info_file)

    time_en = time.time()
    time_de = time_en - time_st
    time_h = time_de // 3600
    time_m = (time_de - time_h * 3600) // 60
    time_s = time_de - time_h * 3600 - time_m * 60
    print("SCF times: ", time_h, " h, ", time_m, " min ", time_s, " s .\n\n\n\n", file = info_file)

    info_file.close()

    return e_loss_avg, d_loss_avg, loss_avg, e_del_avg, d_del_avg

def loss_scf_atm():
    Loss = 0
    E_loss = 0
    D_loss = 0
    E_del_sum = 0
    D_del_sum = 0

    den_tmp = []
    ene_den_dft_tmp = []
    global atm_den_tr
    global atm_rel_ene_ml
    global atm_ene_den_tr

    if N_atm_tr == 0:
        return 0., 0., 0., 0., 0.

    time_st = time.time()

    logger("Train: atoms calculation starts", info_file_name)
    for i in range(N_atm_tr):
        info_file = open(info_file_name, "a+")

        if atm_tr[i].spin == 0:
            mlpbe = dft.RKS(atm_tr[i])
        else:
            mlpbe = dft.UKS(atm_tr[i])
            
        grid = atm_grids_tr[i]
        mlpbe = mlpbe.define_xc_(eval_xc_ml, 'MGGA+R', hyb = 0.2)
        mlpbe.grids = grid
        mlpbe.verbose = verbose
        mlpbe.max_cycle = 500
        mlpbe.diis_space = 24
        print("Ori converged: ", mlpbe.converged, file = info_file)

        for ct in converged_tol_list:
            for dls in diis_levelshift_list:
                try:
                    mlpbe.conv_tol = ct
                    mlpbe.level_shift = dls
                    e_ml_tot = mlpbe.kernel()
                    if mlpbe.converged:
                        print("conv_tol: ", ct, file = info_file)
                        print("level_shift: ", dls, file = info_file)
                        print("Converged: ", mlpbe.converged, file = info_file)
                        break
                except Exception as e:
                    print("Error occurs in SCF.", file=info_file)
                    print(f"Exception details: {e}", file=info_file)
                    info_file.close()
                    return torch.inf, torch.inf, torch.inf, np.inf, torch.inf
            if mlpbe.converged: break
        if not mlpbe.converged:
            print("SCF cannot converged.", file = info_file)
            info_file.close()
            return torch.inf, torch.inf, torch.inf, np.inf, torch.inf
        
        if atm_num_tr[i] == 0:
            atm_rel_ene_ml[0] = e_ml_tot
        elif atm_num_tr[i] < 8:
            atm_rel_ene_ml[atm_num_tr[i] + 1] = e_ml_tot
        elif atm_num_tr[i] < 16:
            atm_rel_ene_ml[atm_num_tr[i] + 2] = e_ml_tot
        elif atm_num_tr[i] == 16:
            atm_rel_ene_ml[1] = e_ml_tot
        elif atm_num_tr[i] == 17:
            atm_rel_ene_ml[9] = e_ml_tot
        else:
            raise("error")
        E_delta = (atm_tot_ene_tr[i] - e_ml_tot) * 627.51
            
        if atm_tr[i].spin == 0:
            e_dft, rho01, rho02, gamma1, gamma12, gamma2, tau1, tau2 = dft_cal(atm_tr[i], mlpbe, grid)
        else:
            e_dft, rho01, rho02, gamma1, gamma12, gamma2, tau1, tau2 = udft_cal(atm_tr[i], mlpbe, grid)

        rhos = rho01 + rho02
        Rm = np.zeros(shape = (grid.weights.shape[0], n_k))
        for i_k in range(n_k):
            Rm[:, i_k] = Rm_cal(k[i_k], rhos, grid.weights, grid.coords)

        ml_input_ = np.concatenate((rho01.reshape((-1, 1)), rho02.reshape((-1, 1)),\
                                gamma1.reshape((-1, 1)), gamma12.reshape((-1, 1)), gamma2.reshape((-1, 1)),\
                                tau1.reshape((-1, 1)), tau2.reshape((-1, 1)), Rm),axis=1)
        ml_input = torch.Tensor(ml_input_).to(device)
        ene_den_dft_tmp.append(e_dft)
        den_tmp.append(ml_input)

        ccsd_t_tmp = torch.Tensor(atm_ccsd_t_tr[i]).to(device)
        nelec_tmp = atm_nelec_tr[i]
        
        d_ml = ml_input[:,0] + ml_input[:,1]
        e_ml_ = s_nn(ml_input, nelec_tmp)
        e_ml = e_ml_.flatten()
        e_ml = torch.Tensor(e_ml * d_ml)
        e_cc = ccsd_t_tmp[0]
        wt_cc = ccsd_t_tmp[1]
        d_cc = ccsd_t_tmp[2] + ccsd_t_tmp[3]

        e_delta = e_cc - torch.Tensor(e_dft) - e_ml
        e_loss = torch.sum(e_delta * wt_cc) * 627.51
        d_delta = torch.abs(d_cc - d_ml)
        d_loss = torch.sum(d_delta * wt_cc) * 1e3
        D_delta = torch.sum(d_delta * wt_cc)
        E_loss += torch.abs(e_loss)
        D_loss += d_loss
        tot_loss = torch.abs(e_loss) + const2 * d_loss
        Loss += tot_loss
        E_del_sum += np.abs(E_delta)
        D_del_sum += D_delta
        
        print("Train: Atom No.", i + 1, file = info_file)
        print("Train: E_SCF_loss: ", e_loss.cpu().detach().numpy(), file = info_file)
        print("Train: D_SCF_loss: ", d_loss.cpu().detach().numpy(), file = info_file)
        print("Train: Total_SCF_loss: ", tot_loss.cpu().detach().numpy(), file = info_file)
        print("Train: E_SCF_delta: ", E_delta, file = info_file)
        print("Train: D_SCF_delta: ", D_delta.cpu().detach().numpy(), file = info_file)
        print("\n\n", file = info_file)

        info_file.close()

    e_loss_avg = E_loss / N_atm_tr
    d_loss_avg = D_loss / N_atm_tr
    loss_avg = Loss / N_atm_tr
    e_del_avg = E_del_sum / N_atm_tr
    d_del_avg = D_del_sum / N_atm_tr

    info_file = open(info_file_name, "a+")

    print("\n\n", file = info_file)
    print("Train: Final_atoms_E_SCF_loss: ", e_loss_avg.cpu().detach().numpy(), file = info_file)
    print("Train: Final_atoms_D_SCF_loss: ", d_loss_avg.cpu().detach().numpy(), file = info_file)
    print("Train: Final_atoms_tot_SCF_loss: ", loss_avg.cpu().detach().numpy(), file = info_file)
    print("Train: Final_atoms_E_SCF_delta: ", e_del_avg, file = info_file)
    print("Train: Final_atoms_D_SCF_delta: ", d_del_avg.cpu().detach().numpy(), file = info_file)
    print("\n\n", file = info_file)
    
    atm_den_tr = den_tmp
    atm_ene_den_tr = ene_den_dft_tmp

    time_en = time.time()
    time_de = time_en - time_st
    time_h = time_de // 3600
    time_m = (time_de - time_h * 3600) // 60
    time_s = time_de - time_h * 3600 - time_m * 60
    print("SCF times: ", time_h, " h, ", time_m, " min ", time_s, " s .\n\n\n\n", file = info_file)

    info_file.close()

    return e_loss_avg, d_loss_avg, loss_avg, e_del_avg, d_del_avg

def loss_postscf_mol_tr():
    Loss = 0
    E_loss = 0
    D_loss = 0
    R_loss = 0
    E_del_sum = 0
    D_del_sum = 0

    global mol_rel_ene_ml

    if N_mol_tr == 0:
        return 0., 0., 0., 0., 0., 0.

    time_st = time.time()
        
    logger("Train: mols calculation starts", info_file_name)
    for i in range(N_mol_tr):
        info_file = open(info_file_name, "a+")
            
        grid = mol_grids_tr[i]

        ml_input = mol_den_tr[i]
        rhos = (ml_input[:,0] + ml_input[:,1]).detach().numpy()
        gamma1 = ml_input[:,2].detach().numpy()
        gamma12 = ml_input[:,3].detach().numpy()
        gamma2 = ml_input[:,4].detach().numpy()

        e_dft = mol_ene_den_tr[i]

        Rm = np.zeros(shape = (grid.weights.shape[0], n_k))
        for i_k in range(n_k):
            Rm[:, i_k] = Rm_cal(k[i_k], rhos, grid.weights, grid.coords)

        ccsd_t_tmp = torch.Tensor(mol_ccsd_t_tr[i]).to(device)
        nelec_tmp = mol_nelec_tr[i]
        
        d_ml = ml_input[:,0] + ml_input[:,1]
        e_ml_ = s_nn(ml_input, nelec_tmp)
        e_ml = e_ml_.flatten()
        e_ml = torch.Tensor(e_ml * d_ml)
        e_cc = ccsd_t_tmp[0]
        wt_cc = ccsd_t_tmp[1]
        d_cc = ccsd_t_tmp[2] + ccsd_t_tmp[3]

        e_delta = e_cc - torch.Tensor(e_dft) - e_ml
        e_loss = torch.sum(e_delta * wt_cc) * 627.51
        E_delta = torch.sum(e_delta * wt_cc) * 627.51
        d_delta = torch.abs(d_cc - d_ml)
        d_loss = torch.sum(d_delta * wt_cc) * 1e3
        D_delta = torch.sum(d_delta * wt_cc)

        e_ml_tot = mol_tot_ene_tr[i] - E_delta.detach().numpy() / 627.51
        mol_rel_ene_ml[mol_num_tr[i]] = 0
        for i_a in range(mol_tr[i].natm):
            mol_rel_ene_ml[mol_num_tr[i]] += atm_rel_ene_ml[mol_tr[i]._atm[i_a][0] - 1]
        mol_rel_ene_ml[mol_num_tr[i]] -= e_ml_tot

        R_delta = (mol_rel_ene[mol_num_tr[i]] - mol_rel_ene_ml[mol_num_tr[i]]) * 627.51
        r_loss = np.abs(R_delta)
        E_loss += torch.abs(e_loss)
        D_loss += d_loss
        R_loss += r_loss
        tot_loss = torch.abs(e_loss) + const1 * r_loss + const2 * d_loss
        Loss += tot_loss
        E_del_sum += torch.abs(E_delta)
        D_del_sum += D_delta
        
        print("Train: Mole No.", i + 1, file = info_file)
        print("Train: E_post-SCF_loss: ", e_loss.cpu().detach().numpy(), file = info_file)
        print("Train: D_post-SCF_loss: ", d_loss.cpu().detach().numpy(), file = info_file)
        print("Train: R_post-SCF_loss: ", r_loss, file = info_file)
        print("Train: Total_post-SCF_loss: ", tot_loss.cpu().detach().numpy(), file = info_file)
        print("Train: E_post-SCF_delta: ", E_delta.cpu().detach().numpy(), file = info_file)
        print("Train: D_post-SCF_delta: ", D_delta.cpu().detach().numpy(), file = info_file)
        print("Train: R_post-SCF_delta: ", R_delta, file = info_file)
        print("\n\n", file = info_file)

        info_file.close()

    e_loss_avg = E_loss / N_mol_tr
    d_loss_avg = D_loss / N_mol_tr
    r_loss_avg = R_loss / N_mol_tr
    loss_avg = Loss / N_mol_tr
    e_del_avg = E_del_sum / N_mol_tr
    d_del_avg = D_del_sum / N_mol_tr

    info_file = open(info_file_name, "a+")

    print("\n\n", file = info_file)
    print("Train: Final_mols_E_post-SCF_loss: ", e_loss_avg.cpu().detach().numpy(), file = info_file)
    print("Train: Final_mols_D_post-SCF_loss: ", d_loss_avg.cpu().detach().numpy(), file = info_file)
    print("Train: Final_mols_R_post-SCF_loss: ", r_loss_avg, file = info_file)
    print("Train: Final_mols_tot_post-SCF_loss: ", loss_avg.cpu().detach().numpy(), file = info_file)
    print("Train: Final_mols_E_post-SCF_delta: ", e_del_avg.cpu().detach().numpy(), file = info_file)
    print("Train: Final_mols_D_post-SCF_delta: ", d_del_avg.cpu().detach().numpy(), file = info_file)
    print("\n\n", file = info_file)

    time_en = time.time()
    time_de = time_en - time_st
    time_h = time_de // 3600
    time_m = (time_de - time_h * 3600) // 60
    time_s = time_de - time_h * 3600 - time_m * 60
    print("SCF times: ", time_h, " h, ", time_m, " min ", time_s, " s .\n\n\n\n", file = info_file)

    info_file.close()

    return e_loss_avg, d_loss_avg, r_loss_avg, loss_avg, e_del_avg, d_del_avg

def loss_scf_mol_tr():
    Loss = 0
    E_loss = 0
    D_loss = 0
    R_loss = 0
    E_del_sum = 0
    D_del_sum = 0

    den_tmp = []
    ene_den_dft_tmp = []
    global mol_den_tr
    global mol_rel_ene_ml
    global mol_ene_den_tr

    if N_mol_tr == 0:
        return 0., 0., 0., 0., 0., 0.

    time_st = time.time()
        
    logger("Train: mols calculation starts", info_file_name)
    for i in range(N_mol_tr):
        info_file = open(info_file_name, "a+")

        if mol_tr[i].spin == 0:
            mlpbe = dft.RKS(mol_tr[i])
        else:
            mlpbe = dft.UKS(mol_tr[i])
            
        grid = mol_grids_tr[i]
        mlpbe = mlpbe.define_xc_(eval_xc_ml, 'MGGA+R', hyb = 0.2)
        mlpbe.grids = grid
        mlpbe.verbose = verbose
        mlpbe.max_cycle = 500
        mlpbe.diis_space = 24

        for ct in converged_tol_list:
            for dls in diis_levelshift_list:
                try:
                    mlpbe.conv_tol = ct
                    mlpbe.level_shift = dls
                    e_ml_tot = mlpbe.kernel()
                    if mlpbe.converged:
                        print("conv_tol: ", ct, file = info_file)
                        print("level_shift: ", dls, file = info_file)
                        break
                except Exception as e:
                    print("Error occurs in SCF.", file=info_file)
                    print(f"Exception details: {e}", file=info_file)
                    info_file.close()
                    return torch.inf, torch.inf, np.inf, torch.inf, np.inf, torch.inf
            if mlpbe.converged: break
        if not mlpbe.converged:
            print("SCF cannot converged.", file = info_file)
            info_file.close()
            return torch.inf, torch.inf, np.inf, torch.inf, np.inf, torch.inf

        mol_rel_ene_ml[mol_num_tr[i]] = 0
        for i_a in range(mol_tr[i].natm):
            mol_rel_ene_ml[mol_num_tr[i]] += atm_rel_ene_ml[mol_tr[i]._atm[i_a][0] - 1]
        mol_rel_ene_ml[mol_num_tr[i]] -= e_ml_tot
        E_delta = (mol_tot_ene_tr[i] - e_ml_tot) * 627.51
            
        if mol_tr[i].spin == 0:
            e_dft, rho01, rho02, gamma1, gamma12, gamma2, tau1, tau2 = dft_cal(mol_tr[i], mlpbe, grid)
        else:
            e_dft, rho01, rho02, gamma1, gamma12, gamma2, tau1, tau2 = udft_cal(mol_tr[i], mlpbe, grid)

        rhos = rho01 + rho02
        Rm = np.zeros(shape = (grid.weights.shape[0], n_k))
        for i_k in range(n_k):
            Rm[:, i_k] = Rm_cal(k[i_k], rhos, grid.weights, grid.coords)

        ml_input_ = np.concatenate((rho01.reshape((-1, 1)), rho02.reshape((-1, 1)),\
                                gamma1.reshape((-1, 1)), gamma12.reshape((-1, 1)), gamma2.reshape((-1, 1)),\
                                tau1.reshape((-1, 1)), tau2.reshape((-1, 1)), Rm),axis=1)
        ml_input = torch.Tensor(ml_input_).to(device)
        ene_den_dft_tmp.append(e_dft)
        den_tmp.append(ml_input)

        ccsd_t_tmp = torch.Tensor(mol_ccsd_t_tr[i]).to(device)
        nelec_tmp = mol_nelec_tr[i]
        
        d_ml = ml_input[:,0] + ml_input[:,1]
        e_ml_ = s_nn(ml_input, nelec_tmp)
        e_ml = e_ml_.flatten()
        e_ml = torch.Tensor(e_ml * d_ml)
        e_cc = ccsd_t_tmp[0]
        wt_cc = ccsd_t_tmp[1]
        d_cc = ccsd_t_tmp[2] + ccsd_t_tmp[3]

        e_delta = e_cc - torch.Tensor(e_dft) - e_ml
        e_loss = torch.sum(e_delta * wt_cc) * 627.51
        d_delta = torch.abs(d_cc - d_ml)
        d_loss = torch.sum(d_delta * wt_cc) * 1e3
        D_delta = torch.sum(d_delta * wt_cc)
        R_delta = (mol_rel_ene[mol_num_tr[i]] - mol_rel_ene_ml[mol_num_tr[i]]) * 627.51
        r_loss = np.abs(R_delta)
        E_loss += torch.abs(e_loss)
        D_loss += d_loss
        R_loss += r_loss
        tot_loss = torch.abs(e_loss) + const1 * r_loss + const2 * d_loss
        Loss += tot_loss
        E_del_sum += np.abs(E_delta)
        D_del_sum += D_delta
        
        print("Train: Mole No.", i + 1, file = info_file)
        print("Train: E_SCF_loss: ", e_loss.cpu().detach().numpy(), file = info_file)
        print("Train: D_SCF_loss: ", d_loss.cpu().detach().numpy(), file = info_file)
        print("Train: R_SCF_loss: ", r_loss, file = info_file)
        print("Train: Total_SCF_loss: ", tot_loss.cpu().detach().numpy(), file = info_file)
        print("Train: E_SCF_delta: ", E_delta, file = info_file)
        print("Train: D_SCF_delta: ", D_delta.cpu().detach().numpy(), file = info_file)
        print("Train: R_SCF_delta: ", R_delta, file = info_file)
        print("\n\n", file = info_file)

        info_file.close()

    e_loss_avg = E_loss / N_mol_tr
    d_loss_avg = D_loss / N_mol_tr
    r_loss_avg = R_loss / N_mol_tr
    loss_avg = Loss / N_mol_tr
    e_del_avg = E_del_sum / N_mol_tr
    d_del_avg = D_del_sum / N_mol_tr

    info_file = open(info_file_name, "a+")

    print("\n\n", file = info_file)
    print("Train: Final_mols_E_SCF_loss: ", e_loss_avg.cpu().detach().numpy(), file = info_file)
    print("Train: Final_mols_D_SCF_loss: ", d_loss_avg.cpu().detach().numpy(), file = info_file)
    print("Train: Final_mols_R_SCF_loss: ", r_loss_avg, file = info_file)
    print("Train: Final_mols_tot_SCF_loss: ", loss_avg.cpu().detach().numpy(), file = info_file)
    print("Train: Final_mols_E_SCF_delta: ", e_del_avg, file = info_file)
    print("Train: Final_mols_D_SCF_delta: ", d_del_avg.cpu().detach().numpy(), file = info_file)
    print("\n\n", file = info_file)
    
    mol_den_tr = den_tmp
    mol_ene_den_tr = ene_den_dft_tmp

    time_en = time.time()
    time_de = time_en - time_st
    time_h = time_de // 3600
    time_m = (time_de - time_h * 3600) // 60
    time_s = time_de - time_h * 3600 - time_m * 60
    print("SCF times: ", time_h, " h, ", time_m, " min ", time_s, " s .\n\n\n\n", file = info_file)

    info_file.close()

    return e_loss_avg, d_loss_avg, r_loss_avg, loss_avg, e_del_avg, d_del_avg

def loss_postscf_mol_val():
    Loss = 0
    E_loss = 0
    D_loss = 0
    R_loss = 0
    E_del_sum = 0
    D_del_sum = 0

    global mol_rel_ene_ml

    if N_mol_val == 0:
        return 0., 0., 0., 0., 0., 0.

    time_st = time.time()

    logger("Validation: mols calculation starts", info_file_name)
    for i in range(N_mol_val):
        info_file = open(info_file_name, "a+")
            
        grid = mol_grids_val[i]

        ml_input = mol_den_val[i]
        rhos = (ml_input[:,0] + ml_input[:,1]).detach().numpy()
        gamma1 = ml_input[:,2].detach().numpy()
        gamma12 = ml_input[:,3].detach().numpy()
        gamma2 = ml_input[:,4].detach().numpy()

        e_dft = mol_ene_den_val[i]

        Rm = np.zeros(shape = (grid.weights.shape[0], n_k))
        for i_k in range(n_k):
            Rm[:, i_k] = Rm_cal(k[i_k], rhos, grid.weights, grid.coords)

        ccsd_t_tmp = torch.Tensor(mol_ccsd_t_val[i]).to(device)
        nelec_tmp = mol_nelec_val[i]

        d_ml = ml_input[:,0] + ml_input[:,1]
        e_ml_ = s_nn(ml_input, nelec_tmp)
        e_ml = e_ml_.flatten()
        e_ml = torch.Tensor(e_ml * d_ml)
        e_cc = ccsd_t_tmp[0]
        wt_cc = ccsd_t_tmp[1]
        d_cc = ccsd_t_tmp[2] + ccsd_t_tmp[3]

        e_delta = e_cc - torch.Tensor(e_dft) - e_ml
        e_loss = torch.sum(e_delta * wt_cc) * 627.51
        E_delta = torch.sum(e_delta * wt_cc) * 627.51
        d_delta = torch.abs(d_cc - d_ml)
        d_loss = torch.sum(d_delta * wt_cc) * 1e3
        D_delta = torch.sum(d_delta * wt_cc)

        e_ml_tot = mol_tot_ene_val[i] - E_delta.detach().numpy() / 627.51
        mol_rel_ene_ml[mol_num_val[i]] = 0 
        for i_a in range(mol_val[i].natm):
            mol_rel_ene_ml[mol_num_val[i]] += atm_rel_ene_ml[mol_val[i]._atm[i_a][0] - 1]
        mol_rel_ene_ml[mol_num_val[i]] -= e_ml_tot

        R_delta = (mol_rel_ene[mol_num_val[i]] - mol_rel_ene_ml[mol_num_val[i]]) * 627.51
        r_loss = np.abs(R_delta)
        E_loss += torch.abs(e_loss)
        D_loss += d_loss
        R_loss += r_loss
        tot_loss = torch.abs(e_loss) + const1 * r_loss + const2 * d_loss
        Loss += tot_loss
        E_del_sum += torch.abs(E_delta)
        D_del_sum += D_delta
        
        print("Validation: Mole No.", i + 1, file = info_file)
        print("Validation: E_post-SCF_loss: ", e_loss.cpu().detach().numpy(), file = info_file)
        print("Validation: D_post-SCF_loss: ", d_loss.cpu().detach().numpy(), file = info_file)
        print("Validation: R_post-SCF_loss: ", r_loss, file = info_file)
        print("Validation: Total_post-SCF_loss: ", tot_loss.cpu().detach().numpy(), file = info_file)
        print("Validation: E_post-SCF_delta: ", E_delta.cpu().detach().numpy(), file = info_file)
        print("Validation: D_post-SCF_delta: ", D_delta.cpu().detach().numpy(), file = info_file)
        print("Validation: R_post-SCF_delta: ", R_delta, file = info_file)
        print("\n\n", file = info_file)

        info_file.close()

    e_loss_avg = E_loss / N_mol_val
    d_loss_avg = D_loss / N_mol_val
    r_loss_avg = R_loss / N_mol_val
    loss_avg = Loss / N_mol_val
    e_del_avg = E_del_sum / N_mol_val
    d_del_avg = D_del_sum / N_mol_val

    info_file = open(info_file_name, "a+")

    print("\n\n", file = info_file)
    print("Validation: Final_mols_E_post-SCF_loss: ", e_loss_avg.cpu().detach().numpy(), file = info_file)
    print("Validation: Final_mols_D_post-SCF_loss: ", d_loss_avg.cpu().detach().numpy(), file = info_file)
    print("Validation: Final_mols_R_post-SCF_loss: ", r_loss_avg, file = info_file)
    print("Validation: Final_mols_tot_post-SCF_loss: ", loss_avg.cpu().detach().numpy(), file = info_file)
    print("Validation: Final_mols_E_post-SCF_delta: ", e_del_avg.cpu().detach().numpy(), file = info_file)
    print("Validation: Final_mols_D_post-SCF_delta: ", d_del_avg.cpu().detach().numpy(), file = info_file)
    print("\n\n", file = info_file)

    time_en = time.time()
    time_de = time_en - time_st
    time_h = time_de // 3600
    time_m = (time_de - time_h * 3600) // 60
    time_s = time_de - time_h * 3600 - time_m * 60
    print("SCF times: ", time_h, " h, ", time_m, " min ", time_s, " s .\n\n\n\n", file = info_file)

    info_file.close()

    return e_loss_avg, d_loss_avg, r_loss_avg, loss_avg, e_del_avg, d_del_avg

def loss_scf_mol_val():
    Loss = 0
    E_loss = 0
    D_loss = 0
    R_loss = 0
    E_del_sum = 0
    D_del_sum = 0

    den_tmp = []
    ene_den_dft_tmp = []
    global mol_den_val
    global mol_rel_ene_ml
    global mol_ene_den_val

    if N_mol_val == 0:
        return 0., 0., 0., 0., 0., 0.

    time_st = time.time()

    logger("Validation: mols calculation starts", info_file_name)
    for i in range(N_mol_val):
        info_file = open(info_file_name, "a+")

        if mol_val[i].spin == 0:
            mlpbe = dft.RKS(mol_val[i])
        else:
            mlpbe = dft.UKS(mol_val[i])
            
        grid = mol_grids_val[i]
        mlpbe = mlpbe.define_xc_(eval_xc_ml, 'MGGA+R', hyb = 0.2)
        mlpbe.grids = grid
        mlpbe.verbose = verbose
        mlpbe.max_cycle = 500
        mlpbe.diis_space = 24

        for ct in converged_tol_list:
            for dls in diis_levelshift_list:
                try:
                    mlpbe.conv_tol = ct
                    mlpbe.level_shift = dls
                    e_ml_tot = mlpbe.kernel()
                    if mlpbe.converged:
                        print("conv_tol: ", ct, file = info_file)
                        print("level_shift: ", dls, file = info_file)
                        break 
                except Exception as e:
                    print("Error occurs in SCF.", file=info_file)                      
                    print(f"Exception details: {e}", file=info_file)
                    info_file.close()
                    return torch.inf, torch.inf, np.inf, torch.inf, np.inf, torch.inf  
            if mlpbe.converged: break
        if not mlpbe.converged:
            print("SCF cannot converged.", file = info_file)
            info_file.close()
            return torch.inf, torch.inf, np.inf, torch.inf, np.inf, torch.inf
       
        mol_rel_ene_ml[mol_num_val[i]] = 0 
        for i_a in range(mol_val[i].natm):
            mol_rel_ene_ml[mol_num_val[i]] += atm_rel_ene_ml[mol_val[i]._atm[i_a][0] - 1]
        mol_rel_ene_ml[mol_num_val[i]] -= e_ml_tot
        E_delta = (mol_tot_ene_val[i] - e_ml_tot) * 627.51
            
        if mol_val[i].spin == 0:
            e_dft, rho01, rho02, gamma1, gamma12, gamma2, tau1, tau2 = dft_cal(mol_val[i], mlpbe, grid)
        else:
            e_dft, rho01, rho02, gamma1, gamma12, gamma2, tau1, tau2 = udft_cal(mol_val[i], mlpbe, grid)

        rhos = rho01 + rho02
        Rm = np.zeros(shape = (grid.weights.shape[0], n_k))
        for i_k in range(n_k):
            Rm[:, i_k] = Rm_cal(k[i_k], rhos, grid.weights, grid.coords)

        ml_input_ = np.concatenate((rho01.reshape((-1, 1)), rho02.reshape((-1, 1)),\
                                gamma1.reshape((-1, 1)), gamma12.reshape((-1, 1)), gamma2.reshape((-1, 1)),\
                                tau1.reshape((-1, 1)), tau2.reshape((-1, 1)), Rm),axis=1)
        ml_input = torch.Tensor(ml_input_).to(device)
        ene_den_dft_tmp.append(e_dft)
        den_tmp.append(ml_input)

        ccsd_t_tmp = torch.Tensor(mol_ccsd_t_val[i]).to(device)
        nelec_tmp = mol_nelec_val[i]

        d_ml = ml_input[:,0] + ml_input[:,1]
        e_ml_ = s_nn(ml_input, nelec_tmp)
        e_ml = e_ml_.flatten()
        e_ml = torch.Tensor(e_ml * d_ml)
        e_cc = ccsd_t_tmp[0]
        wt_cc = ccsd_t_tmp[1]
        d_cc = ccsd_t_tmp[2] + ccsd_t_tmp[3]

        e_delta = e_cc - torch.Tensor(e_dft) - e_ml
        e_loss = torch.sum(e_delta * wt_cc) * 627.51
        d_delta = torch.abs(d_cc - d_ml)
        d_loss = torch.sum(d_delta * wt_cc) * 1e3
        D_delta = torch.sum(d_delta * wt_cc)
        R_delta = (mol_rel_ene[mol_num_val[i]] - mol_rel_ene_ml[mol_num_val[i]]) * 627.51
        r_loss = np.abs(R_delta)
        E_loss += torch.abs(e_loss)
        D_loss += d_loss
        R_loss += r_loss
        tot_loss = torch.abs(e_loss) + const1 * r_loss + const2 * d_loss
        Loss += tot_loss
        E_del_sum += np.abs(E_delta)
        D_del_sum += D_delta
        
        print("Validation: Mole No.", i + 1, file = info_file)
        print("Validation: E_SCF_loss: ", e_loss.cpu().detach().numpy(), file = info_file)
        print("Validation: D_SCF_loss: ", d_loss.cpu().detach().numpy(), file = info_file)
        print("Validation: R_SCF_loss: ", r_loss, file = info_file)
        print("Validation: Total_SCF_loss: ", tot_loss.cpu().detach().numpy(), file = info_file)
        print("Validation: E_SCF_delta: ", E_delta, file = info_file)
        print("Validation: D_SCF_delta: ", D_delta.cpu().detach().numpy(), file = info_file)
        print("Validation: R_SCF_delta: ", R_delta, file = info_file)
        print("\n\n", file = info_file)

        info_file.close()

    e_loss_avg = E_loss / N_mol_val
    d_loss_avg = D_loss / N_mol_val
    r_loss_avg = R_loss / N_mol_val
    loss_avg = Loss / N_mol_val
    e_del_avg = E_del_sum / N_mol_val
    d_del_avg = D_del_sum / N_mol_val

    info_file = open(info_file_name, "a+")

    print("\n\n", file = info_file)
    print("Validation: Final_mols_E_SCF_loss: ", e_loss_avg.cpu().detach().numpy(), file = info_file)
    print("Validation: Final_mols_D_SCF_loss: ", d_loss_avg.cpu().detach().numpy(), file = info_file)
    print("Validation: Final_mols_R_SCF_loss: ", r_loss_avg, file = info_file)
    print("Validation: Final_mols_tot_SCF_loss: ", loss_avg.cpu().detach().numpy(), file = info_file)
    print("Validation: Final_mols_E_SCF_delta: ", e_del_avg, file = info_file)
    print("Validation: Final_mols_D_SCF_delta: ", d_del_avg.cpu().detach().numpy(), file = info_file)
    print("\n\n", file = info_file)
    
    mol_den_val = den_tmp
    mol_ene_den_val = ene_den_dft_tmp

    time_en = time.time()
    time_de = time_en - time_st
    time_h = time_de // 3600
    time_m = (time_de - time_h * 3600) // 60
    time_s = time_de - time_h * 3600 - time_m * 60
    print("SCF times: ", time_h, " h, ", time_m, " min ", time_s, " s .\n\n\n\n", file = info_file)

    info_file.close()

    return e_loss_avg, d_loss_avg, r_loss_avg, loss_avg, e_del_avg, d_del_avg

def loss_scf_mol_pre():
    TE_del_sum = 0
    AE_del_sum = 0
    mol_err_num = 0

    global mol_rel_ene_ml

    if N_mol_pre == 0:
        return 0., 0.

    time_st = time.time()

    logger("Predict: mols calculation starts", info_file_name)
    for i in range(N_mol_pre):
        info_file = open(info_file_name, "a+")
        print("Predict: Mole No.", i + 1, file = info_file)

        if mol_num_pre[i] in mol_num_tr or mol_num_pre[i] in mol_num_val:
            print("Already calculated.", file = info_file)
            e_ml_tot = 0
            for i_a in range(mol_pre[i].natm):
                e_ml_tot += atm_rel_ene_ml[mol_pre[i]._atm[i_a][0] - 1]
            e_ml_tot -= mol_rel_ene_ml[mol_num_pre[i]]
        else:
            info_file = open(info_file_name, "a+")

            if mol_pre[i].spin == 0:
                mlpbe = dft.RKS(mol_pre[i])
            else:
                mlpbe = dft.UKS(mol_pre[i])
                
            grid = mol_grids_pre[i]
            mlpbe = mlpbe.define_xc_(eval_xc_ml, 'MGGA+R', hyb = 0.2)
            mlpbe.grids = grid
            mlpbe.max_cycle = 200

            try:
                e_ml_tot = mlpbe.kernel()
            except:
                print("Predict: Error occurs in this mol SCF test.", file = info_file)
                mol_err_num += 1
                info_file.close()
                continue
        
            mol_rel_ene_ml[mol_num_pre[i]] = 0 
            for i_a in range(mol_pre[i].natm):
                mol_rel_ene_ml[mol_num_pre[i]] += atm_rel_ene_ml[mol_pre[i]._atm[i_a][0] - 1]
            mol_rel_ene_ml[mol_num_pre[i]] -= e_ml_tot

        TE_delta = (mol_tot_ene_pre[i] - e_ml_tot) * 627.51
        AE_delta = (mol_rel_ene[mol_num_pre[i]] - mol_rel_ene_ml[mol_num_pre[i]]) * 627.51
        TE_del_sum += np.abs(TE_delta)
        AE_del_sum += np.abs(AE_delta)

        print("Predict: TE_SCF_delta: ", TE_delta, file = info_file)
        print("Predict: AE_SCF_delta: ", AE_delta, file = info_file)
        print("\n\n", file = info_file)

        info_file.close()

    te_del_avg = TE_del_sum / N_mol_pre
    ae_del_avg = AE_del_sum / N_mol_pre

    info_file = open(info_file_name, "a+")

    print("\n\n", file = info_file)
    print("Predict: Final_mols_TE_SCF_delta: ", te_del_avg, file = info_file)
    print("Predict: Final_mols_AE_SCF_delta: ", ae_del_avg, file = info_file)
    print("Predict: Number of unconverged molecules: ", mol_err_num, file = info_file)
    print("\n\n", file = info_file)

    time_en = time.time()
    time_de = time_en - time_st
    time_h = time_de // 3600
    time_m = (time_de - time_h * 3600) // 60
    time_s = time_de - time_h * 3600 - time_m * 60
    print("SCF times: ", time_h, " h, ", time_m, " min ", time_s, " s .\n\n\n\n", file = info_file)

    info_file.close()

    return te_del_avg, ae_del_avg

def loss_postscf_cal():
    ## units in kcal/mol !!!
    ## do not forget to convert SCF energies from PySCF to kcal/mol !!!

    global i_change
    global i_cycle

    info_file = open(info_file_name, "a+")
    i_cycle += 1
    print("Train: Cycle (use post-SCF): ", i_cycle, file = info_file)
    info_file.close()

    elat, dlat, tlat, edat, ddat = loss_postscf_atm()
    if tlat == torch.inf:
        return np.inf
    elmt, dlmt, rlmt, tlmt, edmt, ddmt = loss_postscf_mol_tr()
    if tlmt == torch.inf:
        return np.inf

    e_loss_tr = (elat * N_atm_tr + elmt * N_mol_tr) / (N_atm_tr + N_mol_tr)
    d_loss_tr = (dlat * N_atm_tr + dlmt * N_mol_tr) / (N_atm_tr + N_mol_tr)
    r_loss_tr = rlmt
    tot_loss_tr = e_loss_tr + const1 * r_loss_tr + const2 * d_loss_tr
    e_delta_tr = (edat * N_atm_tr + edmt * N_mol_tr) / (N_atm_tr + N_mol_tr)
    d_delta_tr = (ddat * N_atm_tr + ddmt * N_mol_tr) / (N_atm_tr + N_mol_tr)

    info_file = open(info_file_name, "a+")

    print("\n\n\n\nFinal results:", file = info_file)
    print("Train: Final_E_post-SCF_loss: ", e_loss_tr.cpu().detach().numpy(), file = info_file)
    print("Train: Final_D_post-SCF_loss: ", d_loss_tr.cpu().detach().numpy(), file = info_file)
    print("Train: Final_R_post-SCF_loss: ", r_loss_tr, file = info_file)
    print("Train: Final_tot_post-SCF_loss: ", tot_loss_tr.cpu().detach().numpy(), file = info_file)
    print("Train: Final_E_post-SCF_delta: ", e_delta_tr.cpu().detach().numpy(), file = info_file)
    print("Train: Final_D_post-SCF_delta: ", d_delta_tr.cpu().detach().numpy(), file = info_file)
    print("\n\n\n\n", file = info_file)

    info_file.close()

    if (best_loss_tr > tot_loss_tr):
        elmv, dlmv, rlmv, tlmv, edmv, ddmv = loss_postscf_mol_val()
        if tlmv == torch.inf:
            return np.inf

        if N_mol_val:
            e_loss_val = elmv
            d_loss_val = dlmv
            r_loss_val = rlmv
            tot_loss_val = e_loss_val + const1 * r_loss_val + const2 * d_loss_val
            e_delta_val = edmv
            d_delta_val = ddmv

            tedap = edat

            info_file = open(info_file_name, "a+")

            print("\n\n\n\nFinal results:", file = info_file)
            print("Best: Train: Final_E_post-SCF_loss: ", e_loss_tr.cpu().detach().numpy(), file = info_file)
            print("Best: Train: Final_D_post-SCF_loss: ", d_loss_tr.cpu().detach().numpy(), file = info_file)
            print("Best: Train: Final_R_post-SCF_loss: ", r_loss_tr, file = info_file)
            print("Best: Train: Final_tot_post-SCF_loss: ", tot_loss_tr.cpu().detach().numpy(), file = info_file)
            print("Best: Train: Final_E_post-SCF_delta: ", e_delta_tr.cpu().detach().numpy(), file = info_file)
            print("Best: Train: Final_D_post-SCF_delta: ", d_delta_tr.cpu().detach().numpy(), file = info_file)
            print("\n\n", file = info_file)
            print("Best: Validation: Final_E_post-SCF_loss: ", e_loss_val.cpu().detach().numpy(), file = info_file)
            print("Best: Validation: Final_D_post-SCF_loss: ", d_loss_val.cpu().detach().numpy(), file = info_file)
            print("Best: Validation: Final_R_post-SCF_loss: ", r_loss_val, file = info_file)
            print("Best: Validation: Final_tot_post-SCF_loss: ", tot_loss_val.cpu().detach().numpy(), file = info_file)
            print("Best: Validation: Final_E_post-SCF_delta: ", e_delta_val.cpu().detach().numpy(), file = info_file)
            print("Best: Validation: Final_D_post-SCF_delta: ", d_delta_val.cpu().detach().numpy(), file = info_file)
            print("\n\n", file = info_file)
            print("Best: Predict: Final_atoms_TE_post-SCF_delta: ", tedap, file = info_file)
            print("\n\n\n\n", file = info_file)

            info_file.close()

        print(atm_rel_ene)
        print(atm_rel_ene_ml)
        print(mol_rel_ene)
        print(mol_rel_ene_ml)


    print(tot_loss_tr.cpu().detach().numpy())

    return tot_loss_tr

def loss_scf_cal():
    ## units in kcal/mol !!!
    ## do not forget to convert SCF energies from PySCF to kcal/mol !!!

    global i_change
    global i_cycle
    global best_loss_tr
    global best_del_ene_tr

    info_file = open(info_file_name, "a+")
    i_cycle += 1
    print("Train: Cycle (use SCF): ", i_cycle, file = info_file)
    info_file.close()

    elat, dlat, tlat, edat, ddat = loss_scf_atm()
    if tlat == torch.inf:
        return np.inf
    elmt, dlmt, rlmt, tlmt, edmt, ddmt = loss_scf_mol_tr()
    if tlmt == torch.inf:
        return np.inf

    e_loss_tr = (elat * N_atm_tr + elmt * N_mol_tr) / (N_atm_tr + N_mol_tr)
    d_loss_tr = (dlat * N_atm_tr + dlmt * N_mol_tr) / (N_atm_tr + N_mol_tr)
    r_loss_tr = rlmt
    tot_loss_tr = e_loss_tr + const1 * r_loss_tr + const2 * d_loss_tr
    e_delta_tr = (edat * N_atm_tr + edmt * N_mol_tr) / (N_atm_tr + N_mol_tr)
    d_delta_tr = (ddat * N_atm_tr + ddmt * N_mol_tr) / (N_atm_tr + N_mol_tr)

    info_file = open(info_file_name, "a+")

    print("\n\n\n\nFinal results:", file = info_file)
    print("Train: Final_E_SCF_loss: ", e_loss_tr.cpu().detach().numpy(), file = info_file)
    print("Train: Final_D_SCF_loss: ", d_loss_tr.cpu().detach().numpy(), file = info_file)
    print("Train: Final_R_SCF_loss: ", r_loss_tr, file = info_file)
    print("Train: Final_tot_SCF_loss: ", tot_loss_tr.cpu().detach().numpy(), file = info_file)
    print("Train: Final_E_SCF_delta: ", e_delta_tr, file = info_file)
    print("Train: Final_D_SCF_delta: ", d_delta_tr.cpu().detach().numpy(), file = info_file)
    print("\n\n\n\n", file = info_file)

    info_file.close()

    if (best_loss_tr > tot_loss_tr):
        elmv, dlmv, rlmv, tlmv, edmv, ddmv = loss_scf_mol_val()
        if tlmv == torch.inf:
            return np.inf
        tedmp, aedmp = loss_scf_mol_pre()

        best_loss_tr = tot_loss_tr
        if i_change <= 0:
            best_del_ene_tr = e_delta_tr

        if N_mol_val:
            e_loss_val = elmv
            d_loss_val = dlmv
            r_loss_val = rlmv
            tot_loss_val = e_loss_val + const1 * r_loss_val + const2 * d_loss_val
            e_delta_val = edmv
            d_delta_val = ddmv

            tedap = edat

            info_file = open(info_file_name, "a+")

            print("\n\n\n\nFinal results:", file = info_file)
            print("Best: Train: Final_E_SCF_loss: ", e_loss_tr.cpu().detach().numpy(), file = info_file)
            print("Best: Train: Final_D_SCF_loss: ", d_loss_tr.cpu().detach().numpy(), file = info_file)
            print("Best: Train: Final_R_SCF_loss: ", r_loss_tr, file = info_file)
            print("Best: Train: Final_tot_SCF_loss: ", tot_loss_tr.cpu().detach().numpy(), file = info_file)
            print("Best: Train: Final_E_SCF_delta: ", e_delta_tr, file = info_file)
            print("Best: Train: Final_D_SCF_delta: ", d_delta_tr.cpu().detach().numpy(), file = info_file)
            print("\n\n", file = info_file)
            print("Best: Validation: Final_E_SCF_loss: ", e_loss_val.cpu().detach().numpy(), file = info_file)
            print("Best: Validation: Final_D_SCF_loss: ", d_loss_val.cpu().detach().numpy(), file = info_file)
            print("Best: Validation: Final_R_SCF_loss: ", r_loss_val, file = info_file)
            print("Best: Validation: Final_tot_SCF_loss: ", tot_loss_val.cpu().detach().numpy(), file = info_file)
            print("Best: Validation: Final_E_SCF_delta: ", e_delta_val, file = info_file)
            print("Best: Validation: Final_D_SCF_delta: ", d_delta_val.cpu().detach().numpy(), file = info_file)
            print("\n\n", file = info_file)
            print("Best: Predict: Final_atoms_TE_SCF_delta: ", tedap, file = info_file)
            print("Best: Predict: Final_mols_TE_SCF_delta: ", tedmp, file = info_file)
            print("Best: Predict: Final_mols_AE_SCF_delta: ", aedmp, file = info_file)
            print("\n\n\n\n", file = info_file)

            info_file.close()

        print(atm_rel_ene)
        print(atm_rel_ene_ml)
        print(mol_rel_ene)
        print(mol_rel_ene_ml)

        file_x_tmp_name = "sa_best_x_tmp_" + str(i_change) + ".log"
        torch.save(s_nn.state_dict(), file_x_tmp_name)
        i_change += 1


    print(tot_loss_tr.cpu().detach().numpy())

    return tot_loss_tr

if __name__ == "__main__":
    param = np.zeros((n_net, ))
    pp = load_param(snn_param, input_dim, hidden, param)
    s_nn.load_state_dict(pp)
    loss_scf_cal()

    opt = Adam(s_nn.parameters(), lr = lr_bp)

    for epoch_rand in range(n_rand):
        logger("New random epoch:", info_file_name)
        print("New epoch:", epoch_rand)
        print(s_nn.state_dict())
        param = np.random.randn(n_net, ) * 1e-4
        pp = load_param(snn_param, input_dim, hidden, param)
        s_nn.load_state_dict(pp)
        for epoch_scf in range(n_scf):
            print("New SCF epoch:", epoch_rand, "%", epoch_scf)
            loss_scf_cal()
            for epoch_postscf in range(n_postscf):
                print("New postscf epoch:", epoch_rand, "%", epoch_scf, "%", epoch_postscf)
                opt.zero_grad()
                loss_tr = loss_postscf_cal()
                loss_tr.backward()
                opt.step()
            if epoch_scf == n_scf - 1:
                print("New SCF epoch:", epoch_rand, "%", epoch_scf + 1)
                loss_scf_cal()

    torch.save(s_nn.state_dict(), "model.log")

    logger("Program ends.", info_file_name)
