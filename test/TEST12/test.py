import numpy as np
import pandas as pd
from pyscf import dft
import torch
from torch.autograd import Variable

from pathlib import Path
import time
import sys

from snn_3h import SNN
from file_input import input_files, logger, trans_time


# basic parameters in test
hidden = [40, 40, 40]

dataset_name = sys.argv[1]
method = sys.argv[2]
class UnsupportedFunctionalError(Exception):  
    pass  
  
if method == "b3lyp":  
    dft_hyb = 0.2  
elif method == "pbe":  
    dft_hyb = 0.0  
else:  
    print("we have not support this functional, please try again.")  
    raise UnsupportedFunctionalError("Unsupported functional encountered. Please use 'b3lyp' or 'pbe'.")

use_cuda = False
basis_path = "/your/path/to/put/this/code/ML-B3LYP-p/"
atm_path = basis_path + "test/TEST12/test_data_set/atoms/"
mol_path = basis_path + "test/TEST12/test_data_set/" + dataset_name + "/"

# fixed basic parameters
device = torch.device("cuda" if use_cuda else "cpu")
lamda = 1e-5
beta = 1.5
mindiv = 1e-10
verbose = 3
input_dim = 6
n_net = (
    hidden[0] * (input_dim + 1)
    + hidden[1] * (hidden[0] + 1)
    + hidden[2] * (hidden[1] + 2)
    + 1
)

info_file_name = "info_" + dataset_name + ".log"
model_name = "model.log"

atm_obj = []
atm_name = []
atm_grids = []

mol_obj = []
mol_name = []
mol_grids = []

Atoms_energy = np.full((18,), np.nan)
TE_energy = pd.DataFrame(columns=["mole", "energy"])
AE_energy = pd.DataFrame(columns=["mole", "energy"])


def atomic_number_mapping():
    """
    Returns a dictionary mapping atomic symbols to their atomic numbers.
    """
    return {
        "H": 0,
        "He": 1,
        "Li": 2,
        "Be": 3,
        "B": 4,
        "C": 5,
        "N": 6,
        "O": 7,
        "F": 8,
        "Ne": 9,
        "Na": 10,
        "Mg": 11,
        "Al": 12,
        "Si": 13,
        "O": 7,
        "F": 8,
        "Ne": 9,
        "Na": 10,
        "Mg": 11,
        "Al": 12,
        "Si": 13,
        "P": 14,
        "S": 15,
        "Cl": 16,
        "Ar": 17,
        "1": 0,  # Hypothetical code for Hydrogen  
        "2": 1,  # Hypothetical code for Helium  
        "3": 2,  # Hypothetical code for Lithium  
        "4": 3,  # Hypothetical code for Beryllium  
        "5": 4,  # Hypothetical code for Boron  
        "6": 5,  # Hypothetical code for Carbon  
        "7": 6,  # Hypothetical code for Nitrogen  
        "8": 7,  # Hypothetical code for Oxygen  
        "9": 8,  # Hypothetical code for Fluorine  
        "10": 9, # Hypothetical code for Neon  
        "11": 10, # Hypothetical code for Sodium  
        "12": 11, # Hypothetical code for Magnesium  
        "13": 12, # Hypothetical code for Aluminum  
        "14": 13, # Hypothetical code for Silicon  
        "15": 14, # Hypothetical code for Phosphorus  
        "16": 15, # Hypothetical code for Sulfur  
        "17": 16, # Hypothetical code for Chlorine  
        "18": 17  # Hypothetical code for Argon  
    }
  
def find_common_atomic_numbers(directory):  
    all_atom_types = set()  
    atom_number_map = atomic_number_mapping()  
    directory_path = Path(directory)  
  
    for file_path in directory_path.glob('*.xyz'):  
        try:  
            with file_path.open('r') as file:  
                for line in file:  
                    atom_type = line.strip().split(maxsplit=1)[0]  
                    if atom_type in atom_number_map:  
                        all_atom_types.add(atom_number_map[atom_type])  
        except Exception as e:  
            print(f"Error reading file {file_path}: {e}")  
  
    if not all_atom_types:  
        return []  
  
    sorted_atomic_numbers = sorted(list(all_atom_types))  
    return sorted_atomic_numbers


def eval_xc_ml(xc_code, rho, spin, relativity=0, deriv=2, verbose=None):
    if spin != 0:
        rho1 = rho[0]
        rho2 = rho[1]
        rho01, dx1, dy1, dz1, lap1, tau1 = rho1[:6]
        rho02, dx2, dy2, dz2, lap2, tau2 = rho2[:6]
        gamma1 = dx1**2 + dy1**2 + dz1**2
        gamma2 = dx2**2 + dy2**2 + dz2**2
        gamma12 = dx1 * dx2 + dy1 * dy2 + dz1 * dz2
    else:
        rho0, dx, dy, dz, lap, tau = rho[:6]
        gamma1 = gamma2 = gamma12 = (dx**2 + dy**2 + dz**2) * 0.25
        rho01 = rho02 = rho0 * 0.5
        tau1 = tau2 = tau * 0.5

    ml_in_ = np.concatenate(
        (
            rho01.reshape((-1, 1)),
            rho02.reshape((-1, 1)),
            gamma1.reshape((-1, 1)),
            gamma12.reshape((-1, 1)),
            gamma2.reshape((-1, 1)),
            tau1.reshape((-1, 1)),
            tau2.reshape((-1, 1)),
        ),
        axis=1,
    )
    ml_in = Variable(torch.Tensor(ml_in_).to(device), requires_grad=True)
    exc_ml_out = s_nn(ml_in)

    ml_exc = exc_ml_out.data[:, 0]
    exc_ml = torch.dot(exc_ml_out[:, 0], ml_in[:, 0] + ml_in[:, 1])
    exc_ml.backward()
    grad = ml_in.grad.cpu().data.numpy()
    grad[np.isnan(grad)] = 0
    grad[np.isinf(grad)] = 0

    if spin != 0:
        vrho_ml = np.hstack((grad[:, 0].reshape((-1, 1)), grad[:, 1].reshape((-1, 1))))
        vgamma_ml = np.hstack(
            (
                grad[:, 2].reshape((-1, 1)),
                grad[:, 3].reshape((-1, 1)),
                grad[:, 4].reshape((-1, 1)),
            )
        )
        vlap_ml = np.zeros((ml_in.shape[0], 2))
        vtau_ml = np.hstack((grad[:, 5].reshape((-1, 1)), grad[:, 6].reshape((-1, 1))))
    else:
        vrho_ml = (grad[:, 0] + grad[:, 1]) / 2
        vgamma_ml = (grad[:, 2] + grad[:, 4] + grad[:, 3]) / 4
        vlap_ml = np.zeros(ml_in.shape[0])
        vtau_ml = (grad[:, 5] + grad[:, 6]) / 2

    # Mix with existing functionals
    dft_xc = dft.libxc.eval_xc(method, rho, spin, relativity, deriv, verbose)
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


# definition of functional #
def cal_atm_te():
    global Atoms_energy

    atm_err_num = 0

    time_wall_st = time.time()
    time_cpu_st = time.process_time()
    logger("Atoms calculation starts", info_file_name)
    for i in range(N_atm):
        info_file = open(info_file_name, "a+")
        print("Atom No.", i + 1, file=info_file)
        info_file = open(info_file_name, "a+")

        if atm_obj[i].spin == 0:
            mlpbe = dft.RKS(atm_obj[i])
        else:
            mlpbe = dft.UKS(atm_obj[i])

        grid = atm_grids[i]
        mlpbe = mlpbe.define_xc_(eval_xc_ml, "MGGA", hyb=dft_hyb)
        mlpbe.grids = grid
        mlpbe.max_cycle = 200

        try:
            e_ml_tot = mlpbe.kernel()
        except Exception as e:
            print(f"Error occurs in this atom SCF test. {e}", file=info_file)
            atm_err_num += 1
            info_file.close()
            continue

        TE_delta = e_ml_tot * 627.51
        Atoms_energy[atm_exist[i]] = TE_delta
        print("TE of ", atm_name[i], " : ", TE_delta, file=info_file)
        print("\n\n", file=info_file)

        info_file.close()

    info_file = open(info_file_name, "a+")
    print("Number of unconverged atoms: ", atm_err_num, file=info_file)
    print("\n\n", file=info_file)

    time_wall_en = time.time()
    time_cpu_en = time.process_time()

    time_wall_h, time_wall_m, time_wall_s = trans_time(time_wall_st, time_wall_en)
    time_cpu_h, time_cpu_m, time_cpu_s = trans_time(time_cpu_st, time_cpu_en)
    print(
        "SCF Wall times: ",
        time_wall_h,
        " h, ",
        time_wall_m,
        " min ",
        time_wall_s,
        " s .\n\n\n\n",
        file=info_file,
    )
    print(
        "SCF CPU times: ",
        time_cpu_h,
        " h, ",
        time_cpu_m,
        " min ",
        time_cpu_s,
        " s .\n\n\n\n",
        file=info_file,
    )

    print("Atoms TE:", file=info_file)
    print(Atoms_energy, file=info_file)
    print("\n\n\n\n", file=info_file)

    info_file.close()

    return 0


def cal_testset_ae():
    global AE_energy
    global TE_energy

    mol_err_num = 0

    time_wall_st = time.time()
    time_cpu_st = time.process_time()
    logger("Mols calculation starts", info_file_name)
    for i in range(N_mol):
        info_file = open(info_file_name, "a+")
        print("Mole No.", i + 1, file=info_file)
        info_file = open(info_file_name, "a+")

        if mol_obj[i].spin == 0:
            mlpbe = dft.RKS(mol_obj[i])
        else:
            mlpbe = dft.UKS(mol_obj[i])

        grid = mol_grids[i]
        mlpbe = mlpbe.define_xc_(eval_xc_ml, "MGGA", hyb=dft_hyb)
        mlpbe.grids = grid
        #mlpbe.diis_space = 12
        #mlpbe = scf.addons.smearing_(mlpbe, sigma=.01, method='fermi')
        mlpbe.max_cycle = 200
        #e_ml_tot = mlpbe.kernel()

        try:
            e_ml_tot = mlpbe.kernel()
            isconv = mlpbe.converged
        except Exception as e:
            print(f"Error occurs in this mol SCF test. {e}", file=info_file)
            mol_err_num += 1
            info_file.close()
            continue

        TE_delta = e_ml_tot * 627.51
        AE_delta = -TE_delta
        for i_a in range(mol_obj[i].natm):
            AE_delta += Atoms_energy[mol_obj[i]._atm[i_a][0] - 1]

        TE_energy = TE_energy.append(
            {"mole": mol_name[i], "energy": TE_delta}, ignore_index=True
        )
        AE_energy = AE_energy.append(
            {"mole": mol_name[i], "energy": AE_delta}, ignore_index=True
        )
        print("TE of ", mol_name[i], " : ", TE_delta, file=info_file)
        print("AE of ", mol_name[i], " : ", AE_delta, file=info_file)
        print("Is converged?", isconv, file=info_file)
        print("\n\n", file=info_file)

        info_file.close()

    info_file = open(info_file_name, "a+")
    print("Number of unconverged molecules: ", mol_err_num, file=info_file)
    print("\n\n", file=info_file)

    time_wall_en = time.time()
    time_cpu_en = time.process_time()

    time_wall_h, time_wall_m, time_wall_s = trans_time(time_wall_st, time_wall_en)
    time_cpu_h, time_cpu_m, time_cpu_s = trans_time(time_cpu_st, time_cpu_en)
    print(
        "SCF Wall times: ",
        time_wall_h,
        " h, ",
        time_wall_m,
        " min ",
        time_wall_s,
        " s .\n\n\n\n",
        file=info_file,
    )
    print(
        "SCF CPU times: ",
        time_cpu_h,
        " h, ",
        time_cpu_m,
        " min ",
        time_cpu_s,
        " s .\n\n\n\n",
        file=info_file,
    )

    info_file.close()

    return 0


if __name__ == "__main__":
    # begin calculation
    open(info_file_name, "w").close()

    logger("Program starts.", info_file_name)
    atm_exist = find_common_atomic_numbers(mol_path)
    logger("\n\nloading atoms: \n\n", info_file_name)
    N_atm = input_files(atm_path, atm_obj, atm_name, atm_grids, info_file_name, obj_limit = atm_exist)
    logger("\n\nloading mols in " + dataset_name + ": \n\n", info_file_name)
    N_mol = input_files(mol_path, mol_obj, mol_name, mol_grids, info_file_name)
    logger("\n\n\n\n", info_file_name)

    s_nn = SNN(input_dim, hidden, lamda, beta, use_cuda)
    snn_param = s_nn.state_dict()
    s_nn.to(device)

    pp = torch.load(model_name)
    s_nn.load_state_dict(pp)
    cal_atm_te()
    cal_testset_ae()

    AE_energy.to_csv("ae_" + dataset_name + ".log", index=False, sep="\t")
    TE_energy.to_csv("te_" + dataset_name + ".log", index=False, sep="\t")

    logger("Program ends.", info_file_name)
