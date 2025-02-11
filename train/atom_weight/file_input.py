import numpy as np
from pyscf import gto, dft
import pandas as pd
from natsort import natsorted, ns
from os import listdir

# basic parameters in test
verbose = 1
basis = "def2tzvpd"
mindiv = 1e-10

def input_files(path, num_list, ene_file, obj_list, nelec_list, grids_list, tot_ene_list, file_name):
    spin = pd.DataFrame([],columns=['spin'])
    spin['spin'] = np.loadtxt(path + 'multi-file', dtype = int) - 1
    tot_ene = pd.read_csv(ene_file, sep = '\t', header = None, usecols = [7], names = ['e_tot'])
    dist = natsorted(listdir(path), alg=ns.PATH)
    tmp_file = open(file_name, "a+")

    i = 0
    for file in dist:
        if file.endswith((".xyz")):
            if i not in num_list:
                i += 1
                continue
            
            print("loading: ", i + 1, file = tmp_file)
            coord_file = path + file
            m = gto.Mole()
            m.verbose = verbose
            m.atom = open(coord_file)
            m.charge = 0
            m.spin = int(spin.loc[i, "spin"])
            m.basis = basis
            m.max_memory = 10000
            m.build(dump_input = False)
            obj_list.append(m)
            nelec_list.append(m.nelectron)
            grid = dft.gen_grid.Grids(m)
            grid.level = 4
            grid.build()
            grids_list.append(grid)
            tot_ene_list.append(float(tot_ene['e_tot'].values[i + 1]))
            
            i += 1

    tmp_file.close()

    return 0

def load_ccsd_t_data(path, num_list, ccsd_t_list, file_name, rho_limit):
    dist = natsorted(listdir(path), alg=ns.PATH)

    i = 0
    for file in dist:
        if file.endswith((".xyz")):
            if i not in num_list:
                i += 1
                continue
            
            data_file = path + file.rstrip(".xyz") + ".train"
            dt, w, ra, rb = np.loadtxt(data_file, delimiter='\t', unpack = True)
            ccsd_t_list.append(np.array([dt, w, ra, rb]))
            
            i += 1
    
    return 0

def logger(string, file_name):
    tmp_file = open(file_name, "a+")
    print(string, file = tmp_file)
    tmp_file.close()

    return 0
