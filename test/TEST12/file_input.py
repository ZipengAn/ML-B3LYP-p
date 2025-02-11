import numpy as np
from pyscf import gto, dft
import pandas as pd
from natsort import natsorted, ns
from os import listdir

# basic parameters in test
verbose = 1
basis = "def2tzvpd"
mindiv = 1e-10

def input_files(path, obj_list, mol_name_list, grids_list, file_name, obj_limit = None):
    mol_data = pd.DataFrame([],columns=['charge', 'spin'])
    mol_data['charge'] = np.loadtxt(path + 'charge-file', dtype = int)
    mol_data['spin'] = np.loadtxt(path + 'multi-file', dtype = int) - 1
    dist = natsorted(listdir(path), alg=ns.PATH)
    tmp_file = open(file_name, "a+")

    i = 0
    for file in dist:
        if file.endswith((".xyz")):
            if obj_limit is not None:
                if i not in obj_limit:
                    i += 1
                    continue
                
            print("loading: ", file, file = tmp_file)
            coord_file = path + file
            m = gto.Mole()
            m.verbose = verbose
            m.atom = open(coord_file)
            m.charge = int(mol_data.loc[i, "charge"])
            m.spin = int(mol_data.loc[i, "spin"])
            m.basis = basis
            m.max_memory = 10000
            m.build(dump_input = False)
            obj_list.append(m)

            mol_name_list.append(file[4:-4])

            grid = dft.gen_grid.Grids(m)
            grid.level = 4
            grid.build()
            grids_list.append(grid)

            i += 1

    N_obj = len(obj_list)

    tmp_file.close()

    return N_obj

def logger(string, file_name):
    tmp_file = open(file_name, "a+")
    print(string, file = tmp_file)
    tmp_file.close()

    return 0

def trans_time(t_st, t_en):
    t_de = t_en - t_st
    th = t_de // 3600
    tm = (t_de - th * 3600) // 60
    ts = t_de - th * 3600 - tm * 60
    return th, tm, ts
