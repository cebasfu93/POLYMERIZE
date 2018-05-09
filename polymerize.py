import numpy as np
from scipy.spatial import distance
import sys

fnames = []
times = []
for i in range(1, len(sys.argv)):
    if i%2==1:
        fnames.append(sys.argv[i])
    else:
        times.append(sys.argv[i])

all_restypes = []
all_names = []
all_resid = []
all_xyz = []
all_tails = []
all_heads =[]

def read_mol2(fname):
    ATOM = False
    xyz, names, restypes, resid = [], [], [], []
    mol2 = np.genfromtxt(fname, delimiter = "\n", dtype="str")
    for i in range(len(mol2)):
        if "@<TRIPOS>BOND" in mol2[i]:
            break
        elif ATOM:
            at_info = mol2[i].split()
            xyz.append(at_info[2:5])
            names.append(at_info[1])
            restypes.append(at_info[7])
            resid.append(at_info[6])
        elif "@<TRIPOS>ATOM" in mol2[i]:
            ATOM = True

    tail, head = [], []
    for i in range(len(mol2)):
        if "@<TRIPOS>HEADTAIL" in mol2[i]:
            tail.append(mol2[i+1].split())
            head.append(mol2[i+2].split())
    return np.array(xyz, dtype="float"), np.array(names), np.array(resid), np.array(restypes), np.array(tail), np.array(head)

for i in range(len(fnames)):
    lig_info = read_mol2(fnames[i])
    all_xyz.append(lig_info[0]), all_names.append(lig_info[1]), all_resid.append(lig_info[2]), all_restypes.append(lig_info[3]), all_tails.append(lig_info[4]), all_heads.append(lig_info[5])
