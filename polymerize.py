import numpy as np
from scipy.spatial import distance
import sys
from sklearn.decomposition import PCA
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fnames = []
times = []
for i in range(1, len(sys.argv)-1):
    if i%2==1:
        fnames.append(sys.argv[i])
    else:
        times.append(int(sys.argv[i]))

seq = []
for i in range(len(times)):
    seq = seq + [i]*times[i]
seq = np.array(seq, dtype='int')

all_restypes = []
all_names = []
all_resid = []
all_xyz = []
all_tails = []
all_heads =[]

def center(coords):
    coords = np.array(coords, dtype='float')
    COM = np.mean(coords, axis=0)
    coords = coords - COM
    return coords

def extend(vec, ext):
    norm = np.linalg.norm(vec)
    return vec/norm*(norm+ext)

def calc_angle(a, b, c):
    #Calculates the angle formed by a-b-c
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    if cosine_angle-1<0.001 and cosine_angle-1>-0.001:
        angle = 0.0
    else:
        angle = np.arccos(cosine_angle)
    return angle

def rot_Y(coords, angle):
    rot_mat = np.array([ [np.cos(angle), 0, np.sin(angle)], \
                        [0,     1,      0], \
                        [-np.sin(angle), 0, np.cos(angle)]])
    coords = np.dot(rot_mat, coords.T).T
    return coords

def rot_Z(coords, angle):
    rot_mat = np.array([ [np.cos(angle), -np.sin(angle), 0], \
                        [np.sin(angle),     np.cos(angle), 0], \
                        [0,         0,        1]])
    coords = np.dot(rot_mat, coords.T).T
    return coords

def D3plot(xyz):
    fig=plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()

def align_2_X(coords, tail_ndx, head_ndx):
    coords = coords - coords[tail_ndx]
    xyz_tail = np.array([0,0,0])

    if head_ndx == -1:
        xyz_head = np.mean(coords, axis = 0)
    else:
        xyz_head = coords[head_ndx].flatten()
    O = np.zeros(3)
    xy = np.zeros(3)
    xy[0:2] = xyz_head[0:2]
    theta = calc_angle(xyz_head, O, xy)
    coords = rot_Y(coords, theta)

    if head_ndx == -1:
        xyz_head = np.mean(coords, axis = 0)
    else:
        xyz_head = coords[head_ndx].flatten()
    O = np.zeros(3)
    x = np.zeros(3)
    x[0] = xyz_head[0]
    theta = calc_angle(xyz_head, O, x)
    coords = rot_Z(coords, theta)
    return coords

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

    tail = np.array(tail).flatten()
    head = np.array(head).flatten()
    names = np.array(names)
    resid = np.array(resid, dtype='int')
    restypes = np.array(restypes)
    ndx_tail = np.where(np.logical_and(names == tail[0], resid == int(tail[1])))[0]
    ndx_head = np.where(np.logical_and(names == head[0], resid == int(head[1])))[0]
    if not ndx_head:
        ndx_head = -1
    xyz = center(np.array(xyz, dtype="float"))
    xyz = align_2_X(xyz, ndx_tail, ndx_head)
    return xyz, names, resid, restypes, ndx_tail, ndx_head

def write_pdb_block(atname_func, res_name_func, xyz_func, resnum, atnum, out_filename):
    #Writes the information of one atom in a pdb file
    xyz_func=np.round(xyz_func, decimals=4)
    coords=open(out_filename, 'a')
    coords.write('ATOM'.ljust(6))
    coords.write(str(atnum).rjust(5))
    coords.write(' ' + str(atname_func).ljust(4))
    coords.write(' '+str(res_name_func).ljust(3))
    coords.write('  '+str(resnum).rjust(4))
    coords.write('    ' + str(round(xyz_func[0],3)).rjust(8))
    coords.write(str(round(xyz_func[1],3)).rjust(8))
    coords.write(str(round(xyz_func[2],3)).rjust(8)+"\n")
    coords.close()

for i in range(len(fnames)):
    lig_info = read_mol2(fnames[i])
    all_xyz.append(lig_info[0]), all_names.append(lig_info[1]), all_resid.append(lig_info[2]), all_restypes.append(lig_info[3]), all_tails.append(lig_info[4]), all_heads.append(lig_info[5])

pol_restypes = all_restypes[0]
pol_names = all_names[0]
pol_resid = all_resid[0]
pol_xyz = all_xyz[0]
N_prev = 0
for i in range(1, len(seq)):
    last_head = N_prev + all_heads[seq[i-1]]
    pol_resid = np.append(pol_resid, all_resid[seq[i]]+pol_resid[-1])
    pol_names = np.append(pol_names, all_names[seq[i]])
    pol_restypes = np.append(pol_restypes, all_restypes[seq[i]])
    moved_xyz = all_xyz[seq[i]] + extend(pol_xyz[last_head], 1.6)
    pol_xyz = np.append(pol_xyz, moved_xyz, axis=0)
    N_prev += len(all_names[seq[i]])

at = 0
pdb = open(sys.argv[-1], "w")
pdb.close()
for i in range(len(pol_names)):
    at+=1
    write_pdb_block(pol_names[i], pol_restypes[i], pol_xyz[i], pol_resid[i], at, sys.argv[-1])
