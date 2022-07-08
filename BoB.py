from collections import defaultdict
import numpy as np
from scipy.spatial.distance import pdist, squareform

elements = []
coords = []
with open('adj_lig_lib.sdf', 'r') as f:
    at_num = 0
    e = []
    c = []
    for line in f.readlines():
        if 'V2000' in line:  # reads the document for V2000 which is at the beginning at every molecule
            at_num = int(line.split()[0])  # the first element in V2000 contains the number of atoms
        if at_num != 0:  # if atom number does not equal 0 runs the loop
            if 'V2000' not in line:
                xyz = [float(i) for i in (line.split()[:3])]  # gets the xyz coords of atom
                c.append(xyz)
                atom_type = line.split()[3]  # gets the atoms element
                at_num -= 1  # subtract 1 from the atom number
                e.append(atom_type)
            if at_num == 0:  # when all atoms info has been pulled, the info will be appended to a new list and clears the original
                elements.append(e)
                coords.append(c)
                e = []
                c = []


def bag_init():
    return {'II': [], 'IBr': [], 'BrI': [], 'ICl': [], 'ClI': [], 'IF': [], 'FI': [], 'IS': [], 'SI': [], 'IO': [],
            'OI': [], 'IN': [], 'NI': [], 'IC': [],
            'CI': [], 'IH': [], 'BrBr': [], 'BrCl': [], 'ClBr': [], 'BrF': [], 'FBr': [], 'BrS': [], 'SBr': [],
            'BrO': [], 'OBr': [],
            'BrN': [], 'NBr': [], 'BrC': [], 'CBr': [], 'BrH': [], 'HBr': [], 'ClCl': [], 'ClF': [], 'FCl': [],
            'ClS': [],
            'SCl': [], 'ClO': [], 'OCl': [], 'ClN': [], 'NCl': [], 'ClC': [], 'CCl': [], 'ClH': [],
            'FF': [], 'SF': [], 'FS': [], 'FO': [], 'OF': [], 'FN': [], 'NF': [], 'FC': [], 'CF': [], 'FH': [],
            'SS': [], 'SO': [], 'OS': [], 'SN': [], 'NS': [], 'SC': [], 'CS': [], 'SH': [],
            'OO': [], 'ON': [], 'NO': [], 'OC': [], 'CO': [], 'OH': [],
            'NN': [], 'NC': [], 'CN': [], 'NH': [],
            'CC': [], 'CH': [],
            'HH': []}


bag_size = {'II': 0, 'BIr': 0, 'CIl': 0, 'FI': 0, 'IS': 0, 'IO': 0, 'IN': 0, 'CI': 0, 'HI': 0,
            'BBrr': 0, 'BClr': 0, 'BFr': 0, 'BSr': 0, 'BOr': 0, 'BNr': 0, 'BCr': 0, 'BHr': 0,
            'CCll': 0, 'CFl': 0, 'CSl': 0, 'COl': 0, 'CNl': 0, 'CCl': 0, 'CHl': 0,
            'FF': 0, 'FS': 0, 'FO': 0, 'FN': 0, 'CF': 0, 'FH': 0,
            'SS': 0, 'OS': 0, 'NS': 0, 'CS': 0, 'HS': 0,
            'OO': 0, 'NO': 0, 'CO': 0, 'HO': 0,
            'NN': 0, 'CN': 0, 'HN': 0,
            'CC': 0, 'CH': 0,
            'HH': 0
            }

periodic_table = {'I': [53, 126.9045],
                  'Br': [35, 79.904],
                  'Cl': [17, 35.453],
                  'S': [16, 32.065],
                  'F': [9, 18.9984],
                  'O': [8, 15.9994],
                  'N': [7, 14.0067],
                  'C': [6, 12.0107],
                  'H': [1, 1.0079]}

all_bob = []


def col_mat(mole, molc):
    natoms = len(molc)
    full_CM = np.zeros((natoms, natoms))  # creates an empty matrix that is the size of the number of atoms by the atoms

    pos = []
    Z = []

    for atom in mole:  # iterates through atoms in molecule getting their type and position
        Z.append(periodic_table[atom][0])
    for xyz in molc:
        pos.append(xyz)  # gets the position of atoms in space

    pos = np.array(pos, dtype=float)
    Z = np.array(Z, dtype=float)

    tiny = 1e-20  # sets a minimum value so not dividing by 0
    dm = pdist(pos)  # pairwise distances (goes from 3D to 1D)

    coulomb_matrix = np.outer(Z, Z) / (squareform(dm) + tiny)  # runs atom through coulombs values to get values
    full_CM[0:natoms, 0:natoms] = coulomb_matrix  # adds these values to the coulombs matrix
    return full_CM


def bob(mole, molc, cm):
    bag_dict = bag_init()
    a = []
    for atom in mole:  # pulls type of atoms and appends to a
        a.append(atom)
    cm_len = len(a)  # defines number of molecules
    for i in range(cm_len - 1):
        for j in range(i + 1, cm_len):  # goes through
            bag = a[i] + a[j]
            bag_dict[bag].append(cm[i, j])
    return bag_dict


def comb_bob(bag_dict):
    out = defaultdict(list)
    for (k, v) in bag_dict.items():
        out[''.join(sorted(k))].extend(v)  # Sorts the bags alphabetically and combines the ones that are the same
    return out


def sorting(out):
    temp_dict = {}
    for bag_key, bag in out.items():  # Sorts BoB values from largest to smallest in each bag
        sorted_bag = sorted(bag, reverse=True)
        temp_dict[bag_key] = sorted_bag
    return temp_dict


def counter():
    for molecule in all_bob:
        for bag_key, bag in molecule.items():
            if len(bag) > bag_size[bag_key]:  # sees if there are more bonds in a specific bag compared to other bags
                bag_size[bag_key] = len(bag)  # if there are more bonds then record that new number in bag size
    return None  # If smaller than continue


def zero_pad():
    k = 0
    for bag_dict in all_bob:  # iterates through all bob in all molecules
        temp_dict = {}
        for bag_key, bag in bag_dict.items():
            new_bag = bag
            if len(bag) < bag_size[bag_key]:  # looks if there are less number of bonds in bag
                new_bag = bag + (bag_size[bag_key] - len(bag)) * [
                    0]  # if there are less then will add 0's to that bag til same size of larges bag
            temp_dict[bag_key] = new_bag
        all_bob[k] = temp_dict
        k = k + 1
    return None


def stacking():
    final_data = []
    for bag_dict in all_bob:
        stack = []
        for bag_key, bag_value in bag_dict.items():  # adds bags values to new dictionary
            stack.append(bag_value)
        stack = np.array(stack, dtype=object)  # converts bag values into an array
        flat_stack = np.hstack(stack)  # stacks arrays in sequence horizontally
        final_data.append(flat_stack)  # adds to final dataset
    return final_data


_cm = []
for k in range(len(elements)):
    cm = col_mat(elements[k], coords[k])
    bd = bob(elements[k], coords[k], cm)
    z = comb_bob(bd)
    sorted_bag_dict = sorting(z)
    all_bob.append(sorted_bag_dict)
counter()
zero_pad()
final_b = stacking()

np.save('BoB',final_b)
