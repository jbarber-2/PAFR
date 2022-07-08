import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt

elements = []
coords = []
with open('adj_lig_lib.sdf', 'r') as f:
    at_num = 0
    e = []
    c = []
    for line in f.readlines():
        if 'V2000' in line:
            at_num = int(line.split()[0])
        if at_num != 0:
            if 'V2000' not in line:
                xyz = [float(i) for i in (line.split()[:3])]
                c.append(xyz)
                atom_type = line.split()[3]
                at_num -= 1
                e.append(atom_type)
            if at_num == 0:
                elements.append(e)
                coords.append(c)
                e = []
                c = []

def number_type(elements):
    periodic_table = {'I','Br','Cl','S','F','O','N','C','H'}
    pres = ['I','Br','S','F','O','N','C','H']
    num_type={}
    for element in periodic_table:
        type_num = 0
        for molecule in elements:
            if element in molecule:
                type_num += 1
                if element not in pres:
                    pres.append(element)
        num_type[element] = type_num
    element = list(num_type.keys())
    number = list(num_type.values())
    plt.bar(element, number)
    plt.xlabel('Elements')
    plt.ylabel('Number of Molecules')
    plt.title('Number of Molecules with Specific Element')
    plt.show()

atoms = []
for molecule in elements:
    atoms.append(len(molecule))

a = np.array(atoms)


def atom_numb(a):
    mean_a = np.mean(a)
    max_a = np.max(a)
    min_a = np.min(a)

    print('The Mean Number of Atoms:', mean_a)
    print('The Maximum Number of Atoms:', max_a)
    print('The Minimum Number of Atoms:', min_a)

    plt.hist(a, 50, range=(0, 100))
    plt.xlabel('Number Of Atoms')
    plt.ylabel('Number of Molecules')
    plt.title('Distribution of Atoms per Ligand')
    plt.show()

zkq = np.loadtxt('energy.txt')
zkp = np.loadtxt('energy5zkp.txt')

def atom_en(zkq, zkp ):
    plt.hist(zkp, 50, range = (-14, -1), histtype = 'step', label = '5zkp')
    plt.hist(zkq, 50, range =(-14, -1), histtype = 'step', label = '5zkq' )
    plt.xlabel('Energy (kcal/mol)')
    plt.ylabel('Number of Molecules')
    plt.title('Distribution of Energy')
    plt.legend()
    plt.show()

error_5zkq = np.loadtxt('error_1.txt')
error_5zkp = np.loadtxt('error_5zkp.txt')

def test_error_8000(error_5zkq, error_5zkp):
    plt.hist(error_5zkp, 50, range = (0, 5), histtype = 'step', label = '5zkp')
    plt.hist(error_5zkq, 50, range =(0, 5), histtype = 'step', label = '5zkq' )
    plt.text(3, 200,'5zkp MAE = 0.504 kcal/mol')
    plt.text(3, 175,'5zkq MAE = 0.482 kcal/mol')
    plt.xlabel('Energy (kcal/mol)')
    plt.ylabel('Number of Molecules')
    plt.title('Test Set Error with 8000 Training')
    plt.legend()
    plt.show()

error_5zkq_1000 = np.loadtxt('error_5zkq_1000.txt')
error_5zkp_1000 = np.loadtxt('error_5zkp_1000.txt')

def test_error_1000(error_5zkq_1000, error_5zkp_1000):
    plt.hist(error_5zkp_1000, 50, range = (0, 5), histtype = 'step', label = '5zkp')
    plt.hist(error_5zkq_1000, 50, range =(0, 5), histtype = 'step', label = '5zkq' )
    plt.text(3, 700,'5zkp MAE = 0.655 kcal/mol')
    plt.text(3, 550,'5zkq MAE = 0.630 kcal/mol')
    plt.xlabel('Energy (kcal/mol)')
    plt.ylabel('Number of Molecules')
    plt.title('Test Set Error with 1000 Training')
    plt.legend()
    plt.show()

pred_energy = np.loadtxt('pred_energy.txt')
pred_energy_5zkp = np.loadtxt('pred_energy_5zkp.txt')

def pred_dist(pred_energy, pred_energy_5zkp):
    plt.hist(pred_energy_5zkp, 50, range = (-14, -1), histtype = 'step', label = '5zkp')
    plt.hist(pred_energy, 50, range =(-14, -1), histtype = 'step', label = '5zkq' )
    plt.xlabel('Energy (kcal/mol)')
    plt.ylabel('Number of Molecules')
    plt.title('Distribution of Predicted Ligands')
    plt.legend()
    plt.show()

pred_error_5zkq = np.loadtxt('pred_error_5zkq.txt')
pred_error_5zkp = np.loadtxt('pred_error_5zkp.txt')

def pred_error(pred_error_5zkq, pred_error_5zkp):
    plt.hist(pred_error_5zkp, 50, range = (0, 5), histtype = 'step', label = '5zkp')
    plt.hist(pred_error_5zkq, 50, range =(0, 5), histtype = 'step', label = '5zkq' )
    plt.text(3, 30,'5zkp MAE = 0.504 kcal/mol')
    plt.text(3, 25,'5zkq MAE = 0.482 kcal/mol')
    plt.xlabel('Energy (kcal/mol)')
    plt.ylabel('Number of Molecules')
    plt.title('Distribution of Top 1000 Predicted Error')
    plt.legend()
    plt.show()

number_type(elements)
atom_numb(a)
number_type(elements[0:10000])
atom_en(zkq, zkp)
test_error_8000(error_5zkq, error_5zkp)
test_error_1000(error_5zkq_1000, error_5zkp_1000)
pred_error(pred_error_5zkq, pred_error_5zkp)
pred_dist(pred_energy, pred_energy_5zkp)