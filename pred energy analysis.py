import numpy as np

with open('pred_energy_5zkp.txt', 'r') as f:
    en = []
    for energy in f:
        en.append(float(energy))

en_pred = np.array(en)

K = 1000

res = sorted(range(len(en)), key = lambda sub: en[sub])[:K]

final_atoms = []
for atoms in res:
    final_atoms.append(atoms + 10000)

with open('to-run.txt', 'w') as f:
    for item in final_atoms:
        f.write("%s\n" % item)
f.close()

energy = []
for ene in en:
    energy.append(str(round(ene, 2)))

energies = []
for i in res:
    energies.append(energy[i])

with open('top_1000_5zkp.txt', 'w') as f:
    for item in energies:
        f.write("%s\n" % item)
f.close()