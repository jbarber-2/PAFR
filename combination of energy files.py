import numpy as np

en1 = np.loadtxt('true_energy1.txt')
en2 = np.loadtxt('true_energy2.txt')
en3 = np.loadtxt('true_energy3.txt')
en4 = np.loadtxt('true_energy4.txt')
en5 = np.loadtxt('true_energy5.txt')
en6 = np.loadtxt('true_energy6.txt')
en7 = np.loadtxt('true_energy7.txt')
en8 = np.loadtxt('true_energy8.txt')
en9= np.loadtxt('true_energy9.txt')
en10 = np.loadtxt('true_energy10.txt')

en = []
en.append(en1)
en.append(en2)
en.append(en3)
en.append(en4)
en.append(en5)
en.append(en6)
en.append(en7)
en.append(en8)
en.append(en9)
en.append(en10)

c = []
for elements in en:
    c.append(len(elements))
b = np.array(c)

fin = []
for i in range(len(en[0])):
    d = []
    for ene in en:
        d.append(ene[i])
    fin.append(d)
print(len(fin))

for elements in fin:
    np.array(elements)

mean = []
rang = []
for elements in fin:
    mean.append(np.mean(elements))
    rang.append(np.ptp(elements))

mean2 = np.round(mean, 3)
print(mean2)
r = np.array(rang)
type(r)

np.argwhere(r >= 2)

with open('true_energy_5zkq.txt', 'w') as f:
    for item in mean2:
        f.write("%s\n" % item)
f.close()