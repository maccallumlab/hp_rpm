import chain
import rmsd
import numpy as np
import sys

n_res = int(sys.argv[1])
base = sys.argv[2]
# list of labelled positions
labels = eval(sys.argv[3])
# list of lists of residues close to each label
contacts = eval(sys.argv[4])

c = chain.Chain('P' * n_res)
confs = []
for conf in chain.enumerate_conf(c):
    conf = np.array(conf)
    keep = True
    for i, conts in zip(labels, contacts):
        for j in range(n_res):
            d = np.linalg.norm(conf[i, :] - conf[j, :])
            if d < 1.01 and j not in conts:
                keep = False
            elif d > 1.01 and j in conts:
                keep = False
    if keep:
        confs.append(conf)
n_confs = len(confs)
print n_confs

array = np.zeros((n_res, n_res, n_confs), dtype=bool)
for n, conf in enumerate(confs):
    for i in range(n_res):
        for j in range(n_res):
            dist = np.linalg.norm(conf[i, :] - conf[j, :])
            if dist < 1.01:  # fudge factor for rounding
                array[i, j, n] = True
                array[j, i, n] = True

with open('contacts_{}.npy'.format(base), 'wb') as outfile:
    np.save(outfile, array)

confs = np.array(confs)
with open('confs_{}.npy'.format(base), 'wb') as outfile:
    np.save(outfile, confs)

rmsds = np.zeros((n_confs, n_confs))
for i in range(n_confs):
    for j in range(i, n_confs):
        r = rmsd.kabsch_rmsd(confs[i, :, :], confs[j, :, :])
        rmsds[i, j] = r
        rmsds[j, i] = r

with open('rmsds_{}.npy'.format(base), 'wb') as outfile:
    np.save(outfile, rmsds)
