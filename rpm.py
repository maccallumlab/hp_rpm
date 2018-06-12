import numpy as np
import sys
from matplotlib import pyplot as pp

n_res = int(sys.argv[1])
base = sys.argv[2]

rmsds = np.load(open('rmsds_{}.npy'.format(base), 'rb'))
n_conf = rmsds.shape[0]
assert rmsds.shape[0] == rmsds.shape[1]

contacts = np.load(open('contacts_{}.npy'.format(base), 'rb'))
assert contacts.shape[0] == n_res
assert contacts.shape[1] == n_res
assert contacts.shape[2] == n_conf


results = []
for probe in range(n_res):
    probe_rmsds = []
    for i in range(n_conf):
        r = 0.0
        n = 0.0
        for j in range(n_conf):
            if (contacts[probe, :, i] == contacts[probe, :, j]).all():
                r += rmsds[i, j]
                n += 1.0
        probe_rmsds.append(r/n)
    results.append(np.mean(probe_rmsds))
pp.plot(results, marker='o')
pp.axhline(np.mean(rmsds))
print np.mean(rmsds)
pp.show()
