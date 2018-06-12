from rmsd import kabsch_rmsd
import chain
import numpy as np
import sys
import random
from matplotlib import pyplot as pp


def create_ensemble(n_res):
    # create a chain with arbitrary sequence
    c = chain.Chain('P' * n_res)
    confs = [np.array(x) for x in chain.enumerate_conf(c)]
    n_conf = len(confs)
    contacts = np.zeros((n_res, n_res, n_conf), dtype=bool)
    for n, conf in enumerate(confs):
        for i in range(n_res):
            for j in range(n_res):
                dist = np.linalg.norm(conf[i, :] - conf[j, :])
                if dist < 1.01:  # fudge factor for rounding
                    contacts[i, j, n] = True
                    contacts[j, i, n] = True
    return np.array(confs), contacts


def compute_pairwise_rmsd(confs):
    n_conf = confs.shape[0]
    rmsds = []
    for i in range(n_conf):
        for j in range(n_conf):
            rmsds.append(kabsch_rmsd(confs[i, :, :], confs[j, :, :]))
    return np.mean(rmsds)


def get_contacts(sequence, label_pos):
    n = len(sequence)
    conf_file = 'conf/hp{length}/{sequence}.conf'.format(
        length=n, sequence=sequence)
    conf = np.array(eval(open(conf_file).read()))
    contacts = []
    for i in range(len(conf)):
        d = np.linalg.norm(conf[i] - conf[label_pos])
        if d < 1.01:
            contacts.append((label_pos, i))
    return contacts


def filter_ensemble(ensemble_contacts, labels, contacts):
    n_res = ensemble_contacts.shape[0]
    n_conf = ensemble_contacts.shape[2]
    contact_matrix = np.zeros((n_res, n_res), dtype=bool)
    for label, label_contacts in zip(labels, contacts):
        for i, j in label_contacts:
            contact_matrix[i, j] = True
            contact_matrix[j, i] = True
    filtered_indices = []
    for struct_ind in range(n_conf):
        keep = True
        for label in labels:
            if not np.all(ensemble_contacts[label, :, struct_ind] == contact_matrix[label, :]):
                keep = False
        if keep:
            filtered_indices.append(struct_ind)
    return filtered_indices


if __name__ == '__main__':
    sequence = sys.argv[1]
    run_id = int(sys.argv[2])
    N = len(sequence)

    residues = list(range(N))
    random.shuffle(residues)

    labeled_residues = []
    contacts = []
    rmsds = []

    # compute the ensemble and the contacts for each structure
    # without restraints
    ensemble, ensemble_contacts = create_ensemble(N)

    # compute the starting pairwise rmsd
    rmsd = compute_pairwise_rmsd(ensemble)
    rmsds.append(rmsd)

    for r in residues:
        contacts.append(get_contacts(sequence, r))
        labeled_residues.append(r)

        filtered_ind = filter_ensemble(ensemble_contacts, labeled_residues, contacts)
        ens = ensemble[filtered_ind, :, :]
        rmsd = compute_pairwise_rmsd(ens)
        rmsds.append(rmsd)

    with open('results_{:03d}.txt'.format(run_id), 'w') as outfile:
        print >>outfile, residues
        print >>outfile, rmsds
