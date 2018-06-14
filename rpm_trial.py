from __future__ import print_function
from rmsd import kabsch_rmsd
import chain
import numpy as np
import sys
import random
import os


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


def compute_rmsd_matrix(confs):
    n_conf = confs.shape[0]
    rmsds = np.zeros((n_conf, n_conf))
    for i in range(n_conf):
        for j in range(n_conf):
            r = kabsch_rmsd(confs[i, :, :], confs[j, :, :])
            rmsds[i, j] = r
            rmsds[j, i] = r
    return rmsds


def compute_pairwise_rmsd(rmsd_matrix):
    return np.mean(rmsd_matrix)


def get_contacts(sequence, label_pos):
    n = len(sequence)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_name = 'conf/hp{length}/{sequence}.conf'.format(
        length=n, sequence=sequence)
    conf_file = os.path.join(dir_path, file_name)
    conf = np.array(eval(open(conf_file).read()))
    return get_contacts_for_conf(conf, label_pos)


def get_contacts_for_conf(conf, label_pos):
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


def rpm_choose(ensemble, ensemble_contacts,
               rmsd_matrix, residues,
               labels, contacts):
    best = None
    best_rmsd = 9e99
    for r in residues:
        rmsds = []
        filtered_ind = filter_ensemble(ensemble_contacts,
                                       labeled_residues,
                                       contacts)
        for i in filtered_ind:
            new_contacts = get_contacts_for_conf(ensemble[i, :, :], r)
            new_filtered_ind = filter_ensemble(ensemble_contacts,
                                               labeled_residues + [r],
                                               contacts + [new_contacts])
            ens = rmsd_matrix[np.ix_(new_filtered_ind, new_filtered_ind)]
            rmsd = compute_pairwise_rmsd(ens)
            print('\t', r, i, rmsd)
            rmsds.append(rmsd)
        rmsd = np.mean(rmsds)
        if rmsd < best_rmsd:
            best_rmsd = rmsd
            best = r
    return best



if __name__ == '__main__':
    sequence = sys.argv[1]
    N = len(sequence)

    residues = list(range(N))
    order = []

    labeled_residues = []
    contacts = []
    rmsds = []

    # compute the ensemble and the contacts for each structure
    # without restraints
    ensemble, ensemble_contacts = create_ensemble(N)

    # compute the starting pairwise rmsd
    rmsd_matrix = compute_rmsd_matrix(ensemble)
    rmsd = compute_pairwise_rmsd(rmsd_matrix)
    rmsds.append(rmsd)

    while residues:
        r = rpm_choose(ensemble, ensemble_contacts,
                       rmsd_matrix, residues,
                       labeled_residues, contacts)
        residues.remove(r)
        order.append(r)
        contacts.append(get_contacts(sequence, r))
        labeled_residues.append(r)

        filtered_ind = filter_ensemble(ensemble_contacts,
                                       labeled_residues,
                                       contacts)
        ens = rmsd_matrix[np.ix_(filtered_ind, filtered_ind)]
        rmsd = compute_pairwise_rmsd(ens)
        rmsds.append(rmsd)

    with open('rpm_results.txt', 'w') as outfile:
        print (order, file=outfile)
        print (rmsds, file=outfile)
