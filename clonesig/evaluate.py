#!/usr/bin/env python
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import scipy as sp
from scipy import stats
import itertools
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.metrics.cluster import v_measure_score
from scipy.spatial.distance import cosine, euclidean


def score1B_base(J_true, J_pred):
    return (J_true + 1 - min([J_true + 1, np.abs(J_true - J_pred)])) / \
        (J_true + 1)


def score1C_base(phi_true_values, phi_pred_values, weights_true=None,
                 weights_pred=None):
    return 1 - sp.stats.wasserstein_distance(phi_true_values, phi_pred_values,
                                             weights_true, weights_pred)


def score2A_base(true_cluster_assign, pred_cluster_assign):
    # co-clustering matrix - hard assignement
    # step1 build the co-clustering matrix
    N = len(true_cluster_assign)
    sim_mat = np.zeros((N + 1, N + 1))

    for u in np.unique(true_cluster_assign):
        sim_mat[tuple(zip(*list(
            itertools.product(np.where(true_cluster_assign == u)[0],
                              np.where(true_cluster_assign == u)[0]))))] = 1
    sim_mat[N, N] = 1
    pred_mat = np.zeros((N + 1, N + 1))
    for u in np.unique(pred_cluster_assign):
        pred_mat[tuple(zip(*list(
            itertools.product(np.where(pred_cluster_assign == u)[0],
                              np.where(pred_cluster_assign == u)[0]))))] = 1
    pred_mat[N, N] = 1
    p = sp.stats.pearsonr(sim_mat.flatten(), pred_mat.flatten())[0]
    m = matthews_corrcoef(sim_mat.flatten(), pred_mat.flatten())
    v = v_measure_score(true_cluster_assign, pred_cluster_assign)

    good_scen_matrix = sim_mat.copy()
    bad1 = np.identity(N + 1)
    bad2 = np.ones((N + 1, N + 1))
    bad2[:-1, -1] = 0
    bad2[-1, :-1] = 0

    p_good = sp.stats.pearsonr(sim_mat.flatten(),
                               good_scen_matrix.flatten())[0]
    p_bad1 = sp.stats.pearsonr(sim_mat.flatten(), bad1.flatten())[0]
    p_bad2 = sp.stats.pearsonr(sim_mat.flatten(), bad2.flatten())[0]
    p_bad = min(p_bad1, p_bad2)
    pn = max(0, - p / (p_bad - p_good) + p_bad / (p_bad - p_good))

    m_good = matthews_corrcoef(sim_mat.flatten(), good_scen_matrix.flatten())
    m_bad1 = matthews_corrcoef(sim_mat.flatten(), bad1.flatten())
    m_bad2 = matthews_corrcoef(sim_mat.flatten(), bad2.flatten())
    m_bad = min(m_bad1, m_bad2)
    mn = max(0, - m / (m_bad - m_good) + m_bad / (m_bad - m_good))

    v_good = v_measure_score(sim_mat.flatten(), good_scen_matrix.flatten())
    v_bad1 = v_measure_score(sim_mat.flatten(), bad1.flatten())
    v_bad2 = v_measure_score(sim_mat.flatten(), bad2.flatten())
    v_bad = min(v_bad1, v_bad2)
    vn = max(0, - v / (v_bad - v_good) + v_bad / (v_bad - v_good))
    return np.mean([pn, mn, vn])


def score2C_base(sim_subclonal, pred_subclonal):
    """
    this metric compares the sensitivity, specificity, precision of mutations
    classified as clonal vs subclonal. We take the convention that the clone
    with the largest phi is clonal and the rest is subclonal
    we take also the convention that subclonal are the positive examples, and
    clonal the negative ones.

    Parameters
    ----------
    sim_subclonal: binary iterable
                   array-like of length N, with 1 if the mutation is subclonal,
                   and 0 else, for the true labels
    pred_subclonal: binary iterable
                    array-like of length N, with 1 if the mutation is subclonal,
                    and 0 else, for the predicted labels
    """
    TP = sum((pred_subclonal == 1) & (sim_subclonal == 1))
    TN = sum((pred_subclonal == 0) & (sim_subclonal == 0))
    FP = sum((pred_subclonal == 1) & (sim_subclonal == 0))
    FN = sum((pred_subclonal == 0) & (sim_subclonal == 1))

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    sensitivity = TP / max(1, (TP + FN))
    specificity = TN / (TN + FP)
    precision = TP / max(1, (TP + FN))
    try:
        auc = roc_auc_score(sim_subclonal, pred_subclonal)
    except ValueError:
        auc = np.nan
    return auc, accuracy, sensitivity, specificity, precision


def score_sig_1A_base(sim_raw_distrib, pred_profile):
    """
    euclidian norm between normalized trinucleotide context counts (empirical),
    and the reconstituted profile
    Parameters
    ----------
    sim_raw_distrib: iterable (length the number of features, e.g. 96 for
                     trinucleotides)
                     normalized count of features in the data.
    pred_profile: iterable (length the number of features, e.g. 96 for
                  trinucleotides)
                  predicted profile (dot product between exposure matrix, and
                  signature matrix)
    """
    return euclidean(sim_raw_distrib, pred_profile)


def score_sig_1B_base(sim_profile, pred_profile):
    """
    euclidian norm between simulated and estimated signature profile (summed
    over all clones)
    Parameters
    ----------
    true_profile: iterable (length the number of features, e.g. 96 for
                  trinucleotides)
                  true profile (dot product between true exposure matrix, and
                  signature matrix)
    pred_profile: iterable (length the number of features, e.g. 96 for
                  trinucleotides)
                  predicted profile (dot product between predicted exposure
                  matrix, and signature matrix)
    """
    return euclidean(sim_profile, pred_profile)


def score_sig_1C_base(true_signatures, pred_signatures, threshold=0.95):
    """
    precision recall for detected signatures
    Parameters
    ----------
    true_signatures: binary iterable
                     array-like of length L (number of signatures), with 1 if
                     the signature is active, 0 else, for the true values
    pred_signatures: binary iterable
                     array-like of length L (number of signatures), with 1 if
                     the signature is active, 0 else, for the predicted values
    """
    threshold = min(threshold, sum(pred_signatures))
    min_thresh = np.sort(
        pred_signatures)[::-1][np.argmax(
            np.cumsum(np.sort(pred_signatures)[::-1]) >= threshold)]
    pred_sig_idx = np.where(pred_signatures >= min_thresh)[0]

    pred_signatures_binary = np.zeros(len(true_signatures))
    pred_signatures_binary[pred_sig_idx] = 1

    TP = sum((pred_signatures_binary == 1) & (true_signatures == 1))
    TN = sum((pred_signatures_binary == 0) & (true_signatures == 0))
    FP = sum((pred_signatures_binary == 1) & (true_signatures == 0))
    FN = sum((pred_signatures_binary == 0) & (true_signatures == 1))

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    sensitivity = TP / (TP + FN + 1)
    specificity = TN / (TN + FP)
    precision = TP / (TP + FP + 1)
    auc = roc_auc_score(true_signatures, pred_signatures)

    return auc, accuracy, sensitivity, specificity, precision


def score_sig_1D_base(true_signatures, pred_signatures):
    """
    percent of mutations with the right signature
    Parameters
    ----------
    true_signatures: iterable
                     array-like of length N (number of mutations), with the
                     true signature for each mutation
    pred_signatures: iterable
                     array-like of length N (number of mutations), with the
                     predicted signature for each mutation
    """
    N = len(true_signatures)
    return sum(true_signatures == pred_signatures) / N


def score_sig_1E_base(true_profile, pred_profile):
    """
    cosine dist between the clone distrib that generated the mutation and the
    reconstituted one.
    true_profile: array-like (NxK)
                  with N the number of mutations, and K the number of features
                  (96 for trinucleotides)
                  representing for each mutation the true profile that
                  generated the mutation in the simulation
    pred_profile: array-like (NxK)
                  with N the number of mutations, and K the number of features
                  (96 for trinucleotides)
                  representing for each mutation the predicted profile that
                  generated the mutation in the estimation
    """
    N = len(true_profile)
    dist_l = np.zeros(N)
    for i in range(N):
        dist_l[i] = cosine(true_profile[i], pred_profile[i])

    min_diff_distrib_mut = np.min(dist_l)
    max_diff_distrib_mut = np.max(dist_l)
    std_diff_distrib_mut = np.std(dist_l)
    median_diff_distrib_mut = np.median(dist_l)
    perc_dist_5 = sum(dist_l < 0.05)/len(dist_l)
    perc_dist_10 = sum(dist_l < 0.10)/len(dist_l)
    return (min_diff_distrib_mut, max_diff_distrib_mut, std_diff_distrib_mut,
            median_diff_distrib_mut, perc_dist_5, perc_dist_10)
