#!/usr/bin/env python
# -*- coding:utf-8 -*-
import pandas as pd
import os
import numpy as np
import pkg_resources
import scipy as sp
from collections import Iterable
import sys
from clonesig.estimator import Estimator, EV_DOF_THRESHOLD
from clonesig import mixin_init_parameters

try:
    rows, columns = os.popen('stty size', 'r').read().split()
    pd.set_option('display.width', int(columns))
    pd.set_option('display.max_columns', 200)
except:
    print("running on server, otherwise please investigate")


def check_parameters(T, B, D, C_normal, C_tumor_tot, C_tumor_minor, purity,
                     inputMU, inputNu, nu_heuristics, nb_fits, seeds,
                     max_nb_clones, return_sig_change_test, min_mut_clone,
                     min_prop_sig, prefit_signatures, prefit_thresh,
                     model_selection_function, model_selection_kws):
    """
    this function checks that parameters for run_clonesig are ok, and performs
    preparatory computations
    """
    N = len(B)
    if isinstance(seeds, Iterable):
        if len(seeds) != nb_fits:
            raise ValueError("Number of seeds is incompatible with number of required fits")
    if seeds is None:
        seeds = list(range(nb_fits))
    if model_selection_kws is None:
        model_selection_kws = dict()
    if nu_heuristics is not None:
        Mmax = int(np.max(C_tumor_tot - C_tumor_minor))
        inputNu = np.zeros((N, Mmax))
        if nu_heuristics == 'ones':
            inputNu[:, 0] = 1
        elif nu_heuristics == 'minor':
            minor_cn_nonzero = np.array(C_tumor_minor).astype(int)-1
            minor_cn_nonzero[minor_cn_nonzero < 0] = 0
            inputNu[np.arange(N), minor_cn_nonzero] = 1
        elif nu_heuristics == 'major':
            C_tumor_major = C_tumor_tot - C_tumor_minor
            major_cn_nonzero = np.array(C_tumor_major).astype(int)-1
            major_cn_nonzero[major_cn_nonzero < 0] = 0
            inputNu[np.arange(N), major_cn_nonzero] = 1
        elif nu_heuristics == 'clonal':
            pre_cn = np.max(
                np.concatenate([np.ones(N).reshape(-1, 1),
                                np.round((B / D) * (purity * C_tumor_tot +
                                                    (1 - purity) * C_normal) /
                                purity, 0).reshape(-1, 1)], axis=1), axis=1)
            pre_cn = np.min(np.concatenate([(C_tumor_tot - C_tumor_minor).reshape(-1, 1) - 1,
                                            pre_cn.reshape(-1, 1) - 1], axis=1),
                            axis=1)
            pre_cn_bis = np.min(
                np.concatenate([np.ones(N).reshape(-1, 1) * (Mmax - 1),
                                pre_cn.reshape(-1, 1)], axis=1), axis=1)
            inputNu[np.arange(N), pre_cn_bis.astype(int)] = 1
        else:
            raise ValueError("invalid nu_heuristics")
    return (T, B, D, C_normal, C_tumor_tot, C_tumor_minor, purity, inputMU,
            inputNu, nb_fits, seeds, max_nb_clones, return_sig_change_test,
            min_mut_clone, min_prop_sig, prefit_signatures, prefit_thresh,
            model_selection_function, model_selection_kws)


def split_clone_initialization(previous_est, inputMU):
    """
    this function splits the clone with the maximal entropy, this is the
    strategy 'Splitting the component with the largest contribution to the
    mixture entropy' described in "EM for mixtures - Initialization requires
    special care", Baudry, Celeux 2015
    https://hal.inria.fr/hal-01113242/document
    """
    nb_clones = previous_est.J + 1
    to_split = np.argmax(-(previous_est.qun *
                         np.log(previous_est.qun)).sum(axis=0))
    mask = np.ones(nb_clones-1, dtype=bool)
    mask[to_split] = 0
    new_phi = np.zeros(nb_clones)
    new_phi[:nb_clones - 2] = previous_est.phi[mask]
    new_phi[-2] = np.random.ranf() * 0.8 + 0.1
    new_phi[-1] = np.random.ranf() * 0.8 + 0.1
    new_xi = np.zeros(nb_clones)
    new_xi[:nb_clones - 2] = previous_est.xi[mask]
    new_xi[-1], new_xi[-2] = [previous_est.xi[to_split]] * 2
    new_pi = np.zeros((nb_clones, inputMU.shape[0]))
    new_pi[:nb_clones - 2, :] = previous_est.pi[mask, :]
    new_pi[-1, :] = np.random.dirichlet(alpha=np.ones(inputMU.shape[0]))
    new_pi[-2, :] = np.random.dirichlet(alpha=np.ones(inputMU.shape[0]))
    return new_phi, new_xi, new_pi


def get_ll_test_dof(dof, nb_clones, nb_mut):
    """
    better model in the future, for now just coming from the simulations with
    30 signatures, without copy number. Will provide a better model later...
    """
    if dof == 21:
        return -4.61 + 7.83 * nb_clones + 0 * dof + 0.00168 * nb_mut
    elif dof <= 13:
        return -14.2 + 5.04 * nb_clones + 1.23 * dof + 0.000992 * nb_mut
    else:
        return -17.6 + 6.44 * nb_clones + 0.909 * dof + 0.00134 * nb_mut


def lrtest(llH0, llH1, dof):
    lr = 2 * (llH1 - llH0)
    sp.stats.chisqprob = lambda chisq, df: sp.stats.chi2.sf(chisq, df)
    p = sp.stats.chisqprob(lr, dof) # llmax has 1 dof more than llmin
    return lr, p

"""
T = data_df.trinucleotide.values 
B = data_df.var_counts.values 
D = data_df.var_counts.values + data_df.ref_counts.values 
C_normal = data_df.normal_cn.values 
C_tumor_tot = data_df.minor_cn.values + data_df.major_cn.values 
C_tumor_minor = data_df.minor_cn.values 
purity = purity 
inputMU = MU 
inputNu=None 
nu_heuristics='clonal' 
nb_fits=1 
seeds=None 
max_nb_clones=6 
return_sig_change_test=True 
min_mut_clone=0 
min_prop_sig=0.0 
prefit_signatures=False, 
prefit_thresh=0.05 
model_selection_function=None 
model_selection_kws={'factor': 0.048}                                                                                                                                                                                                                                
    """
def remove_small_clones(previous_est, min_mut_clone, inputMU):
    """
    this function removes clones smaller than a threshold (min_mut_clone)
    After each post-hoc modification, the estimator is refit with
    initialization to previous parameters (adjusted if needed)

    Parameters
    ----------
    previous_est: Estimator object
                  current estimator fitted to the data, from which to remove
                  clones that are too small
    min_mut_clone: int or float
                   if int, the minimal number of mutations per returned clone
                   by hard assignement (most likely clone). If the threshold
                   is not met for a clone, it is deleted, and attributions to
                   the remaining clones are computed for all mutations.
                   if float, same principle, but the threshold is applied to
                   the \\xi parameters, representing the proportion of each
                   clone with soft assignement.
    inputMU: array-like (L, 96)
             known L signatures to be fit by clonesig
    Returns
    -------
    new_est: new estimator fit with new number of clones.
    """
    if isinstance(min_mut_clone, float):
        future_clones = previous_est.xi > min_mut_clone
    elif isinstance(min_mut_clone, int):
        useful_counts = np.zeros(previous_est.J)
        pre_counts = np.unique(np.argmax(previous_est.qun, axis=1),
                               return_counts=True)
        useful_counts[pre_counts[0]] = pre_counts[1]
        actual_min_mut_clone = min(np.max(useful_counts), min_mut_clone)
        future_clones = useful_counts >= actual_min_mut_clone
    new_phi = previous_est.phi[future_clones]
    new_xi = previous_est.xi[future_clones] /\
        previous_est.xi[future_clones].sum()
    new_pi = previous_est.pi[future_clones, :]
    new_nb_clones = sum(future_clones)
    new_est = Estimator(previous_est.T, previous_est.B,
                        previous_est.C_normal, previous_est.C_tumor_tot,
                        previous_est.C_tumor_minor, previous_est.D,
                        previous_est.p, new_nb_clones,
                        inputMU=inputMU, pi=new_pi, phi=new_phi,
                        xi=new_xi, nu=previous_est.nu,
                        tau=previous_est.tau)
    new_est.fit()
    return new_est


def remove_small_sigs(previous_est, single_clone_est, min_prop_sig, inputMU):
    """
    this function removes signatures with exposure smaller than a threshold
    (min_prop_sig) in all subclones of previous_est and in a global fit
    (single_clone_est).
    After each post-hoc modification, the estimator is refit with
    initialization to previous parameters (adjusted if needed)

    Parameters
    ----------
    previous_est: Estimator object
                  current estimator fitted to the data, from which to remove
                  signatures with too small exposures in the sample
    single_clone_est: Estimator object
                      current single clone estimator fitted to the data.
                      Because of the likelihood test ratio, it is necessary to
                      adjust it as well accordingly.
    min_prop_sig: float
                  minimal exposure for signatures. If the maximal exposure of
                  a given signature among all clones is smaller than
                  min_prop_sig, then it is removed, and the contribution of
                  other signatures is scaled to 1.
    inputMU: array-like (L, 96)
             known L signatures to be fit by clonesig.
    Returns
    -------
    """
    big_pi = np.concatenate((single_clone_est.pi, previous_est.pi), axis=0)
    future_sigs = np.max(big_pi, axis=0) > min_prop_sig
    new_inputMU = inputMU[future_sigs, :]
    pre_new_single_clone_pi = single_clone_est.pi[:, future_sigs]
    pre_new_pi = previous_est.pi[:, future_sigs]
    new_single_clone_pi = pre_new_single_clone_pi /\
        pre_new_single_clone_pi.sum(axis=1)[:, np.newaxis]
    new_pi = pre_new_pi / pre_new_pi.sum(axis=1)[:, np.newaxis]
    new_inputMU = inputMU[future_sigs, :]
    new_est = Estimator(previous_est.T, previous_est.B,
                        previous_est.C_normal, previous_est.C_tumor_tot,
                        previous_est.C_tumor_minor, previous_est.D,
                        previous_est.p, previous_est.J, inputMU=new_inputMU,
                        pi=new_pi, phi=previous_est.phi, xi=previous_est.xi,
                        nu=previous_est.nu, tau=previous_est.tau)
    new_est.fit()
    new_sc_est = Estimator(single_clone_est.T, single_clone_est.B,
                           single_clone_est.C_normal,
                           single_clone_est.C_tumor_tot,
                           single_clone_est.C_tumor_minor, single_clone_est.D,
                           single_clone_est.p, single_clone_est.J,
                           inputMU=new_inputMU, pi=new_single_clone_pi,
                           phi=single_clone_est.phi, xi=single_clone_est.xi,
                           nu=single_clone_est.nu, tau=single_clone_est.tau)
    new_sc_est.fit()
    return new_est, new_sc_est, new_inputMU


def get_MU(cosmic_version=3, cancer_type=None, exome=False):
    """
    this function initializes the MU matrix with v2 or v3 from COSMIC.
    An exome version exists for v3 only.
    The matrix can be filtered on cancer types as available from COSMIC. The
    types are not the same in v2 and v3. Please specify an int, and see doc
    to know which cancer types are available

    Parameters
    ----------
    cosmic_version: int
                    cosmic version to use. 2 or 3 are available
    cancer_type: int
                 cancer type to filter the signature matrix and remove
                 signatures not known to be active in this type. See doc for
                 a precise matching of indexes to cancer types
    exome: bool
           a re-normalized version of signatures exists for v3, to account
           for the different frequencies of trinucleotides in exome comapred to
           whole genome. Set exome to True if you are analyzing exome samples.
    """
    if cosmic_version == 3:
        if exome:
            filename = 'data/sigProfiler_exome_SBS_signatures.csv'
        else:
            filename = 'data/sigProfiler_SBS_signatures_2018_03_28.csv'
        sig = pd.read_csv(
            pkg_resources.resource_stream(
                'clonesig', filename),
            sep=',')
        sig_matrix = sig.values[:, 2:].astype(float).T
        new_sig_matrix = sig_matrix + mixin_init_parameters.ZERO_PADDING * (sig_matrix == 0)
        MU = new_sig_matrix / new_sig_matrix.sum(axis=1)[:, np.newaxis]
        if cancer_type is not None:
            filter_filename = 'data/match_cancer_type_sig_v3.csv'
            cancer_type_sig = pd.read_csv(pkg_resources.resource_stream(
                'clonesig', filter_filename), index_col=0).values
            select = cancer_type_sig[cancer_type, :]
            subMU = MU[select.astype(bool), :]
        else:
            subMU = MU.copy()
    elif cosmic_version == 2:
        if exome:
            raise ValueError("No exome version of signatures v2 available on COSMIC for exomes.")
        filename = 'data/signatures_probabilities.txt'
        sig = pd.read_csv(
            pkg_resources.resource_stream(
                'clonesig', filename
            ),
            sep='\t', index_col=0
        )
        sig_cols = ['Signature {}'.format(i) for i in range(1, 31)]
        sig = sig.assign(sortkey=sig.index + sig.Trinucleotide)
        sig = sig.sort_values(by='sortkey')
        sig_matrix = sig[sig_cols].values.T
        m, k = sig_matrix.shape

        # avoid 0 values so add a small value and renormalize
        new_sig_matrix = sig_matrix + mixin_init_parameters.ZERO_PADDING * (sig_matrix == 0)
        MU = new_sig_matrix / new_sig_matrix.sum(axis=1)[:, np.newaxis]
        if cancer_type is not None:
            filter_filename = 'data/match_cancer_type_sig_v2.csv'
            cancer_type_sig = pd.read_csv(pkg_resources.resource_stream(
                'clonesig', filter_filename), index_col=0).fillna(0).values
            select = cancer_type_sig[cancer_type, :]
            subMU = MU[select.astype(bool), :]
        else:
            subMU = MU.copy()
    else:
        raise ValueError("wrong cosmic version, should be 2 or 3 (int). You provided {}".format(cosmic_version))
    return subMU


def run_clonesig(T, B, D, C_normal, C_tumor_tot, C_tumor_minor, purity,
                 inputMU, inputNu=None, nu_heuristics=None, nb_fits=1,
                 seeds=None, max_nb_clones=6, return_sig_change_test=True,
                 min_mut_clone=0, min_prop_sig=0.0, prefit_signatures=False,
                 prefit_thresh=0.05, model_selection_function=None,
                 model_selection_kws=None):
    """
    this function is a wrapper that takes data (and settings) as input, tries
    to fit clonesig model for a various number of clones, and returns the best
    fit, with some relevant selected post-hoc adjustements. After each
    post-hoc modification, the estimator is refit with initialization to
    previous parameters (adjusted if needed)

    Parameters
    ----------
    T : iterable of length N
        with the trinucleotide context of each mutation, numbered
        from 0 to 95
    B : iterable of length N
        with the variant allele read count for each mutation
    D : iterable of length N
        with the total read count for each mutation
    C_normal : iterable of length N
               copy number of non-tumor cells in the sample at each mutation
               locus
    C_tumor_tot : iterable of length N
                  the total copy number of tumor cells in the sample at each
                  mutation locus
    C_tumor_minor : iterable of length N
                    the minor copy number of tumor cells in the sample at each
                    mutation locus. If this info is not available, set it to
                    zero so that clonesig considers all possible genotypes
    purity : float in [0, 1]
             an estimate of the tumor purity of the sample
    inputMU : array-like (L, 96)
              known L signatures to be fit by clonesig.
    inputNu : array-like (N, Mmax)
              with Mmax = max(C_tumor_tot - C_tumor_minor)
              probablity distribution of number of mutated copies for each
              mutation
              be careful, it is a probability distribution, so one should have
              np.sum(inputNu) / N = 1
    nu_heuristics : string among ('ones', 'minor', 'major', 'clonal')
                    automatic generation of the nu parameter with 3 possible
                    heuristics: set tue number of mutated copy number to 1,
                    or set the number of mutated copy number to major or minor
                    copy number, or set the mutated CN to the max of 1, and the
                    number of mutated copy to get a CCF of 1 given purity and
                    total copy number
                    this option will over-ride any inputNu given by user.
    nb_fits : integer (>1)
              number of independant fits to perform for this sample (as results
              depend on the random initialization, and might be local maxima of
              the EM objective function)
    seeds : iterable, of length nb_fits
            seeds for the different initialization. If not provided, seeds are
            set to 0 to nb_fits-1
    max_nb_clones : integer (>1)
                    maximum number of clones wanted to be found by the model
    return_sig_change_test : boolean
                             perform a statistical test (adapted from a
                             loglikelihood ratio test) to assess whether there
                             is a change of signature in the sample (H1) or if
                             all clones have the same signature exposures (H0)
    min_mut_clone : int or float
                    if int, the minimal number of mutations per returned clone
                    by hard assignement (most likely clone). If the threshold
                    is not met for a clone, it is deleted, and attributions to
                    the remaining clones are computed for all mutations.
                    if float, same principle, but the threshold is applied to
                    the \\xi parameters, representing the proportion of each
                    clone with soft assignement.
    min_prop_sig : float
                   minimal exposure for signatures. If the maximal exposure of
                   a given signature among all clones is smaller than
                   min_prop_sig, then it is removed, and the contribution of
                   other signatures is scaled to 1
    prefit_signatures : boolean
                        fit signatures to the sample (globally, with 1 clone),
                        and then just use the subset of signatures with an
                        exposure of at least prefit_thresh
    prefit_thresh : float
                    minimal threshold to select signature in the prefit step
    model_selection_function : string among (...)
                               model selection function to use
    model_selection_kws : dictionary
                          parameters to pass to the model_selection_function


    Returns
    -------
    """
    (T, B, D, C_normal, C_tumor_tot, C_tumor_minor, purity, inputMU,
     inputNu, nb_fits, seeds, max_nb_clones, return_sig_change_test,
     min_mut_clone, min_prop_sig, prefit_signatures, prefit_thresh,
     model_selection_function, model_selection_kws) = check_parameters(
        T, B, D, C_normal, C_tumor_tot, C_tumor_minor, purity, inputMU,
        inputNu, nu_heuristics, nb_fits, seeds, max_nb_clones,
        return_sig_change_test, min_mut_clone, min_prop_sig, prefit_signatures,
        prefit_thresh, model_selection_function, model_selection_kws)
    # prefit of signatures
    if prefit_signatures:
        prefit_est = Estimator(T, B, C_normal, C_tumor_tot,
                               C_tumor_minor, D, purity, 1,
                               inputMU=inputMU, nu=inputNu)
        prefit_est.fit()
        future_sigs = (prefit_est.pi.T.dot(prefit_est.xi)) > prefit_thresh
        prefit_inputMU = inputMU[future_sigs, :]
    else:
        prefit_inputMU = inputMU.copy()
        future_sigs = None

    criterion = np.zeros((nb_fits, max_nb_clones+2))
    loglikelihood = np.zeros((nb_fits, max_nb_clones+2))
    loglikelihood_nopi = np.zeros((nb_fits, max_nb_clones+2))
    est_matrix = np.zeros((nb_fits, max_nb_clones+2)).astype(object)
    for j, nb_clones in enumerate(range(1, max_nb_clones+3)):
        for i, s in enumerate(seeds):
            print(j, i)
            np.random.seed(s)
            if nb_clones >= 2:
                previous_est = est_matrix[i, j-1]
                new_phi, new_xi, new_pi = \
                    split_clone_initialization(previous_est, prefit_inputMU)
                est = Estimator(T, B, C_normal, C_tumor_tot,
                                C_tumor_minor, D, purity, nb_clones,
                                inputMU=prefit_inputMU, pi=new_pi, phi=new_phi,
                                xi=new_xi, nu=inputNu, tau=previous_est.tau)
            else:
                est = Estimator(T, B, C_normal, C_tumor_tot,
                                C_tumor_minor, D, purity, nb_clones,
                                inputMU=prefit_inputMU, nu=inputNu)

            est.fit()
            criterion[i, j] = est.get_bic_heuristics(**model_selection_kws)
            loglikelihood[i, j] = est.get_loglikelihood
            est_matrix[i, j] = est
        if j > 1:
            bm = criterion.mean(axis=0)
            if (bm[j-2] > bm[j-1]) and (bm[j-2] > bm[j]) and (bm[j-1] > bm[j]):
                print('stopped and chosen number of clones is ', nb_clones - 2)
                print(loglikelihood.mean(axis=0))
                print(bm)
                break
    print('stopped and chosen number of clones is ', nb_clones - 2)
    print(loglikelihood.mean(axis=0))
    print(bm)

    # get best run
    chosen_nb_clones = max(nb_clones - 2, 1)
    chosen_nb_clones_idx = chosen_nb_clones - 1
    i_best = np.argmin(loglikelihood_nopi[:, chosen_nb_clones_idx])
    est_best = est_matrix[i_best, chosen_nb_clones_idx]
    # sc = single clone
    sc_est_best = est_matrix[i_best, 0]

    est_best_big_clones = remove_small_clones(est_best, min_mut_clone,
                                              prefit_inputMU)
    new_est, new_sc_est, new_inputMU = remove_small_sigs(est_best_big_clones,
                                                         sc_est_best,
                                                         min_prop_sig,
                                                         prefit_inputMU)
    print(np.repeat(new_sc_est.pi, new_est.J, axis=0).shape, new_inputMU.shape, new_est.J)
    cst_est = Estimator(T, B, C_normal, C_tumor_tot,
                        C_tumor_minor, D, purity, new_est.J,
                        inputMU=new_inputMU,
                        pi=np.repeat(new_sc_est.pi, new_est.J, axis=0),
                        phi=new_est.phi, tau=new_est.tau, xi=new_est.xi,
                        nu=inputNu)
    dof_test = get_ll_test_dof(new_inputMU, new_est.J)
    lr, p = lrtest(cst_est.get_loglikelihood,
                   new_est.get_loglikelihood, dof_test)
    return new_est, lr, p, new_inputMU, cst_est, future_sigs
