#!/usr/bin/env python
# -*- coding:utf-8 -*-
from collections import Iterable
from enum import Enum
import os
import pkg_resources
import sys

import pandas as pd
import numpy as np
import scipy as sp
from scipy import special, spatial
from clonesig import mixin_init_parameters


try:
    rows, columns = os.popen('stty size', 'r').read().split()
    pd.set_option('display.width', int(columns))
    pd.set_option('display.max_columns', 200)
except:
    print("running on server, otherwise please investigate")

# this is the threshold to consider eigenvalues null to get an approximation
# of the degree of freedom of the cosine distance matrix for signatures
EV_DOF_THRESHOLD = 0.5


def _get_projected(L, R, x):
    x[x < L] = L[x < L]
    x[x > R] = R[x > R]
    return x


def log_binomial_coeff(n, k):
    return (sp.special.gammaln(n+1) - (sp.special.gammaln(k+1) +
            sp.special.gammaln(n-k+1)))


def _beta_binomial_logpmf(x, n, phi, tau):
    alpha = phi / tau
    beta = 1/tau - alpha
    return log_binomial_coeff(n, x)\
        + sp.special.betaln(x + alpha, n - x + beta)\
        - sp.special.betaln(alpha, beta)


def beta_binomial_pmf(x, n, phi, tau):
    return np.exp(_beta_binomial_logpmf(x, n, phi, tau))


class Estimator(mixin_init_parameters.MixinInitParameters):
    def __init__(self, T, B, C_normal, C_tumor_tot, C_tumor_minor, D, p, J,
                 maxiter=10000, pi=None, phi=None, xi=None, rho=None, tau=None,
                 verbose=False, inputMU=None, nu=None, save_trace=False):
        """
        est = Estimator(...)
        est.init_params(...)
        est.fit(...)
        """
        self.T = T
        self.B = B
        self.C_normal = C_normal
        self.C_tumor_tot = C_tumor_tot
        self.C_tumor_minor = C_tumor_minor
        self.C_tumor_major = (self.C_tumor_tot - self.C_tumor_minor).astype(int)
        self.C_est_tumor_mut = C_tumor_minor.copy()
        # this is a choice
        self.C_est_tumor_mut[self.C_est_tumor_mut == 0] = \
            self.C_tumor_tot[self.C_est_tumor_mut == 0]
        # alternative could be
        # self.C_est_tumor_mut[self.C_est_tumor_mut==0] = 1
        self.Mmax = max(self.C_tumor_major)
        self.D = D
        self.p = p
        self.J = J
        self.N = len(B)
        if inputMU is None:
            self.mu_matrix = self.default_mu()
        else:
            self.mu_matrix = inputMU
        self.mu_matrix = self._remove_zeros_distrib(self.mu_matrix)
        self.mu = np.moveaxis(np.tile(self.mu_matrix[:, self.T.astype(int)],
                                      (self.J, self.Mmax, 1, 1)),
                              [0, 1, 2, 3], [1, 3, 2, 0])
        self.L = self.mu.shape[2]
        self.Fs = list()
        self.maxiter = maxiter
        self.init_params(pi, phi, xi, rho, tau, spasePi=False)
        self.init_nu_param(nu)
        self.verbose = verbose
        ev, _ = np.linalg.eig(1-sp.spatial.distance.squareform(sp.spatial.distance.pdist(self.mu_matrix, 'cosine')))
        self.dof = sum(ev > EV_DOF_THRESHOLD)
        self.save_trace = save_trace

    def init_nu_param(self, nu=None):
        if nu is None:
            self.nu = np.ones((self.N, self.Mmax)) * \
                mixin_init_parameters.ZERO_PADDING
            for i in range(self.N):
                self.nu[i, :self.C_tumor_major[i]] = 1 / self.C_tumor_major[i]
                # self.nu[i, min(int(np.round(max(1, self.B[i]/self.D[i] * (self.p*self.C_tumor_tot[i] + (1-self.p) * self.C_normal[i]) / self.p))), self.C_tumor_major[i]) - 1] = 1
        else:
            self.nu = nu
        # make sure there are no null values
        self.nu[self.nu == 0] = mixin_init_parameters.ZERO_PADDING
        self.nu = self.nu / self.nu.sum(axis=1)[:, np.newaxis]
        # get log(nu) in the right dimension (N * J * L * Mmax) for computation
        self.lognu_sig = np.moveaxis(np.tile(np.log(self.nu), [self.J, self.L, 1, 1]),
                                     [0, 1, 2], [1, 2, 0])

        self.Mask = (self.nu > mixin_init_parameters.ZERO_PADDING * 10).astype(bool)
        self.Mask_sig = np.moveaxis(np.tile(self.Mask, [self.J, self.L, 1, 1]),
                                    [0, 1, 2], [1, 2, 0])

        pre_eta = np.repeat(np.arange(1, self.Mmax+1).reshape([-1, 1]), self.N,
                            axis=1).T

        self.eta = self.p * pre_eta / ((1 - self.p) * np.repeat(self.C_normal.reshape([-1, 1]), self.Mmax, axis=1) +
                                       self.p * np.repeat(self.C_tumor_tot.reshape([-1, 1]), self.Mmax, axis=1))
        self.eta[~self.Mask] = 1
        self.qun, self.vmnu, self.rnus = self.get_responsabilities

    @property
    def get_theta(self):
        return np.concatenate((self.xi.flatten(), self.pi.flatten(),
                               self.phi.flatten(), [self.tau]))

    def get_log_xi_sig(self, xi):
        """
        get log(xi) in the right dimension (N * J * L * Mmax) for computation
        """
        return np.moveaxis(np.tile(np.log(xi).reshape(-1, 1).T,
                                   [self.N, self.L, self.Mmax, 1]),
                           [1, 2, 3], [2, 3, 1])

    def get_log_bb_sig(self, phi, tau):
        """
        computes the logbinomial probability of Bn|Dn for all point, in all
        clones in the right dimension (N * J * L * Mmax) for computation
        """
        phi_un_bar = np.rollaxis(np.repeat([self.eta], self.J, axis=0), 0, 2) \
            * np.tile(phi.reshape(-1, 1), [self.N, 1, self.Mmax])
        log_bb = _beta_binomial_logpmf(
            np.rollaxis(np.tile(self.B.reshape(-1, 1), [self.J, 1, self.Mmax]),
                        1, 0),
            np.rollaxis(np.tile(self.D.reshape(-1, 1), [self.J, 1, self.Mmax]),
                        1, 0),
            phi_un_bar, tau)
        log_bb_sig = np.rollaxis(np.repeat([log_bb], self.L, axis=0), 0, 3)
        return log_bb_sig

    def get_log_pi_sig(self, pi):
        """
        get log(pi) in the right dimension (N * J * L * Mmax) for computation
        """
        return np.moveaxis(np.tile(np.log(pi), (self.N, self.Mmax, 1, 1)),
                           [1, 2, 3], [3, 1, 2])

    def compute_F(self, qun, rnus, vmnu, new_xi, new_pi, tau, phi, nu):
        q = self.compute_Q(qun, rnus, vmnu, new_xi, new_pi, tau, phi, nu)

        big_qun = np.moveaxis(np.repeat([qun], self.Mmax, axis=0),
                              [0, 1, 2], [2, 0, 1])

        big_qun_sig = np.moveaxis(np.repeat([big_qun], self.L, axis=0),
                                  [0, 1, 2], [2, 0, 1])
        big_rnus_m = np.moveaxis(np.repeat([rnus], self.Mmax, axis=0),
                                 [0, 1, 2, 3], [3, 0, 1, 2])
        big_vmn_sig = np.moveaxis(np.repeat([vmnu], self.L, axis=0),
                                  [0, 1, 2], [2, 0, 1])

        joint_dist = self._remove_zeros_joint(
            big_qun_sig * big_rnus_m * big_vmn_sig)
        # old version - (joint_dist * np.log(joint_dist)).sum()
        h = - (joint_dist * np.log(joint_dist)).sum()
        return q - h

    def compute_Q(self, qun, rnus, vmnu, new_xi, new_pi, tau, phi, nu):
        log_xi_sig = self.get_log_xi_sig(new_xi)
        log_mu = np.log(self.mu)
        log_pi = self.get_log_pi_sig(new_pi)
        combiln = log_binomial_coeff(self.D, self.B)
        bin_coeff = np.moveaxis(np.tile(combiln, (self.J, self.Mmax, 1)),
                                [0, 1, 2], [1, 2, 0])
        phi_un_bar = np.repeat(self.eta.reshape(self.N, -1, self.Mmax), self.J, axis=1) *\
            np.swapaxes(np.tile(phi, (self.N, self.Mmax, 1)), 1, 2)
        big_b = np.moveaxis(np.tile(self.B, (self.J, self.Mmax, 1)),
                            [0, 1, 2], [1, 2, 0])
        big_d = np.moveaxis(np.tile(self.D, (self.J, self.Mmax, 1)),
                            [0, 1, 2], [1, 2, 0])

        term1 = sp.special.loggamma(big_b + phi_un_bar / tau)
        term2 = sp.special.loggamma((1 - phi_un_bar) / tau + big_d - big_b)
        term3 = sp.special.loggamma(np.ones((self.N, self.J, self.Mmax)) / tau)
        term4 = sp.special.loggamma(np.ones((self.N, self.J, self.Mmax)) / tau + big_d)
        term5 = sp.special.loggamma(phi_un_bar / tau)
        term6 = sp.special.loggamma((1 - phi_un_bar) / tau)

        big_qun = np.moveaxis(np.repeat([qun], self.Mmax, axis=0),
                              [0, 1, 2], [2, 0, 1])

        big_qun_sig = np.moveaxis(np.repeat([big_qun], self.L, axis=0),
                                  [0, 1, 2], [2, 0, 1])
        big_rnus_m = np.moveaxis(np.repeat([rnus], self.Mmax, axis=0),
                                 [0, 1, 2, 3], [3, 0, 1, 2])
        big_vmn_sig = np.moveaxis(np.repeat([vmnu], self.L, axis=0),
                                  [0, 1, 2], [2, 0, 1])

        Q = (big_qun * vmnu * (bin_coeff + term1 + term2 + term3 - term4 - term5 - term6)).sum() + \
            (big_qun_sig * big_rnus_m * big_vmn_sig * (log_xi_sig + log_mu + log_pi + self.lognu_sig)).sum()
        return -Q

    def compute_alternative_Q(self, qun, rnus, vmnu, new_xi, new_pi, tau, phi, nu):
        """
        this function implements another way to compute Q
        on can then test
        self.compute_Q(qun, rnus, new_xi, new_pi, tau, phi) ==\
            self.compute_alternative_Q(qun, rnus, new_xi, new_pi, tau, phi)
        """
        log_xi_sig = self.get_log_xi_sig(new_xi)
        log_bb_sig = self.get_log_bb_sig(phi, tau)
        log_pi = self.get_log_pi_sig(new_pi)
        log_mu = np.log(self.mu)

        big_qun = np.moveaxis(np.repeat([qun], self.Mmax, axis=0),
                              [0, 1, 2], [2, 0, 1])

        big_qun_sig = np.moveaxis(np.repeat([big_qun], self.L, axis=0),
                                  [0, 1, 2], [2, 0, 1])
        big_rnus_m = np.moveaxis(np.repeat([rnus], self.Mmax, axis=0),
                                 [0, 1, 2, 3], [3, 0, 1, 2])
        big_vmn_sig = np.moveaxis(np.repeat([vmnu], self.L, axis=0),
                                  [0, 1, 2], [2, 0, 1])
        Q = (big_qun_sig * big_rnus_m * big_vmn_sig *
             (log_xi_sig + log_mu + log_pi + log_bb_sig + self.lognu_sig) * self.Mask_sig).sum()
        return -Q

    def compute_dQ(self, qun, vmnu, tau, phi):
        # convention : tau and then phi
        dQ = np.zeros(self.J + 1)

        # general stuff
        big_eta = np.repeat(self.eta.reshape(self.N, -1, self.Mmax), self.J, axis=1)
        phi_un_bar = big_eta *\
            np.swapaxes(np.tile(phi, (self.N, self.Mmax, 1)), 1, 2)
        big_b = np.moveaxis(np.tile(self.B, (self.J, self.Mmax, 1)),
                            [0, 1, 2], [1, 2, 0])
        big_d = np.moveaxis(np.tile(self.D, (self.J, self.Mmax, 1)),
                            [0, 1, 2], [1, 2, 0])
        big_qun = np.moveaxis(np.repeat([qun], self.Mmax, axis=0),
                              [0, 1, 2], [2, 0, 1])

        # compute dQ/d\tau
        term1 = sp.special.psi(big_b + phi_un_bar / tau)
        term2 = sp.special.psi((1 - phi_un_bar) / tau + big_d - big_b)
        term3 = sp.special.psi(np.ones((self.N, self.J, self.Mmax)) / tau)
        term4 = sp.special.psi(np.ones((self.N, self.J, self.Mmax)) / tau + big_d)
        term5 = sp.special.psi(phi_un_bar / tau)
        term6 = sp.special.psi((1 - phi_un_bar) / tau)

        dQ[0] = (big_qun * vmnu / (tau**2) * (- phi_un_bar * term1
                                              - (1 - phi_un_bar) * term2
                                              - term3
                                              + term4
                                              + phi_un_bar * term5
                                              + (1 - phi_un_bar) * term6)).sum()

        # compute dQ/f\phi_u
        factor = big_qun * vmnu * big_eta / tau
        dQ[1:] = (factor * (term1 - term2 - term5 + term6)).sum(axis=0).sum(axis=1)

        return -dQ

    def compute_alternative_dQ(self, qun, vmnu, tau, phi):
        """
        this function implements another way to compute dQ
        on can then test
        self.compute_dQ(qun, rnus, new_xi, new_pi, tau, phi)[1:] == \
        self.compute_alternative_dQ(qun, rnus, new_xi, new_pi, tau, phi)[1:]
        # only implemented for dQ/d\phi as \tau=1/\tau, so not the same
        """
        dQ = np.zeros(self.J + 1)
        big_eta = np.repeat(self.eta.reshape(self.N, -1, self.Mmax),
                            self.J, axis=1)
        big_qun = np.moveaxis(np.repeat([qun], self.Mmax, axis=0),
                              [0, 1, 2], [2, 0, 1])
        for mut in range(self.N):
            bn = np.moveaxis(np.tile(np.arange(self.B[mut]),
                                     (self.J, self.Mmax, 1)),
                             [0, 1, 2], [1, 2, 0])
            dn = np.moveaxis(np.tile(np.arange(self.D[mut]),
                                     (self.J, self.Mmax, 1)),
                             [0, 1, 2], [1, 2, 0])
            phi_big1 = np.swapaxes(np.tile(phi, (int(self.B[mut]), self.Mmax, 1)),
                                   1, 2)
            to_add_phi_1 = (big_eta[mut, :, :] /
                            (big_eta[mut, :, :] * phi_big1 + bn * tau)).sum(axis=0)
            bndn = np.moveaxis(np.tile(np.arange(self.D[mut] - self.B[mut]),
                                       (self.J, self.Mmax, 1)),
                               [0, 1, 2], [1, 2, 0])
            phi_big2 = np.swapaxes(np.tile(phi, (int(self.D[mut] - self.B[mut]),
                                                 self.Mmax, 1)), 1, 2)
            to_add_phi_2 = (-big_eta[mut, :, :] /
                            (1 - phi_big2 * big_eta[mut, :, :] + bndn * tau))\
                .sum(axis=0)
            dQ[1:] += (big_qun[mut, :, :] * vmnu[mut, :, :] *
                       (to_add_phi_1 + to_add_phi_2)).sum(axis=1)
            to_add_tau_1 = (bn / (big_eta[mut, :, :] * phi_big1 + bn * tau))\
                .sum(axis=0)
            to_add_tau_2 = (bndn /
                            (1 - phi_big2 * big_eta[mut, :, :] + bndn * tau))\
                .sum(axis=0)
            to_add_tau_3 = (dn / (1 + dn * tau)).sum(axis=0)
            dQ[0] += (big_qun[mut, :, :] * vmnu[mut, :, :] *
                      (to_add_tau_1 + to_add_tau_2 - to_add_tau_3)).sum()
        return -dQ

    def compute_dQ2(self, qun, vmnu, tau, phi):
        dQ2 = np.zeros((self.J + 1, self.J + 1))

        big_eta = np.repeat(self.eta.reshape(self.N, -1, self.Mmax),
                            self.J, axis=1)
        phi_un_bar = big_eta *\
            np.swapaxes(np.tile(phi, (self.N, self.Mmax, 1)), 1, 2)
        big_b = np.moveaxis(np.tile(self.B, (self.J, self.Mmax, 1)),
                            [0, 1, 2], [1, 2, 0])
        big_d = np.moveaxis(np.tile(self.D, (self.J, self.Mmax, 1)),
                            [0, 1, 2], [1, 2, 0])

        term1_0 = sp.special.psi(big_b + phi_un_bar / tau)
        term2_0 = sp.special.psi((1 - phi_un_bar) / tau + big_d - big_b)
        term3_0 = sp.special.psi(np.ones((self.N, self.J, self.Mmax)) / tau)
        term4_0 = sp.special.psi(np.ones((self.N, self.J, self.Mmax)) / tau + big_d)
        term5_0 = sp.special.psi(phi_un_bar / tau)
        term6_0 = sp.special.psi((1 - phi_un_bar) / tau)

        term1_1 = sp.special.polygamma(1, big_b + phi_un_bar / tau)
        term2_1 = sp.special.polygamma(1, (1 - phi_un_bar) / tau + big_d - big_b)
        term3_1 = sp.special.polygamma(1, np.ones((self.N, self.J, self.Mmax)) / tau)
        term4_1 = sp.special.polygamma(1, np.ones((self.N, self.J, self.Mmax)) / tau + big_d)
        term5_1 = sp.special.polygamma(1, phi_un_bar / tau)
        term6_1 = sp.special.polygamma(1, (1 - phi_un_bar) / tau)

        u_prime_v = (2 / (tau**3)) * (phi_un_bar * term1_0
                                      + (1 - phi_un_bar) * term2_0
                                      + term3_0
                                      - term4_0
                                      - phi_un_bar * term5_0
                                      - (1 - phi_un_bar) * term6_0)

        v_prime_u = (phi_un_bar**2 / tau**4 * term1_1
                     + (1 - phi_un_bar)**2 / tau**4 * term2_1
                     + term3_1 / tau**4
                     - term4_1 / tau**4
                     - phi_un_bar**2 / tau**4 * term5_1
                     - (1 - phi_un_bar)**2 / tau**4 * term6_1)

        big_qun = np.moveaxis(np.repeat([qun], self.Mmax, axis=0),
                              [0, 1, 2], [2, 0, 1])

        dQ2[0] = (big_qun * vmnu * (u_prime_v + v_prime_u)).sum()

        factor = big_qun * vmnu * big_eta / tau**2
        dQ2[0, 1:] = dQ2[1:, 0] = (factor * (- term1_0
                                             - phi_un_bar / tau * term1_1
                                             + term2_0
                                             + (1 - phi_un_bar) / tau * term2_1
                                             + term5_0
                                             + phi_un_bar / tau * term5_1
                                             - term6_0
                                             - (1 - phi_un_bar) / tau * term6_1)).sum(axis=0).sum(axis=1)

        dQ2[np.arange(1, self.J+1), np.arange(1, self.J+1)] = \
            (big_qun * vmnu * big_eta**2 / tau**2 * (term1_1 + term2_1 - term5_1 - term6_1)).sum(axis=0).sum(axis=1)

        return -dQ2

    def compute_alternative_dQ2(self, qun, vmnu, tau, phi):
        """
        this function implements another way to compute dQ
        on can then test
        self.compute_dQ(qun, rnus, new_xi, new_pi, tau, phi)[1:] == \
        elf.compute_alternative_dQ(qun, rnus, new_xi, new_pi, tau, phi)[1:]
        # only implemented for dQ/d\phi as \tau=1/\tau, so not the same
        """
        dQ2 = np.zeros((self.J + 1, self.J + 1))
        big_eta = np.repeat(self.eta.reshape(self.N, -1, self.Mmax),
                            self.J, axis=1)
        big_qun = np.moveaxis(np.repeat([qun], self.Mmax, axis=0),
                              [0, 1, 2], [2, 0, 1])
        for mut in range(self.N):
            bn = np.moveaxis(np.tile(np.arange(self.B[mut]),
                                     (self.J, self.Mmax, 1)),
                             [0, 1, 2], [1, 2, 0])
            dn = np.moveaxis(np.tile(np.arange(self.D[mut]),
                                     (self.J, self.Mmax, 1)),
                             [0, 1, 2], [1, 2, 0])
            phi_big1 = np.swapaxes(np.tile(phi, (int(self.B[mut]), self.Mmax, 1)),
                                   1, 2)
            to_add_phi_1 = (big_eta[mut, :, :]**2 / (big_eta[mut, :, :] * phi_big1 + bn * tau)**2).sum(axis=0)
            bndn = np.moveaxis(np.tile(np.arange(self.D[mut] - self.B[mut]),
                                       (self.J, self.Mmax, 1)),
                               [0, 1, 2], [1, 2, 0])
            phi_big2 = np.swapaxes(np.tile(phi, (int(self.D[mut] - self.B[mut]), self.Mmax, 1)),
                                   1, 2)
            to_add_phi_2 = (big_eta[mut, :, :]**2 / (1 - phi_big2 * big_eta[mut, :, :] + bndn * tau)**2).sum(axis=0)
            dQ2[np.arange(1, self.J+1), np.arange(1, self.J+1)] += (big_qun[mut, :, :] * vmnu[mut, :, :] * (- to_add_phi_1 - to_add_phi_2)).sum(axis=1)

            to_add_tau_1 = (bn**2 / ((big_eta[mut, :, :] * phi_big1 + bn * tau))**2).sum(axis=0)
            to_add_tau_2 = (bndn**2 / ((1 - phi_big2 * big_eta[mut, :, :] + bndn * tau))**2).sum(axis=0)
            to_add_tau_3 = (dn**2 / ((1 + dn * tau))**2).sum(axis=0)
            dQ2[0, 0] += (big_qun[mut, :, :] * vmnu[mut, :, :] * (- to_add_tau_1 - to_add_tau_2 + to_add_tau_3)).sum()

            to_add_phi_tau_1 = (big_eta[mut, :, :] * bn / ((big_eta[mut, :, :] * phi_big1 + bn * tau)**2)).sum(axis=0)
            to_add_phi_tau_2 = (big_eta[mut, :, :] * bndn / ((1 - phi_big2 * self.eta[mut] + bndn * tau)**2)).sum(axis=0)
            dQ2[1:, 0] += (big_qun[mut, :, :] * vmnu[mut, :, :] * (- to_add_phi_tau_1 + to_add_phi_tau_2)).sum(axis=1)
            dQ2[0, 1:] += (big_qun[mut, :, :] * vmnu[mut, :, :] * (- to_add_phi_tau_1 + to_add_phi_tau_2)).sum(axis=1)
        return -dQ2


    @staticmethod
    def _remove_zeros_joint(joint):
        pre_joint = joint + mixin_init_parameters.ZERO_PADDING * (joint == 0)
        joint_norm = pre_joint / pre_joint.sum(axis=3).sum(axis=2).sum(axis=1)[:, np.newaxis, np.newaxis, np.newaxis]
        return joint_norm

    @staticmethod
    def _get_binded_variables(L, R, x, dQ, epsilon):
        B_left = (x <= L + epsilon) & (dQ > 0)
        B_right = (x >= R - epsilon) & (dQ < 0)
        return B_left | B_right

    @staticmethod
    def _compute_right_term(x, x_new, B, dQ, dQ2, L, R):
        pre_lam = np.linalg.inv(dQ2).dot(dQ)[~B]
        left_part = np.sum((dQ[~B]).dot(pre_lam))
        right_part = np.sum(((dQ[B]).dot((x - x_new)[B])))
        return left_part, right_part

    @property
    def get_responsabilities(self):
        # compute qun
        log_xi_sig = self.get_log_xi_sig(self.xi)
        log_bb_sig = self.get_log_bb_sig(self.phi, self.tau)
        log_pi = self.get_log_pi_sig(self.pi)

        pre_altfinal = np.exp(log_xi_sig + log_bb_sig + log_pi + np.log(self.mu) + self.lognu_sig)
        pre_altfinal[pre_altfinal == 0] = 2 * sys.float_info.min 
        final = pre_altfinal * self.Mask_sig
        pre_qun = final.sum(axis=3).sum(axis=2)
        row_sums = pre_qun.sum(axis=1)
        qun = pre_qun / row_sums[:, np.newaxis]

        # compute vmnu
        pre_vmnu = final.sum(axis=2)
        row_sums = pre_vmnu.sum(axis=2)
        vmnu = pre_vmnu / np.expand_dims(row_sums, axis=2)

        # compute rnus
        pre_rnus = np.exp(log_pi[:, :, :, 0] + np.log(self.mu)[:, :, :, 0])
        row_sums = pre_rnus.sum(axis=2)
        rnus = pre_rnus / row_sums[:, :, np.newaxis]
        return qun, vmnu, rnus

    def fit(self, epsilon_em=None, epsilon_newton=10**-6, epsilon_box=10**-6,
            beta=0.5, sigma=0.25):
        if epsilon_em is None:
            epsilon_em = 10**-5 * self.J * self.L
        new_theta = self.get_theta * 10000
        em = 0
        while (np.sqrt(np.sum((new_theta-self.get_theta)**2)) > epsilon_em) and (em < self.maxiter):
            if self.verbose:
                print(em, self.xi, self.phi, self.tau, np.sqrt(np.sum((new_theta - self.get_theta)**2)))
            em = em + 1
            new_theta = self.get_theta
            ###########
            # E-phase #
            ###########
            qun, vmnu, rnus = self.get_responsabilities

            ###########
            # M-phase #
            ###########
            new_xi = qun.sum(axis=0) / self.N

            pre_new_pi = (rnus*np.rollaxis(np.repeat([qun], self.L, axis=0), 0, 3)).sum(axis=0)/np.rollaxis(np.repeat([qun], self.L, axis=0), 0, 3).sum(axis=0)
            new_pi = self._remove_zeros_distrib(pre_new_pi)


            L = np.zeros(self.J + 1) + 1e-5
            R = np.ones(self.J + 1) - 1e-5
            R[0] = 0.5

            # newton method
            x_0 = np.concatenate(([self.tau], self.phi))
            x = x_0
            currentQ = self.compute_Q(qun, rnus, vmnu, new_xi, new_pi, x[0], x[1:], self.nu)
            newt = 0
            while True:
                if self.verbose:
                    print('newt', newt, x, currentQ)
                newt = newt + 1
                dQ2 = self.compute_dQ2(qun, vmnu, x[0], x[1:])
                dQ = self.compute_dQ(qun, vmnu, x[0], x[1:])
                if (np.sum(dQ) == 0) & (em == 1):
                    break

                # get epsilon
                tmp_new_x = _get_projected(L, R, x - dQ)
                eps_k = min(epsilon_box, np.sqrt(np.sum((x - tmp_new_x)**2)))

                # get I^{sharp} (binded variables)
                B = self._get_binded_variables(L, R, x, dQ, eps_k)

                # get D
                dQ2[:, B] = dQ2[B, :] = 0
                dQ2[B, B] = 1

                # get alpha k
                m = 1
                alpha = 1
                x_new = _get_projected(L, R, x - alpha * np.linalg.inv(dQ2).dot(dQ))
                new_Q = self.compute_Q(qun, rnus, vmnu, new_xi, new_pi, x_new[0], x_new[1:], self.nu)
                left_part, right_part = self._compute_right_term(x, x_new, B, dQ, dQ2, L, R)

                # deal with cases where the hessian in indefinite!
                if left_part < 0:
                    eigenvalues = np.linalg.eigvals(dQ2)
                    if min(eigenvalues) < 0:
                        to_add_eig = np.abs(min(eigenvalues)) + np.finfo(np.float32).eps
                        dQ2 = dQ2 + to_add_eig * np.identity(len(x))
                        x_new = _get_projected(L, R, x - alpha * np.linalg.inv(dQ2).dot(dQ))
                        new_Q = self.compute_Q(qun, rnus, vmnu, new_xi, new_pi, x_new[0], x_new[1:], self.nu)
                        left_part, right_part = \
                            self._compute_right_term(x, x_new, B, dQ, dQ2, L, R)
                # stopping criterion
                if left_part + right_part < epsilon_newton:
                    break
                # line search
                right_term = sigma * (beta * alpha * left_part + right_part)
                if self.verbose:
                    print('linesearch_before', m, currentQ, new_Q, right_term,
                          x, x_new)
                while not ((currentQ - new_Q) >= right_term):
                    if self.verbose:
                        print('linesearch', m, currentQ, new_Q,
                              right_term, x, x_new)
                    m += 1
                    alpha = beta * alpha
                    x_new = _get_projected(L, R, x - alpha * np.linalg.inv(dQ2).dot(dQ))
                    new_Q = self.compute_Q(qun, rnus, vmnu, new_xi, new_pi, x_new[0], x_new[1:], self.nu)
                    left_part, right_part = self._compute_right_term(x, x_new, B, dQ, dQ2, L, R)
                    right_term = sigma * (beta * alpha * left_part + right_part)
                x = x_new
                currentQ = self.compute_Q(qun, rnus, vmnu, new_xi, new_pi, x[0], x[1:], self.nu)

            new_tau = x[0]
            new_phi = x[1:]

            self.xi = new_xi
            self.pi = new_pi
            self.tau = new_tau
            self.phi = new_phi
            self.qun = qun
            self.rnus = rnus
            self.vmnu = vmnu
            currentQ = self.compute_F(qun, rnus, vmnu, new_xi, new_pi, x[0], x[1:], self.nu)
            if self.save_trace:
                self.Fs.append(currentQ)
            self.Fs.append(currentQ)

    @property
    def get_k(self):
        return self.J * (self.L - 1 + 2)

    @property
    def get_k_cn(self):
        """
        k is the number of parameters of the model
        using this function to test several values
        wrt data. True value is
        self.J * (self.L - 1 + 2)
        (self.L - 1) because of 1 degree of liberty in pi
        2 for phi and xi
        -1 because xi lacks a degree of freedom
        1 on top for tau/rho
        np.mean(self.C_tumor_major) to account for the extra degree of freedom
        of the model due to fitting the copy number.
        """
        return self.J * (self.L - 1 + 2) * np.mean(self.C_tumor_major)

    @property
    def get_k_dof_cn(self):
        """
        same as get_k_cn but with the degree of freedom of the input signature
        matrix instead of the number of signatures.
        """
        return self.J * (self.dof - 1 + 2) * np.mean(self.C_tumor_major)

    def get_bic(self, dof=False, cn=False):
        if not cn:
            k = self.get_k
        else:
            if dof:
                k = self.get_k_dof_cn
            else:
                k = self.get_k_cn
        return - k * np.log(self.N) / 2 + self.get_loglikelihood

    def get_bic_heuristics(self, dof=True, factor=0.042, cn=False):
        """
        the factor is valid for dof=True
        (O.O65 for a subset, O.O34 for 47 signatures)
        otherwise, we advise factor around 0.015 for the 47 signatures
        or around 0.040 for a subset of signatures.
        """
        if not cn:
            k = self.get_k
        else:
            if dof:
                k = self.get_k_dof_cn
            else:
                k = self.get_k_cn
        return - factor * k * np.log(self.N) + self.get_loglikelihood

    def get_aicc(self, dof=False, cn=False):
        if not cn:
            k = self.get_k
        else:
            if dof:
                k = self.get_k_dof_cn
            else:
                k = self.get_k_cn
        try:
            return self.get_aic(dof, cn) - k * (k + 1) / (self.N - k - 1)
        except ZeroDivisionError:
            return np.nan

    def get_aic(self, dof=False, cn=False):
        if not cn:
            k = self.get_k
        else:
            if dof:
                k = self.get_k_dof_cn
            else:
                k = self.get_k_cn
        return - k + self.get_loglikelihood

    def get_icl(self, norm=False):
        clo = self.qun.argmax(axis=1)
        sig = self.rnus[np.arange(self.N), self.qun.argmax(axis=1), :].argmax(axis=1)
        cn = self.vmnu[np.arange(self.N), self.qun.argmax(axis=1), :].argmax(axis=1)

        big_qun = np.moveaxis(np.repeat([self.qun], self.Mmax, axis=0),
                              [0, 1, 2], [2, 0, 1])

        big_qun_sig = np.moveaxis(np.repeat([big_qun], self.L, axis=0),
                                  [0, 1, 2], [2, 0, 1])
        big_rnus_m = np.moveaxis(np.repeat([self.rnus], self.Mmax, axis=0),
                                 [0, 1, 2, 3], [3, 0, 1, 2])
        big_vmn_sig = np.moveaxis(np.repeat([self.vmnu], self.L, axis=0),
                                  [0, 1, 2], [2, 0, 1])

        joint_dist = self._remove_zeros_joint(
            big_qun_sig * big_rnus_m * big_vmn_sig)
        nn = np.zeros(joint_dist.shape)
        nn[np.arange(self.N), clo, sig, cn] = 1
        # old version - (joint_dist * np.log(joint_dist)).sum()
        h = - (nn * np.log(joint_dist)).sum()
        if norm:
            return self.get_bic() + h / self.N
        else:
            return self.get_bic() + h

    @property
    def get_loglikelihood(self):
        log_xi_sig = self.get_log_xi_sig(self.xi)
        log_bb_sig = self.get_log_bb_sig(self.phi, self.tau)
        log_pi = self.get_log_pi_sig(self.pi)
        big_mask = np.swapaxes(np.repeat([self.Mask], self.J, axis=0), 0, 1)

        final = np.exp(log_xi_sig + log_bb_sig + log_pi + np.log(self.mu) + self.lognu_sig)
        return np.log((final.sum(axis=2) * big_mask).sum(axis=2).sum(axis=1)).sum()


def main():
    pass

if __name__ == '__main__':
    main()

"""
epsilon_em=10**-3
epsilon_newton=10**-6
epsilon_box=10**-6
beta=0.5
sigma=0.25
epsilon_xi=1e-10
"""
