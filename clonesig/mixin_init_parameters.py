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

XI_THRESHOLD = 0.05
ZERO_PADDING = 10**-10


class MixinInitParameters:
    L = None

    @staticmethod
    def default_mu():
        # open COSMIC signature matrix
        sig = pd.read_csv(
            pkg_resources.resource_stream(
                __name__, 'data/sigProfiler_SBS_signatures_2018_03_28.csv'),
            sep=',')
        sig_matrix = sig.values[:, 2:].astype(float).T

        # avoid 0 values so add a small value and renormalize
        new_sig_matrix = sig_matrix + ZERO_PADDING * (sig_matrix == 0)
        return (
            new_sig_matrix / new_sig_matrix.sum(axis=1)[:, np.newaxis]
        )

    @staticmethod
    def _remove_zeros_distrib(M):
        pre_M = M + ZERO_PADDING * (M == 0)
        M_norm = pre_M / pre_M.sum(axis=1)[:, np.newaxis]
        return M_norm

    def init_params(self, pi_param=None, phi_param=None, xi_param=None,
                    rho_param=None, tau_param=None, spasePi=True,
                    change_sig_activity=True):
        # get xi
        if xi_param is None:
            self.xi = np.random.dirichlet(alpha=np.ones(self.J))
            while min(self.xi) < XI_THRESHOLD:
                self.xi = np.random.dirichlet(alpha=np.ones(self.J))
        else:
            self.xi = xi_param

        # get pi
        if pi_param is None:
            if spasePi:
                nb_active_sig = np.min((np.random.poisson(7) + 1, self.L))
                active_signatures = np.random.choice(self.L, nb_active_sig, replace=False)
            else:
                nb_active_sig = self.L
                active_signatures = np.arange(self.L)
            self.pi = np.zeros((self.J, self.L))
            self.pi[0, active_signatures] = np.random.dirichlet(alpha=np.ones(nb_active_sig))
            if change_sig_activity:
                for i in range(1, self.J):
                    self.pi[i, active_signatures] = \
                        np.random.dirichlet(alpha=np.ones(nb_active_sig))
            else:
                self.pi = np.repeat([self.pi[0, :]], self.J)
        elif isinstance(pi_param, Iterable) and (len(np.array(pi_param).shape) == 2):
            self.pi = self._remove_zeros_distrib(pi_param)
        else:
            raise ValueError("Wrong type or size for argument pi.")

        # get phi in decreasing order
        if phi_param is None:
            self.phi = np.zeros(self.J)
            self.phi[0] = 1.0 - 1e-5
            for i in range(1, self.J):
                self.phi[i] = np.random.uniform(low=0.1 + 0.05*(self.J - i - 1),
                                                high=self.phi[i-1] - 0.05)
        else:
            self.phi = phi_param
        # get rho and tau
        if (rho_param is None) and (tau_param is None):
            self.tau = 1 / (np.random.randn() * 5 + 60)
        elif (rho_param is not None) and (tau_param is None):
            self.tau = 1 / rho_param
        elif (tau_param is not None) and (rho_param is None):
            self.tau = tau_param
        else:
            raise ValueError('either rho or tau, or none of them should be specified, not both.')
        self.rho = 1 / self.tau
