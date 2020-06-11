#!/usr/bin/env python
# -*- coding:utf-8 -*-
# TODO(lpe): Include in package
import pandas as pd
import os
from sklearn import preprocessing
import numpy as np
from scipy.stats import beta
import pathlib
import pickle
from clonesig.estimator import Estimator
from clonesig import mixin_init_parameters

try:
    rows, columns = os.popen('stty size', 'r').read().split()
    pd.set_option('display.width', int(columns))
    pd.set_option('display.max_columns', 200)
except:
    print("running on server, otherwise please investigate")


def get_all_patterns():
    pat = list()
    for ref in ('C', 'T'):
        alt_list = ['A', 'C', 'G', 'T']
        alt_list.remove(ref)
        for alt in alt_list:
            for b1 in ('A', 'C', 'G', 'T'):
                for b2 in ('A', 'C', 'G', 'T'):
                    pat.append('{}[{}>{}]{}'.format(b1, ref, alt, b2))
    return pat
PAT_LIST = get_all_patterns()
LE = preprocessing.LabelEncoder()
LE.fit(PAT_LIST)


def get_context(x):
    match_dict = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    if x['ref'] in ('C', 'T'):
        context = x['CONTEXT'][4] + '[' + x['ref'] + '>' + x['alt'] + ']' + \
            x['CONTEXT'][6]
    else:
        context = match_dict[x['CONTEXT'][6]] + '[' + match_dict[x['ref']] +\
            '>' + match_dict[x['alt']] + ']' + match_dict[x['CONTEXT'][4]]
    return context


def beta_binomial(n, phi, rho, size=None):
    alpha = phi * rho
    beta = rho - alpha
    p = np.random.beta(alpha, beta, size)
    X = np.random.binomial(n, p, size)
    return X


""" test values
filename_maf = 'tmp/useful_final_qc_merge_cnv_purity.csv'
patient = 'TCGA-CV-A45Y'
folder = 'protected_hg38_vcf'
"""


class MAFLoader:
    """
    abstract class for loader. For now it is not abstract, but it will
    be to allow to make similar classes to load from other types of input
    like vcf etc.
    """
    def __init__(self, filename_maf, patient, folder):
        self.filename_maf = filename_maf
        self.patient = patient
        self.folder = folder
        self.B = None
        self.D = None
        self.T = None
        self.C_normal = None
        self.C_tumor_minor = None
        self.C_tumor_tot = None
        self.purity = None

    def load(self):
        maf = pd.read_csv(self.filename_maf, sep='\t')
        mut_filename = 'results/{}/pyclone/{}/input.tsv'.format(self.patient,
                                                                self.folder)
        mutations = pd.read_csv(mut_filename, sep='\t')
        maf = maf.assign(
            mutation_id=maf['mutation_id.1'].str[29:-2] +
            maf['mutation_id.1'].str[-1])
        mutations_context = pd.merge(
            mutations, maf[['mutation_id', 'CONTEXT']].drop_duplicates(),
            left_on='mutation_id', right_on='mutation_id', how='left')
        mutations_context = mutations_context.assign(
            ref=mutations_context.mutation_id.str[-2])
        mutations_context = mutations_context.assign(
            alt=mutations_context.mutation_id.str[-1])
        mutations_context = mutations_context.assign(
            pattern=pd.Categorical(
                mutations_context.apply(get_context, axis=1),
                categories=PAT_LIST, ordered=True))
        mutations_context_nona = mutations_context.dropna(how='any', axis=0)
        self.B = mutations_context_nona.var_counts.values
        self.D = self.B + mutations_context_nona.ref_counts.values
        self.T = LE.transform(mutations_context_nona.pattern)
        self.C_normal = mutations_context_nona.normal_cn
        self.C_tumor_minor = mutations_context_nona.minor_cn
        self.C_tumor_tot = self.C_tumor_minor + mutations_context_nona.major_cn
        with open('results/{}/pyclone/ascat_purity.txt'
                  .format(self.patient), 'r') as pf:
            self.purity = float(pf.read())


class DataWriter():
    def _get_data_df(self):
        return self.data

    def _get_cn_profile_df(self):
        return self.cn_profile

    def write_clonesig(self, foldername):
        data_df = self._get_data_df()
        pathlib.Path(foldername).mkdir(parents=True, exist_ok=True)
        data_df.to_csv('{}/input_t.tsv'.format(foldername), sep='\t',
                       index=False)
        with open('{}/purity.txt'.format(foldername), 'w') as f:
                f.write('{}'.format(self.purity))

    def write_tracksig(self, foldername, sample_id=None):
        data_df = self._get_data_df()
        pathlib.Path(foldername).mkdir(parents=True, exist_ok=True)
        if sample_id is None:
            sample_id = foldername

        data_df = data_df.assign(vaf=data_df.var_counts /
                                 (data_df.ref_counts + data_df.var_counts))
        data_df = data_df.assign(
            total_cn=lambda x: x['minor_cn'] + x['major_cn'])
        data_df = data_df.assign(
            vaf_cn=data_df.vaf * data_df['total_cn'] / data_df['mut_cn'])
        data_df = data_df.assign(
            vaf_purity=data_df.apply(
                lambda x: x['vaf']/self.purity *
                ((1 - self.purity) * 2 + self.purity * x['total_cn']) /
                x['mut_cn'], axis=1))
        data_df.sort_values(by='vaf_purity', inplace=True)
        data_df.reset_index(inplace=True, drop=True)

        data_df = data_df.assign(mutation_group=lambda x: x.index//100)
        nbin = len(data_df)//100
        data_df = data_df[data_df.mutation_group <= nbin - 1]
        trinucleotide_count = pd.pivot_table(index=['mutation_group'],
                                             columns=['trinucleotide'],
                                             values=['mutation_id'],
                                             aggfunc='count',
                                             data=data_df,
                                             dropna=False)\
            .fillna(0).astype(int)
        trinucleotide_count.columns = trinucleotide_count.columns\
            .droplevel().astype('str')
        trinucleotide_count_vaf = pd.merge(
            data_df.groupby('mutation_group').vaf_purity.mean().to_frame(),
            trinucleotide_count, left_index=True, right_index=True)
        trinucleotide_count_vaf.insert(
            loc=0, column='sample_name',
            value=['{}_tracksig'.format(sample_id)] *
            len(trinucleotide_count_vaf))
        tracksig_outdir = '{}/tracksig'.format(foldername)
        pathlib.Path(tracksig_outdir).mkdir(parents=True, exist_ok=True)
        trinucleotide_count_vaf.to_csv(
            '{}/batch_100_pattern96.csv'.format(tracksig_outdir),
            sep='\t', index=False, header=False)

    def write_tracksigfreq(self, foldername, sample_id=None):
        data_df = self._get_data_df()
        pathlib.Path(foldername).mkdir(parents=True, exist_ok=True)
        if sample_id is None:
            sample_id = foldername

        data_df = data_df.assign(phat=beta.rvs(data_df.var_counts + 1,
                                               data_df.ref_counts + 1))
        data_df = data_df.assign(phi=(2 + self.purity *
                                      (data_df.major_cn +
                                       data_df.minor_cn - 2)) *
                                 data_df.phat)
        data_df = data_df.assign(qi=beta.rvs(data_df.var_counts + 1,
                                             data_df.ref_counts + 1))
        data_df = data_df.assign(
            qi=data_df.apply(lambda x: min(1, x.qi), axis=1))
        data_df.sort_values(by='phi', inplace=True, ascending=False)
        data_df.reset_index(inplace=True, drop=True)
        data_df = data_df.assign(bin=lambda x: x.index//100 + 1)
        nbin = len(data_df)//100
        data_df = data_df[data_df.bin <= nbin]
        trinucleotide_count = pd.pivot_table(index=['bin'],
                                             columns=['trinucleotide'],
                                             values=['mutation_id'],
                                             aggfunc='count',
                                             data=data_df,
                                             dropna=False)\
            .fillna(0).astype(int)
        LE_PAT_tracksig = [c[2] + '_' + c[4] + '_' + c[0] + c[2] + c[6]
                           for c in PAT_LIST]
        trinucleotide_count.columns = trinucleotide_count.columns\
            .droplevel().astype('str')
        trinucleotide_count.columns = LE_PAT_tracksig
        tracksigfreq_outdir = '{}/tracksigfreq'.format(foldername)
        pathlib.Path(tracksigfreq_outdir).mkdir(parents=True, exist_ok=True)
        trinucleotide_count.T.to_csv(
            '{}/batch_100_pattern96.csv'.format(tracksigfreq_outdir),
            sep='\t', index=True, header=True)
        data_df.to_csv('{}/vcaf.csv'.format(tracksigfreq_outdir),
                       sep='\t', index=False, header=True)

    def write_palimpsest(self, foldername):
        # deal with mutation data
        data_df = self._get_data_df()
        palimpsest_snv_fields = ['Sample', 'Type', 'CHROM', 'POS', 'REF',
                                 'ALT', 'Tumor_Varcount', 'Tumor_Depth',
                                 'Normal_Depth',  # 'Gene_Name', 'Driver',
                                 'substype', 'context3', 'mutcat3']
        data_df = data_df.assign(Sample=foldername.split('/')[1])
        data_df = data_df.assign(Type='SNV')
        data_df = data_df.assign(CHROM='chr' + data_df.chromosome.astype(str))
        data_df = data_df.assign(POS=data_df.position)
        data_df = data_df.assign(
            REF=data_df.apply(
                lambda x: PAT_LIST[x['trinucleotide']][2], axis=1))
        data_df = data_df.assign(
            ALT=data_df.apply(
                lambda x: PAT_LIST[x['trinucleotide']][4], axis=1))
        data_df = data_df.assign(Tumor_Varcount=data_df.var_counts)
        data_df = data_df.assign(
            Tumor_Depth=data_df.var_counts + data_df.ref_counts)
        data_df = data_df.assign(Normal_Depth=data_df.Tumor_Depth)
        # data_df = data_df.assign(Gene_Name='')
        # data_df = data_df.assign(Driver='')
        data_df = data_df.assign(
            substype=data_df.apply(
                lambda x: PAT_LIST[x['trinucleotide']][2:5].replace('>', ''),
                axis=1))
        data_df = data_df.assign(
            context3=data_df.apply(
                lambda x: ''.join(
                    [PAT_LIST[x['trinucleotide']][i] for i in [0, 2, 6]]),
                axis=1))
        data_df = data_df.assign(
            mutcat3=data_df.substype + '_' + data_df.context3)
        pathlib.Path(foldername).mkdir(parents=True, exist_ok=True)
        palimpsest_outdir = '{}/palimpsest'.format(foldername)
        pathlib.Path(palimpsest_outdir).mkdir(parents=True, exist_ok=True)
        data_df[palimpsest_snv_fields].to_csv(
            '{}/vcf_table.csv'.format(palimpsest_outdir),
            sep='\t', index=False)

        # deal with copy number data
        cn_df = self._get_cn_profile_df()
        palimpsest_cna_fields = ['Sample', 'CHROM', 'POS_START', 'POS_END',
                                 'LogR', 'ntot', 'Nmin', 'Nmaj', 'Ploidy']
        cn_df = cn_df.assign(Sample=foldername.split('/')[1])
        cn_df = cn_df.assign(CHROM='chr' + cn_df.chromosome.astype(str))
        cn_df = cn_df.assign(POS_START=cn_df.start)
        cn_df = cn_df.assign(POS_END=cn_df.end)
        cn_df = cn_df.assign(LogR=np.log2((cn_df.minor + cn_df.major)/2))
        cn_df = cn_df.assign(ntot=cn_df.minor + cn_df.major)
        cn_df = cn_df.assign(Nmin=cn_df.minor)
        cn_df = cn_df.assign(Nmaj=cn_df.major)
        cn_df = cn_df.assign(Ploidy=cn_df.ntot.mean())
        cn_df[palimpsest_cna_fields].to_csv(
            '{}/cna_table.csv'.format(palimpsest_outdir),
            sep='\t', index=False)

        # deal with annotation data
        annot_df = pd.DataFrame([[foldername.split('/')[1], 'F', self.purity]],
                                columns=['Sample', 'Gender', 'Purity'])
        annot_df.to_csv('{}/annot_table.csv'.format(palimpsest_outdir),
                        sep='\t', index=False)

    def write_deconstructsig(self, foldername):
        data_df = self._get_data_df()
        pathlib.Path(foldername).mkdir(parents=True, exist_ok=True)
        deconstructsigs_outdir = '{}/deconstructsigs'.format(foldername)
        pathlib.Path(deconstructsigs_outdir).mkdir(parents=True, exist_ok=True)
        data_df.groupby('trinucleotide').mutation_id.count().to_frame().T\
            .to_csv('{}/pattern96.csv'.format(deconstructsigs_outdir),
                    sep='\t', index=False)

    def write_pyclone_sciclone_ccube(self, foldername):
        data_df = self._get_data_df()
        cn_df = self._get_cn_profile_df()
        pyclone_fields = ['mutation_id', 'ref_counts', 'var_counts',
                          'normal_cn', 'minor_cn', 'major_cn']
        pyclone_df = data_df[pyclone_fields]
        pyclone_outdir = '{}/pyclone'.format(foldername)
        sciclone_outdir = '{}/sciclone'.format(foldername)
        ccube_outdir = '{}/ccube'.format(foldername)
        pathlib.Path(pyclone_outdir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(sciclone_outdir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(ccube_outdir).mkdir(parents=True, exist_ok=True)
        for f in pyclone_fields[1:]:
            pyclone_df = pyclone_df.assign(**{f: pyclone_df[f].astype(int)})
        pyclone_df.to_csv('{}/input.tsv'.format(pyclone_outdir),
                          sep='\t', index=False)
        cn_df.to_csv('{}/cnv_table.csv'.format(foldername),
                     sep='\t', index=False)
        ccube_fields = ["mutation_id", "minor_cn", "major_cn",
                        "total_cn", "purity", "normal_cn",
                        "total_counts", "var_counts", "ref_counts"]
        data_df = data_df.assign(total_counts=data_df.total_cn)
        data_df = data_df.assign(purity=self.purity)
        ccube_df = data_df[ccube_fields]
        ccube_df.to_csv('{}/input.tsv'.format(ccube_outdir),
                        sep='\t', index=False)

    def write_dpclust(self, foldername):
        data_df = self._get_data_df()
        dpclust_outdir = '{}/dpclust'.format(foldername)
        pathlib.Path(dpclust_outdir).mkdir(parents=True, exist_ok=True)
        dpclust_fields = ['chr', 'end', 'WT.count', 'mut.count',
                          'subclonal.CN', 'mutation.copy.number',
                          'subclonal.fraction', 'no.chrs.bearing.mut']
        data_df = data_df.assign(vaf=data_df.var_counts /
                                 (data_df.ref_counts + data_df.var_counts))
        data_df = data_df.assign(
            total_cn=lambda x: x['minor_cn'] + x['major_cn'])
        data_df = data_df.assign(
            vaf_purity=data_df.apply(
                lambda x: x['vaf']/self.purity *
                ((1 - self.purity) * 2 + self.purity * x['total_cn']), axis=1))
        data_df = data_df.assign(multi=data_df.apply(
            lambda x: int(np.round(x.vaf_purity)) if x.vaf_purity > 1 else 1,
            axis=1))
        data_df = data_df.assign(ccf=data_df.apply(
            lambda x: min(1, x.vaf_purity / x.multi), axis=1))

        data_df = data_df.assign(**{'chr': data_df.chromosome,
                                    'end': data_df.position,
                                    'WT.count': data_df.ref_counts,
                                    'mut.count': data_df.var_counts,
                                    'subclonal.CN': data_df.total_cn,
                                    'mutation.copy.number': data_df.vaf_purity,
                                    'subclonal.fraction': data_df.ccf,
                                    'no.chrs.bearing.mut': data_df.multi})
        dpclust_df = data_df[dpclust_fields]
        dpclust_df.to_csv('{}/input.tsv'.format(dpclust_outdir),
                          sep='\t', index=False)
        short_name = foldername.split('/')[-1]
        info_df = pd.DataFrame(data=[[short_name, short_name, 'input.tsv',
                                      self.purity]],
                               columns=['sample', 'subsample', 'datafile',
                                        'cellularity'])
        info_df.to_csv('{}/info.tsv'.format(dpclust_outdir),
                       sep='\t', index=False)

    def write_phylogicndt(self, foldername):
        phylogicndt_outdir = '{}/phylogicndt'.format(foldername)
        pathlib.Path(phylogicndt_outdir).mkdir(parents=True, exist_ok=True)
        phylo_fields = ['Hugo_Symbol', 'Chromosome', 'Start_position',
                        'Reference_Allele', 'Tumor_Seq_Allele2', 't_ref_count',
                        't_alt_count', 'local_cn_a1', 'local_cn_a2']
        data_df = self._get_data_df()
        data_df = data_df.assign(
            total_cn=lambda x: x['minor_cn'] + x['major_cn'])

        data_df = data_df. \
            assign(** {'Hugo_Symbol': 'Unknown',
                       'Chromosome': 'chr' + data_df.chromosome.astype(str),
                       'Start_position': data_df.position,
                       'Reference_Allele': data_df.apply(
                          lambda x: PAT_LIST[x['trinucleotide']][2], axis=1),
                       'Tumor_Seq_Allele2': data_df.apply(
                          lambda x: PAT_LIST[x['trinucleotide']][4], axis=1),
                       't_ref_count': data_df.ref_counts,
                       't_alt_count': data_df.var_counts,
                       'local_cn_a1': data_df.minor_cn,
                       'local_cn_a2': data_df.major_cn})
        data_df[phylo_fields].to_csv('{}/input.maf'.format(phylogicndt_outdir),
                                     sep='\t', index=False)

    def write_object(self, foldername):
        pathlib.Path(foldername).mkdir(parents=True, exist_ok=True)
        with open('{}/sim_data'.format(foldername), 'wb') as sim_data_file:
            my_pickler = pickle.Pickler(sim_data_file)
            my_pickler.dump(self)


class SimLoader(mixin_init_parameters.MixinInitParameters, DataWriter):
    """
    class to simulate data
    """
    def __init__(self, N, J, inputMU=None, xi_param=None, pi_param=None,
                 phi_param=None, rho_param=None, purity_param=None,
                 change_sig_activity=True, cn=True, D_param=None,
                 dip_prop=None):
        """
        N is the number of mutations
        J is the number of clones
        xi_param is parameters to simulate xi,the proportion of mutations
            belonging to each clone if xi is None, xi is drawn from a dirichlet
            process with each alpha set to 1 (equivalent to no prior)
            otherwise, xi_param is a vector of length J, summing to 1
            and we set xi = xi_param
        pi_param is the parameters to set pi, a MxJ proportion of each
            signature in each clone. pi_param can alternatively be a list of
            indexes of active signatures in the truncal clone. pi_param can
            also be an int setting the number of active
            signatures in the truncal clone.
        phi_param is the parameters to set phi, a J-long vector representing
            the cellular prevalence of the clones.
            if phi_param is a vector, we set phi=phi_param
            if phi_param is None, we draw phi as phi[0]=1
                (assume clonal mutations), and iteratively
                we draw phi[i] as a uniform variable on
                [0, phi[i-1]], i=1...J-1
        rho_param is the overdispersion parameter.
            if rho_param is a float, rho=rho_param
            if rho is None, we draw rho from a normal distribution of mean 60
                and variance 5
        purity_param is the purity
            if purity_param is a float, purity=purity_param
            if purity_param is None, we draw purity from a normal distrib of
                mean 0.7 and variance 0.1
        change_sig_activity  is a boolean used to simulate pi, with either a
            constant signature activity between clones
            or with a different signature activity
        dip_prop is the proportion of the genome that is diploid
        """
        self.N = N
        self.J = J
        self.xi_param = xi_param
        self.pi_param = pi_param
        self.phi_param = phi_param
        self.rho_param = rho_param
        self.purity_param = purity_param
        self.change_sig_activity = change_sig_activity
        self.xi = None
        self.pi = None
        self.phi = None
        self.rho = None
        self.purity = None
        self.B = None
        self.D = None
        self.C_normal = None
        self.C_tumor_mut = None
        self.C_tumor_tot = None
        self.C_tumor_minor = None
        self.purity = None
        self.cn = cn
        if inputMU is None:
            self.MU = self.default_mu()
        else:
            self.MU = inputMU
        self.L = self.MU.shape[0]
        self.K = self.MU.shape[1]
        self.D_param = D_param
        self.dip_prop = dip_prop

    def _get_unobserved_nodes(self):
        self.init_params(pi_param=self.pi_param, phi_param=self.phi_param,
                         xi_param=self.xi_param, rho_param=self.rho_param,
                         spasePi=True,
                         change_sig_activity=self.change_sig_activity)

    def _get_observed_nodes(self):
        if self.purity_param is None:
            self.purity = min(np.random.randn() * 0.1 + 0.7, 0.99)
        else:
            self.purity = self.purity_param
        if self.D_param is None:
            self.D = np.random.lognormal(5, 0.7, self.N).astype(int) + 1
        else:
            self.D = (self.D_param * np.ones(self.N)).astype(int)
        # parameters and distribution chosen at random, worth fitting on TCGA
        # data?
        # I made some arbitrary choices of distribution, in particular for D
        # (the coverage) and for copy number. Maybe this should be further
        # parametrized?
        # otherwise it is rather self explanatory with the code.
        self.C_normal = 2 * np.ones(self.N)
        if self.cn:
            if self.dip_prop is None:
                self.C_tumor_tot = np.max(
                    ((np.random.lognormal(1, 0.3, self.N)).astype(int),
                     np.ones(self.N)), axis=0)
                C_strand1 = (np.random.beta(5, 3, self.N) * self.C_tumor_tot)\
                    .astype(int)  # un peu trop de 0 pas forcément super grave
            elif self.dip_prop < 1:
                p = np.zeros(10)
                infl_dip = np.min((1, self.dip_prop * 1.1))
                p[2] = infl_dip
                p[3] = (1 - infl_dip) / 2
                p[4] = ((1 - infl_dip) / 2) / 3 * 2
                p[1] = (1 - p.sum()) / 2
                p[5:] = (1 - p.sum()) * \
                    np.array([0.5, 0.25, 0.125, 1/16, 1/32]) / \
                    np.array([0.5, 0.25, 0.125, 1/16, 1/32]).sum()
                self.C_tumor_tot = np.random.choice(10, p=p, size=self.N)
                idx = False * np.ones(self.N).astype(bool)
                idx[self.C_tumor_tot == 2] = np.random.choice(
                    [True, False], p=[9/11, 2/11],
                    size=sum(self.C_tumor_tot == 2))
                C_strand1 = np.zeros(self.N)
                C_strand1[idx] = 1
                C_strand1[~idx] = (np.random.beta(5, 3, len(C_strand1[~idx])) *
                                   self.C_tumor_tot[~idx]).astype(int)
            else:
                self.C_tumor_tot = 2 * np.ones(self.N)
                C_strand1 = np.ones(self.N)
            max_mut = np.max(np.vstack((self.C_tumor_tot-C_strand1,
                                        C_strand1)), axis=0)
            self.C_tumor_mut = (np.random.rand(self.N) * max_mut)\
                .astype(int) + 1
        else:
            self.C_tumor_tot = 2 * np.ones(self.N)
            C_strand1 = np.ones(self.N)
            self.C_tumor_mut = np.ones(self.N)

        self.C_tumor_minor = np.min(np.vstack((self.C_tumor_tot-C_strand1,
                                               C_strand1)), axis=0)

        self.U = np.random.choice(self.J, self.N, replace=True, p=self.xi)
        self.B = np.zeros(self.N)
        for i in range(self.J):
            phi_bar = self.phi[i] * self.purity * \
                      self.C_tumor_mut[self.U == i] / \
                      ((1-self.purity) * self.C_normal[self.U == i] +
                       self.purity * self.C_tumor_tot[self.U == i])
            self.B[self.U == i] = beta_binomial(self.D[self.U == i],
                                                phi_bar, self.rho,
                                                sum(self.U == i))

        self.S = np.zeros(self.N)
        for i in range(self.J):
            self.S[self.U == i] = np.random.choice(self.L, sum(self.U == i),
                                                   replace=True,
                                                   p=self.pi[i, :])

        self.T = np.zeros(self.N)
        for i in range(self.L):
            self.T[self.S == i] = np.random.choice(self.K, sum(self.S == i),
                                                   replace=True,
                                                   p=self.MU[i, :])

    @property
    def get_loglikelihood(self):
        est = Estimator(self.T, self.B, self.C_normal, self.C_tumor_tot,
                        self.C_tumor_minor, self.D, self.purity, self.J,
                        inputMU=self.MU, pi=self.pi, phi=self.phi, xi=self.xi,
                        tau=self.tau)
        return est.get_loglikelihood

    def _get_data_df(self):
        data_df = pd.DataFrame({'mutation_id': ["mut_{}".format(i) for i in
                                                range(1, self.N+1)],
                                'chromosome': [1] * self.N,
                                'position': np.arange(5, 5 + 10*self.N, 10),
                                'ref_counts': self.D - self.B,
                                'var_counts': self.B,
                                'normal_cn': self.C_normal,
                                'minor_cn': self.C_tumor_minor,
                                'mut_cn': self.C_tumor_mut,
                                'major_cn': self.C_tumor_tot -
                                self.C_tumor_minor,
                                'total_cn': self.C_tumor_tot,
                                'trinucleotide': pd.Categorical(
                                    self.T.astype(int),
                                    categories=list(range(96)),
                                    ordered=True),
                                'signature': self.S.astype(int),
                                'clone': self.U.astype(int)})
        return data_df

    def _get_cn_profile_df(self):
        cn_df = pd.DataFrame({'chromosome': [1] * self.N,
                              'start': np.arange(1, 1 + 10*self.N, 10),
                              'end': np.arange(9, 9 + 10*self.N, 10),
                              'minor': self.C_tumor_minor,
                              'major': self.C_tumor_tot - self.C_tumor_minor})
        return cn_df
