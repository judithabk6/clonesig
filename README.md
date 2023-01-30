# CloneSig

The possibility to sequence DNA in cancer samples has triggered much effort recently to identify the forces at the genomic level that shape tumor apparition and evolution. Two main approaches have been followed for that purpose: (i) deciphering the clonal composition of each tumour by using the observed prevalences of somatic mutations, and (ii) elucidating the mutational processes involved in the generation of those same somatic mutations. Currently, both subclonal and mutational signatures deconvolutions are performed separately, while they are both manifestations of the same underlying process.

We present CloneSig, the first method that jointly infers subclonal and mutational signature composition evolution of a tumor sample form bulk sequencing. CloneSig is based on a probabilistic graphical model that models somatic mutations as derived from a mixture of subclones where different mutational signatures are active. Parameters of the model are estimated using an EM algorithm. 

Details of the model, and results on real and simulated data can be found in the corresponding [publication](https://doi.org/10.1038/s41467-021-24992-y), and [its companion repository for analyses](https://github.com/judithabk6/Clonesig_analysis).

Abécassis, J., Reyal, F. & Vert, JP. CloneSig can jointly infer intra-tumor heterogeneity and mutational signature activity in bulk tumor sequencing data. Nat Commun 12, 5352 (2021). https://doi.org/10.1038/s41467-021-24992-y

In release 1.0.1 we have changed the license to the MIT license to enhance share and reuse of CloneSig.

## Installation

Clonesig can be installed by executing
```
python setup.py install
```

Or the package can be directly installed from the GitHub repository using
```
pip install git+git://github.com/judithabk6/clonesig.git
```

Installation time is a few minutes on a standard personal computer.

## Usage
### Input
The easiest was to run CloneSig is by using the wrapper function `run_clonesig` in a python script. The required inputs of this function are 
- `T` : iterable of length N with the trinucleotide context of each mutation, numbered from 0 to 95
- `B` : iterable of length N with the variant allele read count for each mutation
- `D` : iterable of length N with the total read count for each mutation
- `C_normal` : iterable of length N copy number of non-tumor cells in the sample at each mutation locus
- `C_tumor_tot`  : iterable of length N the total copy number of tumor cells in the sample at each mutation locus
- `C_tumor_minor` : iterable of length N the minor copy number of tumor cells in the sample at each mutation locus. If this info is not available, set it to zero so that clonesig considers all possible genotypes
- `purity` : float in [0, 1] an estimate of the tumor purity of the sample
- `inputMU` : array-like (L, 96) known L signatures to be fit by clonesig (can be obtained from the function `get_MU`)

If you use the package `pandas` to read those data from a csv-like file, be careful to pass numpy arrays as input (matrices), and note pandas Series (see [the example run](https://github.com/judithabk6/clonesig/blob/master/examples/full_CloneSig_run_with_simulated_data.ipynb)).

A basic running code is

``` python
import numpy as np
import pandas as pd
from clonesig.run_clonesig import get_MU, run_clonesig

# use pandas to read your data
sim_mutation_table = pd.read_csv("examples/example_data.csv")
with open('examples/purity.txt', 'r') as f:
    purity = float(f.read())

default_MU = get_MU()
est, lr, pval, new_inputMU, cst_est, future_sigs = run_clonesig(
    np.array(sim_mutation_table.trinucleotide),
    np.array(sim_mutation_table.var_counts),
    np.array(sim_mutation_table.var_counts + sim_mutation_table.ref_counts),
    np.array(sim_mutation_table.normal_cn),
    np.array(sim_mutation_table.total_cn),
    np.array(sim_mutation_table.total_cn - sim_mutation_table.major_cn),
    sim_object.purity, default_MU)
```

The different estimates for the parameter can then be accessed by 
``` python
# signature activity by clone
est.pi
# clone proportions
est.xi
# clone CCFs
est.phi
# beta-binomial overdispersion
est.tau

# attribution of mutations to clones
est.qun.argmax(axis=1)
# probabilistic version is est.qun

# attribution of mutations to signatures
np.arange(default_MU.shape[0])[est.rnus[np.arange(est.N), est.qun.argmax(axis=1), :].argmax(axis=1)]
# probabilistic version is est.rnus

# mutation multiplicity
est.vmnu[np.arange(est.N), est.qun.argmax(axis=1), :].argmax(axis=1) +1
# probabilistic version is est.vmnu
```



### Examples
Feel free to check this jupyter notebook with [a full example run](https://github.com/judithabk6/clonesig/blob/master/examples/full_CloneSig_run_with_simulated_data.ipynb)

The full pipeline (simulation, conversion of simulation to a dataframe, and launch of CloneSig fit from an input table) is detailed, and simple plots to visualize the results are shown.


## Get in touch
You can contact me through the issues for any concern or question regarding CloneSig.
