import setuptools

with open('README.md', 'r') as readme:
    long_description = readme.read()

setuptools.setup(
    name='clonesig',
    version='0.1',
    author='Judith Abécassis',
    # author_email='author@example.com',
    description='Clone Signature',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/judithabk6/clonesig',
    packages=setuptools.find_packages('.'),
    install_requires=[
        'pandas>=0.24.2,<0.25',
        'scikit-learn>=0.20.3,<0.21',
        'numpy>=1.16.2,<1.17',
    ],
    package_data={
        'clonesig': ['data/signatures_probabilities.txt',
                     'data/sigProfiler_exome_SBS_signatures.csv',
                     'data/sigProfiler_SBS_signatures_2019_05_22.csv',
                     'data/match_cancer_type_sig_v2.csv',
                     'data/match_cancer_type_sig_v3.csv',
                     'data/sigProfiler_SBS_signatures_2018_03_28.csv'],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
