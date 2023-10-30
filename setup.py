from skbuild import setup
import argparse

import io,os,sys
this_directory = os.path.abspath(os.path.dirname(__file__))
with io.open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="pfmatch",
    version="0.1",
    include_package_data=True,
    author=['Ka Vang (Patrick) Tsang, Carolyn Smith, Sam Young, Kazuhiro Terao'],
    description='Photon transportation physics models',
    license='MIT',
    keywords='Interface software for photon libraries in LArTPC experiments',
    scripts=['bin/pfmatch_generate_toymc.py'],
    packages=['pfmatch','pfmatch/algorithms','pfmatch/datatypes','pfmatch/apps','pfmatch/io','pfmatch/utils'],
    package_data={'pfmatch': ['config/*.yaml']},
    install_requires=[
        'numpy',
        'scikit-build',
        'torch',
        'h5py',
        'photonlib',
        'slar',
    ],
    long_description=long_description,
    long_description_content_type='text/markdown',
)
