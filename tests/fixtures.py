import os
from functools import reduce
from tempfile import NamedTemporaryFile

import h5py
import numpy as np
import pytest
import torch
import yaml
from pfmatch.datatypes import Flash, QCluster
from slar.nets import SirenVis

from photonlib import PhotonLib

GLOBAL_SEED = 123

@pytest.fixture
def rng():
    return  np.random.default_rng(GLOBAL_SEED)

@pytest.fixture
def torch_rng():
    return torch.Generator().manual_seed(GLOBAL_SEED)

def writable_temp_file(suffix=None):
    return NamedTemporaryFile('w', suffix=suffix, delete=False).name

@pytest.fixture
def fake_flashmatch_data(rng):
    nevents = int(rng.random()*95) + 5
    out = []
    for _ in range(nevents):
        ntracks = int(rng.random()*8)+2 # at least 2 tracks
        qcluster_v = []

        for _ in range(ntracks):
            qpt_v = rng.random(size=(int(rng.random()*100)+1,4))
            qcluster_v.append(QCluster(qpt_v))
        nflash = int(rng.random()*10)+1 
        flash_v = []
        for _ in range(nflash):
            
            pe_v = rng.random(180)*20_000
            flash_v.append(Flash(pe_v))
            
        out.append((qcluster_v,flash_v))
    return out

@pytest.fixture
def num_pmt():
    """can't change because of hardcoded values in the SIREN model"""
    return 180

@pytest.fixture
def fake_photon_library(rng, num_pmt):
    """
    h5 file has the following structure:
       - numvox: number of voxels in each dimension with shape (3,)
       - vis: 3D array of visibility values with shape (numvox, Npmt)
       - min: minimum coordinate of the active volume with shape (3,)
       - max: maximum coordinate of the active volume with shape (3,)
    """
    fake_h5 = writable_temp_file(suffix='.h5')
    with h5py.File(fake_h5, 'w') as f:
        f.create_dataset('numvox', shape=(3,), data=[10, 10, 10])
        total_numvox = np.prod(f['numvox'][:])

        # fake vis data -- random numbers uniformly distributed from 10^-7 to 10^-3
        vis = 10**rng.uniform(low=-7, high=-3, size=(total_numvox, num_pmt))
        f.create_dataset('vis', shape=(total_numvox, num_pmt), data=vis)
        
        # fake min/max data
        f.create_dataset('min', shape=(3,), data=[-400, -200, -1000])
        f.create_dataset('max', shape=(3,), data=[-35, 170, 1000])
    yield fake_h5
    os.remove(fake_h5)
    
@pytest.fixture
def test_data_dir():
    return reduce(os.path.join, [os.path.dirname(__file__), 'data'])

@pytest.fixture
def plib(fake_photon_library):
    return PhotonLib.load(fake_photon_library)

@pytest.fixture
def siren(test_data_dir, num_pmt):
    cfg=f'''
    model:
        network:
            in_features: 3
            hidden_features: 512
            hidden_layers: 5
            out_features: {num_pmt}
        ckpt_file: "{os.path.join(test_data_dir, 'siren.ckpt')}"
    '''
    cfg = yaml.safe_load(cfg)
    
    return SirenVis(cfg)