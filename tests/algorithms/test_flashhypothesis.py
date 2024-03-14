from functools import reduce
from math import e
from tempfile import NamedTemporaryFile

import h5py
import numpy as np
import pytest
import torch
import os
import hashlib

import yaml

from pfmatch.algorithms import FlashHypothesis
from photonlib import PhotonLib
from slar.nets import SirenVis
from tests.fixtures import rng, num_pmt, writable_temp_file, plib, siren, fake_photon_library


def writable_temp_file(suffix=None):
    return NamedTemporaryFile('w', suffix=suffix, delete=False).name

""" ------------------------ test-specific fixtures ------------------------ """
@pytest.fixture
def track_good():
    return torch.tensor([[-60, 100, 300, 23000],
                         [-100, 0, 500, 18000],
                         [-300, -100, -500, 19000]], dtype=torch.float32)

@pytest.fixture
def track_bad_xmin():
    return torch.tensor([[-23982479, 100, 300, 23000],
                         [-200, 0, 500, 18000],
                         [-300, -100, -500, 19000]], dtype=torch.float32)
@pytest.fixture
def track_bad_xmax():
    return torch.tensor([[-60, 100, 300, 23000],
                         [+203984283, 0, 500, 18000],
                         [-300, -100, -500, 19000]], dtype=torch.float32)

@pytest.fixture
def flash_algo_plib(plib):
    return FlashHypothesis(dict(), plib)

@pytest.fixture
def flash_algo_siren(siren):
    return FlashHypothesis(dict(), siren)

@pytest.fixture
def flash_algos(flash_algo_plib, flash_algo_siren):
    return (flash_algo_plib, flash_algo_siren)


""" --------------------------------- tests -------------------------------- """

def test_FlashHypothesis_init(flash_algos):
    # this is tested by the fixture itself
    pass

def test_FlashHypothesis_dx(rng, flash_algos, track_good, track_bad_xmin, track_bad_xmax):
    for flash_algo in flash_algos:
        # this is tested by the fixture itself
        
        # test dx getter & setter
        old_dx = flash_algo.dx
        new_dx = rng.random()*100
        flash_algo.dx = new_dx
        assert not np.allclose(flash_algo.dx, old_dx), 'dx setter does not work'
        assert np.allclose(flash_algo.dx, new_dx), 'dx setter does not work'
        
        # test dx_range correct input
        dx_min, dx_max = flash_algo.dx_range(track_good)
        assert torch.allclose((track_good + dx_min)[:,0].min(), torch.tensor(flash_algo._xmin)), 'dx_min is not correct'
        assert torch.allclose((track_good + dx_max)[:,0].max(), torch.tensor(flash_algo._xmax)), 'dx_max is not correct'
        
        # test dx_range incorrect input
        with pytest.raises(AssertionError):
            flash_algo.dx_range(track_bad_xmin)
        with pytest.raises(AssertionError):
            flash_algo.dx_range(track_bad_xmax)
        
def test_FlashHypothesis_forward(num_pmt, flash_algos, track_good, track_bad_xmin, track_bad_xmax):
    for flash_algo in flash_algos:
        # test good tracks
        print('Using plib of type',type(flash_algo.plib))
        print('Meta',flash_algo.plib.meta)
        fwd = flash_algo(track_good)
        assert fwd.shape == (num_pmt,), 'forward output (p.e.) shape is not correct'
        assert torch.all(fwd >= 0), 'forward output (p.e.) is negative'
        
        # test bad tracks
        with pytest.raises(AssertionError):
            flash_algo(track_bad_xmin)
        with pytest.raises(AssertionError):
            flash_algo(track_bad_xmax)
            
def test_FlashHypothesis_backward(flash_algos, track_good):
    for flash_algo in flash_algos:
        # test good tracks
        fwd = flash_algo(track_good)

        fwd.sum().backward()
        assert flash_algo._dx.grad is not None, 'backward gradient is None'