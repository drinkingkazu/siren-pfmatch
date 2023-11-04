import numpy as np
import torch

from pfmatch.datatypes import FlashMatch
from pfmatch.datatypes import FlashMatchInput

# import pytest fixtures; do not remove
from tests.fixtures import rng, torch_rng, num_pmt
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from pfmatch.datatypes import FlashMatch


def test_flashmatch_filter_loss_matrix(rng, torch_rng, num_pmt):
    # Create a FlashMatch object with some fake data
    num_qclusters = int(rng.random()*50)+2
    num_flashes = int(rng.random()*50)+2
    flashmatch = FlashMatch(num_qclusters, num_flashes, num_pmt)
    flashmatch.loss_matrix = torch.rand(size=(num_qclusters, num_flashes), generator=torch_rng)
    
    # Set a loss threshold 10-90%
    loss_threshold = rng.random()*0.8 + 0.1

    # Filter the loss matrix
    *__, LM = flashmatch.filter_loss_matrix(loss_threshold)
    
    # check that all values <=loss_threshold are in the filtered loss matrix
    assert np.all([x in LM for x in flashmatch.loss_matrix[flashmatch.loss_matrix<=loss_threshold].ravel()])
    
    # filter none out
    *__, LM = flashmatch.filter_loss_matrix(1.5)
    assert np.allclose(LM, flashmatch.loss_matrix)
    
    # filter all out
    *__, LM = flashmatch.filter_loss_matrix(-np.inf)
    assert len(LM) == 0

def test_local_match(rng, torch_rng, num_pmt):
    # Create a FlashMatch object with some fake data
    num_qclusters = int(rng.random()*50)+2
    num_flashes = int(rng.random()*50)+2
    flashmatch = FlashMatch(num_qclusters, num_flashes, num_pmt)
    flashmatch.loss_matrix = torch.rand(size=(num_qclusters, num_flashes), generator=torch_rng)

    # Run local_match
    flashmatch.local_match()

    # Check that the output has the correct shape
    assert flashmatch.tpc_ids.shape == (num_qclusters,)
    assert flashmatch.flash_ids.shape == (num_qclusters,)
    assert flashmatch.loss_v.shape == (num_qclusters,)
    assert flashmatch.reco_x_v.shape == (num_qclusters,)
    assert flashmatch.reco_pe_v.shape == (num_qclusters, num_pmt)

    # Check that the output has the correct values
    for i in range(num_qclusters):
        j = flashmatch.flash_ids[i]
        assert flashmatch.loss_v[i] == flashmatch.loss_matrix[i, j]
        assert flashmatch.reco_x_v[i] == flashmatch.reco_x_matrix[i, j]
        assert np.allclose(flashmatch.reco_pe_v[i], flashmatch.reco_pe_matrix[i, j])

    assert torch.allclose(flashmatch.tpc_ids,torch.arange(num_qclusters))
    assert torch.allclose(flashmatch.flash_ids,torch.argmin(flashmatch.loss_matrix, axis = 1))


def test_bipartite_match(rng, torch_rng, num_pmt):
    # Create a FlashMatch object with some fake data
    num_qclusters = int(rng.random()*50)+10
    num_flashes = int(rng.random()*50)+10
    flashmatch = FlashMatch(num_qclusters, num_flashes, num_pmt)
    flashmatch.loss_matrix = torch.rand(size=(num_qclusters, num_flashes), generator=torch_rng)
    
    row_idx, col_idx = linear_sum_assignment(flashmatch.loss_matrix.detach().numpy())

    # Run global_match
    loss_threshold = rng.random()*0.8 + 0.1
    flashmatch.bipartite_match(loss_threshold)
    assert np.allclose(flashmatch.tpc_ids, row_idx)
    assert np.allclose(flashmatch.flash_ids, col_idx)
    assert np.allclose(flashmatch.loss_v, flashmatch.loss_matrix[row_idx, col_idx])
    assert np.allclose(flashmatch.reco_x_v, flashmatch.reco_x_matrix[row_idx, col_idx])
    assert np.allclose(flashmatch.reco_pe_v, flashmatch.reco_pe_matrix[row_idx, col_idx])
    assert flashmatch.duration.shape == (num_flashes,)

"""TODO: tests for
global_match... how do I test this?
"""